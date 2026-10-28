"""Training, honest probability evaluation, and trusted-local persistence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_names: list
    # Kept for import compatibility; ordinary random-fold CV is no longer run.
    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None


def make_model(kind: str = "logreg") -> Pipeline:
    if kind == "rf":
        return Pipeline([("model", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))])
    if kind != "logreg":
        raise ValueError("Model kind must be logreg or rf")
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, random_state=42))])


def train(X: pd.DataFrame, y: pd.Series, kind: str = "logreg") -> TrainedModel:
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data needs both binary target classes")
    pipe = make_model(kind)
    pipe.fit(X, y)
    return TrainedModel(pipe, list(X.columns))


def probability_metrics(y, probability) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    prediction = (probability >= .5).astype(int)
    return {"rows": len(y), "positive_rate": float(np.mean(y)),
            "accuracy": float(accuracy_score(y, prediction)),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)),
            "report": classification_report(y, prediction, labels=[0, 1], digits=3, zero_division=0),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def evaluate(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> dict:
    if list(X.columns) != model.feature_names:
        raise ValueError("Prediction feature order does not match the trained contract")
    return probability_metrics(y, model.pipeline.predict_proba(X)[:, 1])


def feature_importance(model: TrainedModel) -> pd.Series:
    step = model.pipeline.named_steps["model"]
    values = step.coef_[0] if hasattr(step, "coef_") else step.feature_importances_
    return pd.Series(values, index=model.feature_names).sort_values(key=abs, ascending=False)


def save_model(model: TrainedModel, folder: str | Path, metadata: dict):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.joblib"
    joblib.dump(model, path)
    manifest = {**metadata, "artifact_format": "trusted-local-joblib-v1",
                "feature_order": model.feature_names,
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "metadata.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_trusted_local_model(folder: str | Path) -> TrainedModel:
    """Only load artifacts you generated and trust: joblib can execute code."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    path = folder / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model artifact differs from its manifest")
    model = joblib.load(path)  # Trusted local artifacts only; never user uploads.
    if not isinstance(model, TrainedModel) or model.feature_names != metadata["feature_order"]:
        raise ValueError("Saved model feature contract differs from manifest")
    return model
