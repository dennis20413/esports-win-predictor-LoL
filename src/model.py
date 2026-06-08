"""
model.py
========
Train and evaluate a win-prediction classifier.

We use a pipeline (scaler + model) and report the metrics a DS interviewer
expects for a binary classifier: accuracy, ROC-AUC, precision/recall, plus a
calibration view and feature importances for interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix,
)


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_names: list
    cv_auc_mean: float
    cv_auc_std: float


def make_model(kind: str = "logreg") -> Pipeline:
    """Build a scaler + classifier pipeline."""
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
        return Pipeline([("model", clf)])          # trees don't need scaling
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def train(X: pd.DataFrame, y: pd.Series, kind: str = "logreg") -> TrainedModel:
    """Fit a model and report cross-validated ROC-AUC."""
    pipe = make_model(kind)
    cv = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    pipe.fit(X, y)
    return TrainedModel(
        pipeline=pipe,
        feature_names=list(X.columns),
        cv_auc_mean=float(cv.mean()),
        cv_auc_std=float(cv.std()),
    )


def evaluate(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate on a held-out set and return headline metrics."""
    proba = model.pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "report": classification_report(y, pred, digits=3),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def feature_importance(model: TrainedModel) -> pd.Series:
    """Return per-feature importance / coefficient for interpretability."""
    step = model.pipeline.named_steps["model"]
    if hasattr(step, "coef_"):
        vals = step.coef_[0]
    elif hasattr(step, "feature_importances_"):
        vals = step.feature_importances_
    else:  # pragma: no cover
        vals = np.zeros(len(model.feature_names))
    return pd.Series(vals, index=model.feature_names).sort_values(key=abs, ascending=False)
