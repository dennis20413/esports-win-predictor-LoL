"""One frozen chronological holdout with a training-period class-prior baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn

from features import build_features
from model import evaluate, feature_importance, probability_metrics, save_model, train
from validation import chronological_split, validate_snapshots


def run_experiment(frame, kind="logreg", output="artifacts", reports="reports"):
    frame = validate_snapshots(frame)
    train_frame, test_frame, split = chronological_split(frame)
    model = train(build_features(train_frame), train_frame.blue_win, kind)
    result = evaluate(model, build_features(test_frame), test_frame.blue_win)
    prior = float(train_frame.blue_win.mean())
    baseline = probability_metrics(test_frame.blue_win, np.full(len(test_frame), prior))
    source = str(frame.data_source.iloc[0])
    report = {"data_source": source, "real_data_evaluated": source == "riot_api",
              "warning": "Synthetic results only validate this simulation and code path" if source == "synthetic"
                         else "Local supplied Riot-data evaluation; not external validation or evidence of current live API compatibility",
              "source_sha256": hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
              "split": split, "model": kind, "threshold": .5,
              "test_metrics": result, "training_prior_baseline": baseline,
              "feature_importance": feature_importance(model).to_dict(),
              "game_versions": sorted(frame.game_version.unique().tolist()),
              "simulation": frame.attrs.get("simulation"),
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
              "limitations": ["No player-disjoint isolation; recurring players may cross the cutoff",
                              "Multiple snapshots of one test match receive row-weighted metrics",
                              "Timestamp order does not isolate game patches",
                              "Only games lasting to the requested snapshot can enter the sample",
                              "No calibrated-probability or production-quality claim"]}
    manifest = save_model(model, output, {"source": source, "source_sha256": report["source_sha256"],
                                         "split": split, "environment": report["environment"],
                                         "test_metrics": result})
    report["model_sha256"] = manifest["model_sha256"]
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    name = "synthetic-evaluation.json" if source == "synthetic" else "local-riot-evaluation.json"
    (reports / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
