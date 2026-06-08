"""
test_model.py
=============
Validates feature engineering and that the model learns real signal.

Run:  python tests/test_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.model_selection import train_test_split

from data_loader import synthetic_matches
from features import build_features, feature_names
from model import train, evaluate


def test_features_built_correctly():
    df = synthetic_matches(n=200, seed=1)
    X = build_features(df)
    assert list(X.columns) == feature_names()
    assert not X.isnull().any().any()


def test_model_beats_baseline():
    """ROC-AUC should be clearly better than 0.5 (random guessing)."""
    df = synthetic_matches(n=3000, seed=42)
    X = build_features(df)
    y = df["blue_win"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = train(Xtr, ytr)
    metrics = evaluate(model, Xte, yte)
    assert metrics["roc_auc"] > 0.7


def test_rf_also_works():
    df = synthetic_matches(n=2000, seed=7)
    X = build_features(df)
    y = df["blue_win"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    model = train(Xtr, ytr, kind="rf")
    metrics = evaluate(model, Xte, yte)
    assert metrics["accuracy"] > 0.6


if __name__ == "__main__":
    test_features_built_correctly()
    test_model_beats_baseline()
    test_rf_also_works()
    print("All tests passed - features valid and model learns real signal.")
