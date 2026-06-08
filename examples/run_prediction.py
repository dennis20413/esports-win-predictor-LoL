"""
run_prediction.py
=================
Train and evaluate the esports win-prediction model end to end: load data,
engineer features, train (with cross-validation), evaluate on a held-out set,
and show feature importances.

Usage:
    python run_prediction.py                      # synthetic data
    python run_prediction.py --csv matches.csv    # your own match data
    python run_prediction.py --model rf           # random forest instead of logreg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.model_selection import train_test_split

from data_loader import load_csv, synthetic_matches
from features import build_features
from model import train, evaluate, feature_importance


def main() -> None:
    p = argparse.ArgumentParser(description="Esports win prediction.")
    p.add_argument("--csv", help="Path to a match CSV (else synthetic data).")
    p.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    args = p.parse_args()

    df = load_csv(args.csv) if args.csv else synthetic_matches()
    X = build_features(df)
    y = df["blue_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nEsports Win Predictor — {args.model}")
    print(f"{len(df)} matches  |  {X.shape[1]} features  |  blue win rate {y.mean():.1%}\n")

    model = train(X_train, y_train, kind=args.model)
    print(f"Cross-validated ROC-AUC: {model.cv_auc_mean:.3f} +/- {model.cv_auc_std:.3f}")

    metrics = evaluate(model, X_test, y_test)
    print(f"\nHeld-out test set:")
    print(f"  accuracy : {metrics['accuracy']:.3f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"\nClassification report:\n{metrics['report']}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    print("\nFeature importance (most influential first):")
    print(feature_importance(model).round(3).to_string())

    print("\nNote: synthetic data unless --csv given; illustrative.")


if __name__ == "__main__":
    main()
