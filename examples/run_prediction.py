"""Offline synthetic smoke experiment, or validated metadata-preserving CSV."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_csv, synthetic_matches
from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Validated snapshot CSV including match identity, chronology and source")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    frame = load_csv(args.csv) if args.csv else synthetic_matches()
    report = run_experiment(frame, args.model, args.out, args.reports)
    print(json.dumps({key: report[key] for key in ["data_source", "warning", "split", "test_metrics",
                                                "training_prior_baseline"]}, indent=2))


if __name__ == "__main__":
    main()
