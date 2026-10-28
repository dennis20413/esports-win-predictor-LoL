"""Optional Riot collection; requires a key and has not been live-validated here."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_csv
from experiment import run_experiment
from riot_api import RiotClient, collect_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=20)
    parser.add_argument("--matches", type=int, default=5)
    parser.add_argument("--minute", type=int, default=15)
    parser.add_argument("--out", default="data/real_matches.csv")
    parser.add_argument("--use-cache")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    args = parser.parse_args()
    if args.use_cache:
        frame = load_csv(args.use_cache)
    else:
        frame = collect_dataset(RiotClient(), args.players, args.matches, args.minute)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        path.with_suffix(".collection.json").write_text(json.dumps(frame.attrs.get("collection", {}), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_experiment(frame, args.model), indent=2))


if __name__ == "__main__":
    main()
