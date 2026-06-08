"""
fetch_and_train.py
===================
Pull REAL Challenger ranked match data from the Riot API, save it to CSV, then
train and evaluate the win-prediction model on it.

Prerequisites:
    1. A Riot API key from https://developer.riotgames.com
    2. export RIOT_API_KEY="RGAPI-..."   (do NOT commit your key)
    3. pip install -r requirements.txt

Usage:
    python fetch_and_train.py --players 20 --matches 5 --minute 15
    python fetch_and_train.py --use-cache real_matches.csv   # reuse a prior pull

The fetch step is rate-limited and can take several minutes. The data is cached
to CSV so you only pull once.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from features import build_features
from model import train, evaluate, feature_importance


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch real Riot data and train.")
    p.add_argument("--players", type=int, default=20, help="Challenger players to sample.")
    p.add_argument("--matches", type=int, default=5, help="Matches per player.")
    p.add_argument("--minute", type=int, default=15, help="Snapshot minute.")
    p.add_argument("--out", default="real_matches.csv", help="Where to cache the data.")
    p.add_argument("--use-cache", help="Skip fetching; load this CSV instead.")
    p.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    args = p.parse_args()

    if args.use_cache:
        df = pd.read_csv(args.use_cache)
        print(f"Loaded {len(df)} cached matches from {args.use_cache}")
    else:
        from riot_api import RiotClient, collect_dataset
        if not os.environ.get("RIOT_API_KEY"):
            print("ERROR: set RIOT_API_KEY first (see script header).")
            sys.exit(1)
        client = RiotClient()
        print("Fetching real Challenger match data (this can take a few minutes)...")
        df = collect_dataset(client, n_players=args.players,
                             matches_per_player=args.matches, minute=args.minute)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} matches to {args.out}")

    if len(df) < 50:
        print(f"Only {len(df)} matches — pull more for a reliable model.")

    X = build_features(df)
    y = df["blue_win"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = train(Xtr, ytr, kind=args.model)
    metrics = evaluate(model, Xte, yte)

    print(f"\nReal-data model ({args.model}) @ minute {args.minute}")
    print(f"  CV ROC-AUC : {model.cv_auc_mean:.3f} +/- {model.cv_auc_std:.3f}")
    print(f"  test acc   : {metrics['accuracy']:.3f}")
    print(f"  test AUC   : {metrics['roc_auc']:.3f}")
    print("\nFeature importance:")
    print(feature_importance(model).round(3).to_string())


if __name__ == "__main__":
    main()
