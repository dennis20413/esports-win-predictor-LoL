"""
data_loader.py
==============
Load match-snapshot data from a CSV, or generate a synthetic dataset whose
outcomes follow a realistic (but noisy) relationship to in-game leads — so the
model has genuine, learnable signal without being trivially separable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import RAW_COLUMNS


def load_csv(path: str) -> pd.DataFrame:
    """Load a match dataset with RAW_COLUMNS plus a 'blue_win' label."""
    df = pd.read_csv(path)
    return df


def synthetic_matches(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate match snapshots. The true win probability depends on gold lead,
    objectives, form, and game time — with realistic noise (upsets happen), so
    a well-built model reaches strong but not perfect accuracy.
    """
    rng = np.random.default_rng(seed)

    game_minute = rng.integers(8, 40, n)
    blue_form = rng.uniform(0.3, 0.7, n)
    red_form = rng.uniform(0.3, 0.7, n)

    # Gold scales with game time; leads widen later.
    base_gold = 500 * game_minute
    gold_spread = rng.normal(0, 1500, n)
    blue_gold = base_gold + gold_spread / 2
    red_gold = base_gold - gold_spread / 2

    blue_kills = rng.poisson(game_minute / 3, n)
    red_kills = rng.poisson(game_minute / 3, n)
    blue_towers = rng.integers(0, 11, n)
    red_towers = rng.integers(0, 11, n)
    blue_dragons = rng.integers(0, 5, n)
    red_dragons = rng.integers(0, 5, n)

    # True latent advantage -> win probability via logistic link.
    advantage = (
        0.0008 * (blue_gold - red_gold)
        + 0.25 * (blue_towers - red_towers)
        + 0.20 * (blue_dragons - red_dragons)
        + 0.15 * (blue_kills - red_kills)
        + 3.0 * (blue_form - red_form)
    )
    # Scale and add noise (upsets).
    prob = 1 / (1 + np.exp(-advantage / 1.2))
    blue_win = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "blue_recent_winrate": blue_form,
        "red_recent_winrate": red_form,
        "blue_gold": blue_gold,
        "red_gold": red_gold,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "blue_towers": blue_towers,
        "red_towers": red_towers,
        "blue_dragons": blue_dragons,
        "red_dragons": red_dragons,
        "game_minute": game_minute,
        "blue_win": blue_win,
    })
    return df[RAW_COLUMNS + ["blue_win"]]
