"""Validated CSV loading and explicitly simulated offline match snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import SCHEMA, validate_snapshots


def load_csv(path: str) -> pd.DataFrame:
    return validate_snapshots(pd.read_csv(path))


def synthetic_matches(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Invented matches for testing only; never evidence of real-game accuracy."""
    if n < 2:
        raise ValueError("At least two synthetic matches are required")
    rng = np.random.default_rng(seed)
    minute = rng.integers(8, 40, n)
    lead = rng.normal(0, 1500, n)
    blue_kills, red_kills = rng.poisson(minute / 3), rng.poisson(minute / 3)
    blue_towers, red_towers = rng.integers(0, 11, n), rng.integers(0, 11, n)
    blue_dragons, red_dragons = rng.integers(0, 5, n), rng.integers(0, 5, n)
    advantage = (.0008 * lead + .25 * (blue_towers - red_towers)
                 + .20 * (blue_dragons - red_dragons) + .15 * (blue_kills - red_kills))
    probability = 1 / (1 + np.exp(-advantage / 1.2))
    starts = 1700000000000 + np.arange(n, dtype=np.int64) * 3600000
    frame = pd.DataFrame({
        "match_id": [f"SYNTHETIC_{i:06d}" for i in range(n)],
        "game_start_timestamp": starts,
        "game_end_timestamp": starts + (minute + 5) * 60000,
        "snapshot_timestamp": minute * 60000,
        "game_version": "synthetic-v2", "data_source": "synthetic",
        "blue_gold": 500 * minute + lead / 2, "red_gold": 500 * minute - lead / 2,
        "blue_kills": blue_kills, "red_kills": red_kills,
        "blue_towers": blue_towers, "red_towers": red_towers,
        "blue_dragons": blue_dragons, "red_dragons": red_dragons,
        "game_minute": minute, "blue_win": (rng.random(n) < probability).astype(int),
    })
    result = validate_snapshots(frame[SCHEMA])
    result.attrs["simulation"] = {"generator": "synthetic-v2", "seed": seed, "matches": n}
    return result
