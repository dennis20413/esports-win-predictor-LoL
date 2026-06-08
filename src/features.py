"""
features.py
===========
Feature engineering for esports match win prediction.

Given pre-match information about two teams (recent form, economy, objectives,
and in-game state at a snapshot), build a feature vector that a classifier can
use to predict whether the BLUE side wins.

Domain note: features are chosen from a competitive-analysis perspective — the
same lens used when scouting opponents (form, resource leads, objective control,
and tempo), not arbitrary columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the model expects (raw match-snapshot schema).
RAW_COLUMNS = [
    "blue_recent_winrate", "red_recent_winrate",
    "blue_gold", "red_gold",
    "blue_kills", "red_kills",
    "blue_towers", "red_towers",
    "blue_dragons", "red_dragons",
    "game_minute",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw snapshot columns into model features. We emphasize *differences*
    and *ratios* between the two sides, because in competitive games the relative
    lead matters more than absolute values.
    """
    f = pd.DataFrame(index=df.index)

    # Form differential.
    f["form_diff"] = df["blue_recent_winrate"] - df["red_recent_winrate"]

    # Economic lead (gold), normalized by game time so early/late leads compare.
    minute = df["game_minute"].clip(lower=1)
    f["gold_diff_per_min"] = (df["blue_gold"] - df["red_gold"]) / minute

    # Combat lead.
    f["kill_diff"] = df["blue_kills"] - df["red_kills"]

    # Objective control.
    f["tower_diff"] = df["blue_towers"] - df["red_towers"]
    f["dragon_diff"] = df["blue_dragons"] - df["red_dragons"]

    # Tempo: is the game early or late? Leads late are more decisive.
    f["game_minute"] = df["game_minute"]

    # Interaction: a gold lead late in the game is worth more.
    f["gold_x_time"] = (df["blue_gold"] - df["red_gold"]) * df["game_minute"] / 1000.0

    return f


def feature_names() -> list[str]:
    return ["form_diff", "gold_diff_per_min", "kill_diff", "tower_diff",
            "dragon_diff", "game_minute", "gold_x_time"]
