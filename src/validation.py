"""Shared snapshot contract and match-isolated chronological evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import RAW_COLUMNS

METADATA = ["match_id", "game_start_timestamp", "game_end_timestamp",
            "snapshot_timestamp", "game_version", "data_source"]
SCHEMA = METADATA + RAW_COLUMNS + ["blue_win"]


def validate_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate before feature engineering; deduplicate only identical snapshots."""
    missing = set(SCHEMA) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Nonempty snapshot data with required columns is needed; missing={sorted(missing)}")
    frame = frame[SCHEMA].copy()
    if frame.isna().any().any():
        raise ValueError("Snapshots must not contain missing values")
    for column in ["match_id", "game_version", "data_source"]:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")
    if not frame.data_source.isin(["synthetic", "riot_api"]).all() or frame.data_source.nunique() != 1:
        raise ValueError("Use one explicit source: synthetic or riot_api; do not mix sources")
    numeric = RAW_COLUMNS + ["game_start_timestamp", "game_end_timestamp", "snapshot_timestamp", "blue_win"]
    for column in numeric:
        if (not pd.api.types.is_numeric_dtype(frame[column]) or
                pd.api.types.is_bool_dtype(frame[column]) or
                not np.isfinite(frame[column].to_numpy(dtype=float)).all()):
            raise ValueError(f"{column} must contain finite numeric values")
    counts = ["blue_kills", "red_kills", "blue_towers", "red_towers", "blue_dragons", "red_dragons"]
    integers = counts + ["game_start_timestamp", "game_end_timestamp", "snapshot_timestamp", "blue_win"]
    if not all((frame[column] == np.floor(frame[column])).all() for column in integers):
        raise ValueError("Counts, timestamps and labels must be integers")
    if (frame[RAW_COLUMNS] < 0).any().any() or not frame.blue_win.isin([0, 1]).all():
        raise ValueError("Snapshot values must be nonnegative and labels binary")
    if not frame.game_minute.between(1, 90).all():
        raise ValueError("Snapshot minute must lie between 1 and 90")
    if not np.allclose(frame.game_minute * 60000, frame.snapshot_timestamp, atol=1e-4, rtol=0):
        raise ValueError("Minute and elapsed snapshot timestamp must agree")
    if ((frame.game_start_timestamp <= 0).any() or
            (frame.game_end_timestamp <= frame.game_start_timestamp + frame.snapshot_timestamp).any()):
        raise ValueError("Snapshot must be strictly before match end")
    # Every snapshot of a match must have one identity, chronology, patch and outcome.
    stable = ["game_start_timestamp", "game_end_timestamp", "game_version", "data_source", "blue_win"]
    if (frame.groupby("match_id")[stable].nunique() > 1).any().any():
        raise ValueError("Conflicting metadata or labels within a match")
    keys = ["match_id", "snapshot_timestamp"]
    for _, group in frame.groupby(keys, sort=False):
        if len(group.drop_duplicates()) > 1:
            raise ValueError("Conflicting feature values for duplicate match snapshots")
    result = frame.drop_duplicates(keys).sort_values(
        ["game_start_timestamp", "match_id", "snapshot_timestamp"], kind="stable").reset_index(drop=True)
    result.attrs.update(frame.attrs)
    return result


def chronological_split(frame: pd.DataFrame, test_fraction: float = .25):
    """Separate start-time groups; purge training labels unavailable at cutoff."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    frame = validate_snapshots(frame)
    times = np.sort(frame.game_start_timestamp.unique())
    if len(times) < 2:
        raise ValueError("At least two distinct match start times are required")
    position = min(len(times) - 1, max(1, int(len(times) * (1 - test_fraction))))
    cutoff = int(times[position])
    train = frame[(frame.game_start_timestamp < cutoff) & (frame.game_end_timestamp < cutoff)].copy()
    test = frame[frame.game_start_timestamp >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split is empty after purging overlapping match outcomes")
    if set(train.match_id) & set(test.match_id):
        raise AssertionError("Same match crossed evaluation partitions")
    metadata = {"protocol": "Ordered match-start groups; equal timestamps stay together; training outcomes must precede test cutoff",
                "cutoff_timestamp": cutoff,
                "train_matches": int(train.match_id.nunique()), "test_matches": int(test.match_id.nunique()),
                "train_rows": len(train), "test_rows": len(test),
                "purged_matches": int(frame.match_id.nunique() - train.match_id.nunique() - test.match_id.nunique()),
                "latest_train_label_timestamp": int(train.game_end_timestamp.max()),
                "first_test_start_timestamp": int(test.game_start_timestamp.min())}
    return train.reset_index(drop=True), test.reset_index(drop=True), metadata
