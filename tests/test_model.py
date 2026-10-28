"""Offline synthetic and mocked-API regression checks; no real-game benchmark."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
import requests

from data_loader import synthetic_matches
from experiment import run_experiment
from features import build_features, feature_names
from model import evaluate, load_trusted_local_model, train
from riot_api import RiotClient, collect_dataset, snapshot_from_timeline
from validation import chronological_split, validate_snapshots


def test_features_and_sources_align():
    frame = synthetic_matches(100)
    features = build_features(frame)
    assert list(features.columns) == feature_names()
    assert "form_diff" not in features.columns
    assert not any("recent_winrate" in name for name in frame.columns)
    assert np.isfinite(features).all().all()
    assert frame.data_source.eq("synthetic").all()


@pytest.mark.parametrize("column,value", [("blue_gold", np.nan), ("red_gold", np.inf),
                                         ("blue_kills", -1), ("blue_kills", 1.5),
                                         ("blue_win", 2), ("game_minute", 0)])
def test_invalid_numeric_data_is_rejected(column, value):
    frame = synthetic_matches(30)
    if isinstance(value, float):
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_snapshots(frame)


def test_missing_metadata_and_mixed_sources_are_rejected():
    frame = synthetic_matches(30)
    with pytest.raises(ValueError):
        validate_snapshots(frame.drop(columns="match_id"))
    frame.loc[0, "data_source"] = "riot_api"
    with pytest.raises(ValueError):
        validate_snapshots(frame)


def test_identical_duplicates_deduplicate_but_conflicts_fail():
    frame = synthetic_matches(30)
    assert len(validate_snapshots(pd.concat([frame, frame.iloc[:1]]))) == 30
    for column in ["blue_win", "blue_gold"]:
        duplicate = frame.iloc[:1].copy()
        duplicate[column] = 1 - duplicate[column] if column == "blue_win" else duplicate[column] + 1
        with pytest.raises(ValueError):
            validate_snapshots(pd.concat([frame, duplicate]))


def test_multiple_snapshots_and_equal_start_times_stay_together():
    frame = synthetic_matches(80)
    extra = frame.iloc[[12]].copy()
    extra["snapshot_timestamp"] -= 60000
    extra["game_minute"] -= 1
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.loc[61, ["game_start_timestamp", "game_end_timestamp"]] = frame.loc[60, ["game_start_timestamp", "game_end_timestamp"]].to_numpy()
    training, test, metadata = chronological_split(frame)
    assert not set(training.match_id) & set(test.match_id)
    assert training.game_start_timestamp.max() < test.game_start_timestamp.min()
    assert training.game_end_timestamp.max() < test.game_start_timestamp.min()
    assert set(training[training.match_id.eq("SYNTHETIC_000012")].snapshot_timestamp) == set(
        frame[frame.match_id.eq("SYNTHETIC_000012")].snapshot_timestamp)
    assert metadata["train_matches"] + metadata["test_matches"] == 80


def test_outcomes_not_available_at_cutoff_are_purged():
    frame = synthetic_matches(40)
    _, _, split = chronological_split(frame)
    frame.loc[0, "game_end_timestamp"] = split["cutoff_timestamp"] + 1
    training, test, metadata = chronological_split(frame)
    assert "SYNTHETIC_000000" not in set(training.match_id) | set(test.match_id)
    assert metadata["purged_matches"] == 1


def test_model_metrics_baseline_and_trusted_local_roundtrip(tmp_path):
    frame = synthetic_matches(300, seed=17)
    report = run_experiment(frame, output=tmp_path / "artifacts", reports=tmp_path / "reports")
    assert report["data_source"] == "synthetic" and report["real_data_evaluated"] is False
    assert report["training_prior_baseline"]["roc_auc"] == .5
    assert report["test_metrics"]["log_loss"] > 0
    training, test, _ = chronological_split(frame)
    restored = load_trusted_local_model(tmp_path / "artifacts")
    assert np.allclose(restored.pipeline.named_steps["scaler"].mean_, build_features(training).mean())
    assert evaluate(restored, build_features(test), test.blue_win)["brier"] == pytest.approx(report["test_metrics"]["brier"])
    with pytest.raises(ValueError):
        evaluate(restored, build_features(test)[feature_names()[::-1]], test.blue_win)
    path = tmp_path / "artifacts/model.joblib"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError):
        load_trusted_local_model(tmp_path / "artifacts")


def test_test_labels_do_not_affect_fitted_pipeline(tmp_path):
    frame = synthetic_matches(300, seed=21)
    _, _, split = chronological_split(frame)
    first = run_experiment(frame, output=tmp_path / "one", reports=tmp_path / "r1")
    changed = frame.copy()
    mask = changed.game_start_timestamp >= split["cutoff_timestamp"]
    changed.loc[mask, "blue_win"] = 1 - changed.loc[mask, "blue_win"]
    second = run_experiment(changed, output=tmp_path / "two", reports=tmp_path / "r2")
    assert first["model_sha256"] == second["model_sha256"]
    assert first["threshold"] == second["threshold"] == .5


def test_random_forest_path_and_single_class_test_are_supported():
    frame = synthetic_matches(150)
    training, test, _ = chronological_split(frame)
    fitted = train(build_features(training), training.blue_win, kind="rf")
    result = evaluate(fitted, build_features(test), pd.Series(np.zeros(len(test), dtype=int)))
    assert result["roc_auc"] is None
    assert len(result["confusion_matrix"]) == 2


def _payloads():
    players = [{"participantId": i, "teamId": 100 if i < 200 else 200}
               for i in list(range(101, 106)) + list(range(201, 206))]
    detail = {"metadata": {"matchId": "MOCK_MATCH"}, "info": {
        "queueId": 420, "gameStartTimestamp": 1700000000000, "gameEndTimestamp": 1700001800000,
        "gameVersion": "mock-patch", "participants": players,
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}]}}
    frames = []
    for timestamp, gold in [(0, 0), (60000, 100), (120000, 200), (180000, 9999)]:
        frames.append({"timestamp": timestamp,
                       "participantFrames": {str(p["participantId"]): {"totalGold": gold} for p in players},
                       "events": []})
    frames[1]["events"] = [{"timestamp": 50000, "type": "CHAMPION_KILL", "killerId": 101}]
    frames[2]["events"] = [
        {"timestamp": 110000, "type": "BUILDING_KILL", "buildingType": "TOWER_BUILDING", "teamId": 200},
        {"timestamp": 115000, "type": "ELITE_MONSTER_KILL", "monsterType": "DRAGON", "killerId": 0, "killerTeamId": 100}]
    frames[3]["events"] = [{"timestamp": 130000, "type": "CHAMPION_KILL", "killerId": 201}]
    return {"metadata": {"matchId": "MOCK_MATCH"}, "info": {"frames": frames}}, detail


def test_snapshot_uses_actual_cutoff_and_explicit_participant_sides():
    timeline, detail = _payloads()
    row = snapshot_from_timeline(timeline, detail, minute=2.5)
    assert row["snapshot_timestamp"] == 120000 and row["game_minute"] == 2
    assert row["blue_gold"] == row["red_gold"] == 1000
    assert row["blue_kills"] == 1 and row["red_kills"] == 0
    assert row["blue_towers"] == row["blue_dragons"] == 1
    assert row["match_id"] == "MOCK_MATCH" and row["data_source"] == "riot_api"


def test_future_frames_and_events_do_not_change_snapshot():
    timeline, detail = _payloads()
    original = snapshot_from_timeline(timeline, detail, 2.5)
    timeline["info"]["frames"][-1]["participantFrames"]["101"]["totalGold"] = 1234567
    timeline["info"]["frames"][-1]["events"] *= 10
    assert snapshot_from_timeline(timeline, detail, 2.5) == original


@pytest.mark.parametrize("bad", ["participant", "gold", "winner", "mismatch"])
def test_missing_observations_and_outcomes_never_default(bad):
    timeline, detail = _payloads()
    if bad == "participant":
        del timeline["info"]["frames"][2]["participantFrames"]["101"]
    elif bad == "gold":
        timeline["info"]["frames"][2]["participantFrames"]["101"]["totalGold"] = np.nan
    elif bad == "winner":
        detail["info"]["teams"] = []
    else:
        timeline["metadata"]["matchId"] = "OTHER"
    with pytest.raises(ValueError):
        snapshot_from_timeline(timeline, detail, 2.5)


@pytest.mark.parametrize("duration", [60000, 150000])
def test_games_ending_at_or_before_requested_cutoff_are_skipped(duration):
    timeline, detail = _payloads()
    detail["info"]["gameEndTimestamp"] = detail["info"]["gameStartTimestamp"] + duration
    assert snapshot_from_timeline(timeline, detail, 2.5) is None


def test_terminal_snapshot_is_rejected_by_csv_contract():
    frame = synthetic_matches(30)
    frame.loc[0, "game_end_timestamp"] = frame.loc[0, "game_start_timestamp"] + frame.loc[0, "snapshot_timestamp"]
    with pytest.raises(ValueError):
        validate_snapshots(frame)


@pytest.mark.parametrize("options", [{"sleep": np.nan}, {"sleep": -1}, {"max_retries": .5}, {"max_retries": True}])
def test_invalid_client_retry_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        RiotClient(api_key="offline-test-token", **options)


def _response(status, retry=None):
    response = requests.Response()
    response.status_code, response.url = status, "https://mock.invalid/test"
    response._content = b'{"ok": true}'
    if retry is not None:
        response.headers["Retry-After"] = retry
    return response


def test_429_retry_after_and_bounded_attempts(monkeypatch):
    client = RiotClient(api_key="offline-test-token", sleep=0, max_retries=1)
    waiting = []
    monkeypatch.setattr("riot_api.time.sleep", waiting.append)
    responses = iter([_response(429, "2.5"), _response(200)])
    monkeypatch.setattr(client.session, "get", lambda *a, **k: next(responses))
    assert client._get("https://mock.invalid/test") == {"ok": True}
    assert waiting == [2.5, 0]
    responses = iter([_response(429, "0"), _response(429, "0")])
    monkeypatch.setattr(client.session, "get", lambda *a, **k: next(responses))
    with pytest.raises(requests.HTTPError):
        client._get("https://mock.invalid/test")


def test_collector_uses_puuid_and_deduplicates_mocked_matches():
    timeline, detail = _payloads()

    class FakeClient:
        def challenger_league(self):
            return {"entries": [{"puuid": "offline-player"}, {"puuid": "offline-player"}]}
        def match_ids(self, puuid, count):
            assert puuid == "offline-player"
            return ["MOCK_MATCH", "MOCK_MATCH"]
        def match_timeline(self, mid):
            return copy.deepcopy(timeline)
        def match_detail(self, mid):
            return copy.deepcopy(detail)

    result = collect_dataset(FakeClient(), n_players=2, matches_per_player=2, minute=2.5)
    assert len(result) == 1
    assert result.attrs["collection"]["duplicate_matches"] == 3
    assert result.attrs["collection"]["live_api_validated_by_repository_tests"] is False
