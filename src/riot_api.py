"""Optional PUUID-based Riot collection. Live compatibility is not claimed."""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
import time

import numpy as np
import pandas as pd
import requests

from validation import SCHEMA, validate_snapshots

# Explicit default routing: NA ranked games. Collection is not Taiwan validation.
PLATFORM_HOST = "https://na1.api.riotgames.com"
REGION_HOST = "https://americas.api.riotgames.com"


class RiotClient:
    def __init__(self, api_key=None, sleep=1.3, max_retries=3):
        self.api_key = api_key or os.environ.get("RIOT_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set RIOT_API_KEY locally; never commit a key")
        if (not isinstance(sleep, (int, float)) or not np.isfinite(sleep) or sleep < 0 or
                type(max_retries) is not int or not 0 <= max_retries <= 10):
            raise ValueError("Nonnegative pacing and 0-10 retries are required")
        self.sleep, self.max_retries = sleep, max_retries
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})

    def _get(self, url, params=None):
        for attempt in range(self.max_retries + 1):
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                if attempt == self.max_retries:
                    response.raise_for_status()
                value = response.headers.get("Retry-After", "5")
                try:
                    wait = float(value)
                except ValueError:
                    wait = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                if not 0 <= wait <= 120:
                    raise RuntimeError("Retry-After outside bounded wait; retry collection later")
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(self.sleep)
            return response.json()
        raise RuntimeError("Retry budget exhausted")

    def challenger_league(self, queue="RANKED_SOLO_5x5"):
        return self._get(f"{PLATFORM_HOST}/lol/league/v4/challengerleagues/by-queue/{queue}")

    def grandmaster_league(self, queue="RANKED_SOLO_5x5"):
        return self._get(f"{PLATFORM_HOST}/lol/league/v4/grandmasterleagues/by-queue/{queue}")

    def match_ids(self, puuid, count=10, queue=420):
        return self._get(f"{REGION_HOST}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                         params={"count": count, "queue": queue})

    def match_timeline(self, match_id):
        return self._get(f"{REGION_HOST}/lol/match/v5/matches/{match_id}/timeline")

    def match_detail(self, match_id):
        return self._get(f"{REGION_HOST}/lol/match/v5/matches/{match_id}")


def _team_winner(detail):
    teams = detail["info"]["teams"]
    if len(teams) != 2 or {t["teamId"] for t in teams} != {100, 200}:
        raise ValueError("Two explicit team outcomes are required")
    if any(type(t.get("win")) is not bool for t in teams) or sum(t["win"] for t in teams) != 1:
        raise ValueError("Exactly one boolean team winner is required")
    return int(next(t["win"] for t in teams if t["teamId"] == 100))


def snapshot_from_timeline(timeline, detail, minute=15):
    """Use the last complete frame at/before cutoff; never fill missing players."""
    if not isinstance(minute, (int, float)) or not np.isfinite(minute) or not 1 <= minute <= 90:
        raise ValueError("Requested minute must be finite and between 1 and 90")
    info = detail["info"]
    match_id = detail["metadata"]["matchId"]
    if timeline["metadata"]["matchId"] != match_id:
        raise ValueError("Timeline and detail refer to different matches")
    if info.get("queueId") != 420:
        raise ValueError("Only ranked solo/duo queue420 is supported")
    start, end = info["gameStartTimestamp"], info["gameEndTimestamp"]
    cutoff = int(minute * 60000)
    if end - start <= cutoff:
        return None
    frames = timeline["info"]["frames"]
    eligible = [f for f in frames if 60000 <= f["timestamp"] <= cutoff]
    if not eligible:
        return None
    chosen = max(eligible, key=lambda f: f["timestamp"])
    timestamp = chosen["timestamp"]
    if cutoff - timestamp > 60000:
        raise ValueError("No recent complete frame at the requested cutoff")
    participants = info["participants"]
    side = {p["participantId"]: p["teamId"] for p in participants}
    if (len(participants) != 10 or len(side) != 10 or
            sum(team == 100 for team in side.values()) != 5 or
            sum(team == 200 for team in side.values()) != 5):
        raise ValueError("Exactly ten unique participants with five per side are required")
    player_frames = chosen["participantFrames"]
    gold = {100: 0, 200: 0}
    for pid, team in side.items():
        player = player_frames.get(str(pid))
        if player is None or "totalGold" not in player:
            raise ValueError("Complete gold observations are required for every participant")
        value = player["totalGold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Participant gold must be finite and nonnegative")
        gold[team] += value
    counts = {100: {"kills": 0, "towers": 0, "dragons": 0},
              200: {"kills": 0, "towers": 0, "dragons": 0}}
    for frame in frames:
        for event in frame.get("events", []):
            if event["timestamp"] > timestamp:
                continue
            kind = event.get("type")
            if kind == "CHAMPION_KILL":
                team = side.get(event.get("killerId"))
                if team in counts:
                    counts[team]["kills"] += 1
            elif kind == "BUILDING_KILL" and event.get("buildingType") == "TOWER_BUILDING":
                lost = event.get("teamId")
                if lost not in (100, 200):
                    raise ValueError("Tower event must identify the side losing the tower")
                counts[300 - lost]["towers"] += 1
            elif kind == "ELITE_MONSTER_KILL" and event.get("monsterType") == "DRAGON":
                team = event.get("killerTeamId") or side.get(event.get("killerId"))
                if team not in counts:
                    raise ValueError("Dragon event must identify a valid killing side")
                counts[team]["dragons"] += 1
    snapshot = {"match_id": match_id, "game_start_timestamp": start, "game_end_timestamp": end,
                "snapshot_timestamp": timestamp, "game_version": info["gameVersion"],
                "data_source": "riot_api", "game_minute": timestamp / 60000,
                "blue_gold": gold[100], "red_gold": gold[200], "blue_win": _team_winner(detail)}
    for name, team in [("blue", 100), ("red", 200)]:
        for objective, value in counts[team].items():
            snapshot[f"{name}_{objective}"] = value
    return validate_snapshots(pd.DataFrame([snapshot])).iloc[0].to_dict()


def collect_dataset(client, n_players=20, matches_per_player=5, minute=15):
    if n_players < 1 or not 1 <= matches_per_player <= 100:
        raise ValueError("Positive player count and 1-100 matches per player are required")
    entries = client.challenger_league()["entries"][:n_players]
    rows, seen = [], set()
    audit = {"players_attempted": len(entries), "player_failures": 0, "match_failures": 0,
             "duplicate_matches": 0, "short_or_missing_cutoff_matches": 0, "collected": 0,
             "live_api_validated_by_repository_tests": False}
    for entry in entries:
        try:
            puuid = entry["puuid"]
            ids = client.match_ids(puuid, count=matches_per_player)
        except (KeyError, requests.RequestException, ValueError, RuntimeError):
            audit["player_failures"] += 1
            continue
        for mid in ids:
            if mid in seen:
                audit["duplicate_matches"] += 1
                continue
            seen.add(mid)
            try:
                snapshot = snapshot_from_timeline(client.match_timeline(mid), client.match_detail(mid), minute)
                if snapshot is None:
                    audit["short_or_missing_cutoff_matches"] += 1
                else:
                    rows.append(snapshot)
            except (KeyError, TypeError, requests.RequestException, ValueError, RuntimeError):
                audit["match_failures"] += 1
    if not rows:
        raise ValueError(f"No valid snapshots collected; collection counts={audit}")
    result = validate_snapshots(pd.DataFrame(rows, columns=SCHEMA))
    audit["collected"] = len(result)
    result.attrs["collection"] = audit
    return result
