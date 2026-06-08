"""
riot_api.py
===========
A client for the Riot Games API that collects real League of Legends ranked
match data and converts it into the mid-game snapshot schema used by this
project's model.

You need a free API key from https://developer.riotgames.com (sign in with your
Riot account). Set it as an environment variable:

    export RIOT_API_KEY="RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"   # macOS/Linux
    setx  RIOT_API_KEY "RGAPI-..."                                     # Windows

NEVER hard-code your key or commit it to GitHub — it is tied to your account and
expires/rotates. This client reads it from the environment only.

Pipeline:
  high-elo league (Challenger/Grandmaster)
    -> summoner IDs -> PUUIDs
    -> recent ranked match IDs
    -> match timeline (per-minute frames)
    -> snapshot row at a chosen game minute + the eventual winner label

Rate limits: the development key allows ~20 req/s and 100 req/2 min. This client
sleeps between calls to stay within limits; large pulls take time by design.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd
import requests

# Routing: platform host (e.g. na1) for league/summoner; regional host
# (americas/asia/europe) for match-v5. Adjust to your region.
PLATFORM = "na1"
REGION = "americas"

PLATFORM_HOST = f"https://{PLATFORM}.api.riotgames.com"
REGION_HOST = f"https://{REGION}.api.riotgames.com"


class RiotClient:
    def __init__(self, api_key: str | None = None, sleep: float = 1.3):
        self.api_key = api_key or os.environ.get("RIOT_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set RIOT_API_KEY env var (get one at "
                "https://developer.riotgames.com)."
            )
        self.sleep = sleep                       # polite delay between calls
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET with basic rate-limit handling (honors 429 Retry-After)."""
        while True:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                time.sleep(wait + 1)
                continue
            resp.raise_for_status()
            time.sleep(self.sleep)
            return resp.json()

    # --- discovery -----------------------------------------------------------
    def challenger_league(self, queue: str = "RANKED_SOLO_5x5") -> dict:
        url = f"{PLATFORM_HOST}/lol/league/v4/challengerleagues/by-queue/{queue}"
        return self._get(url)

    def grandmaster_league(self, queue: str = "RANKED_SOLO_5x5") -> dict:
        url = f"{PLATFORM_HOST}/lol/league/v4/grandmasterleagues/by-queue/{queue}"
        return self._get(url)

    def puuid_by_summoner_id(self, summoner_id: str) -> str:
        url = f"{PLATFORM_HOST}/lol/summoner/v4/summoners/{summoner_id}"
        return self._get(url)["puuid"]

    def match_ids(self, puuid: str, count: int = 10, queue: int = 420) -> list[str]:
        # queue 420 = ranked solo/duo
        url = f"{REGION_HOST}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        return self._get(url, params={"count": count, "queue": queue})

    def match_timeline(self, match_id: str) -> dict:
        url = f"{REGION_HOST}/lol/match/v5/matches/{match_id}/timeline"
        return self._get(url)

    def match_detail(self, match_id: str) -> dict:
        url = f"{REGION_HOST}/lol/match/v5/matches/{match_id}"
        return self._get(url)


# --- snapshot extraction -----------------------------------------------------
def _team_winner(match_detail: dict) -> int:
    """Return 1 if team 100 (blue) won, else 0."""
    for team in match_detail["info"]["teams"]:
        if team["teamId"] == 100:
            return int(team["win"])
    return 0


def snapshot_from_timeline(timeline: dict, detail: dict, minute: int = 15) -> dict | None:
    """
    Build one snapshot row at `minute` from a match timeline + detail.
    Participants 1-5 are blue (team 100), 6-10 are red (team 200).
    """
    frames = timeline["info"]["frames"]
    if len(frames) <= minute:
        return None
    frame = frames[minute]["participantFrames"]

    def side_sum(pids, key):
        total = 0
        for pid in pids:
            pf = frame.get(str(pid), {})
            if key == "gold":
                total += pf.get("totalGold", 0)
            elif key == "level":
                total += pf.get("level", 0)
        return total

    blue_ids, red_ids = range(1, 6), range(6, 11)
    blue_gold = side_sum(blue_ids, "gold")
    red_gold = side_sum(red_ids, "gold")

    # Kills / objectives up to `minute` from timeline events.
    blue_kills = red_kills = 0
    blue_towers = red_towers = 0
    blue_dragons = red_dragons = 0
    for f in frames[: minute + 1]:
        for ev in f.get("events", []):
            if ev.get("type") == "CHAMPION_KILL":
                killer = ev.get("killerId", 0)
                if 1 <= killer <= 5:
                    blue_kills += 1
                elif 6 <= killer <= 10:
                    red_kills += 1
            elif ev.get("type") == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
                # killerTeamId 100 means blue destroyed a red tower.
                if ev.get("teamId") == 200:      # team that LOST the building
                    blue_towers += 1
                else:
                    red_towers += 1
            elif ev.get("type") == "ELITE_MONSTER_KILL" and ev.get("monsterType") == "DRAGON":
                if 1 <= ev.get("killerId", 0) <= 5:
                    blue_dragons += 1
                else:
                    red_dragons += 1

    return {
        "blue_recent_winrate": 0.5,   # placeholder; fill via summoner history if desired
        "red_recent_winrate": 0.5,
        "blue_gold": blue_gold,
        "red_gold": red_gold,
        "blue_kills": blue_kills,
        "red_kills": red_kills,
        "blue_towers": blue_towers,
        "red_towers": red_towers,
        "blue_dragons": blue_dragons,
        "red_dragons": red_dragons,
        "game_minute": minute,
        "blue_win": _team_winner(detail),
    }


def collect_dataset(
    client: RiotClient,
    n_players: int = 20,
    matches_per_player: int = 5,
    minute: int = 15,
) -> pd.DataFrame:
    """
    Walk the Challenger ladder -> players -> matches -> snapshots.
    Returns a DataFrame ready for features.build_features().

    Note: this makes many API calls and is rate-limited; a few hundred matches
    can take several minutes on a development key.
    """
    league = client.challenger_league()
    entries = league["entries"][:n_players]

    seen_matches: set[str] = set()
    rows: list[dict] = []

    for i, entry in enumerate(entries):
        try:
            puuid = client.puuid_by_summoner_id(entry["summonerId"])
            match_ids = client.match_ids(puuid, count=matches_per_player)
        except Exception as exc:
            print(f"  [skip player {i}] {exc}")
            continue

        for mid in match_ids:
            if mid in seen_matches:
                continue
            seen_matches.add(mid)
            try:
                tl = client.match_timeline(mid)
                detail = client.match_detail(mid)
                snap = snapshot_from_timeline(tl, detail, minute=minute)
                if snap:
                    rows.append(snap)
                    print(f"  collected {len(rows)} snapshots", end="\r")
            except Exception as exc:
                print(f"  [skip match {mid}] {exc}")
                continue

    print()
    return pd.DataFrame(rows)
