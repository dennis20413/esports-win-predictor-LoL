# Esports Match Win Predictor

[![offline tests](https://github.com/dennis20413/esports-win-predictor-LoL/actions/workflows/ci.yml/badge.svg)](https://github.com/dennis20413/esports-win-predictor-LoL/actions/workflows/ci.yml)

A mid-game snapshot classification project with explicit data provenance, independent-match chronological evaluation, a probability baseline, and reproducible local model artifacts. Domain features reflect economy, kills and objective control.

**Current evidence:** offline tests and the checked-in evaluation use **synthetic data only**. Riot collection is implemented and tested with mocked payloads; it has **not** been validated against the live API in this release. There are no claimed real-match results, live predictions, calibrated probabilities or deployed service.

## Run the offline experiment

Tested with Python 3.11.6; dependencies are pinned to the versions used locally.

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python examples/run_prediction.py
```

The last command generates 3,000 invented matches (`synthetic-v2`, seed 42), fits a logistic regression on the earlier 2,250, and evaluates the last 750. It saves `artifacts/model.joblib`, `artifacts/metadata.json` and a clearly named [synthetic evaluation report](reports/synthetic-evaluation.json). Random forest is available with `--model rf`; no model is automatically chosen from final-test results.

| Synthetic holdout only | Accuracy | ROC AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7307 | 0.8133 | 0.5363 | 0.1789 |
| Training-period class prior | 0.4893 | 0.5000 | 0.6949 | 0.2509 |

These are simulation results used to exercise the pipeline. They do not estimate professional, ranked or current-patch match accuracy and should not be quoted as real-game performance on a resume.

## What changed in the evaluation

- A shared schema requires match identity, start/end times, observation time, game version and data source before feature engineering.
- Identical `(match_id, snapshot_timestamp)` rows are deduplicated. Conflicting snapshots or outcome labels are rejected.
- The holdout boundary is defined by match-start timestamp groups. Equal start timestamps stay together; every snapshot of one match stays in one partition.
- Training matches whose outcome was not available **before** the test cutoff are purged. Snapshot timestamps must be strictly before match end.
- The scaler fits only the training partition. The probability threshold is fixed at 0.5; random folds and the former unverified CV headline were removed.
- A prior fitted only on training labels is reported alongside accuracy, ROC AUC, log loss, Brier score and confusion counts. A one-class test partition produces `null` AUC rather than a misleading number.
- Current features omit recent-form inputs: live collection did not implement historical form, so the old constant-0.5 placeholders are no longer mixed with a simulated form signal.

The split targets approximately 25% of distinct start-time groups for test; the row fraction may differ when groups have different sizes. This does not ensure player isolation or patch isolation. Repeated players may occur on either side. Multiple observations from a test match receive row-weighted metrics; prefer one fixed observation point for match-level comparisons.

## Optional Riot data collection

Set `RIOT_API_KEY` in your local environment from the [Riot Developer Portal](https://developer.riotgames.com/). Do not put keys in code, issue text, screenshots, CSVs or model reports.

```sh
python examples/fetch_and_train.py --players 20 --matches 5 --minute 15
python examples/fetch_and_train.py --use-cache data/real_matches.csv
```

Collection uses league-entry **PUUIDs** directly, Match-v5 details and timelines, unique match IDs, bounded 429 retries with `Retry-After`, request timeouts, and a collection-count audit file. Defaults target NA (`na1` / `americas`) ranked solo/duo queue 420; this is not Taiwan data validation. Current endpoint schemas and authorization remain subject to the [official Riot API documentation](https://developer.riotgames.com/apis).

The snapshot extractor selects the last complete frame at or before the requested timestamp, refuses stale frames more than one minute earlier, maps participants through their explicit team IDs, and counts only events available by the selected frame's time. Future frames and their events cannot affect the observation. Missing gold, participants, team outcomes or inconsistent match IDs are rejected rather than filled with zero or a loss label. Games ending at or before the requested observation are excluded.

The collector records failed players, failed matches, repeated match IDs and excluded observation points. It raises an error when no valid data is collected. Tests simulate these API behaviors without a key or network access. Live schema compatibility, rate-limit behavior and real-data model quality remain unverified here.

## CSV contract and compatibility

| Field | Meaning |
|---|---|
| `match_id` | Nonempty match identifier; all snapshots share stable match metadata and label |
| `game_start_timestamp`, `game_end_timestamp` | Integer epoch milliseconds; the outcome is available only after the end |
| `snapshot_timestamp` | Integer milliseconds elapsed since game start |
| `game_minute` | Actual selected frame time, equal to `snapshot_timestamp / 60000` |
| `game_version` | Explicit source patch/version, or `synthetic-v2` for the simulator |
| `data_source` | Exactly one source per dataset: `synthetic` or `riot_api` |
| `blue_gold`, `red_gold` | Finite, nonnegative observed team gold |
| `blue_kills`, `red_kills`, `blue_towers`, `red_towers`, `blue_dragons`, `red_dragons` | Nonnegative integer counts observed by the snapshot |
| `blue_win` | Binary eventual outcome; never an input feature |

Use a validated external file with `python examples/run_prediction.py --csv your_matches.csv`. **Old CSVs without identity and time metadata are deliberately rejected.** Regenerate them from source or recover genuine metadata; do not invent timestamps or relabel synthetic data as real.

The normalized, validated CSV content receives a SHA-256 digest in the report. `data_source=riot_api` records the supplied data's provenance claim; the tag is not proof that this repository independently verified its origin.

## Local model artifacts

The artifact manifest includes feature order, model hash, source digest, split boundaries, library versions and measured evaluation. Predict in a fresh local Python process:

```python
import sys
sys.path.insert(0, "src")
from data_loader import synthetic_matches
from features import build_features
from model import load_trusted_local_model

model = load_trusted_local_model("artifacts")
probability = model.pipeline.predict_proba(build_features(synthetic_matches(2)))[:, 1]
```

**Only load joblib artifacts you generated and trust.** Joblib/pickle loading can execute code. The hash catches accidental changes relative to the manifest; it does not authenticate an untrusted artifact or make uploaded models safe. Artifacts, collected datasets and keys are ignored by Git.

## Portfolio use

This project supports discussion of time-aware validation, identity-level leakage, outcome availability, feature parity, failure handling and model reproducibility. For a real-match accuracy claim, first collect authorized data, inspect collection failures, choose an evaluation protocol before looking at test outcomes, and retain the resulting provenance and report. Synthetic test scores alone are not that evidence.

中文：这个专案目前展示的是可检查的数据工程与验证流程。模拟资料用于离线测试；Riot 采集程式尚未实测，不应写成已完成真实比赛验证。面试可以解释同一场比赛为何不能跨训练／测试、为什么要排除测试时尚未结束的训练比赛，以及为什么移除了未实现的近期胜率特征。

See [validation notes](docs/validation.md) for executed checks. CI runs offline tests and a synthetic quickstart; it does not collect live Riot data.
