# Validation record

This upgrade was developed on 2026-10-28 with Python 3.11.6, NumPy 2.3.2, pandas 2.3.3, scikit-learn 1.9.0, requests 2.34.2, joblib 1.5.3 and pytest 9.2.0. All API payloads used by tests are invented mocks.

Executed checks cover numeric/schema errors, source mixing, duplicate snapshots and conflicting labels, equal-start-time groups, multiple snapshots of one match, outcome-availability purging, training-only feature fitting, held-out-label independence, local persistence and hash mismatch, feature order, one-class test metrics, actual frame/event cutoff, explicit participant sides, missing participants/gold/winners, terminal snapshots, bounded mocked 429 retries, PUUID collection and match deduplication.

The offline quickstart generates 3,000 synthetic matches and reports an earlier/later holdout against a training-period class-prior baseline. The six observed-state features are gold lead per minute, kill differential, tower differential, dragon differential, actual game minute and gold/time interaction. Recent form is excluded because no corresponding real historical feature was implemented.

| Executed command | Result |
|---|---|
| `python -m pytest -q` | 29 passed in 11.70 seconds; process exit code 0; no pytest warnings |
| `python examples/run_prediction.py` | Completed 3,000 synthetic matches, 2,250 training matches and 750 test matches; report and local artifacts written |
| Local artifact roundtrip and feature order | Passing; restored predictions match the original metrics; accidental corruption is rejected before loading |
| Changing only final-test labels | Passing; fitted model artifact digest and fixed threshold remain unchanged |
| API ingestion | Mocked tests only; no API key or outbound request used |

The restricted Windows execution environment printed a `pyreadline3` console-handle finalizer message after successful test completion. This occurred outside the test results; the pytest process still returned exit code 0.

No live Riot requests, real-match benchmark, cloud deployment, player-disjoint split, patch-isolated split or probability calibration have been validated. The workflow is supplied for GitHub Actions; local results alone do not establish its remote status.

Before changes, the complete Git history at `8fa5a5290ba7819a938240b710949c1c8f56bed5` was saved in a verified external Git bundle. The bundle is kept outside this repository.
