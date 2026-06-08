# Esports Match Win Predictor

A machine-learning model that predicts the winner of an esports match from a
mid-game snapshot (economy, objectives, kills, team form). An end-to-end
classification project — feature engineering, model training with
cross-validation, evaluation, and interpretability.

Built from a competitive-analysis perspective: the features mirror how a coach
or referee actually reads a game (resource leads, objective control, tempo),
informed by hands-on esports experience.

> ⚠️ Uses synthetic match data by default; plug in real data with `--csv`.

---

## 🧠 Problem

**Binary classification:** given the state of a match at minute *m*, will the
BLUE side win? Useful for live win-probability, post-game analysis, and scouting.

## 🛠️ Pipeline

1. **Feature engineering** (`features.py`) — convert raw snapshot columns into
   *relative* features (gold lead per minute, kill/tower/dragon differentials,
   form differential, and a gold×time interaction, since leads matter more late).
2. **Model** (`model.py`) — a scaler + classifier pipeline; logistic regression
   (interpretable) or random forest, evaluated with 5-fold cross-validated
   ROC-AUC.
3. **Evaluation** — accuracy, ROC-AUC, precision/recall, confusion matrix.
4. **Interpretability** — feature importances / coefficients, so the model's
   logic can be explained, not just trusted.

## 📊 Example output

```
Esports Win Predictor — logreg
3000 matches | 7 features | blue win rate 49.8%

Cross-validated ROC-AUC: 0.829 +/- 0.011
Held-out accuracy: 0.724   ROC-AUC: 0.799

Feature importance (most influential first):
tower_diff           0.950
gold_x_time          0.649
gold_diff_per_min    0.556
kill_diff            0.495
form_diff            0.382
```

The model's most important feature — tower differential — matches competitive
intuition: objective control is a strong win indicator. Interpretability like
this is what makes a model trustworthy.

## 🛡️ Good-practice notes

- **Relative features, not raw values** — competitive outcomes depend on the
  *lead*, so differentials generalize better than absolute numbers.
- **Cross-validation**, not a single train/test split, for a stable estimate.
- **Honest signal** — synthetic outcomes include realistic noise (upsets happen),
  so accuracy is strong but not suspiciously perfect.
- **Interpretability first** — coefficients/importances are reported so results
  can be explained to non-technical stakeholders.

## ⚙️ Install & run

```bash
pip install -r requirements.txt

python examples/run_prediction.py              # logistic regression
python examples/run_prediction.py --model rf   # random forest
python examples/run_prediction.py --csv your_matches.csv
```

```python
from features import build_features
from model import train, evaluate

X = build_features(matches_df)
model = train(X, matches_df["blue_win"])
print(evaluate(model, X_test, y_test))
```

## ✅ Validation

```bash
python tests/test_model.py
```

Confirms features are built correctly with no nulls, and the model achieves
ROC-AUC well above the 0.5 random baseline on held-out data.

## 🗂️ Project layout

```
src/
  features.py      # competitive-analysis feature engineering
  model.py         # pipeline, CV training, evaluation, importances
  data_loader.py   # CSV loader + realistic synthetic match generator
examples/
  run_prediction.py
tests/
  test_model.py
```

## 🔭 Roadmap
- [ ] Time-series win probability across a full match (per-minute curve)
- [ ] Real data ingestion (e.g., Riot API / match histories)
- [ ] Model comparison with gradient boosting (XGBoost / LightGBM) and SHAP

## 🌐 Real data from the Riot API

The project ships with a full pipeline to pull **real Challenger ranked matches**
and train on them — demonstrating real-world data engineering, not just modeling.

```bash
# 1. Get a free key at https://developer.riotgames.com (sign in with Riot)
# 2. Set it as an environment variable (never commit it):
export RIOT_API_KEY="RGAPI-..."        # macOS/Linux
#    setx RIOT_API_KEY "RGAPI-..."      # Windows

# 3. Fetch real data, then train on it:
python examples/fetch_and_train.py --players 20 --matches 5 --minute 15

# Reuse a prior pull without hitting the API again:
python examples/fetch_and_train.py --use-cache real_matches.csv
```

**Pipeline:** Challenger league → summoner PUUIDs → recent ranked match IDs →
match timeline (per-minute frames) → mid-game snapshot + winner label.

**Engineering details handled:**
- **API key from environment only** — never hard-coded or committed (`.gitignore`
  excludes keys and pulled data).
- **Rate-limit handling** — respects HTTP 429 `Retry-After` and paces requests to
  stay within the development key's limits.
- **Robust collection** — skips failed matches, de-duplicates, caches to CSV so
  you pull once.

> The same `features.py` / `model.py` work on both synthetic and real data, so you
> can develop offline and switch to live data by changing one command.

---
**Author:** Dennis Chen (陳雲皓) · [LinkedIn](https://linkedin.com/in/dennischen2507) · [GitHub](https://github.com/dennis20413)

*This project draws on my background as an esports referee (CTESA, EVO France) and competitive coach — domain knowledge directly shaped the feature design.*
