"""
Premier League Match Outcome Predictor
---------------------------------------
Predicts Home win / Draw / Away win for EPL fixtures.

Data source : football-data.co.uk (free, per-season CSVs, includes shots)
Stack       : pandas -> scikit-learn (split) -> xgboost (classifier) -> matplotlib

Two jobs live here:
  * train.py   trains on history and SAVES model.json + features.json
  * app.py     loads the saved model and serves predictions (no retraining)
"""

import datetime as _dt
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
N_SEASONS = 8        # how many seasons of history to train on


def _season_codes(n=N_SEASONS, today=None):
    """football-data season codes, oldest -> current.

    A season starting in year Y is coded YY(Y+1). The EPL runs Aug-May, so
    from July onward the current season is the one starting this year.
    Derived from the date rather than hardcoded, so the current season is
    picked up automatically instead of going stale every August.
    """
    today = today or _dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}"
            for y in range(start - n + 1, start + 1)]


SEASONS = _season_codes()
BASE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
USECOLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
           "HS", "AS", "HST", "AST"]
WINDOW = 5           # "recent form" = last 5 matches
MODEL_PATH = "model.json"
FEATURES_PATH = "features.json"
CLASS_NAMES = ["Home win", "Draw", "Away win"]   # target 0 / 1 / 2

# Each rolling feature: output name -> (source column, aggregation).
# build_features (training) and team_current_form (prediction) BOTH read this,
# so the two paths can never drift apart.
FEATURE_SPEC = {
    "f_gf":      ("gf",    "mean"),   # Goals Scored   (attack output)
    "f_ga":      ("ga",    "mean"),   # Goals Conceded (defensive record)
    "f_shots":   ("shots", "mean"),   # Shots
    "f_sot":     ("sot",   "mean"),   # Shots on target
    "f_winrate": ("win",   "mean"),   # NEW: win rate over last 5
    "f_form":    ("pts",   "sum"),    # Recent Form (points in last 5)
}


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(seasons=SEASONS):
    frames = []
    for s in seasons:
        try:
            df = pd.read_csv(BASE.format(season=s))
        except Exception:
            continue  # season not published yet, skip
        df = df[[c for c in USECOLS if c in df.columns]].copy()
        df["Season"] = s
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date", "FTR"]).sort_values("Date").reset_index(drop=True)
    return data


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
# Golden rule: a match's features may only use games that happened BEFORE it.
def _team_long(data):
    """Explode each match into two team-rows carrying that match's raw stats."""
    home = pd.DataFrame({
        "Date": data["Date"], "match_id": data.index, "team": data["HomeTeam"],
        "gf": data["FTHG"], "ga": data["FTAG"],
        "shots": data.get("HS"), "sot": data.get("HST"), "is_home": 1,
    })
    away = pd.DataFrame({
        "Date": data["Date"], "match_id": data.index, "team": data["AwayTeam"],
        "gf": data["FTAG"], "ga": data["FTHG"],
        "shots": data.get("AS"), "sot": data.get("AST"), "is_home": 0,
    })
    home["pts"] = np.select([data["FTR"] == "H", data["FTR"] == "D"], [3, 1], 0)
    away["pts"] = np.select([data["FTR"] == "A", data["FTR"] == "D"], [3, 1], 0)
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["pts"] == 3).astype(int)
    return long.sort_values("Date")


def build_features(data):
    long = _team_long(data)

    for name, (col, agg) in FEATURE_SPEC.items():
        def _roll(s, agg=agg):
            r = s.shift(1).rolling(WINDOW, min_periods=1)   # shift(1) = pre-match only
            return r.mean() if agg == "mean" else r.sum()
        long[name] = long.groupby("team")[col].transform(_roll)

    feat = list(FEATURE_SPEC)
    home_feats = long[long.is_home == 1].set_index("match_id")[feat].add_prefix("home_")
    away_feats = long[long.is_home == 0].set_index("match_id")[feat].add_prefix("away_")

    X = data.join(home_feats).join(away_feats)
    feature_cols = list(home_feats.columns) + list(away_feats.columns)
    X = X.dropna(subset=feature_cols)
    y = X["FTR"].map({"H": 0, "D": 1, "A": 2})
    return X[feature_cols], y, feature_cols


def team_current_form(long, team):
    """Latest form for a team = its last WINDOW completed matches (for prediction)."""
    rows = long[long.team == team].tail(WINDOW)
    if rows.empty:
        return None
    out = {}
    for name, (col, agg) in FEATURE_SPEC.items():
        out[name] = rows[col].mean() if agg == "mean" else rows[col].sum()
    return out


# ----------------------------------------------------------------------
# 3. TRAIN + 4. EVALUATE + SAVE
# ----------------------------------------------------------------------
def _new_model():
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )


def evaluate(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    m = _new_model().fit(X_tr, y_tr)
    preds = m.predict(X_te)
    print(f"Accuracy: {accuracy_score(y_te, preds):.3f}  (home-win baseline ~0.43)\n")
    print(classification_report(y_te, preds, target_names=CLASS_NAMES))
    cm = confusion_matrix(y_te, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Home", "Draw", "Away"]).plot(cmap="Blues")
    import matplotlib.pyplot as plt
    plt.title("EPL Outcome Predictor -- Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion_matrix.png")


def train_and_save(X, y, feature_cols):
    """Refit on ALL matches (not just the train split) and persist for serving."""
    model = _new_model().fit(X, y)
    model.save_model(MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved {MODEL_PATH} and {FEATURES_PATH}")
    return model


def load_model():
    model = _new_model()
    model.load_model(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    return model, feature_cols


# ----------------------------------------------------------------------
# 5. PREDICT A FIXTURE
# ----------------------------------------------------------------------
def predict_fixture(model, feature_cols, data, home, away):
    """Return {'Home win': p, 'Draw': p, 'Away win': p} for a matchup."""
    long = _team_long(data)
    h, a = team_current_form(long, home), team_current_form(long, away)
    if h is None:
        raise ValueError(f"No recent data for '{home}'. Check spelling / team is in the data.")
    if a is None:
        raise ValueError(f"No recent data for '{away}'. Check spelling / team is in the data.")
    row = {f"home_{k}": v for k, v in h.items()}
    row.update({f"away_{k}": v for k, v in a.items()})
    X_row = pd.DataFrame([row])[feature_cols]         # enforce training column order
    proba = model.predict_proba(X_row)[0]
    return {name: round(float(p), 3) for name, p in zip(CLASS_NAMES, proba)}


if __name__ == "__main__":
    data = load_data()
    print(f"Loaded {len(data)} matches across {data['Season'].nunique()} seasons.\n")
    X, y, feature_cols = build_features(data)
    print(f"Built {X.shape[1]} features for {X.shape[0]} matches.\n")
    evaluate(X, y)
    model = train_and_save(X, y, feature_cols)
    print("\nExample:", predict_fixture(model, feature_cols, data, "Chelsea", "Arsenal"))
