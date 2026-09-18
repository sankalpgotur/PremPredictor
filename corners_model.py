"""
Corner-count model for EPL fixtures.
--------------------------------------
Predicts home corners, away corners and total corners as DISTRIBUTIONS,
not just point estimates -- "most likely" for a count needs the full pmf.

Why Negative Binomial rather than Poisson:
  corners are overdispersed (var/mean ~ 1.67 per side), so Poisson would
  quote far too little uncertainty. NB adds a dispersion parameter.

Why total is modelled directly instead of home + away:
  home and away corners are negatively correlated (~ -0.31) -- teams trade
  territory -- so summing two independent fits would overstate the spread.

Signals used (all strictly pre-match):
  1. rolling form over the last FORM_WINDOW matches, either venue
  2. venue-specific form  (home team at home, away team away)
  3. head-to-head corners the last time these two met
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from epl_predictor import SEASONS, BASE

FORM_WINDOW = 10      # "last 10 games"; min_periods=1 so early season still works
VENUE_WINDOW = 10     # last 10 at that venue specifically
RAW_COLS = ["Date", "HomeTeam", "AwayTeam", "HC", "AC", "HS", "AS", "HST", "AST"]

TARGETS = ["HC", "AC", "TC"]          # home / away / total corners


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def load_corner_data(seasons=SEASONS):
    frames = []
    for s in seasons:
        try:
            df = pd.read_csv(BASE.format(season=s))
        except Exception:
            continue
        keep = [c for c in RAW_COLS if c in df.columns]
        df = df[keep].copy()
        df["Season"] = s
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date", "HC", "AC"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["TC"] = data["HC"] + data["AC"]
    return data


# ----------------------------------------------------------------------
# 2. FEATURES  (golden rule: shift(1) everywhere -- pre-match only)
# ----------------------------------------------------------------------
def _long(data):
    """One row per team per match, carrying that match's corner/shot stats."""
    home = pd.DataFrame({
        "match_id": data.index, "Date": data["Date"], "team": data["HomeTeam"],
        "opp": data["AwayTeam"], "is_home": 1,
        "cf": data["HC"], "ca": data["AC"],
        "sf": data.get("HS"), "sa": data.get("AS"),
    })
    away = pd.DataFrame({
        "match_id": data.index, "Date": data["Date"], "team": data["AwayTeam"],
        "opp": data["HomeTeam"], "is_home": 0,
        "cf": data["AC"], "ca": data["HC"],
        "sf": data.get("AS"), "sa": data.get("HS"),
    })
    return pd.concat([home, away], ignore_index=True).sort_values(["Date", "match_id"])


def _roll(g, col, window):
    """Mean of `col` over the previous `window` rows -- excludes the current row."""
    return g[col].shift(1).rolling(window, min_periods=1).mean()


def _team_rollups(long):
    """Overall and venue-specific rolling means for every team-match row."""
    long = long.sort_values(["team", "Date"]).copy()
    gb = long.groupby("team", group_keys=False)

    # overall form, any venue
    for c in ["cf", "ca", "sf", "sa"]:
        long[f"form_{c}"] = gb.apply(lambda g: _roll(g, c, FORM_WINDOW))

    # venue-specific: same team, same venue only
    long = long.sort_values(["team", "is_home", "Date"])
    gv = long.groupby(["team", "is_home"], group_keys=False)
    for c in ["cf", "ca"]:
        long[f"venue_{c}"] = gv.apply(lambda g: _roll(g, c, VENUE_WINDOW))

    return long.sort_values(["Date", "match_id"])


def _h2h(data):
    """Corners in the previous meeting of the same two teams.

    Oriented to the CURRENT fixture: h2h_hc is corners won by this match's
    home team last time out, whichever ground it was played on.
    """
    pair = data.apply(lambda r: tuple(sorted([r.HomeTeam, r.AwayTeam])), axis=1)
    out_hc, out_ac, out_tc = [], [], []
    last = {}                                    # pair -> (home_team, hc, ac)
    for idx, r in data.iterrows():
        key = pair.loc[idx]
        prev = last.get(key)
        if prev is None:
            out_hc.append(np.nan); out_ac.append(np.nan); out_tc.append(np.nan)
        else:
            p_home, p_hc, p_ac = prev
            # flip if the fixture was the other way round last time
            if p_home == r.HomeTeam:
                out_hc.append(p_hc); out_ac.append(p_ac)
            else:
                out_hc.append(p_ac); out_ac.append(p_hc)
            out_tc.append(p_hc + p_ac)
        last[key] = (r.HomeTeam, r.HC, r.AC)     # update AFTER recording
    return pd.DataFrame(
        {"h2h_hc": out_hc, "h2h_ac": out_ac, "h2h_tc": out_tc},
        index=data.index,
    )


FEATURE_COLS = [
    "h_form_cf", "h_form_ca", "h_form_sf", "h_form_sa",
    "a_form_cf", "a_form_ca", "a_form_sf", "a_form_sa",
    "h_venue_cf", "h_venue_ca", "a_venue_cf", "a_venue_ca",
    "h2h_hc", "h2h_ac",
]


def build_features(data):
    """Return (X, y_dict) with one row per match and no post-match information."""
    long = _team_rollups(_long(data))

    cols = ["form_cf", "form_ca", "form_sf", "form_sa", "venue_cf", "venue_ca"]
    h = long[long.is_home == 1].set_index("match_id")[cols].add_prefix("h_")
    a = long[long.is_home == 0].set_index("match_id")[cols].add_prefix("a_")

    X = data.join(h).join(a).join(_h2h(data))

    # No previous meeting (promoted sides): fall back to that fixture's own
    # rolling form rather than dropping the row or injecting a global constant.
    X["h2h_hc"] = X["h2h_hc"].fillna(X["h_form_cf"])
    X["h2h_ac"] = X["h2h_ac"].fillna(X["a_form_cf"])

    X = X.dropna(subset=FEATURE_COLS)
    y = {t: X[t] for t in TARGETS}
    return X, y


# ----------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------
def fit_nb(X, y):
    """Negative Binomial GLM (log link). alpha estimated from Pearson dispersion."""
    Xd = sm.add_constant(X[FEATURE_COLS], has_constant="add")
    poisson = sm.GLM(y, Xd, family=sm.families.Poisson()).fit()
    alpha = max(poisson.pearson_chi2 / poisson.df_resid - 1, 1e-6) / max(poisson.mu.mean(), 1e-9)
    nb = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    nb._alpha = alpha
    return nb


def predict_mu(model, X):
    return model.predict(sm.add_constant(X[FEATURE_COLS], has_constant="add"))


def nb_pmf(mu, alpha, kmax=30):
    """Negative Binomial pmf over 0..kmax for a single mean mu."""
    from scipy.stats import nbinom
    if alpha <= 1e-8:
        from scipy.stats import poisson
        return poisson.pmf(np.arange(kmax + 1), mu)
    n = 1.0 / alpha
    p = n / (n + mu)
    return nbinom.pmf(np.arange(kmax + 1), n, p)


# ----------------------------------------------------------------------
# 4. PERSIST  (coefficients as plain JSON -- no pickle, so no statsmodels
#    version coupling between training here and serving on Streamlit Cloud)
# ----------------------------------------------------------------------
CORNERS_PATH = "corners_model.json"


def train_and_save(path=CORNERS_PATH, seasons=SEASONS):
    import json
    data = load_corner_data(seasons)
    X, y = build_features(data)
    out = {"features": FEATURE_COLS, "targets": {}}
    for tgt in TARGETS:
        m = fit_nb(X, y[tgt])
        out["targets"][tgt] = {
            "params": {k: float(v) for k, v in m.params.items()},
            "alpha": float(m._alpha),
        }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def load_corners_model(path=CORNERS_PATH):
    import json
    with open(path) as f:
        return json.load(f)


def _mu_from_params(params, row):
    """mu = exp(b0 + sum(b_i * x_i)) -- the GLM log link, applied by hand."""
    z = params.get("const", 0.0)
    for k, v in params.items():
        if k != "const":
            z += v * float(row[k])
    return float(np.exp(z))


# ----------------------------------------------------------------------
# 5. FEATURES FOR AN UPCOMING (UNPLAYED) FIXTURE
# ----------------------------------------------------------------------
def fixture_features(data, home, away):
    """Build the same 14 features for a fixture that hasn't been played."""
    long = _long(data)
    out = {}

    for side, team, is_home in [("h", home, 1), ("a", away, 0)]:
        recent = long[long.team == team].tail(FORM_WINDOW)
        if recent.empty:
            raise ValueError(f"No match history for '{team}'.")
        for c in ["cf", "ca", "sf", "sa"]:
            out[f"{side}_form_{c}"] = recent[c].mean()
        # venue-specific: that team's last matches at THIS venue only
        at_venue = long[(long.team == team) & (long.is_home == is_home)].tail(VENUE_WINDOW)
        src = at_venue if not at_venue.empty else recent
        for c in ["cf", "ca"]:
            out[f"{side}_venue_{c}"] = src[c].mean()

    prev = data[(((data.HomeTeam == home) & (data.AwayTeam == away)) |
                 ((data.HomeTeam == away) & (data.AwayTeam == home)))]
    if prev.empty:
        out["h2h_hc"], out["h2h_ac"] = out["h_form_cf"], out["a_form_cf"]
        out["h2h_meetings"] = 0
    else:
        p = prev.iloc[-1]
        out["h2h_hc"] = p.HC if p.HomeTeam == home else p.AC
        out["h2h_ac"] = p.AC if p.HomeTeam == home else p.HC
        out["h2h_meetings"] = len(prev)
        out["h2h_last_date"] = p.Date
    return out


# ----------------------------------------------------------------------
# 6. PREDICT -- full distribution, not just a point estimate
# ----------------------------------------------------------------------
def predict_corners(model_json, data, home, away, lines=(8.5, 9.5, 10.5, 11.5, 12.5)):
    """Return mean / most-likely / interval / over-under probs for each target."""
    row = fixture_features(data, home, away)
    res = {"features": row, "targets": {}}

    for tgt in TARGETS:
        spec = model_json["targets"][tgt]
        mu = _mu_from_params(spec["params"], row)
        alpha = spec["alpha"]
        pmf = nb_pmf(mu, alpha, kmax=40)
        cdf = np.cumsum(pmf)

        lo = int(np.searchsorted(cdf, 0.10))      # 80% central interval
        hi = int(np.searchsorted(cdf, 0.90))
        entry = {
            "mean": round(mu, 2),
            "most_likely": int(np.argmax(pmf)),
            "interval80": (lo, hi),
            "pmf": pmf,
        }
        if tgt == "TC":
            entry["over_under"] = {
                ln: {"over": round(float(1 - cdf[int(np.floor(ln))]), 3),
                     "under": round(float(cdf[int(np.floor(ln))]), 3)}
                for ln in lines
            }
        res["targets"][tgt] = entry
    return res
