"""
Generic count model for EPL match markets.
-------------------------------------------
One Negative Binomial GLM per side per market. Corners was the first of
these; the same machinery covers goals, shots on target, yellows and fouls,
since all five are overdispersed counts driven by the same rolling signals.

Per market we fit three targets: home count, away count, and the total.
Total is fitted DIRECTLY rather than summed -- home and away counts are
negatively correlated for territorial markets (corners ~ -0.31), so adding
two independent fits would overstate the spread.

Everything is strictly pre-match: rolling form over the last N games,
same-venue record, last head-to-head, and (for cards/fouls) the referee's
own rate relative to the league.
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom

from leagues import LEAGUES, SEASONS, DEFAULT_LEAGUE, url as league_url, has_referee

FORM_WINDOW = 10
VENUE_WINDOW = 10
MODEL_PATH = "count_model.json"

# market key -> (home column, away column, display name, feature tag)
MARKETS = {
    "goals":   ("FTHG", "FTAG", "Total goals",       "FTHG / FTAG"),
    "sot":     ("HST",  "AST",  "Shots on target",   "HST / AST"),
    "corners": ("HC",   "AC",   "Corners",           "HC / AC"),
    "yellows": ("HY",   "AY",   "Yellow cards",      "HY / AY"),
    "fouls":   ("HF",   "AF",   "Fouls",             "HF / AF"),
}
# markets whose rate is materially set by the official
REF_SENSITIVE = ("yellows", "fouls")

RAW_COLS = ["Date", "Time", "HomeTeam", "AwayTeam", "Referee",
            "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
            "HS", "AS", "HST", "AST", "HC", "AC",
            "HF", "AF", "HY", "AY", "HR", "AR"]


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def load_match_data(league=DEFAULT_LEAGUE, seasons=SEASONS):
    frames = []
    for s in seasons:
        try:
            df = pd.read_csv(league_url(league, s))
        except Exception:
            continue
        df = df[[c for c in RAW_COLS if c in df.columns]].copy()
        df["Season"] = s
        frames.append(df)
    if not frames:
        raise ValueError(f"No data for league {league!r}.")
    data = pd.concat(frames, ignore_index=True)
    if "Referee" not in data.columns:
        data["Referee"] = pd.NA
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    need = ["Date"] + [c for m in MARKETS.values() for c in m[:2]]
    data = data.dropna(subset=[c for c in need if c in data.columns])
    data = data.sort_values("Date").reset_index(drop=True)
    for k, (hc, ac, _, _) in MARKETS.items():
        data[f"tot_{k}"] = data[hc] + data[ac]
    return data


# ----------------------------------------------------------------------
# REFEREE RATES
# ----------------------------------------------------------------------
def referee_table(data):
    """Per-official rates, shrunk toward the league mean (few-match officials)."""
    d = data.dropna(subset=["Referee"]).copy()
    d["y_tot"] = d.HY + d.AY
    d["f_tot"] = d.HF + d.AF
    d["r_tot"] = d.HR + d.AR
    lg = {"y": d.y_tot.mean(), "f": d.f_tot.mean(), "r": d.r_tot.mean()}
    g = d.groupby("Referee").agg(
        games=("y_tot", "size"), y=("y_tot", "mean"),
        f=("f_tot", "mean"), r=("r_tot", "mean"),
    )
    k = 20.0                       # shrinkage strength, in matches
    for c, prior in [("y", lg["y"]), ("f", lg["f"]), ("r", lg["r"])]:
        g[f"{c}_sh"] = (g["games"] * g[c] + k * prior) / (g["games"] + k)
    g["y_factor"] = g["y_sh"] / lg["y"]
    g["f_factor"] = g["f_sh"] / lg["f"]
    return g, lg


# ----------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------
def _long(data):
    """One row per team per match with that match's for/against counts."""
    rows = {}
    for side, is_home in [("h", 1), ("a", 0)]:
        d = {
            "match_id": data.index, "Date": data["Date"], "is_home": is_home,
            "team": data["HomeTeam"] if is_home else data["AwayTeam"],
            "opp": data["AwayTeam"] if is_home else data["HomeTeam"],
        }
        for k, (hc, ac, _, _) in MARKETS.items():
            own, oth = (hc, ac) if is_home else (ac, hc)
            d[f"{k}_f"] = data[own]
            d[f"{k}_a"] = data[oth]
        d["shots_f"] = data["HS"] if is_home else data["AS"]
        d["shots_a"] = data["AS"] if is_home else data["HS"]
        rows[side] = pd.DataFrame(d)
    return pd.concat(rows.values(), ignore_index=True).sort_values(["Date", "match_id"])


STAT_COLS = [f"{k}_{s}" for k in MARKETS for s in ("f", "a")] + ["shots_f", "shots_a"]


def _rollups(long):
    long = long.sort_values(["team", "Date"]).copy()
    g = long.groupby("team", group_keys=False)
    for c in STAT_COLS:
        long[f"form_{c}"] = g[c].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    long = long.sort_values(["team", "is_home", "Date"])
    gv = long.groupby(["team", "is_home"], group_keys=False)
    for c in STAT_COLS:
        long[f"venue_{c}"] = gv[c].transform(
            lambda s: s.shift(1).rolling(VENUE_WINDOW, min_periods=1).mean())
    return long.sort_values(["Date", "match_id"])


def _h2h(data, market):
    """Counts in the previous meeting of these two teams, oriented to this fixture."""
    hc, ac = MARKETS[market][:2]
    out_h, out_a, last = [], [], {}
    for idx, r in data.iterrows():
        key = (min(r.HomeTeam, r.AwayTeam), max(r.HomeTeam, r.AwayTeam))
        prev = last.get(key)
        if prev is None:
            out_h.append(np.nan); out_a.append(np.nan)
        else:
            p_home, p_h, p_a = prev
            if p_home == r.HomeTeam:
                out_h.append(p_h); out_a.append(p_a)
            else:
                out_h.append(p_a); out_a.append(p_h)
        last[key] = (r.HomeTeam, r[hc], r[ac])
    return pd.Series(out_h, index=data.index), pd.Series(out_a, index=data.index)


def market_features(market, league=DEFAULT_LEAGUE):
    """Feature names used by one market's models.

    The referee term is dropped for leagues with no Referee column
    (football-data publishes it for the Premier League only).
    """
    return [
        f"h_form_{market}_f", f"h_form_{market}_a",
        f"a_form_{market}_f", f"a_form_{market}_a",
        "h_form_shots_f", "a_form_shots_f",
        f"h_venue_{market}_f", f"a_venue_{market}_f",
        f"h2h_{market}_h", f"h2h_{market}_a",
    ] + (["ref_factor"] if market in REF_SENSITIVE and has_referee(league) else [])


def build_features(data, market, league=DEFAULT_LEAGUE):
    long = _rollups(_long(data))
    cols = [f"form_{market}_f", f"form_{market}_a", "form_shots_f",
            f"venue_{market}_f"]
    h = long[long.is_home == 1].set_index("match_id")[cols].add_prefix("h_")
    a = long[long.is_home == 0].set_index("match_id")[cols].add_prefix("a_")
    X = data.join(h).join(a)

    hh, aa = _h2h(data, market)
    X[f"h2h_{market}_h"] = hh.fillna(X[f"h_form_{market}_f"])
    X[f"h2h_{market}_a"] = aa.fillna(X[f"a_form_{market}_f"])

    if market in REF_SENSITIVE and has_referee(league):
        tbl, _ = referee_table(data)
        col = "y_factor" if market == "yellows" else "f_factor"
        X["ref_factor"] = X["Referee"].map(tbl[col]).fillna(1.0)

    feats = market_features(market, league)
    X = X.dropna(subset=feats)
    hc, ac = MARKETS[market][:2]
    return X, {"h": X[hc], "a": X[ac], "t": X[f"tot_{market}"]}, feats


# ----------------------------------------------------------------------
# FIT
# ----------------------------------------------------------------------
def fit_nb(X, y, feats):
    Xd = sm.add_constant(X[feats], has_constant="add")
    p = sm.GLM(y, Xd, family=sm.families.Poisson()).fit()
    alpha = max(p.pearson_chi2 / p.df_resid - 1, 1e-6) / max(p.mu.mean(), 1e-9)
    m = sm.GLM(y, Xd, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    m._alpha = alpha
    return m


def train_league(data, league, verbose=True):
    """Fit every market for one league."""
    out = {"markets": {}, "has_referee": has_referee(league)}
    for mk in MARKETS:
        X, ys, feats = build_features(data, mk, league)
        entry = {"features": feats, "sides": {}}
        for side, y in ys.items():
            m = fit_nb(X, y, feats)
            entry["sides"][side] = {
                "params": {k: float(v) for k, v in m.params.items()},
                "alpha": float(m._alpha),
            }
        out["markets"][mk] = entry
        if verbose:
            print(f"    {mk:<8} n={len(X):<5} alpha(total)={entry['sides']['t']['alpha']:.4f}")

    if has_referee(league):
        tbl, lg = referee_table(data)
        out["league_means"] = {k: float(v) for k, v in lg.items()}
        out["referees"] = {
            r: {"games": int(v.games), "y": float(v.y_sh), "f": float(v.f_sh),
                "r": float(v.r), "y_factor": float(v.y_factor),
                "f_factor": float(v.f_factor)}
            for r, v in tbl.iterrows()
        }
    else:
        d = data.copy()
        out["league_means"] = {
            "y": float((d.HY + d.AY).mean()), "f": float((d.HF + d.AF).mean()),
            "r": float((d.HR + d.AR).mean()),
        }
        out["referees"] = {}
    out["teams"] = sorted(set(data.HomeTeam) | set(data.AwayTeam))
    out["n_matches"] = int(len(data))
    out["last_date"] = str(data.Date.max().date())
    return out


def train_and_save(path=MODEL_PATH, seasons=SEASONS, verbose=True):
    """Fit all five leagues and write one file keyed by league code."""
    out = {"leagues": {}}
    for code in LEAGUES:
        if verbose:
            print(f"  {LEAGUES[code]['name']}:")
        data = load_match_data(code, seasons)
        out["leagues"][code] = train_league(data, code, verbose)
    return _write(out, path)


def _write(out, path):
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def load_count_model(path=MODEL_PATH):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# PREDICT
# ----------------------------------------------------------------------
def _mu(params, row):
    z = params.get("const", 0.0)
    for k, v in params.items():
        if k != "const":
            z += v * float(row[k])
    return float(np.exp(z))


def nb_pmf(mu, alpha, kmax=40):
    if alpha <= 1e-8:
        from scipy.stats import poisson
        return poisson.pmf(np.arange(kmax + 1), mu)
    n = 1.0 / alpha
    return nbinom.pmf(np.arange(kmax + 1), n, n / (n + mu))


def fixture_row(data, home, away, market, referee=None, model=None, league=DEFAULT_LEAGUE):
    """Assemble one market's feature row for an unplayed fixture."""
    long = _long(data)
    row = {}
    for side, team, is_home in [("h", home, 1), ("a", away, 0)]:
        recent = long[long.team == team].tail(FORM_WINDOW)
        if recent.empty:
            raise ValueError(f"No match history for '{team}'.")
        row[f"{side}_form_{market}_f"] = recent[f"{market}_f"].mean()
        row[f"{side}_form_{market}_a"] = recent[f"{market}_a"].mean()
        row[f"{side}_form_shots_f"] = recent["shots_f"].mean()
        venue = long[(long.team == team) & (long.is_home == is_home)].tail(VENUE_WINDOW)
        src = venue if not venue.empty else recent
        row[f"{side}_venue_{market}_f"] = src[f"{market}_f"].mean()

    hc, ac = MARKETS[market][:2]
    prev = data[((data.HomeTeam == home) & (data.AwayTeam == away)) |
                ((data.HomeTeam == away) & (data.AwayTeam == home))]
    if prev.empty:
        row[f"h2h_{market}_h"] = row[f"h_form_{market}_f"]
        row[f"h2h_{market}_a"] = row[f"a_form_{market}_f"]
    else:
        p = prev.iloc[-1]
        row[f"h2h_{market}_h"] = p[hc] if p.HomeTeam == home else p[ac]
        row[f"h2h_{market}_a"] = p[ac] if p.HomeTeam == home else p[hc]

    if market in REF_SENSITIVE and has_referee(league):
        f = 1.0
        if referee and model:
            key = "y_factor" if market == "yellows" else "f_factor"
            f = model.get("referees", {}).get(referee, {}).get(key, 1.0)
        row["ref_factor"] = f
    return row


def predict_market(model, data, home, away, market, referee=None, league=DEFAULT_LEAGUE):
    """mu / most-likely / interval / pmf / over-under for one market."""
    spec = model["markets"][market]
    row = fixture_row(data, home, away, market, referee, model, league)
    out = {"features": row, "sides": {}}
    for side in ("h", "a", "t"):
        s = spec["sides"][side]
        mu = _mu(s["params"], row)
        pmf = nb_pmf(mu, s["alpha"])
        cdf = np.cumsum(pmf)
        out["sides"][side] = {
            "mu": mu,
            "most_likely": int(np.argmax(pmf)),
            "lo": int(np.searchsorted(cdf, 0.10)),
            "hi": int(np.searchsorted(cdf, 0.90)),
            "pmf": pmf, "cdf": cdf,
        }
    t = out["sides"]["t"]
    line = np.floor(t["mu"]) + 0.5
    out["line"] = float(line)
    out["ladder"] = []
    for j in (-1, 0, 1, 2):
        ln = line + j
        if ln < 0.4:
            continue
        over = float(1 - t["cdf"][int(np.floor(ln))])
        out["ladder"].append({"line": float(ln), "over": over, "under": 1 - over})
    out["p_over"] = float(1 - t["cdf"][int(np.floor(line))])
    return out


def predict_result(model, data, home, away, league=DEFAULT_LEAGUE):
    """1X2 and half-time 1X2, from the goals market's home/away means."""
    g = predict_market(model, data, home, away, "goals", league=league)
    lh, la = g["sides"]["h"]["mu"], g["sides"]["a"]["mu"]

    def grid(mh, ma):
        from scipy.stats import poisson
        ph = poisson.pmf(np.arange(10), mh)
        pa = poisson.pmf(np.arange(10), ma)
        M = np.outer(ph, pa)
        H = np.tril(M, -1).sum()
        D = np.trace(M)
        A = np.triu(M, 1).sum()
        tot = H + D + A
        return (H / tot, D / tot, A / tot), M

    ft, _ = grid(lh, la)
    # ~44% of goals land in the first half
    ht, M = grid(lh * 0.44, la * 0.44)
    i, j = np.unravel_index(np.argmax(M), M.shape)
    return {"ft": ft, "ht": ht, "ht_top": f"{i}-{j}", "ht_top_p": float(M.max()),
            "lh": lh, "la": la}
