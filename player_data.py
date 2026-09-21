"""
Player-level data for the market drill-downs, across the top five leagues.
---------------------------------------------------------------------------
football-data.co.uk is match-level only -- no lineups, no player names -- so
per-player projections need a second source. FBref supplies season totals
(minutes, starts, shots, SoT, cards, fouls).

Scraping is slow and rate-limited, so this runs OFFLINE:
`python fetch_players.py` writes players.json and the app only ever reads
that file. Nothing in the serving path touches the network, and soccerdata
stays out of requirements.txt.

Team names differ between FBref and football-data (Internazionale vs Inter,
Nott'ham Forest vs Nott'm Forest, ...). resolve_teams() normalises both
sides and matches them, rather than hand-maintaining ~100 aliases across
five countries; MANUAL_ALIASES covers what normalisation cannot reach.
"""

import difflib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from leagues import LEAGUES

PLAYERS_PATH = "players.json"

# Cases normalisation cannot reach -- different words, not just spelling.
MANUAL_ALIASES = {
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Manchester United": "Man United",
    "Newcastle Utd": "Newcastle",
    "Sheffield Utd": "Sheffield United",
    "Nott'ham Forest": "Nott'm Forest",
    "Nottingham": "Nott'm Forest",
    "Internazionale": "Inter",
    "Hellas Verona": "Verona",
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Racing Sant": "Santander",
    "Racing Santander": "Santander",
    "Monchengladbach": "M'gladbach",
    "Mönchengladbach": "M'gladbach",
    "Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Paris S-G": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Saint-Étienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
}

_SUFFIXES = (r"\b(fc|cf|ac|as|ss|ssc|sc|cd|ud|rcd|sd|afc|bsc|vfb|vfl|tsg|sv|fsv|"
             r"1899|1900|1904|1909|calcio|club|de|futbol|football)\b")


def _norm(name):
    """Lowercase, de-accent, drop club-type words and punctuation."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(_SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_teams(fbref_names, target_names, cutoff=0.72):
    """Map FBref team names onto football-data names.

    Exact-normalised matches win, then containment, then fuzzy. Anything
    left over is RETURNED rather than silently dropped -- a bad mapping
    shows up as an empty squad, which is worse than a loud failure.
    """
    mapping, unmatched = {}, []
    remaining = {_norm(t): t for t in target_names}

    for fb in fbref_names:
        alias = MANUAL_ALIASES.get(fb)
        if alias and alias in target_names:
            mapping[fb] = alias
            remaining.pop(_norm(alias), None)
            continue
        n = _norm(alias or fb)
        if n in remaining:
            mapping[fb] = remaining.pop(n)
            continue
        hit = next((tn for tn in remaining
                    if tn.startswith(n) or n.startswith(tn)
                    or (len(n) > 4 and n in tn) or (len(tn) > 4 and tn in n)), None)
        if hit:
            mapping[fb] = remaining.pop(hit)
            continue
        close = difflib.get_close_matches(n, list(remaining), n=1, cutoff=cutoff)
        if close:
            mapping[fb] = remaining.pop(close[0])
        else:
            unmatched.append(fb)
    return mapping, unmatched


# ----------------------------------------------------------------------
# FETCH (offline only -- see fetch_players.py)
# ----------------------------------------------------------------------
def _col(df, *path):
    """Pull a column whether or not the header is a MultiIndex."""
    for c in df.columns:
        if isinstance(c, tuple) and tuple(x for x in c if x) == tuple(path):
            return df[c]
    for c in df.columns:
        last = c[-1] if isinstance(c, tuple) else c
        if last == path[-1]:
            return df[c]
    raise KeyError(path)


def fetch_league(code, season="2627"):
    import logging
    import soccerdata as sd
    logging.disable(logging.CRITICAL)

    fb = sd.FBref(leagues=LEAGUES[code]["fbref"], seasons=season)
    std = fb.read_player_season_stats(stat_type="standard").reset_index()
    sht = fb.read_player_season_stats(stat_type="shooting").reset_index()
    msc = fb.read_player_season_stats(stat_type="misc").reset_index()

    out = pd.DataFrame({
        "player": std["player"], "team": std["team"], "pos": _col(std, "pos"),
        "mp": _col(std, "Playing Time", "MP"),
        "starts": _col(std, "Playing Time", "Starts"),
        "minutes": _col(std, "Playing Time", "Min"),
        "goals": _col(std, "Performance", "Gls"),
    })
    k = ["player", "team"]
    out = out.merge(pd.DataFrame({
        "player": sht["player"], "team": sht["team"],
        "shots": _col(sht, "Standard", "Sh"), "sot": _col(sht, "Standard", "SoT"),
    }), on=k, how="left").merge(pd.DataFrame({
        "player": msc["player"], "team": msc["team"],
        "cards_y": _col(msc, "Performance", "CrdY"),
        "fouls": _col(msc, "Performance", "Fls"),
    }), on=k, how="left")

    for c in ["mp", "starts", "minutes", "goals", "shots", "sot", "cards_y", "fouls"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out[out["minutes"] > 0].reset_index(drop=True)


def save(per_league, path=PLAYERS_PATH, season="2627"):
    payload = {"season": season, "leagues": {}}
    for code, d in per_league.items():
        df = d["df"]
        payload["leagues"][code] = {
            "n_players": int(len(df)),
            "teams": sorted(df["team"].unique().tolist()),
            "unmatched": d.get("unmatched", []),
            "players": df.to_dict(orient="records"),
        }
    with open(path, "w") as f:
        json.dump(payload, f)
    return payload


def load(path=PLAYERS_PATH):
    with open(path) as f:
        p = json.load(f)
    for lg in p["leagues"].values():
        lg["df"] = pd.DataFrame(lg["players"])
    return p


def league_players(players, code):
    """One league's slice, or None if absent/empty in the cache."""
    if not players:
        return None
    lg = players["leagues"].get(code)
    if not lg or lg["df"].empty:
        return None
    return lg


# ----------------------------------------------------------------------
# PER-PLAYER RATES
# ----------------------------------------------------------------------
# k is in 90s played, so the positional prior dominates until a player has
# a real sample: one card in a 20-minute cameo must not top the table.
SHRINK_90S = {"sot": 5.0, "shots": 5.0, "cards_y": 8.0, "fouls": 5.0, "goals": 6.0}
POS_GROUPS = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}


def pos_group(pos):
    return POS_GROUPS.get(str(pos).split(",")[0].strip().upper(), "MF")


def player_rates(df):
    """Per-90 rates, shrunk toward the positional mean."""
    d = df.copy()
    d["n90"] = d["minutes"] / 90.0
    d["grp"] = d["pos"].map(pos_group)
    for stat in ["sot", "shots", "cards_y", "fouls", "goals"]:
        raw = d[stat] / d["n90"].clip(lower=0.1)
        prior = d.groupby("grp").apply(
            lambda g, s=stat: g[s].sum() / max(g["n90"].sum(), 1e-6))
        k = SHRINK_90S[stat]
        d[f"{stat}_p90"] = (d["n90"] * raw + k * d["grp"].map(prior)) / (d["n90"] + k)
    d["min_per_app"] = d["minutes"] / d["mp"].clip(lower=1)
    return d


def expected_minutes(row, team_matches):
    """Typical minutes load -- NOT a starting-XI prediction.

    Lineups are unpublished until ~1h before kickoff, so this blends the
    player's start rate and sub rate; a rotated starter still reads high.
    """
    starts = row["starts"]
    mp = max(row["mp"], 1)
    start_share = starts / max(team_matches, 1)
    sub_share = max(mp - starts, 0) / max(team_matches, 1)
    mn_start = row["min_per_app"] if starts else 0
    return float(np.clip(start_share * max(mn_start, 60) + sub_share * 22, 0, 95))
