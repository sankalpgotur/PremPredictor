"""
Player-level data for the market drill-downs.
----------------------------------------------
football-data.co.uk is match-level only -- no lineups, no player names -- so
per-player projections need a second source. FBref supplies season totals
(minutes, starts, shots, SoT, cards, fouls) and Understat supplies shot
events with xG.

Both are scraped, which is slow and rate-limited, so this runs OFFLINE:
`python fetch_players.py` writes players.json and the app only ever reads
that file. Nothing in the serving path touches the network.

Team names differ between sources; TEAM_ALIASES maps FBref/Understat naming
onto the football-data.co.uk names the rest of the app uses.
"""

import json
import unicodedata

import numpy as np
import pandas as pd

PLAYERS_PATH = "players.json"

# FBref/Understat name -> football-data.co.uk name
TEAM_ALIASES = {
    "Manchester City": "Man City", "Manchester Utd": "Man United",
    "Manchester United": "Man United", "Newcastle Utd": "Newcastle",
    "Newcastle United": "Newcastle", "Nott'ham Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest", "Nottingham": "Nott'm Forest", "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Brighton & Hove Albion": "Brighton", "West Bromwich Albion": "West Brom",
    "Sheffield Utd": "Sheffield United", "Sheffield United": "Sheffield United",
    "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "Luton Town": "Luton",
    "Norwich City": "Norwich", "Cardiff City": "Cardiff",
    "Swansea City": "Swansea", "Stoke City": "Stoke",
    "Hull City": "Hull", "Birmingham City": "Birmingham",
    "Coventry City": "Coventry",
}


def canon_team(name):
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


# ----------------------------------------------------------------------
# FETCH (offline only -- see fetch_players.py)
# ----------------------------------------------------------------------
def fetch(season="2627", league="ENG-Premier League"):
    import logging
    import soccerdata as sd
    logging.disable(logging.CRITICAL)

    fb = sd.FBref(leagues=league, seasons=season)
    std = fb.read_player_season_stats(stat_type="standard").reset_index()
    sht = fb.read_player_season_stats(stat_type="shooting").reset_index()
    msc = fb.read_player_season_stats(stat_type="misc").reset_index()

    def col(df, *path):
        """Pull a column whether or not the header is a MultiIndex."""
        for c in df.columns:
            if isinstance(c, tuple):
                if tuple(x for x in c if x) == tuple(path) or c[-1] == path[-1] and path[0] in c:
                    return df[c]
            elif c == path[-1]:
                return df[c]
        for c in df.columns:
            last = c[-1] if isinstance(c, tuple) else c
            if last == path[-1]:
                return df[c]
        raise KeyError(path)

    out = pd.DataFrame({
        "player": std["player"],
        "team": std["team"].map(canon_team),
        "pos": col(std, "pos"),
        "mp": col(std, "Playing Time", "MP"),
        "starts": col(std, "Playing Time", "Starts"),
        "minutes": col(std, "Playing Time", "Min"),
        "goals": col(std, "Performance", "Gls"),
    })
    key = ["player", "team"]
    sht2 = pd.DataFrame({
        "player": sht["player"], "team": sht["team"].map(canon_team),
        "shots": col(sht, "Standard", "Sh"), "sot": col(sht, "Standard", "SoT"),
    })
    msc2 = pd.DataFrame({
        "player": msc["player"], "team": msc["team"].map(canon_team),
        "cards_y": col(msc, "Performance", "CrdY"),
        "fouls": col(msc, "Performance", "Fls"),
    })
    out = out.merge(sht2, on=key, how="left").merge(msc2, on=key, how="left")

    for c in ["mp", "starts", "minutes", "goals", "shots", "sot", "cards_y", "fouls"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out = out[out["minutes"] > 0].reset_index(drop=True)
    return out


def save(df, path=PLAYERS_PATH, season="2627"):
    payload = {
        "season": season,
        "n_players": int(len(df)),
        "teams": sorted(df["team"].unique().tolist()),
        "players": df.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return payload


def load(path=PLAYERS_PATH):
    with open(path) as f:
        p = json.load(f)
    p["df"] = pd.DataFrame(p["players"])
    return p


# ----------------------------------------------------------------------
# PER-PLAYER RATES
# ----------------------------------------------------------------------
# Rate per 90 shrunk toward the positional mean. A player with 120 minutes
# and one yellow is not a 0.75/90 booking risk; k is in 90s-played, so the
# prior dominates until a player has a real sample.
SHRINK_90S = {"sot": 5.0, "shots": 5.0, "cards_y": 8.0, "fouls": 5.0, "goals": 6.0}

POS_GROUPS = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}


def pos_group(pos):
    p = str(pos).split(",")[0].strip().upper()
    return POS_GROUPS.get(p, "MF")


def player_rates(df):
    """Add per-90 rates, shrunk toward the positional mean."""
    d = df.copy()
    d["n90"] = d["minutes"] / 90.0
    d["grp"] = d["pos"].map(pos_group)
    for stat in ["sot", "shots", "cards_y", "fouls", "goals"]:
        raw = d[stat] / d["n90"].clip(lower=0.1)
        prior = d.groupby("grp").apply(
            lambda g, s=stat: g[s].sum() / max(g["n90"].sum(), 1e-6))
        pri = d["grp"].map(prior)
        k = SHRINK_90S[stat]
        d[f"{stat}_p90"] = (d["n90"] * raw + k * pri) / (d["n90"] + k)
    # share of the team's matches the player has started
    d["start_rate"] = (d["starts"] / d.groupby("team")["mp"].transform("max").clip(lower=1))
    d["min_per_app"] = d["minutes"] / d["mp"].clip(lower=1)
    return d


def expected_minutes(row, team_matches):
    """Crude but honest: recent usage, not a lineup prediction.

    Lineups are not published until ~1h before kickoff, so this is the
    player's typical minutes load, NOT knowledge of whether they start.
    """
    starts = row["starts"]
    mp = max(row["mp"], 1)
    start_share = starts / max(team_matches, 1)
    sub_share = max(mp - starts, 0) / max(team_matches, 1)
    mn_start = row["min_per_app"] if starts else 0
    return float(np.clip(start_share * max(mn_start, 60) + sub_share * 22, 0, 95))
