"""
Per-player projections for the market drill-downs.
---------------------------------------------------
The team-level NB model says how many SoT / cards / fouls / goals a side is
projected to produce. This module splits that total across the squad.

Method, per player:
  1. rate per 90, shrunk toward the positional mean (few-minutes players
     must not top the table off one card in 20 minutes)
  2. expected minutes from start/sub usage
  3. raw expectation  = rate_p90 * expected_minutes / 90
  4. rescale so the squad sums to the TEAM model's projection, which keeps
     the player view consistent with the market card above it
  5. P(at least one) = 1 - exp(-lambda), the Poisson zero-complement

Step 4 matters: it means opponent strength, venue and referee -- everything
the team model knows -- flows through to the player numbers.

LIMITATION, stated in the UI too: lineups are not published until ~1h before
kickoff. Expected minutes is a usage average, NOT knowledge of who starts.
A rotated-out starter still shows near-full minutes here.
"""

import numpy as np

import lineup as LU
from player_data import player_rates, expected_minutes

# market key -> (player stat column, label, verb)
PLAYER_MARKETS = {
    "sot":     ("sot",     "Shots on target", "a shot on target"),
    "yellows": ("cards_y", "Yellow cards",    "a yellow card"),
    "fouls":   ("fouls",   "Fouls",           "a foul"),
    "goals":   ("goals",   "Goals",           "a goal"),
}


def squad_rows(lg, team, market, team_matches=None, ref_factor=1.0):
    """Baseline per-player rates and usage-average minutes for one squad.

    Separated from the projection so a lineup override can replace the
    minutes without recomputing the rates.
    """
    stat, _, _ = PLAYER_MARKETS[market]
    df = player_rates(lg["df"])
    squad = df[df["team"] == team].copy()
    if squad.empty:
        return []
    if team_matches is None:
        team_matches = max(int(squad["mp"].max()), 1)

    squad["exp_min"] = squad.apply(lambda r: expected_minutes(r, team_matches), axis=1)
    rate = squad[f"{stat}_p90"]
    # cards and fouls scale with the official; shots and goals do not
    if market in ("yellows", "fouls"):
        rate = rate * ref_factor

    rows = []
    for (_, r), rt in zip(squad.iterrows(), rate):
        rows.append({
            "player": r["player"], "pos": str(r["pos"]).split(",")[0],
            "exp_min": float(r["exp_min"]), "rate_p90": float(rt),
            "lam_base": float(rt) * float(r["exp_min"]) / 90.0,
            "season_total": float(r[stat]), "minutes": float(r["minutes"]),
            "starts": int(r["starts"]), "mp": int(r["mp"]),
        })
    return rows


def squad_projection(lg, team, market, team_lambda, team_matches=None,
                     ref_factor=1.0, top=12, starters=None, bench=None,
                     returning=None):
    """Rank a squad for one market, optionally under a published XI."""
    rows = squad_rows(lg, team, market, team_matches, ref_factor)
    if not rows:
        return [], 1.0

    if starters:
        minutes = LU.set_lineup(rows, starters, bench)
    else:
        minutes = LU.default_minutes(rows)

    minutes, rows = LU.apply_returns(rows, minutes, returning)
    # the multiplier is computed AFTER the return adjustment, so a returner
    # on capped minutes correctly reads as reduced availability
    mult = LU.team_multiplier(rows, minutes) if starters else 1.0

    out = LU.apply(rows, team_lambda, minutes, mult)
    return out[:top], mult


def market_players(lg, home, away, market, result, ref_factor=1.0, top=10,
                   home_xi=None, away_xi=None, home_bench=None, away_bench=None,
                   home_returning=None, away_returning=None):
    """Both squads for one market, plus the single most likely player overall.

    home_xi / away_xi are published starting elevens. When given, minutes
    become facts rather than usage averages and the team projection is
    shrunk by the share of baseline output that is missing.
    """
    if lg is None:
        return {"home": [], "away": [], "top": None, "mult": {"home": 1.0, "away": 1.0}}
    h, mh = squad_projection(lg, home, market, result["sides"]["h"]["mu"],
                             ref_factor=ref_factor, top=top,
                             starters=home_xi, bench=home_bench,
                             returning=home_returning)
    a, ma = squad_projection(lg, away, market, result["sides"]["a"]["mu"],
                             ref_factor=ref_factor, top=top,
                             starters=away_xi, bench=away_bench,
                             returning=away_returning)
    combined = sorted(h + a, key=lambda r: -r["p_any"])
    return {"home": h, "away": a, "top": combined[0] if combined else None,
            "mult": {"home": mh, "away": ma}}


def squad_names(lg, team):
    """Every player on a team's books, most-used first -- for the XI picker."""
    df = lg["df"]
    s = df[df["team"] == team].sort_values("minutes", ascending=False)
    return s["player"].tolist()
