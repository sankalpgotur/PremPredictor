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

from player_data import player_rates, expected_minutes

# market key -> (player stat column, label, verb)
PLAYER_MARKETS = {
    "sot":     ("sot",     "Shots on target", "a shot on target"),
    "yellows": ("cards_y", "Yellow cards",    "a yellow card"),
    "fouls":   ("fouls",   "Fouls",           "a foul"),
    "goals":   ("goals",   "Goals",           "a goal"),
}


def squad_projection(lg, team, market, team_lambda, team_matches=None,
                     ref_factor=1.0, top=12):
    """Rank a team's squad for one market.

    team_lambda is that side's projection from the team-level NB model.
    Returns rows with expected count and P(at least one).
    """
    stat, _, _ = PLAYER_MARKETS[market]
    df = player_rates(lg["df"])
    squad = df[df["team"] == team].copy()
    if squad.empty:
        return []

    if team_matches is None:
        team_matches = max(int(squad["mp"].max()), 1)

    squad["exp_min"] = squad.apply(
        lambda r: expected_minutes(r, team_matches), axis=1)
    squad["raw"] = squad[f"{stat}_p90"] * squad["exp_min"] / 90.0

    # cards and fouls scale with the official; shots and goals do not
    if market in ("yellows", "fouls"):
        squad["raw"] *= ref_factor

    total = squad["raw"].sum()
    if total <= 0:
        return []
    # rescale the squad to the team model's projection
    squad["lam"] = squad["raw"] * (team_lambda / total)
    squad["p_any"] = 1 - np.exp(-squad["lam"])
    squad["share"] = squad["lam"] / squad["lam"].sum()

    squad = squad.sort_values("lam", ascending=False).head(top)
    return [
        {
            "player": r["player"],
            "pos": str(r["pos"]).split(",")[0],
            "exp_min": round(float(r["exp_min"]), 1),
            "rate_p90": round(float(r[f"{stat}_p90"]), 3),
            "lam": float(r["lam"]),
            "p_any": float(r["p_any"]),
            "share": float(r["share"]),
            "season_total": float(r[stat]),
            "minutes": float(r["minutes"]),
            "starts": int(r["starts"]),
        }
        for _, r in squad.iterrows()
    ]


def market_players(lg, home, away, market, result, ref_factor=1.0, top=10):
    """Both squads for one market, plus the single most likely player overall."""
    if lg is None:
        return {"home": [], "away": [], "top": None}
    h = squad_projection(lg, home, market, result["sides"]["h"]["mu"],
                         ref_factor=ref_factor, top=top)
    a = squad_projection(lg, away, market, result["sides"]["a"]["mu"],
                         ref_factor=ref_factor, top=top)
    combined = sorted(h + a, key=lambda r: -r["p_any"])
    return {"home": h, "away": a, "top": combined[0] if combined else None}
