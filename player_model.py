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

# ----------------------------------------------------------------------
# LINEUP ADJUSTMENT
# ----------------------------------------------------------------------
# Kept in this module rather than its own file: a separate module is one
# more thing a deploy has to resolve, and a new top-level module failing to
# import is exactly how this broke on Streamlit Cloud.
#
# Without a lineup, expected minutes is a usage average. Once the XI is
# published that guess becomes a fact, which sharpens every player number.
#
# TWO EFFECTS, separately uncertain:
#   1. MINUTES -- starter ~90, sub ~20, absentee 0. Near-certain; does most
#      of the work.
#   2. TEAM RATE -- the team NB model cannot know a player is missing, so the
#      named XI is compared with the side's usual strongest XI and the team
#      projection shrinks by the shortfall, damped by REPLACEMENT.

# A replacement plays; they are simply worse. 0 would mean an absent player's
# output vanishes entirely, 1 would mean absences do not matter at all.
REPLACEMENT = 0.65

START_MINUTES = 90.0
BENCH_MINUTES = 20.0


def default_minutes(squad_rows):
    """Usage-average minutes -- what we use when no XI has been entered."""
    return {r["player"]: r["exp_min"] for r in squad_rows}


def set_lineup(squad_rows, starters, bench=None, start_minutes=START_MINUTES,
               bench_minutes=BENCH_MINUTES):
    """Expected minutes given a published XI.

    Anyone not named as a starter or a sub is treated as absent (0 minutes),
    which is the whole point: the model should stop crediting them.
    """
    bench = set(bench or [])
    starters = set(starters)
    out = {}
    for r in squad_rows:
        p = r["player"]
        if p in starters:
            out[p] = float(start_minutes)
        elif p in bench:
            out[p] = float(bench_minutes)
        else:
            out[p] = 0.0
    return out


def reference_xi(squad_rows, n=11):
    """The side's usual strongest XI: the n most-used players.

    This is the yardstick a published lineup is measured against. Comparing
    instead against usage-average minutes would make ANY named XI look
    stronger than baseline, simply because 11 confirmed starters play more
    minutes than a rotation average -- which says nothing about who they are.
    """
    return sorted(squad_rows, key=lambda r: -r["minutes"])[:n]


def team_multiplier(squad_rows, minutes, n=11, bench_minutes=BENCH_MINUTES):
    """How this lineup compares with the side's usual strongest XI.

    1.0 means full strength. Below 1 means key output is missing. The
    shortfall is damped by REPLACEMENT, because whoever comes in is worse,
    not worthless.
    """
    ref = reference_xi(squad_rows, n)
    ref_out = sum(r["rate_p90"] for r in ref)
    if ref_out <= 0:
        return 1.0
    # bench contributes a part-match share on both sides of the comparison
    ref_out += sum(r["rate_p90"] for r in squad_rows
                   if r not in ref) * (bench_minutes / 90.0) * 0.35

    now = sum(r["rate_p90"] * minutes.get(r["player"], 0.0) / 90.0
              for r in squad_rows)
    raw = now / ref_out
    adjusted = 1.0 - (1.0 - raw) * (1.0 - REPLACEMENT)
    return float(np.clip(adjusted, 0.70, 1.15))


def apply_lineup(squad_rows, team_lambda, minutes, multiplier=1.0):
    """Rescale a squad to the (possibly adjusted) team projection.

    Players on 0 minutes drop out entirely rather than absorbing a share.
    """
    eff = team_lambda * multiplier
    raw = {r["player"]: r["rate_p90"] * minutes.get(r["player"], 0.0) / 90.0
           for r in squad_rows}
    total = sum(raw.values())
    out = []
    for r in squad_rows:
        m = minutes.get(r["player"], 0.0)
        if m <= 0:
            continue
        lam = (raw[r["player"]] / total * eff) if total > 0 else 0.0
        row = dict(r)
        row["exp_min"] = m
        row["lam"] = lam
        row["p_any"] = 1 - np.exp(-lam)
        out.append(row)
    total_lam = sum(r["lam"] for r in out) or 1.0
    for r in out:
        r["share"] = r["lam"] / total_lam
    return sorted(out, key=lambda r: -r["lam"])


# ----------------------------------------------------------------------
# RETURNING FROM A LAYOFF
# ----------------------------------------------------------------------
# These two numbers are ASSUMPTIONS, not fitted effects, and the UI says so.
#
# The minutes cap is the well-supported half: managers demonstrably ease
# players back, and a first start after a layoff is routinely 60-70 minutes
# rather than 90. Capping minutes is the bulk of the adjustment and it flows
# straight through the Poisson, so P(1+) falls accordingly.
#
# The rate haircut -- "rust" -- is the speculative half. Evidence that a fit
# returner produces at a lower PER-MINUTE rate is much weaker than evidence
# they play fewer minutes, so it is deliberately small. Set RETURN_RATE to
# 1.0 to model minutes only.
#
# Measuring the real effect needs per-match appearance history (who missed
# which fixtures, and what they did in the first match back). That is one
# FBref request per match, which is why it is not fitted here.
RETURN_MINUTES_CAP = 65.0
RETURN_RATE = 0.92


def apply_returns(squad_rows, minutes, returning,
                  cap=RETURN_MINUTES_CAP, rate_factor=RETURN_RATE):
    """Cap minutes and shade the rate for players flagged as returning.

    Returns (minutes, adjusted_rows) -- rows are copied, never mutated in
    place, so the caller's baseline stays intact for comparison.
    """
    returning = set(returning or [])
    if not returning:
        return minutes, squad_rows

    out_min = dict(minutes)
    rows = []
    for r in squad_rows:
        p = r["player"]
        row = dict(r)
        if p in returning:
            out_min[p] = min(out_min.get(p, 0.0), cap)
            row["rate_p90"] = r["rate_p90"] * rate_factor
            row["returning"] = True
        rows.append(row)
    return out_min, rows


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
        minutes = set_lineup(rows, starters, bench)
    else:
        minutes = default_minutes(rows)

    minutes, rows = apply_returns(rows, minutes, returning)
    # the multiplier is computed AFTER the return adjustment, so a returner
    # on capped minutes correctly reads as reduced availability
    mult = team_multiplier(rows, minutes) if starters else 1.0

    out = apply_lineup(rows, team_lambda, minutes, mult)
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
