"""
Lineup-aware adjustment of the player projections.
---------------------------------------------------
Without a lineup, expected minutes is a usage average: start rate blended
with sub rate. Once the XI is published that guess can be replaced with the
fact, which sharpens every player number on the page.

No free feed publishes confirmed XIs before kickoff -- FBref only has them
after the match -- so the XI is entered by hand in the UI. set_lineup()
takes the chosen starters, bench and absentees and returns per-player
expected minutes.

TWO EFFECTS, kept separate because they are separately uncertain:

  1. MINUTES. A named starter plays ~90, a benched player ~20, an absentee
     0. This part is near-certain and does most of the work.

  2. TEAM RATE. The team-level NB model is fitted on rolling team form; it
     does not know a key player is missing. Rescaling an unchanged team
     lambda across a weakened squad would hand the absentee's share to
     whoever remains, overstating them. team_multiplier() compares the named
     XI with the side's usual strongest XI and shrinks the team projection by
     the shortfall, damped by REPLACEMENT -- a replacement is worse than the
     player they replace, not worthless.
"""

import numpy as np

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


def apply(squad_rows, team_lambda, minutes, multiplier=1.0):
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
