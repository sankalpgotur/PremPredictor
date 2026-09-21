"""Fetch FBref player season stats -> players.json.

Run OFFLINE (scrapes, rate-limited); the app only reads the cached file.

    python fetch_players.py            # all five leagues
    python fetch_players.py F1 E0      # just these

MERGES into any existing players.json, so a league that fails (FBref rate
limits are aggressive) leaves the others intact -- rerun for that code alone.
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from count_model import load_count_model
from leagues import LEAGUES
from player_data import PLAYERS_PATH, fetch_league, resolve_teams, save


def existing():
    if not os.path.exists(PLAYERS_PATH):
        return {}
    try:
        with open(PLAYERS_PATH) as f:
            return json.load(f).get("leagues", {})
    except (ValueError, OSError):
        return {}


if __name__ == "__main__":
    codes = [c.upper() for c in sys.argv[1:]] or list(LEAGUES)
    bad = [c for c in codes if c not in LEAGUES]
    if bad:
        sys.exit(f"Unknown league code(s): {bad}. Valid: {list(LEAGUES)}")

    model = load_count_model()
    prior = existing()
    import pandas as pd
    per_league = {c: {"df": pd.DataFrame(v["players"]),
                      "unmatched": v.get("unmatched", [])}
                  for c, v in prior.items()}

    failed = []
    for code in codes:
        print(f"\n{LEAGUES[code]['name']}...", flush=True)
        try:
            df = fetch_league(code)
        except Exception as e:
            failed.append(code)
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}")
            print("  (keeping any previously cached data for this league)")
            continue
        targets = model["leagues"][code]["teams"]
        mapping, unmatched = resolve_teams(sorted(df["team"].unique()), targets)
        df["team"] = df["team"].map(mapping).fillna(df["team"])
        per_league[code] = {"df": df, "unmatched": unmatched}
        print(f"  {len(df)} players, {df['team'].nunique()} teams mapped")
        if unmatched:
            print(f"  UNMATCHED (these teams get no player view): {unmatched}")

    if per_league:
        p = save(per_league)
        total = sum(l["n_players"] for l in p["leagues"].values())
        print(f"\nSaved players.json — {len(p['leagues'])} leagues, {total} players")
    if failed:
        print(f"Failed: {failed}. Rerun with: python fetch_players.py {' '.join(failed)}")
