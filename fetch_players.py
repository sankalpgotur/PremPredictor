"""Fetch FBref player season stats for all five leagues -> players.json.

Run OFFLINE (scrapes, rate-limited); the app only reads the cached file.
    python fetch_players.py
"""
import json
import warnings
warnings.filterwarnings("ignore")

from count_model import load_count_model
from leagues import LEAGUES
from player_data import fetch_league, resolve_teams, save

if __name__ == "__main__":
    model = load_count_model()
    per_league = {}
    for code, lg in LEAGUES.items():
        print(f"\n{lg['name']}...", flush=True)
        try:
            df = fetch_league(code)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}")
            continue
        # football-data team names for this league, from the trained model
        targets = model["leagues"][code]["teams"]
        mapping, unmatched = resolve_teams(sorted(df["team"].unique()), targets)
        df["team"] = df["team"].map(mapping).fillna(df["team"])
        per_league[code] = {"df": df, "unmatched": unmatched}
        print(f"  {len(df)} players, {df['team'].nunique()} teams mapped")
        if unmatched:
            print(f"  UNMATCHED (no player data for these): {unmatched}")

    if per_league:
        p = save(per_league)
        print(f"\nSaved players.json — {len(p['leagues'])} leagues, "
              f"{sum(l['n_players'] for l in p['leagues'].values())} players")
