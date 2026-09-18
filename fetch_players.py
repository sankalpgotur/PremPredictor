"""Fetch player season stats from FBref and cache to players.json.

Run OFFLINE (it scrapes and is rate-limited); the app only reads the file.
    python fetch_players.py
"""
import warnings
warnings.filterwarnings("ignore")

from player_data import fetch, save

if __name__ == "__main__":
    print("Fetching FBref player stats (slow, rate-limited)...")
    df = fetch()
    p = save(df)
    print(f"Saved players.json: {p['n_players']} players, {len(p['teams'])} teams")
    print("Teams:", ", ".join(p["teams"]))
