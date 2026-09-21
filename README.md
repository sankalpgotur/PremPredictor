# Match Model Lab

Pre-match projection engine for Europe's top five leagues — Premier League,
La Liga, Serie A, Bundesliga and Ligue 1.

Projects **goals, shots on target, corners, yellow cards and fouls** as full
distributions, plus 1X2 and half-time 1X2, and drills into **which player**
is most likely to record a shot on target, card, foul or goal.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

The app is passcode-gated. Set `APP_PASSCODE` in `.streamlit/secrets.toml`
locally, or in **Manage app → Settings → Secrets** on Streamlit Cloud. It
fails closed: with no passcode configured nobody gets in, including you.

## How it works

**Team markets.** One Negative Binomial GLM per side per market. NB rather
than Poisson because the counts are overdispersed (fouls var/mean 1.29,
corners 1.16); goals are the exception at 0.96, where alpha fits near zero
and it collapses to Poisson. Totals are fitted *directly* rather than summed,
because home and away counts are negatively correlated (corners ≈ −0.31) and
adding two independent fits would overstate the spread.

Features are strictly pre-match: rolling form over the last 10 matches,
same-venue record, last head-to-head, and — Premier League only — the
referee's own rate relative to the league.

**Player markets.** Per-90 rates from FBref season totals, shrunk toward the
positional mean, scaled by expected minutes, then rescaled so each squad sums
to that side's team projection. Opponent, venue and referee therefore flow
through into the player numbers.

## Honest performance

Test-set MAE against a league-mean baseline, time-ordered 80/20 split:

| Side | Gain |
|---|---|
| Home / away | 3–7.5% |
| Totals | under 1% (goals total is 1.2% *worse*) |

The home and away sides carry the real signal. **Totals are barely better
than guessing the mean** — the negative correlation between sides cancels
most of what the features know. Head-to-head tested statistically
insignificant (p>0.5) and is kept only for display. Past *shots* predict
counts better than past counts do.

## Retraining

```bash
python train_counts.py     # all five leagues -> count_model.json
python fetch_players.py    # FBref scrape -> players.json (needs requirements-dev.txt)
```

`train_counts.py` re-pulls from football-data.co.uk. `fetch_players.py`
scrapes FBref, which is slow and rate-limited — run it offline every few
weeks. Nothing in the serving path touches the network, so `soccerdata` stays
out of `requirements.txt`.

Season codes derive from the current date, so the active season is picked up
automatically rather than going stale each August.

## Known limits

- **Lineups.** Not published until ~1h before kickoff, so expected minutes is
  a usage average, not a starting-XI prediction. A rotated starter still
  reads high.
- **Referees.** football-data.co.uk publishes the column for the Premier
  League only. The other four leagues carry no official adjustment, and the
  referee panel says so rather than showing blanks.
- **Corners have no player view.** No feed attributes a corner to a player.
- **Early season.** Card rates lean on the shrinkage prior until players
  accumulate minutes; rankings are driven more by minutes than discipline.

## Data

- [football-data.co.uk](https://www.football-data.co.uk/) — match results and
  team counts, 8 seasons per league
- [FBref](https://fbref.com/) via `soccerdata` — player season totals
