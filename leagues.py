"""
League configuration for the top-five European divisions.
----------------------------------------------------------
Season codes are derived from the date, not hardcoded, so the current
season is picked up automatically instead of going stale every August.

One structural difference to know about: football-data.co.uk publishes a
Referee column for the Premier League ONLY. The other four divisions have
the column absent or entirely empty, so referee features and the referee
panel are gated on `has_referee`.

Bundesliga and Ligue 1 are 18-team divisions (306 matches/season); the
other three are 20-team (380).
"""

import datetime as _dt

N_SEASONS = 8
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUES = {
    "E0": {
        "name": "Premier League", "country": "England", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "fbref": "ENG-Premier League", "teams": 20, "has_referee": True,
    },
    "SP1": {
        "name": "La Liga", "country": "Spain", "flag": "🇪🇸",
        "fbref": "ESP-La Liga", "teams": 20, "has_referee": False,
    },
    "I1": {
        "name": "Serie A", "country": "Italy", "flag": "🇮🇹",
        "fbref": "ITA-Serie A", "teams": 20, "has_referee": False,
    },
    "D1": {
        "name": "Bundesliga", "country": "Germany", "flag": "🇩🇪",
        "fbref": "GER-Bundesliga", "teams": 18, "has_referee": False,
    },
    "F1": {
        "name": "Ligue 1", "country": "France", "flag": "🇫🇷",
        "fbref": "FRA-Ligue 1", "teams": 18, "has_referee": False,
    },
}

DEFAULT_LEAGUE = "E0"


def season_codes(n=N_SEASONS, today=None):
    """football-data season codes, oldest -> current.

    A season starting in year Y is coded YY(Y+1). European leagues run
    Aug-May, so from July onward the current season is the one starting
    this year.
    """
    today = today or _dt.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}"
            for y in range(start - n + 1, start + 1)]


SEASONS = season_codes()


def url(code, season):
    return BASE.format(season=season, code=code)


def label(code):
    lg = LEAGUES[code]
    return f"{lg['flag']}  {lg['name']}"


def has_referee(code):
    return LEAGUES[code]["has_referee"]
