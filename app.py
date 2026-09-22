"""
Match Model Lab -- pre-match projection engine.
Run locally:  streamlit run app.py
"""
import hmac
import os

import streamlit as st

import design as D
from count_model import (MARKETS, FORM_WINDOW, load_match_data, load_count_model,
                         predict_market, predict_result, _long)
from leagues import LEAGUES, DEFAULT_LEAGUE, label as league_label, has_referee
from player_data import PLAYERS_PATH, load as load_players, league_players
from player_model import PLAYER_MARKETS, market_players, squad_names

st.set_page_config(page_title="Match Model Lab", page_icon="📊", layout="wide")

MAX_ATTEMPTS = 5


# ----------------------------------------------------------------- auth
def _expected_passcode():
    """Passcode comes from secrets/env, never from source (repo is public)."""
    try:
        code = st.secrets.get("APP_PASSCODE")
    except Exception:
        code = None
    return str(code) if code else os.environ.get("APP_PASSCODE")


def require_passcode():
    if st.session_state.get("authed"):
        return
    expected = _expected_passcode()
    if not expected:
        # Fail closed: an unset secret must never mean "let everyone in".
        st.error("No passcode configured. Set APP_PASSCODE in the app secrets.")
        st.stop()

    st.title("🔒 Locked")
    if st.session_state.get("attempts", 0) >= MAX_ATTEMPTS:
        st.error("Too many incorrect attempts. Reload the page to try again.")
        st.stop()
    entered = st.text_input("Passcode", type="password", max_chars=4)
    if st.button("Unlock", type="primary"):
        if hmac.compare_digest(entered, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.session_state["attempts"] = st.session_state.get("attempts", 0) + 1
            st.error(f"Incorrect. {MAX_ATTEMPTS - st.session_state['attempts']} attempt(s) left.")
    st.stop()


require_passcode()


# ----------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner="Pulling season data…")
def get_data(league):
    return load_match_data(league)


@st.cache_resource
def get_model():
    return load_count_model()


def _players_stamp():
    """mtime+size of players.json, so the cache key changes when it does."""
    try:
        st_ = os.stat(PLAYERS_PATH)
        return (st_.st_mtime_ns, st_.st_size)
    except OSError:
        return None


@st.cache_resource(show_spinner=False)
def _load_players_cached(stamp):
    """Keyed on `stamp` so a rewritten players.json invalidates the cache.

    Caching the failure was a real bug: when new code met an old-format file,
    the None stuck for the whole process lifetime and every player view
    silently disappeared, even after the file was fixed.
    """
    if stamp is None:
        return None
    try:
        p = load_players()
        return p if "leagues" in p else None
    except (FileNotFoundError, KeyError, ValueError):
        return None


def get_players():
    return _load_players_cached(_players_stamp())


st.html(D.page_css())

all_models = get_model()
players = get_players()

st.session_state.setdefault("market", "corners")

# ----------------------------------------------------------------- controls
codes = list(LEAGUES)
ctl = st.columns([1.6, 2, 2, 1.8])
league = ctl[0].selectbox("league", codes, format_func=league_label,
                          index=codes.index(DEFAULT_LEAGUE))

st.html(D.header(league_name=LEAGUES[league]["name"]))

model = all_models["leagues"][league]
data = get_data(league)
teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))

# only this season's sides -- a date window would reach back into last
# season's final matchday and pick up relegated teams
current = data[data["Season"] == data["Season"].max()]
active = sorted(set(current["HomeTeam"]) | set(current["AwayTeam"])) or teams

home = ctl[1].selectbox("home team", active, index=0)
away = ctl[2].selectbox("away team", active, index=min(1, len(active) - 1))

refs = sorted(model.get("referees", {}),
              key=lambda r: -model["referees"][r]["games"])
if has_referee(league) and refs:
    referee = ctl[3].selectbox("referee", refs)
else:
    referee = None
    ctl[3].selectbox("referee", ["not published"], disabled=True,
                     help="football-data.co.uk carries referees for the "
                          "Premier League only.")

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

# ----------------------------------------------------------------- compute
try:
    results = {k: predict_market(model, data, home, away, k, referee, league)
               for k in MARKETS}
    res = predict_result(model, data, home, away, league)
except ValueError as e:
    st.error(str(e))
    st.stop()

ref_stats = model.get("referees", {}).get(referee) if referee else None
league_means = model["league_means"]

st.html(D.summary_strip(home, away, res, referee or "not published",
                        ref_stats, FORM_WINDOW))

# ----------------------------------------------------------------- market cards
# Streamlit cannot make arbitrary HTML clickable, so each card gets a real
# button beneath it; the button sets the selected market and reruns.
st.html(D._lbl("markets · select one to drill in", D.ACCENT, "10px"))
cols = st.columns(len(MARKETS))
for col, key in zip(cols, MARKETS):
    with col:
        st.html(D.market_card(key, MARKETS[key][2], MARKETS[key][3],
                              results[key], key == st.session_state["market"]))
        has_players = key in PLAYER_MARKETS
        label = "▸ players" if has_players else "▸ detail"
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state["market"] = key
            st.rerun()

market_key = st.session_state["market"]
r = results[market_key]

# ----------------------------------------------------------------- lineup
st.session_state.setdefault("xi", {})
lgp_ctl = league_players(players, league)
if lgp_ctl is not None:
    with st.expander("Team news — enter the starting XI once it is published", expanded=False):
        st.caption("Leave blank to use usage averages. Naming an XI replaces "
                   "guessed minutes with facts and reweights the whole page.")
        for side in (home, away):
            names = squad_names(lgp_ctl, side)
            st.markdown(f"**{side}**")
            a1, a2 = st.columns([3, 2])
            start = a1.multiselect(f"starting XI · {side}", names,
                                   default=st.session_state["xi"].get(f"{side}_start", []),
                                   max_selections=11, key=f"ms_start_{side}")
            bench = a2.multiselect(f"substitutes · {side}", [n for n in names if n not in start],
                                   default=[b for b in st.session_state["xi"].get(f"{side}_bench", [])
                                            if b not in start],
                                   key=f"ms_bench_{side}")
            ret = st.multiselect(f"returning from a layoff · {side}", start + bench,
                                 default=[x for x in st.session_state["xi"].get(f"{side}_ret", [])
                                          if x in start + bench],
                                 key=f"ms_ret_{side}",
                                 help="Caps minutes at 65 and shades the per-90 rate by 8%. "
                                      "These are stated assumptions, not fitted effects.")
            st.session_state["xi"][f"{side}_start"] = start
            st.session_state["xi"][f"{side}_bench"] = bench
            st.session_state["xi"][f"{side}_ret"] = ret
        if st.button("Clear team news", use_container_width=True):
            st.session_state["xi"] = {}
            st.rerun()

# ----------------------------------------------------------------- detail
left, right = st.columns([2.15, 1])

with left:
    f = r["features"]
    drivers = [
        (f"form {market_key} · {home}", f"{f[f'h_form_{market_key}_f']:.2f}"),
        (f"form {market_key} · {away}", f"{f[f'a_form_{market_key}_f']:.2f}"),
        (f"at venue · {home}", f"{f[f'h_venue_{market_key}_f']:.2f}"),
        (f"at venue · {away}", f"{f[f'a_venue_{market_key}_f']:.2f}"),
        ("last h2h", f"{f[f'h2h_{market_key}_h']:.0f} – {f[f'h2h_{market_key}_a']:.0f}"),
    ]
    if "ref_factor" in f:
        drivers.append(("referee factor", f"{f['ref_factor']:.3f}"))

    note = ("Negative Binomial GLM per side on rolling form, same-venue record and "
            "last head-to-head. Head-to-head tested statistically insignificant "
            "(p&gt;0.5). Totals beat a league-mean baseline by under 1% for most "
            "markets — the home and away sides carry the real signal.")
    st.html(D.detail_panel(MARKETS[market_key][2], r, FORM_WINDOW, drivers, note))

    # ---- player breakdown ----
    if market_key not in PLAYER_MARKETS:
        st.html(D.no_player_data(MARKETS[market_key][2]))
    elif league_players(players, league) is None:
        st.html(D.no_player_data(MARKETS[market_key][2]))
    else:
        rf = 1.0
        if market_key in ("yellows", "fouls") and ref_stats:
            rf = ref_stats["y_factor" if market_key == "yellows" else "f_factor"]
        lgp = league_players(players, league)
        xi = st.session_state.get("xi", {})
        out = market_players(
            lgp, home, away, market_key, r, rf, top=10,
            home_xi=xi.get(f"{home}_start"), away_xi=xi.get(f"{away}_start"),
            home_bench=xi.get(f"{home}_bench"), away_bench=xi.get(f"{away}_bench"),
            home_returning=xi.get(f"{home}_ret"), away_returning=xi.get(f"{away}_ret"),
        )
        st.html(D.lineup_banner(home, away, out["mult"],
                                active=bool(xi.get(f"{home}_start") or xi.get(f"{away}_start"))))
        _, label, verb = PLAYER_MARKETS[market_key]
        pnote = (
            f"Per-90 rates from {lgp['n_players']} players' {players['season']} "
            "season totals (FBref), shrunk toward the positional mean so a small "
            "sample cannot top the table. Squad expectations are rescaled to sum to "
            "the team projection above, so opponent, venue and referee all flow "
            "through. <b>Lineups are not published until ~1h before kickoff</b> — "
            "expected minutes is a usage average, not knowledge of who starts, so a "
            "rested or rotated player will still appear here."
        )
        if not out["home"] and not out["away"]:
            st.html(D.no_player_data(label))
        else:
            st.html(D.player_section(home, away, out, label, verb, pnote))

# ----------------------------------------------------------------- sidebar
with right:
    long = _long(data)
    frows = []
    for key, col in [("shots", "shots_f"), ("sot", "sot_f"), ("corners", "corners_f"),
                     ("goals", "goals_f"), ("fouls", "fouls_f"), ("yellows", "yellows_f")]:
        h = long[long.team == home].tail(FORM_WINDOW)[col].mean()
        a = long[long.team == away].tail(FORM_WINDOW)[col].mean()
        frows.append((key, float(h), float(a)))
    st.html(D.feature_table(home, away, frows, FORM_WINDOW))

    traces = []
    for team, col in [(home, D.ACCENT), (away, D.AWAY)]:
        recent = data[(data.HomeTeam == team) | (data.AwayTeam == team)].tail(FORM_WINDOW)
        pts = []
        for _, m in recent.iterrows():
            is_home = m.HomeTeam == team
            gf, ga = (m.FTHG, m.FTAG) if is_home else (m.FTAG, m.FTHG)
            pts.append(3 if gf > ga else 1 if gf == ga else 0)
        wins = sum(1 for p in pts if p == 3)
        traces.append((team, col, f"{sum(pts)} pts · {wins/len(pts)*100:.0f}% win", pts))
    st.html(D.form_trace(traces))
    st.html(D.referee_panel(referee, ref_stats, league_means,
                            available=has_referee(league)))

st.html(D.footer(len(data), data["Date"].max().date()))
