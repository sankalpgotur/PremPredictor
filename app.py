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
def get_data():
    return load_match_data()


@st.cache_resource
def get_model():
    return load_count_model()


st.html(D.page_css())

data = get_data()
model = get_model()
teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
refs = sorted(model["referees"], key=lambda r: -model["referees"][r]["games"])

st.html(D.header())

# ----------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
home = c1.selectbox("home team", teams,
                    index=teams.index("Chelsea") if "Chelsea" in teams else 0)
away = c2.selectbox("away team", teams,
                    index=teams.index("Arsenal") if "Arsenal" in teams else 1)
referee = c3.selectbox("referee", refs)
market_key = c4.selectbox("market detail", list(MARKETS),
                          format_func=lambda k: MARKETS[k][2],
                          index=list(MARKETS).index("corners"))

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

# ----------------------------------------------------------------- compute
try:
    results = {k: predict_market(model, data, home, away, k, referee) for k in MARKETS}
    res = predict_result(model, data, home, away)
except ValueError as e:
    st.error(str(e))
    st.stop()

ref_stats = model["referees"].get(referee)
league = model["league"]

st.html(D.summary_strip(home, away, res, referee, ref_stats, FORM_WINDOW))

# ----------------------------------------------------------------- markets
cards = "".join(
    D.market_card(k, MARKETS[k][2], MARKETS[k][3], results[k], k == market_key)
    for k in MARKETS
)
left, right = st.columns([2.15, 1])

with left:
    st.html(D.market_grid(cards))

    r = results[market_key]
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

    note = ("Negative Binomial GLM per side, fitted on rolling form, same-venue record "
            "and last head-to-head. Head-to-head tested as statistically insignificant "
            "(p&gt;0.5) — it is shown for reference and barely moves the projection. "
            "Totals beat a league-mean baseline by under 1% for most markets; the "
            "home and away sides are the numbers with real signal.")
    st.html(D.detail_panel(MARKETS[market_key][2], r, FORM_WINDOW, drivers, note))

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
    st.html(D.referee_panel(referee, ref_stats, league))

st.html(D.footer(len(data), data["Date"].max().date()))
