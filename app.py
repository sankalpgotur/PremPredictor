"""
Streamlit UI for the EPL predictor.
Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io -> pick app.py
"""
import hmac
import os

import streamlit as st
import pandas as pd

from epl_predictor import load_data, load_model, predict_fixture, _team_long
from corners_model import (load_corner_data, load_corners_model,
                           predict_corners, FORM_WINDOW)

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽")

MAX_ATTEMPTS = 5


def _expected_passcode():
    """Passcode comes from secrets/env, never from the source (repo is public)."""
    try:
        code = st.secrets.get("APP_PASSCODE")
    except Exception:          # no secrets.toml configured at all
        code = None
    return str(code) if code else os.environ.get("APP_PASSCODE")


def require_passcode():
    """Block the app until the right 4-digit code is entered."""
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
        # compare_digest: constant-time, so response timing can't leak the code
        if hmac.compare_digest(entered, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.session_state["attempts"] = st.session_state.get("attempts", 0) + 1
            left = MAX_ATTEMPTS - st.session_state["attempts"]
            st.error(f"Incorrect. {left} attempt(s) left.")
    st.stop()


require_passcode()

st.title("⚽ Premier League Match Predictor")
st.caption("Home win / Draw / Away win from recent form. Data: football-data.co.uk")

# ttl=6h => the app re-pulls the current-season CSV on its own. No git push needed
# to stay current; the numbers refresh as new results land in the source file.
@st.cache_data(ttl=6 * 3600)
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data(ttl=6 * 3600)
def get_corner_data():
    return load_corner_data()

@st.cache_resource
def get_corner_model():
    return load_corners_model()

data = get_data()
model, feature_cols = get_model()

teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 0)
away = c2.selectbox("Away team", teams, index=1)

tab_result, tab_corners = st.tabs(["Match outcome", "Corners"])

# ----------------------------------------------------------------- outcome
with tab_result:
    if st.button("Predict outcome", type="primary", use_container_width=True):
        if home == away:
            st.warning("Pick two different teams.")
        else:
            probs = predict_fixture(model, feature_cols, data, home, away)
            st.subheader(f"{home} (H) vs {away} (A)")
            cols = st.columns(3)
            for col, (label, p) in zip(cols, probs.items()):
                col.metric(label, f"{p*100:.0f}%")
            st.bar_chart(probs)
            pick = max(probs, key=probs.get)
            st.success(f"Most likely: **{pick}** ({probs[pick]*100:.0f}%)")

    with st.expander("Recent form used"):
        long = _team_long(data)
        for t in (home, away):
            last = long[long.team == t].tail(5)[["Date", "gf", "ga", "shots", "pts"]]
            st.write(f"**{t}** \u2014 last 5"); st.dataframe(last, hide_index=True)

# ----------------------------------------------------------------- corners
with tab_corners:
    st.caption(
        f"Negative Binomial model. Inputs: last {FORM_WINDOW} matches, "
        "same-venue record, and the last head-to-head."
    )
    if st.button("Predict corners", type="primary", use_container_width=True):
        if home == away:
            st.warning("Pick two different teams.")
        else:
            cdata = get_corner_data()
            cmodel = get_corner_model()
            try:
                r = predict_corners(cmodel, cdata, home, away)
            except ValueError as e:
                st.error(str(e)); st.stop()

            st.subheader(f"{home} (H) vs {away} (A)")
            labels = {"HC": f"{home} corners", "AC": f"{away} corners", "TC": "Total corners"}
            cols = st.columns(3)
            for col, t in zip(cols, ["HC", "AC", "TC"]):
                e = r["targets"][t]
                col.metric(labels[t], e["most_likely"], f"mean {e['mean']}")
                col.caption(f"80% range: {e['interval80'][0]}\u2013{e['interval80'][1]}")

            st.write("**Total corners \u2014 probability of each count**")
            pmf = r["targets"]["TC"]["pmf"][:23]
            st.bar_chart(pd.DataFrame({"probability": pmf}, index=range(len(pmf))))

            st.write("**Over / under**")
            ou = r["targets"]["TC"]["over_under"]
            st.dataframe(pd.DataFrame([
                {"line": ln, "over": f"{v['over']*100:.1f}%", "under": f"{v['under']*100:.1f}%"}
                for ln, v in ou.items()
            ]), hide_index=True)

            f = r["features"]
            with st.expander("Inputs the model used"):
                st.dataframe(pd.DataFrame([
                    {"signal": f"last {FORM_WINDOW} \u2014 corners won",
                     home: round(f["h_form_cf"], 2), away: round(f["a_form_cf"], 2)},
                    {"signal": f"last {FORM_WINDOW} \u2014 shots",
                     home: round(f["h_form_sf"], 1), away: round(f["a_form_sf"], 1)},
                    {"signal": "at this venue \u2014 corners won",
                     home: round(f["h_venue_cf"], 2), away: round(f["a_venue_cf"], 2)},
                    {"signal": "last meeting \u2014 corners",
                     home: f["h2h_hc"], away: f["h2h_ac"]},
                ]), hide_index=True)
                st.caption(
                    f"{f.get('h2h_meetings', 0)} prior meetings in the data. "
                    "Note: head-to-head tested as statistically insignificant "
                    "(p>0.5) \u2014 shown for reference, it barely moves the forecast."
                )
