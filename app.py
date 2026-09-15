"""
Streamlit UI for the EPL predictor.
Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io -> pick app.py
"""
import streamlit as st
from epl_predictor import load_data, load_model, predict_fixture, _team_long

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽")
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

data = get_data()
model, feature_cols = get_model()

teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 0)
away = c2.selectbox("Away team", teams, index=1)

if st.button("Predict", type="primary", use_container_width=True):
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
        st.write(f"**{t}** — last 5"); st.dataframe(last, hide_index=True)
