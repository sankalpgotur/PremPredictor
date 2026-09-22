"""
HTML renderer for the Match Model Lab layout.
----------------------------------------------
Streamlit widgets cannot be restyled into this design, so the panels are
emitted as raw HTML and injected with st.html; only the controls stay as
native widgets. Colour tokens and type scale are taken from the source
design file so the two stay in step.
"""

# ---- design tokens -------------------------------------------------------
BG = "#08090a"
PANEL = "#0d100f"
TINT = "#141817"
LINE = "#1d2422"
LINE2 = "#222a27"
INK = "#e6ece9"
MUTE = "#8b9793"
DIM = "#848f8b"
ACCENT = "#00e08a"
AWAY = "#f5c341"
WARN = "#ff5a36"
BAR_OFF = "#2a332f"

MONO = "'IBM Plex Mono', ui-monospace, monospace"
SANS = "Archivo, Helvetica, sans-serif"

FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800"
         "&family=IBM+Plex+Mono:wght@400;500;600&display=swap")


def _lbl(text, color=DIM, size="9.5px"):
    return (f'<div style="font-family:{MONO};font-size:{size};letter-spacing:.14em;'
            f'color:{color};text-transform:uppercase;">{text}</div>')


def page_css():
    """Global CSS: restyles Streamlit's own chrome to match the design."""
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="{FONTS}" rel="stylesheet">
<style>
  .stApp {{ background: {BG}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1560px; }}
  html, body, [class*="css"] {{ font-family: {SANS}; color: {INK}; }}

  /* native widgets, pulled toward the design */
  .stSelectbox label, .stRadio label, .stSlider label {{
      font-family: {MONO} !important; font-size: 9.5px !important;
      letter-spacing: .14em; text-transform: uppercase; color: {DIM} !important;
  }}
  .stSelectbox div[data-baseweb="select"] > div {{
      background: {PANEL}; border: 1px solid {LINE}; border-radius: 7px;
      font-family: {MONO}; font-size: 13px; color: {INK};
  }}
  div[data-baseweb="popover"] li {{ font-family: {MONO}; font-size: 13px; }}
  .stRadio [role="radiogroup"] {{ gap: 2px; }}
  .stButton button {{
      background: {PANEL}; color: {ACCENT}; border: 1px solid {LINE};
      border-radius: 7px; font-family: {MONO}; font-size: 12px;
      letter-spacing: .08em; text-transform: uppercase;
  }}
  .stButton button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
  #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def header(title="Pre-match projection engine", sub=None, league_name=""):
    sub = sub or "negative binomial counts · rolling-window features"
    return f"""
<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;
     justify-content:space-between;border-bottom:1px solid {LINE};padding-bottom:14px;
     margin-bottom:18px;font-family:{SANS};">
  <div style="display:flex;flex-direction:column;gap:4px;">
    {_lbl("match model lab" + (" · " + league_name if league_name else ""), ACCENT, "11px")}
    <div style="font-size:26px;font-weight:700;letter-spacing:-.02em;color:{INK};">{title}</div>
    <div style="font-family:{MONO};font-size:11px;color:{DIM};">{sub}</div>
  </div>
</div>
"""


def _result_tiles(label, probs, home, away, extra=""):
    names = [(f"{home} (H)", ACCENT), ("DRAW", MUTE), (f"{away} (A)", AWAY)]
    tiles = ""
    for (nm, col), p in zip(names, probs):
        fair = f"fair {1/p:.2f}" if p > 0.004 else "fair —"
        tiles += f"""
      <div style="flex:1;background:{TINT};border:1px solid {LINE2};border-radius:8px;
           padding:8px 9px;display:flex;flex-direction:column;gap:3px;min-width:0;">
        <span style="font-family:{MONO};font-size:9.5px;color:{DIM};letter-spacing:.1em;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{nm}</span>
        <span style="font-size:17px;font-weight:700;color:{col};">{p*100:.1f}%</span>
        <span style="font-family:{MONO};font-size:10px;color:{DIM};">{fair}</span>
      </div>"""
    return f"""
  <div style="display:flex;flex-direction:column;gap:7px;min-width:0;">
    {_lbl(label)}
    <div style="display:flex;gap:8px;">{tiles}</div>
    {extra}
  </div>"""


def summary_strip(home, away, res, ref_name, ref_stats, window):
    chips = ""
    if ref_stats:
        y_col = WARN if ref_stats["y_factor"] > 1.1 else ACCENT
        for txt, col in [(f"{ref_stats['games']} matches", MUTE),
                         (f"{ref_stats['y']:.1f} Y/match", y_col),
                         (f"{ref_stats['r']:.2f} R/match", MUTE)]:
            chips += (f'<span style="font-family:{MONO};font-size:10.5px;padding:4px 8px;'
                      f'border-radius:5px;background:{TINT};border:1px solid {LINE2};'
                      f'color:{col};">{txt}</span>')

    ht_extra = (f'<div style="font-family:{MONO};font-size:10px;color:{DIM};">'
                f'most likely HT score {res["ht_top"]} · {res["ht_top_p"]*100:.1f}%</div>')

    return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:18px 20px;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
     gap:18px;align-items:start;font-family:{SANS};margin-bottom:18px;">
  <div style="display:flex;flex-direction:column;gap:6px;min-width:0;">
    {_lbl("fixture")}
    <div style="font-size:21px;font-weight:700;letter-spacing:-.01em;color:{INK};">
      {home} <span style="color:{MUTE};font-weight:500;">v</span> {away}</div>
    <div style="font-family:{MONO};font-size:11px;color:{MUTE};">window {window} matches · EPL</div>
  </div>
  <div style="display:flex;flex-direction:column;gap:7px;min-width:0;">
    {_lbl("referee")}
    <div style="font-size:16px;font-weight:600;color:{INK};">{ref_name or "—"}</div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>
  </div>
  {_result_tiles("full-time result", res["ft"], home, away)}
  {_result_tiles("half-time (HTR)", res["ht"], home, away, ht_extra)}
</section>
"""


def market_card(key, name, feat, r, selected):
    """One compact market tile: lambda, home/away split, mini distribution."""
    t, h, a = r["sides"]["t"], r["sides"]["h"], r["sides"]["a"]
    mu = t["mu"]
    hp = h["mu"] / mu * 100 if mu else 50
    ap = 100 - hp
    border = ACCENT if selected else LINE

    lo, hi = max(0, t["lo"] - 2), t["hi"] + 2
    seg = t["pmf"][lo:hi + 1]
    mx = seg.max() if len(seg) and seg.max() > 0 else 1
    bars = ""
    for i, p in enumerate(seg):
        n = lo + i
        col = ACCENT if n > r["line"] else BAR_OFF
        bars += (f'<div style="flex:1;min-width:0;height:{max(2, p/mx*100):.1f}%;'
                 f'background:{col};border-radius:1.5px;"></div>')

    return f"""
<div style="background:{PANEL};border:1px solid {border};border-radius:12px;
     padding:14px 15px 12px;display:flex;flex-direction:column;gap:11px;
     min-width:0;font-family:{SANS};">
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
    <span style="font-size:13.5px;font-weight:600;letter-spacing:-.01em;color:{INK};">{name}</span>
    <span style="font-family:{MONO};font-size:10px;color:{DIM};">{feat}</span>
  </div>
  <div style="display:flex;align-items:flex-end;gap:10px;">
    <span style="font-size:34px;font-weight:800;letter-spacing:-.03em;line-height:.9;
          color:{ACCENT};">{mu:.2f}</span>
    <span style="font-family:{MONO};font-size:10.5px;color:{MUTE};padding-bottom:3px;">
      proj. total<br>{h['mu']:.2f} / {a['mu']:.2f}</span>
  </div>
  <div style="display:flex;gap:3px;height:6px;border-radius:3px;overflow:hidden;background:{TINT};">
    <div style="width:{hp:.1f}%;background:{ACCENT};border-radius:3px;"></div>
    <div style="width:{ap:.1f}%;background:{AWAY};border-radius:3px;"></div>
  </div>
  <div style="display:flex;align-items:flex-end;gap:2px;height:46px;">{bars}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;
       border-top:1px solid {LINE};padding-top:9px;">
    <span style="font-family:{MONO};font-size:11px;color:{MUTE};">Over {r['line']:.1f}</span>
    <span style="font-family:{MONO};font-size:12px;font-weight:600;color:{INK};">{r['p_over']*100:.1f}%</span>
    <span style="font-family:{MONO};font-size:11px;color:{DIM};">likely {t['most_likely']}</span>
  </div>
</div>
"""


def market_grid(cards_html):
    return (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(258px,1fr));'
            f'gap:12px;margin-bottom:18px;">{cards_html}</div>')


def detail_panel(name, r, window, drivers, note):
    """Large distribution + over/under ladder + driver list."""
    t = r["sides"]["t"]
    lo, hi = max(0, t["lo"] - 3), t["hi"] + 3
    seg = t["pmf"][lo:hi + 1]
    mx = seg.max() if len(seg) and seg.max() > 0 else 1

    bars = ""
    for i, p in enumerate(seg):
        n = lo + i
        col = ACCENT if n > r["line"] else BAR_OFF
        lab = f"{p*100:.0f}" if p > 0.055 else ""
        bars += f"""
      <div style="flex:1;min-width:0;display:flex;flex-direction:column;
           justify-content:flex-end;align-items:center;gap:5px;height:100%;">
        <span style="font-family:{MONO};font-size:9px;color:{DIM};">{lab}</span>
        <div style="width:100%;height:{max(2, p/mx*100):.1f}%;background:{col};
             border-radius:2px 2px 0 0;"></div>
        <span style="font-family:{MONO};font-size:10px;color:{MUTE};">{n}</span>
      </div>"""

    rows = ""
    for l in r["ladder"]:
        bg = TINT if abs(l["line"] - r["line"]) < 1e-9 else "transparent"
        rows += f"""
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;
           font-family:{MONO};font-size:12px;padding:7px 8px;border-radius:6px;background:{bg};">
        <span style="color:{INK};">{l['line']:.1f}</span>
        <span style="text-align:right;color:{ACCENT};">{l['over']*100:.1f}%</span>
        <span style="text-align:right;color:{MUTE};">{l['under']*100:.1f}%</span>
      </div>"""

    drv = ""
    for label, val in drivers:
        drv += f"""
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;
           border-bottom:1px solid {LINE};padding:6px 0;">
        <span style="font-family:{MONO};font-size:11px;color:{MUTE};">{label}</span>
        <span style="font-family:{MONO};font-size:12px;font-weight:600;color:{INK};">{val}</span>
      </div>"""

    return f"""
<section style="background:{PANEL};border:1px solid {LINE2};border-radius:12px;
     padding:18px 20px;display:flex;flex-direction:column;gap:16px;font-family:{SANS};">
  <div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:baseline;">
    <div style="display:flex;flex-direction:column;gap:3px;">
      {_lbl("market detail", ACCENT)}
      <span style="font-size:19px;font-weight:700;letter-spacing:-.01em;color:{INK};">{name}</span>
    </div>
    <span style="font-family:{MONO};font-size:11px;color:{MUTE};">
      μ = {t['mu']:.2f} · 80% {t['lo']}–{t['hi']} · window {window}</span>
  </div>
  <div style="display:flex;align-items:flex-end;gap:3px;height:150px;">{bars}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;">
    <div style="display:flex;flex-direction:column;gap:8px;">
      {_lbl("over / under ladder")}
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-family:{MONO};
           font-size:9.5px;color:{DIM};letter-spacing:.08em;padding:0 8px 5px;">
        <span>line</span><span style="text-align:right;">over</span><span style="text-align:right;">under</span>
      </div>
      {rows}
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;">
      {_lbl("what drives this number")}
      {drv}
      <div style="font-family:{MONO};font-size:10.5px;color:{DIM};line-height:1.5;">{note}</div>
    </div>
  </div>
</section>
"""


def feature_table(home, away, rows, window):
    body = ""
    for key, hv, av in rows:
        mx = max(hv, av) * 1.18 or 1
        body += f"""
    <div style="display:flex;flex-direction:column;gap:4px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
        <span style="font-family:{MONO};font-size:12px;font-weight:600;color:{INK};">{hv:.2f}</span>
        <span style="font-family:{MONO};font-size:10px;color:{DIM};">{key}</span>
        <span style="font-family:{MONO};font-size:12px;font-weight:600;color:{INK};">{av:.2f}</span>
      </div>
      <div style="display:flex;gap:3px;align-items:center;">
        <div style="flex:1;display:flex;justify-content:flex-end;height:5px;background:{TINT};
             border-radius:3px;overflow:hidden;">
          <div style="width:{hv/mx*100:.1f}%;background:{ACCENT};"></div></div>
        <div style="flex:1;display:flex;height:5px;background:{TINT};
             border-radius:3px;overflow:hidden;">
          <div style="width:{av/mx*100:.1f}%;background:{AWAY};"></div></div>
      </div>
    </div>"""

    return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:16px 17px;display:flex;flex-direction:column;gap:14px;font-family:{SANS};
     margin-bottom:18px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    {_lbl("feature table")}
    <span style="font-family:{MONO};font-size:10px;color:{MUTE};">last {window}</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600;">
    <span style="color:{ACCENT};">{home}</span><span style="color:{AWAY};">{away}</span>
  </div>
  {body}
</section>
"""


def form_trace(traces):
    body = ""
    for name, col, summary, pts in traces:
        bars = ""
        for p in pts:
            h = {3: 100, 1: 52}.get(p, 16)
            c = col if p == 3 else (MUTE if p == 1 else "#232a28")
            bars += (f'<div style="flex:1;min-width:0;height:{h}%;background:{c};'
                     f'border-radius:1.5px;"></div>')
        body += f"""
    <div style="display:flex;flex-direction:column;gap:7px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <span style="font-size:13px;font-weight:600;color:{col};">{name}</span>
        <span style="font-family:{MONO};font-size:10.5px;color:{MUTE};">{summary}</span>
      </div>
      <div style="display:flex;align-items:flex-end;gap:2px;height:40px;">{bars}</div>
    </div>"""

    return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:16px 17px;display:flex;flex-direction:column;gap:14px;font-family:{SANS};
     margin-bottom:18px;">
  {_lbl("form trace · newest right")}
  {body}
</section>
"""


def referee_panel(name, stats, league, available=True):
    if not available:
        return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:16px 17px;display:flex;flex-direction:column;gap:10px;font-family:{SANS};">
  {_lbl("referee effect")}
  <div style="font-family:{MONO};font-size:10.5px;color:{DIM};line-height:1.55;">
    football-data.co.uk publishes a Referee column for the Premier League only,
    so card and foul projections for this league carry no official adjustment.
  </div>
</section>
"""
    if not stats:
        return ""
    rows = [("yellows / match", stats["y"], league["y"], 6.5),
            ("fouls / match", stats["f"], league["f"], 30.0),
            ("reds / match", stats["r"], league["r"], 0.4)]
    body = ""
    for label, v, avg, mx in rows:
        col = WARN if v > avg * 1.1 else ACCENT
        body += f"""
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">
        <span style="font-family:{MONO};font-size:11px;color:{MUTE};">{label}</span>
        <span style="font-family:{MONO};font-size:12px;font-weight:600;color:{col};">{v:.2f}</span>
      </div>
      <div style="position:relative;height:5px;background:{TINT};border-radius:3px;">
        <div style="position:absolute;left:0;top:0;bottom:0;width:{min(100, v/mx*100):.1f}%;
             background:{col};border-radius:3px;"></div>
        <div style="position:absolute;left:{min(100, avg/mx*100):.1f}%;top:-3px;bottom:-3px;
             width:1.5px;background:#5a6562;"></div>
      </div>
    </div>"""

    shift = (stats["y_factor"] - 1) * 100
    return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:16px 17px;display:flex;flex-direction:column;gap:12px;font-family:{SANS};">
  {_lbl("referee effect")}
  {body}
  <div style="font-family:{MONO};font-size:10.5px;color:{DIM};line-height:1.5;">
    Marker = league average. Card and foul projections are multiplied by this
    official's rate, so {name} moves yellows by {shift:+.0f}%.</div>
</section>
"""


def footer(n_matches, last_date):
    return f"""
<div style="font-family:{MONO};font-size:10px;color:#7f8a86;border-top:1px solid {LINE};
     padding-top:12px;margin-top:22px;">
  {n_matches} EPL matches through {last_date} · football-data.co.uk ·
  negative binomial GLM per side, fitted on rolling form, same-venue record and head-to-head.
</div>
"""


# ----------------------------------------------------------------------
# PLAYER DRILL-DOWN
# ----------------------------------------------------------------------
def player_headline(top, verb, market_name):
    if not top:
        return ""
    return f"""
<div style="background:{PANEL};border:1px solid {ACCENT};border-radius:12px;
     padding:16px 18px;display:flex;flex-wrap:wrap;gap:18px;align-items:center;
     justify-content:space-between;font-family:{SANS};margin-bottom:14px;">
  <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
    {_lbl("most likely to record " + verb, ACCENT)}
    <span style="font-size:24px;font-weight:800;letter-spacing:-.02em;color:{INK};">{top['player']}</span>
    <span style="font-family:{MONO};font-size:10.5px;color:{MUTE};">
      {top['pos']} · {top['exp_min']:.0f} exp. min · {top['rate_p90']:.2f} per 90 ·
      {top['season_total']:.0f} in {top['minutes']:.0f} min this season</span>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;">
    <span style="font-size:34px;font-weight:800;letter-spacing:-.03em;line-height:.9;
          color:{ACCENT};">{top['p_any']*100:.0f}%</span>
    <span style="font-family:{MONO};font-size:10px;color:{DIM};">chance of 1+</span>
  </div>
</div>
"""


def player_table(team, rows, color, market_name):
    if not rows:
        return (f'<div style="font-family:{MONO};font-size:11px;color:{DIM};">'
                f'No player data for {team}.</div>')
    body = ""
    mx = max(r["p_any"] for r in rows) or 1
    for i, r in enumerate(rows):
        bar = r["p_any"] / mx * 100
        body += f"""
    <div style="display:grid;grid-template-columns:18px 1fr 52px 54px;gap:8px;
         align-items:center;padding:6px 0;border-bottom:1px solid {LINE};">
      <span style="font-family:{MONO};font-size:10px;color:{DIM};">{i+1}</span>
      <div style="min-width:0;display:flex;flex-direction:column;gap:3px;">
        <span style="font-size:12.5px;font-weight:600;color:{INK};white-space:nowrap;
              overflow:hidden;text-overflow:ellipsis;">{r['player']}</span>
        <div style="height:4px;background:{TINT};border-radius:2px;overflow:hidden;">
          <div style="width:{bar:.1f}%;height:100%;background:{color};"></div></div>
      </div>
      <span style="font-family:{MONO};font-size:10.5px;color:{MUTE};text-align:right;">
        {r['pos']} {r['exp_min']:.0f}'</span>
      <span style="font-family:{MONO};font-size:12.5px;font-weight:600;color:{color};
            text-align:right;">{r['p_any']*100:.0f}%</span>
    </div>"""

    return f"""
<div style="display:flex;flex-direction:column;gap:6px;min-width:0;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;
       border-bottom:1px solid {LINE2};padding-bottom:6px;">
    <span style="font-size:13.5px;font-weight:700;color:{color};">{team}</span>
    <span style="font-family:{MONO};font-size:9.5px;color:{DIM};letter-spacing:.1em;">P(1+)</span>
  </div>
  {body}
</div>
"""


def player_section(home, away, out, market_name, verb, note):
    return f"""
<section style="background:{PANEL};border:1px solid {LINE2};border-radius:12px;
     padding:18px 20px;display:flex;flex-direction:column;gap:14px;font-family:{SANS};
     margin-top:18px;">
  <div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;align-items:baseline;">
    <div style="display:flex;flex-direction:column;gap:3px;">
      {_lbl("player breakdown", ACCENT)}
      <span style="font-size:19px;font-weight:700;letter-spacing:-.01em;color:{INK};">{market_name}</span>
    </div>
    <span style="font-family:{MONO};font-size:10.5px;color:{MUTE};">
      squad totals rescaled to the team projection</span>
  </div>
  {player_headline(out['top'], verb, market_name)}
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:22px;">
    {player_table(home, out['home'], ACCENT, market_name)}
    {player_table(away, out['away'], AWAY, market_name)}
  </div>
  <div style="font-family:{MONO};font-size:10.5px;color:{DIM};line-height:1.55;
       border-top:1px solid {LINE};padding-top:10px;">{note}</div>
</section>
"""


def no_player_data(market_name):
    return f"""
<section style="background:{PANEL};border:1px solid {LINE};border-radius:12px;
     padding:18px 20px;font-family:{MONO};font-size:11px;color:{DIM};line-height:1.6;
     margin-top:18px;">
  <span style="color:{INK};font-size:13px;font-family:{SANS};font-weight:600;">
    No player breakdown for {market_name}</span><br><br>
  Corners are awarded to a team, not attributed to a player in any of the
  underlying feeds, so there is nothing to rank. Player views exist for
  goals, shots on target, yellow cards and fouls.
</section>
"""


def lineup_banner(home, away, mult, active):
    """State whether projections are running on a published XI or a usage average."""
    if not active:
        return f"""
<div style="background:{PANEL};border:1px solid {LINE};border-left:3px solid {MUTE};
     border-radius:8px;padding:10px 14px;font-family:{MONO};font-size:10.5px;
     color:{DIM};line-height:1.5;margin-bottom:12px;">
  <b style="color:{INK};">No lineup entered.</b> Expected minutes is a usage
  average, so a rotated or rested player still reads high. Enter the XI below
  once team news lands and every number on this page sharpens.
</div>
"""
    def band(m):
        if m >= 1.02: return ACCENT, "stronger than usual"
        if m <= 0.95: return WARN, "weakened"
        return ACCENT, "about full strength"
    ch, th = band(mult["home"]); ca, ta = band(mult["away"])
    return f"""
<div style="background:{PANEL};border:1px solid {ACCENT};border-radius:8px;
     padding:10px 14px;font-family:{MONO};font-size:10.5px;color:{DIM};
     line-height:1.6;margin-bottom:12px;display:flex;flex-wrap:wrap;gap:16px;
     justify-content:space-between;align-items:center;">
  <span><b style="color:{ACCENT};">Lineup applied.</b> Minutes are facts, not averages.</span>
  <span>
    <b style="color:{ch};">{home} ×{mult['home']:.3f}</b> <span style="color:{DIM};">{th}</span>
    &nbsp;·&nbsp;
    <b style="color:{ca};">{away} ×{mult['away']:.3f}</b> <span style="color:{DIM};">{ta}</span>
  </span>
</div>
"""
