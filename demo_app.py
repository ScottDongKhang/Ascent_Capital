"""
Ascent Capital — Interactive Demo
Run: .venv/bin/streamlit run demo_app.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Ascent Capital",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  .stApp { background-color: #080808; color: #C8C8C8; }
  section[data-testid="stSidebar"] {
    background-color: #0C0C0C;
    border-right: 1px solid #1A1A1A;
  }
  .block-container { padding-top: 3rem; max-width: 1080px; }
  #MainMenu, footer, header { visibility: hidden; }
  h1, h2, h3 { color: #E8E8E8 !important; font-weight: 500 !important; }

  /* Divider */
  .div { border: none; border-top: 1px solid #1A1A1A; margin: 32px 0; }

  /* Section header */
  .sec-head {
    font-size: 10px; font-weight: 500; letter-spacing: 2px;
    text-transform: uppercase; color: #444; margin-bottom: 20px;
  }

  /* Mono numbers */
  .mono { font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace; }

  /* Position row */
  .pos-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #111; font-size: 13px;
  }
  .pos-sym { color: #E0E0E0; font-weight: 500; }
  .pos-wt  { font-family: 'JetBrains Mono', monospace; color: #B8962E; font-size: 12px; }
  .pos-more { color: #333; font-size: 11px; padding-top: 8px; }

  /* Stat strip */
  .stat-strip { display: flex; gap: 0; margin-bottom: 36px; border: 1px solid #1A1A1A; }
  .stat-item {
    flex: 1; padding: 14px 20px; border-right: 1px solid #1A1A1A;
  }
  .stat-item:last-child { border-right: none; }
  .stat-lbl { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: #3A3A3A; margin-bottom: 6px; }
  .stat-val { font-family: 'JetBrains Mono', monospace; font-size: 18px; color: #C8C8C8; }
  .stat-val.pos { color: #4A7C59; }
  .stat-val.neg { color: #7C4A4A; }
  .stat-val.gold { color: #B8962E; }

  /* Regime pill */
  .regime-pill {
    display: inline-block; padding: 3px 10px;
    font-size: 10px; font-weight: 500; letter-spacing: 1px;
    text-transform: uppercase; border-radius: 2px;
  }
  .regime-calm_bull  { background: #0D1F14; color: #4A7C59; border: 1px solid #1A3324; }
  .regime-stressed   { background: #1F180D; color: #8A6A2E; border: 1px solid #332A1A; }
  .regime-crisis     { background: #1F0D0D; color: #7C4A4A; border: 1px solid #331A1A; }
  .regime-uncertain  { background: #150D1F; color: #6A4A7C; border: 1px solid #271A33; }

  /* Signal row */
  .signal-row { display: flex; align-items: baseline; gap: 12px; margin: 16px 0; }
  .signal-key { font-size: 10px; letter-spacing: 1px; text-transform: uppercase; color: #333; width: 110px; flex-shrink: 0; }
  .signal-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #C8C8C8; }

  /* Sleeve table */
  .sleeve-table { width: 100%; border-collapse: collapse; margin-top: 4px; }
  .sleeve-table th {
    font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase;
    color: #333; font-weight: 500; padding: 0 0 10px 0; text-align: left;
    border-bottom: 1px solid #1A1A1A;
  }
  .sleeve-table th:not(:first-child) { text-align: right; }
  .sleeve-table td { padding: 8px 0; font-size: 12px; color: #888; border-bottom: 1px solid #0F0F0F; }
  .sleeve-table td:not(:first-child) { text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
  .sleeve-table td:first-child { color: #C0C0C0; }
  .sleeve-up   { color: #4A7C59 !important; }
  .sleeve-down { color: #7C4A4A !important; }
  .sleeve-flat { color: #2A2A2A !important; }

  /* Analyst card */
  .analyst-wrap { margin: 12px 0; }
  .analyst-meta {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 8px;
  }
  .analyst-role {
    font-size: 9px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase;
  }
  .analyst-role.bull   { color: #4A7C59; }
  .analyst-role.bear   { color: #7C4A4A; }
  .analyst-role.devil  { color: #6A4A7C; }
  .analyst-role.regime { color: #2A5A7C; }
  .analyst-role.quant  { color: #3A3A3A; }
  .analyst-line { flex: 1; height: 1px; background: #141414; }
  .analyst-r2   { opacity: 0.65; }
  .analyst-text {
    font-size: 13px; color: #888; line-height: 1.75;
    padding-left: 0; padding-bottom: 20px;
    border-bottom: 1px solid #111;
  }
  .analyst-text.last { border-bottom: none; }

  /* Verdict */
  .verdict-wrap { margin-top: 32px; padding: 24px 0; border-top: 1px solid #1A1A1A; }
  .verdict-status {
    font-size: 10px; font-weight: 600; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 16px;
  }
  .verdict-status.proceed { color: #4A7C59; }
  .verdict-status.reduce  { color: #8A6A2E; }
  .verdict-status.halt    { color: #7C4A4A; }
  .verdict-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #333; margin-bottom: 16px;
  }
  .verdict-text { font-size: 13px; color: #666; line-height: 1.75; max-width: 680px; }
  .verdict-risks { margin-top: 14px; }
  .verdict-risk  { font-size: 11px; color: #3A3A3A; margin: 4px 0; font-family: 'JetBrains Mono', monospace; }

  /* Sidebar */
  .sb-logo {
    padding: 28px 0 20px 0; margin-bottom: 4px;
    border-bottom: 1px solid #1A1A1A;
  }
  .sb-logo-mark { font-size: 18px; font-weight: 600; color: #B8962E; letter-spacing: 3px; }
  .sb-logo-sub  { font-size: 9px; color: #2A2A2A; letter-spacing: 4px; margin-top: 2px; }
  .sb-section   { font-size: 9px; color: #2A2A2A; letter-spacing: 2px;
                  text-transform: uppercase; margin: 24px 0 10px 0; }

  /* Run button override */
  .stButton > button {
    background: #0E0E0E !important; border: 1px solid #B8962E !important;
    color: #B8962E !important; font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    border-radius: 2px !important; padding: 10px !important;
  }
  .stButton > button:hover {
    background: #B8962E !important; color: #080808 !important;
  }

  /* Debate prompt */
  .debate-prompt {
    border: 1px solid #141414; padding: 48px;
    text-align: center; margin-top: 8px;
  }
  .debate-prompt-text { font-size: 12px; color: #2A2A2A; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


# ── Live data ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_live_data() -> tuple:
    root = Path(__file__).parent
    weights, regime, vix, spy_mom = {}, "stressed", 20.0, "Neutral"
    try:
        with open(root / "execution" / "merged_weights.json") as f:
            weights = json.load(f).get("weights", {})
    except Exception:
        pass
    try:
        with open(root / "dashboard" / "regime_signal.json") as f:
            regime = json.load(f).get("regime", "stressed")
    except Exception:
        pass
    try:
        import yfinance as yf
        vix = float(yf.Ticker("^VIX").fast_info.get("last_price", 20.0))
        spy = yf.Ticker("SPY").history(period="60d")["Close"]
        if len(spy) >= 20:
            pct = (spy.iloc[-1] - spy.iloc[-20:].mean()) / spy.iloc[-20:].mean()
            if pct > 0.03:    spy_mom = "Strong positive"
            elif pct > 0.01:  spy_mom = "Weak positive"
            elif pct < -0.03: spy_mom = "Strong negative"
            elif pct < -0.01: spy_mom = "Weak negative"
            else:             spy_mom = "Neutral"
    except Exception:
        pass
    return weights, regime, vix, spy_mom


_live_weights, _live_regime, _live_vix, _live_spy_mom = _load_live_data()

_REGIME_OPTIONS  = ["calm_bull", "stressed", "crisis", "uncertain"]
_SPY_MOM_OPTIONS = ["Strong negative", "Weak negative", "Neutral", "Weak positive", "Strong positive"]
_live_regime_idx  = _REGIME_OPTIONS.index(_live_regime) if _live_regime in _REGIME_OPTIONS else 1
_live_spy_mom_idx = _SPY_MOM_OPTIONS.index(_live_spy_mom) if _live_spy_mom in _SPY_MOM_OPTIONS else 2

REGIME_META = {
    "calm_bull": {"risk": 1.00, "max_wt": 0.15, "label": "Calm Bull"},
    "stressed":  {"risk": 0.65, "max_wt": 0.10, "label": "Stressed"},
    "crisis":    {"risk": 0.40, "max_wt": 0.08, "label": "Crisis"},
    "uncertain": {"risk": 0.75, "max_wt": 0.12, "label": "Uncertain"},
}

BASE_SLEEVES = {
    "Trend":           0.44,
    "Stat-Arb":        0.15,
    "ML (XGBoost)":    0.10,
    "Vol-Regime":      0.05,
    "Mean Reversion":  0.05,
    "Fundamental":     0.05,
    "Earnings (PEAD)": 0.05,
    "Analyst":         0.05,
    "Options Flow":    0.02,
    "Insider":         0.02,
    "Short Interest":  0.02,
}

SLEEVE_DELTA = {
    "calm_bull": {k: 0.0 for k in BASE_SLEEVES},
    "stressed":  {"Trend": .08, "Stat-Arb": -.03, "ML (XGBoost)": -.07, "Vol-Regime": .03, "Mean Reversion": .02, "Fundamental": -.02, "Earnings (PEAD)": -.01, "Analyst": -.01, "Options Flow": 0, "Insider": 0, "Short Interest": 0},
    "crisis":    {"Trend": .12, "Stat-Arb": -.05, "ML (XGBoost)": -.08, "Vol-Regime": .05, "Mean Reversion": .03, "Fundamental": -.04, "Earnings (PEAD)": -.02, "Analyst": -.01, "Options Flow": 0, "Insider": 0, "Short Interest": 0},
    "uncertain": {"Trend": .05, "Stat-Arb": -.02, "ML (XGBoost)": -.03, "Vol-Regime": .02, "Mean Reversion": 0,   "Fundamental": -.01, "Earnings (PEAD)": -.01, "Analyst": 0,    "Options Flow": 0, "Insider": 0, "Short Interest": 0},
}

_FALLBACK = {
    "KMLM": 0.097, "IFRA": 0.087, "PDBC": 0.082, "AMKR": 0.056,
    "FIX": 0.056, "VAL": 0.056, "VICR": 0.056, "VRT": 0.056, "WDC": 0.056,
    "CNC": 0.054, "EWY": 0.041, "DBB": 0.043, "VC": 0.038,
    "DBA": 0.033, "EWC": 0.033, "PVH": 0.032, "TIP": 0.022,
    "NUE": 0.020, "XPO": 0.019, "APA": 0.019, "AAXJ": 0.015, "EEM": 0.015, "EWZ": 0.012,
}

PRESETS = {
    f"Live — {date.today().strftime('%b %d, %Y')}": _live_weights or _FALLBACK,
    "Tech-Heavy": {"AAPL": 0.15, "MSFT": 0.13, "NVDA": 0.12, "GOOGL": 0.10, "META": 0.10, "ADBE": 0.08, "CRWD": 0.08, "AMD": 0.07, "INTC": 0.06, "TXN": 0.06, "PANW": 0.05},
    "Defensive":  {"JNJ": 0.12, "WMT": 0.12, "KO": 0.10, "PG": 0.10, "NEE": 0.09, "MRK": 0.09, "ABT": 0.08, "DUK": 0.08, "TLT": 0.12, "GLD": 0.10},
    "Macro Stress": {"TLT": 0.25, "GLD": 0.20, "BIL": 0.15, "IEF": 0.15, "VIXY": 0.10, "UUP": 0.08, "PDBC": 0.07},
}


# ── Debate copy ──────────────────────────────────────────────────────────────────
def _debate_copy(regime: str, vix: float, holdings: list, spy_mom: str) -> dict:
    h = holdings[:4]
    hs = ", ".join(h) if h else "the portfolio"
    first = h[0] if h else "top holding"
    md = {"Strong positive": "strong upward momentum", "Weak positive": "weakening upward momentum",
          "Neutral": "flat momentum", "Weak negative": "early downtrend signals",
          "Strong negative": "sustained downtrend"}.get(spy_mom, "mixed signals")
    n = len(holdings)

    scenarios = {
        "calm_bull": {
            "bull":   f"The regime is unambiguous. {hs} — these are compounders with durable cash generation, not rate-sensitive cyclicals. VIX at {vix:.0f} is benign. Institutional flows are constructive. The trend signal is intact. Over-hedging a calm bull regime has historically cost more in missed upside than any drawdown protection delivered.",
            "bear":   f"Calm bull is precisely when complacency gets punished. {first} is pricing in a forward earnings trajectory that assumes no margin compression, no multiple contraction, and no macro shock. VIX at {vix:.0f} means tail risk is unpriced, not absent. The last three calm-bull regimes ended with an average -14% drawdown within 8 weeks of the signal peak.",
            "devil":  f"Both arguments assume the correlation structure holds. It won't. In 2018, 2020, and 2022, cross-portfolio correlation moved from 0.32 to 0.79 within 11 trading days of regime transition. Our {n}-name book becomes a single-factor bet at exactly the wrong moment. Neither the bull nor the bear is modeling forced unwinds at illiquid prints.",
            "regime": f"Calm bull playbook: full deployment, momentum tilt, max position at 15%. The regime classifier has stayed in this state for 14+ days with entropy well below the 0.90 threshold — this is a high-conviction signal. Set the trip-wire at VIX 22 or SPY below the 50-day MA. Do not act until the signal moves.",
        },
        "stressed": {
            "bull":   f"Stressed regimes create the separation between quality and noise. {hs} have balance sheets that don't require credit markets to function. VIX at {vix:.0f} is elevated but structurally, this is a repricing event, not a solvency event. The market is discounting a macro outcome with a 35% probability as though it's certain.",
            "bear":   f"VIX at {vix:.0f} is real institutional hedging, not retail fear. SPY showing {md}. When VIX has crossed 25 in a stressed regime, the median forward drawdown from entry has been -14% over 6 weeks. {first} is still pricing in growth assumptions that are now at risk. The model says 65% gross exposure. That's the number.",
            "devil":  f"The credit spread argument is what both sides are missing. If HY spreads widen 150bps from current levels — entirely consistent with this regime — equity correlations spike and bid/ask on our smaller positions widens to 80bps. The scenario neither the bull nor bear is running: forced exit at -10% slippage on three names simultaneously. Monte Carlo p5 shows -21% in that path.",
            "regime": f"Stressed regime protocol: 65% gross exposure, 10% max position, quality tilt. SPY showing {md} — no recovery confirmation. Typical stressed duration is 4–12 weeks. We are early. The most common error in this regime is re-risking on the first green day. Wait for entropy to drop below 0.55 before adding back.",
        },
        "crisis": {
            "bull":   f"Crisis regimes produce the best forward returns from entry — if you have dry powder. {hs} will be structurally intact on the other side. VIX at {vix:.0f} is extreme fear, not rational pricing. 2009 and 2020 both saw 40%+ returns in the 12 months following maximum VIX. Exiting here means locking in the loss and missing the recovery.",
            "bear":   f"VIX at {vix:.0f} is a data point, not a prediction. The 2008 crisis peaked at 89.53 — we may not be at maximum stress. In a crisis regime, the primary risk is permanent capital impairment, not opportunity cost. Correlation across this book approaches 1.0 in a forced liquidation. Cash is not a missed opportunity; it is the instrument that makes recovery participation possible.",
            "devil":  f"Both sides are anchoring on historical analogs with defined endpoints. Some crises resolve in weeks. Some take 18 months. The fatal assumption is that {first} survives without permanent impairment. In a true tail event, our highest-conviction names carry the most model risk — they're also the most owned institutionally, meaning the forced selling is worst there. 40% gross exposure. No exceptions.",
            "regime": f"Crisis protocol: 40% gross exposure, 8% max position, no speculative names. Capital preservation is the only objective. Re-risking criteria are explicit: Fed intervention with balance sheet expansion, HY spreads rolling over, VIX closing below 30 on consecutive days. None of those conditions are met. Hold the line.",
        },
        "uncertain": {
            "bull":   f"Regime uncertainty is not a sell signal. {hs} are earnings-driven compounders — the fundamental case is independent of whether the classifier has conviction. The cost of being too defensive in an uncertain regime that resolves bullish is real and permanent. 75% exposure, hold the quality book, let the regime clarify.",
            "bear":   f"Uncertain regime with {md} in SPY is the worst risk/reward setup we can hold. We don't know the resolution direction, but we're carrying near-full exposure into that uncertainty. Asymmetry is wrong: crisis resolution means -20%+, calm bull resolution means capped upside from sizing constraints. Reduce to 60% and buy the optionality.",
            "devil":  f"The regime uncertainty is itself diagnostic. The classifier is uncertain because the features that trained it are no longer predictive — this is a structural break signal, not rotation noise. Neither the bull nor bear is modeling a world where our regime framework stops working. That is the actual tail risk. A framework that cannot identify its own failure mode is the most dangerous kind.",
            "regime": f"Uncertain regime resolution time: historically 1–3 weeks. Posture: 75% gross exposure, 12% max position, no new positions until entropy drops below 0.70. Do not force a thesis. The signal does not have sufficient conviction in either direction. Patience is a position.",
        },
    }
    return scenarios.get(regime, scenarios["uncertain"])


# ── Verdict ──────────────────────────────────────────────────────────────────────
def _verdict(regime: str, vix: float) -> dict:
    rm = REGIME_META[regime]
    if regime == "crisis" or vix > 40:
        return {
            "rec": "halt_and_review", "conf": 0.78,
            "risks": ["Regime signal: crisis — risk envelope exceeded", f"VIX {vix:.0f} indicates tail risk is actively being priced", "Correlation structure breakdown — diversification benefit near zero"],
            "text": f"The crisis regime signal combined with VIX={vix:.0f} places this portfolio outside the parameters for normal execution. The devil's advocate argument regarding forced-unwind correlation dynamics was not adequately addressed by either side. Halting for principal review is the correct procedural response. Resume only after explicit sign-off.",
        }
    elif regime == "stressed" or vix > 25:
        return {
            "rec": "reduce_size", "conf": 0.71,
            "risks": [f"Target gross exposure: {rm['risk']:.0%} (current: ~100%)", f"VIX {vix:.0f} — institutional hedging activity elevated", "Bear's forward drawdown estimate (-14%) not rebutted"],
            "text": f"The bull's quality argument is noted and partially valid, but it does not address the regime-level sizing mandate. The model calls for {rm['risk']:.0%} gross exposure in a stressed regime. That is the output of 160 walk-forward folds — it is not a suggestion. Reduce gross exposure to {rm['risk']:.0%}, maintain the quality names, max position {rm['max_wt']:.0%}. Re-evaluate at next rebalance.",
        }
    else:
        return {
            "rec": "proceed", "conf": 0.82,
            "risks": ["Monitor for regime transition: VIX >22 or SPY break of 50-day MA", "Top-3 concentration warrants position-level stop-loss discipline"],
            "text": "The calm bull regime signal is high-conviction. The bull case is well-supported and the bear's tail-risk concerns, while structurally valid, are not sufficient to override the primary signal in this environment. The quant check is clean. Proceed with full deployment. Maintain the trip-wires specified by the regime specialist.",
        }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-mark">▲ ASCENT</div>
      <div class="sb-logo-sub">CAPITAL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Scenario</div>', unsafe_allow_html=True)

    regime = st.selectbox(
        "Regime", _REGIME_OPTIONS, index=_live_regime_idx,
        format_func=lambda x: REGIME_META[x]["label"],
        help="HMM regime classification. Drives sleeve weights, gross exposure, and position caps.",
    )
    vix = st.slider("VIX", 10.0, 55.0, round(_live_vix, 1), 0.5)
    spy_momentum = st.select_slider("SPY Momentum", _SPY_MOM_OPTIONS, value=_live_spy_mom)

    st.markdown('<div class="sb-section">Portfolio</div>', unsafe_allow_html=True)
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    weights = PRESETS[preset_name].copy()

    st.markdown('<div class="sb-section">Display</div>', unsafe_allow_html=True)
    show_r2 = st.toggle("Round 2 rebuttals", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run debate", use_container_width=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────────
rm = REGIME_META[regime]
gross = rm["risk"]
top_holdings = sorted(weights.items(), key=lambda x: -x[1])

# Header
st.markdown(f"""
<div style="margin-bottom: 32px;">
  <div style="font-size: 11px; color: #2A2A2A; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px;">
    Ascent Capital
  </div>
  <div style="font-size: 22px; font-weight: 300; color: #E0E0E0; letter-spacing: -0.3px;">
    Pre-Rebalance Decision Engine
  </div>
  <div style="font-size: 12px; color: #2E2E2E; margin-top: 6px;">
    {date.today().strftime('%A, %B %d, %Y')} &nbsp;·&nbsp; 4 agents &nbsp;·&nbsp; 11 alpha sleeves &nbsp;·&nbsp; S&P 500 universe
  </div>
</div>
""", unsafe_allow_html=True)

# OOS stat strip
st.markdown("""
<div class="stat-strip">
  <div class="stat-item">
    <div class="stat-lbl">OOS CAGR</div>
    <div class="stat-val gold">22.4%</div>
  </div>
  <div class="stat-item">
    <div class="stat-lbl">Sharpe Ratio</div>
    <div class="stat-val gold">0.887</div>
  </div>
  <div class="stat-item">
    <div class="stat-lbl">Max Drawdown</div>
    <div class="stat-val neg">−35.8%</div>
  </div>
  <div class="stat-item">
    <div class="stat-lbl">vs SPY (CAGR)</div>
    <div class="stat-val pos">+8.9%</div>
  </div>
  <div class="stat-item">
    <div class="stat-lbl">Walk-fwd Folds</div>
    <div class="stat-val">160</div>
  </div>
  <div class="stat-item">
    <div class="stat-lbl">Period</div>
    <div class="stat-val" style="font-size:13px;">2020–2026</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Portfolio + signals
col1, col2 = st.columns([5, 3])

with col1:
    st.markdown('<div class="sec-head">Current Portfolio</div>', unsafe_allow_html=True)
    rows_html = ""
    for sym, w in top_holdings[:10]:
        rows_html += f"""<div class="pos-row"><span class="pos-sym">{sym}</span><span class="pos-wt">{w:.1%}</span></div>"""
    if len(top_holdings) > 10:
        rows_html += f'<div class="pos-more">+{len(top_holdings)-10} positions</div>'
    st.markdown(rows_html, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-head">Market State</div>', unsafe_allow_html=True)
    spy_color = "#4A7C59" if "positive" in spy_momentum.lower() else ("#7C4A4A" if "negative" in spy_momentum.lower() else "#3A3A3A")
    st.markdown(f"""
    <div class="signal-row">
      <span class="signal-key">Regime</span>
      <span class="regime-pill regime-{regime}">{rm['label']}</span>
    </div>
    <div class="signal-row">
      <span class="signal-key">VIX</span>
      <span class="signal-val">{vix:.1f}</span>
    </div>
    <div class="signal-row">
      <span class="signal-key">SPY Momentum</span>
      <span class="signal-val" style="color:{spy_color};font-size:11px;">{spy_momentum}</span>
    </div>
    <div class="signal-row">
      <span class="signal-key">Gross Exposure</span>
      <span class="signal-val">{gross:.0%}</span>
    </div>
    <div class="signal-row">
      <span class="signal-key">Max Position</span>
      <span class="signal-val">{rm['max_wt']:.0%}</span>
    </div>
    <div class="signal-row">
      <span class="signal-key">Positions</span>
      <span class="signal-val">{len(weights)}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# Sleeve table
st.markdown('<div class="sec-head">Alpha Allocation — Regime Adjusted</div>', unsafe_allow_html=True)
delta_map = SLEEVE_DELTA.get(regime, {})
rows = ""
for sleeve, base in BASE_SLEEVES.items():
    d = delta_map.get(sleeve, 0.0)
    adj = max(0.0, base + d)
    if d > 0:   shift = f'<span class="sleeve-up">+{d:.0%}</span>'
    elif d < 0: shift = f'<span class="sleeve-down">{d:.0%}</span>'
    else:       shift = f'<span class="sleeve-flat">—</span>'
    rows += f"<tr><td>{sleeve}</td><td>{base:.0%}</td><td>{adj:.0%}</td><td>{shift}</td></tr>"

st.markdown(f"""
<table class="sleeve-table">
  <tr>
    <th style="width:40%">Sleeve</th>
    <th>Base</th>
    <th>Live</th>
    <th>Δ</th>
  </tr>
  {rows}
</table>
<div style="font-size:10px;color:#222;margin-top:12px;">
  ML sleeve downweighted in stress/crisis — trained on calm-bull distributions.
  SPY 200MA overlay: {"0.70× applied" if regime in ("stressed","crisis") else "1.00×"}.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# Debate
st.markdown('<div class="sec-head">Pre-Rebalance Debate</div>', unsafe_allow_html=True)

if not run_btn:
    st.markdown(f"""
    <div class="debate-prompt">
      <div class="debate-prompt-text">
        Five agents debate the proposed trades before every rebalance.<br>
        Verdict gates execution. &nbsp;·&nbsp; Current scenario: {rm['label']} · VIX {vix:.0f} · {spy_momentum}
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    top_syms = [s for s, _ in top_holdings[:5]]
    copy = _debate_copy(regime, vix, top_syms, spy_momentum)

    def _card(role: str, label: str, text: str, r2: bool = False, last: bool = False) -> str:
        r2_cls = " analyst-r2" if r2 else ""
        last_cls = " last" if last else ""
        prefix = "↩ " if r2 else ""
        return f"""
<div class="analyst-wrap{r2_cls}">
  <div class="analyst-meta">
    <span class="analyst-role {role}">{prefix}{label}</span>
    <div class="analyst-line"></div>
  </div>
  <div class="analyst-text{last_cls}">{text}</div>
</div>"""

    # Round 1
    ph_bull = st.empty(); ph_bear = st.empty(); ph_devil = st.empty()
    ph_regime = st.empty(); ph_quant = st.empty()

    with st.spinner(""):
        time.sleep(0.3)
    ph_bull.markdown(_card("bull", "Bull", copy["bull"]), unsafe_allow_html=True)
    with st.spinner(""):
        time.sleep(0.3)
    ph_bear.markdown(_card("bear", "Bear", copy["bear"]), unsafe_allow_html=True)
    with st.spinner(""):
        time.sleep(0.3)
    ph_devil.markdown(_card("devil", "Devil's Advocate", copy["devil"]), unsafe_allow_html=True)
    with st.spinner(""):
        time.sleep(0.2)
    ph_regime.markdown(_card("regime", "Regime Specialist", copy["regime"]), unsafe_allow_html=True)
    with st.spinner(""):
        time.sleep(0.15)
    total = sum(weights.values())
    max_sym, max_w = max(weights.items(), key=lambda x: x[1])
    cap_ok = max_w <= rm["max_wt"]
    quant_text = (
        f"{len(weights)} positions · weight sum {total:.4f} · "
        f"max {max_sym} {max_w:.1%} {'✓' if cap_ok else '⚠ exceeds regime cap'}"
    )
    ph_quant.markdown(_card("quant", "Quant Sanity", quant_text, last=True), unsafe_allow_html=True)

    # Round 2
    if show_r2:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        ph_b2 = st.empty(); ph_r2 = st.empty(); ph_d2 = st.empty()
        with st.spinner(""):
            time.sleep(0.3)
        bull_r2  = f"The bear's drawdown estimate assumes uniform exit conditions. These names have $200M+ daily volume — we can exit within 2 days at normal slippage. The -14% figure conflates liquid large-caps with the illiquid tail. The quality argument stands. VIX at {vix:.0f} is not the level at which we alter the fundamental thesis."
        bear_r2  = "The bull's liquidity argument fails in the relevant scenario. In 2020, MSFT — a $1.4T stock — saw bid/ask widen 4x intraday at peak stress. Daily volume figures are normal-market statistics. The scenario that matters is the 99th percentile exit, not the median. That's exactly where the -14% estimate comes from."
        devil_r2 = "Both sides are still debating direction and liquidity. The structural point stands: the correlation matrix we're relying on for diversification was estimated in a different regime. When it matters most, the matrix fails. That's not a prediction — it's a mathematical property of conditional correlation. No rebuttal addressed it."
        ph_b2.markdown(_card("bull",  "Bull",            bull_r2,  r2=True), unsafe_allow_html=True)
        ph_r2.markdown(_card("bear",  "Bear",            bear_r2,  r2=True), unsafe_allow_html=True)
        ph_d2.markdown(_card("devil", "Devil's Advocate", devil_r2, r2=True, last=True), unsafe_allow_html=True)

    # Verdict
    with st.spinner(""):
        time.sleep(0.25)
    v = _verdict(regime, vix)
    rec_label = {"proceed": "Proceed", "reduce_size": "Reduce Size", "halt_and_review": "Halt — Pending Review"}.get(v["rec"], v["rec"])
    rec_cls   = {"proceed": "proceed", "reduce_size": "reduce",      "halt_and_review": "halt"}.get(v["rec"], "reduce")
    risks_html = "".join(f'<div class="verdict-risk">— {r}</div>' for r in v["risks"])

    st.markdown(f"""
    <div class="verdict-wrap">
      <div class="verdict-status {rec_cls}">{rec_label}</div>
      <div class="verdict-conf">Confidence: {int(v['conf']*100)}%</div>
      <div class="verdict-text">{v['text']}</div>
      <div class="verdict-risks">{risks_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:48px;padding-top:20px;border-top:1px solid #111;
                font-size:10px;color:#222;letter-spacing:1px;display:flex;justify-content:space-between;">
      <span>Ascent Capital · Pre-Rebalance Analysis</span>
      <span>{date.today().strftime('%B %d, %Y')}</span>
    </div>
    """, unsafe_allow_html=True)
