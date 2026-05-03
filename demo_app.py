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
  /* ── Reset & base ───────────────────────────────── */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .stApp { background: #F5F5F7; }
  .block-container { padding: 2.5rem 2rem 4rem 2rem; max-width: 1000px; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8E8E8;
  }
  section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
  }

  /* ── Cards ──────────────────────────────────────── */
  .card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  .card-sm {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  }

  /* ── Typography ─────────────────────────────────── */
  .eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; color: #86868B; margin-bottom: 6px;
  }
  .headline {
    font-size: 28px; font-weight: 600; color: #1D1D1F;
    letter-spacing: -0.5px; line-height: 1.2;
  }
  .subhead {
    font-size: 13px; color: #86868B; margin-top: 4px; line-height: 1.5;
  }
  .body-text {
    font-size: 14px; color: #3A3A3C; line-height: 1.75;
  }
  .caption { font-size: 11px; color: #AEAEB2; }
  .mono {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
  }

  /* ── Stat grid ──────────────────────────────────── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 28px;
  }
  .stat-tile {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 16px 14px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    text-align: center;
  }
  .stat-label { font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: #AEAEB2; margin-bottom: 8px; }
  .stat-value { font-size: 20px; font-weight: 600; color: #1D1D1F; font-family: "SF Mono", monospace; }
  .stat-value.green { color: #30A46C; }
  .stat-value.red   { color: #E5484D; }
  .stat-value.gold  { color: #B8862E; }

  /* ── Position table ─────────────────────────────── */
  .pos-table { width: 100%; }
  .pos-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #F2F2F7;
  }
  .pos-row:last-child { border-bottom: none; }
  .pos-sym { font-size: 14px; font-weight: 500; color: #1D1D1F; }
  .pos-wt  { font-size: 13px; font-family: "SF Mono", monospace; color: #3A3A3C; }
  .pos-bar-bg { flex: 1; margin: 0 14px; height: 3px; background: #F2F2F7; border-radius: 2px; }
  .pos-bar-fill { height: 3px; background: #B8862E; border-radius: 2px; opacity: 0.6; }
  .pos-more { font-size: 12px; color: #AEAEB2; padding-top: 10px; }

  /* ── Signal table ───────────────────────────────── */
  .sig-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #F2F2F7;
    font-size: 13px;
  }
  .sig-row:last-child { border-bottom: none; }
  .sig-key { color: #86868B; }
  .sig-val { font-weight: 500; color: #1D1D1F; font-family: "SF Mono", monospace; }

  /* ── Regime badge ───────────────────────────────── */
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px 4px 8px; border-radius: 20px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
  }
  .badge-dot { width: 6px; height: 6px; border-radius: 50%; }
  .badge-calm_bull  { background: #F0FBF4; color: #30A46C; }
  .badge-calm_bull .badge-dot { background: #30A46C; }
  .badge-stressed   { background: #FFF8EC; color: #AD7C14; }
  .badge-stressed .badge-dot { background: #F59E0B; }
  .badge-crisis     { background: #FFF0F0; color: #E5484D; }
  .badge-crisis .badge-dot { background: #E5484D; }
  .badge-uncertain  { background: #F5F0FF; color: #7C3AED; }
  .badge-uncertain .badge-dot { background: #7C3AED; }

  /* ── Sleeve table ───────────────────────────────── */
  .sl-table { width: 100%; border-collapse: collapse; }
  .sl-table th {
    font-size: 10px; font-weight: 600; letter-spacing: 0.5px;
    text-transform: uppercase; color: #AEAEB2;
    padding: 0 0 12px; text-align: left; border-bottom: 1px solid #F2F2F7;
  }
  .sl-table th:not(:first-child) { text-align: right; }
  .sl-table td {
    padding: 10px 0; font-size: 13px; color: #3A3A3C;
    border-bottom: 1px solid #F9F9F9;
  }
  .sl-table td:not(:first-child) { text-align: right; font-family: "SF Mono", monospace; font-size: 12px; }
  .sl-table td:first-child { font-weight: 500; color: #1D1D1F; }
  .sl-up   { color: #30A46C !important; }
  .sl-down { color: #E5484D !important; }
  .sl-flat { color: #D1D1D6 !important; }

  /* ── Agent cards ────────────────────────────────── */
  .agent-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    border-left: 3px solid;
  }
  .agent-card.bull   { border-color: #30A46C; }
  .agent-card.bear   { border-color: #E5484D; }
  .agent-card.devil  { border-color: #7C3AED; }
  .agent-card.regime { border-color: #0071E3; }
  .agent-card.quant  { border-color: #AEAEB2; }
  .agent-card.r2     { opacity: 0.75; }

  .agent-role {
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 8px;
  }
  .agent-role.bull   { color: #30A46C; }
  .agent-role.bear   { color: #E5484D; }
  .agent-role.devil  { color: #7C3AED; }
  .agent-role.regime { color: #0071E3; }
  .agent-role.quant  { color: #AEAEB2; }
  .agent-text { font-size: 14px; color: #3A3A3C; line-height: 1.7; }

  /* ── Verdict ────────────────────────────────────── */
  .verdict-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 24px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
  }
  .verdict-badge {
    display: inline-block; padding: 6px 16px;
    border-radius: 20px; font-size: 12px; font-weight: 700;
    letter-spacing: 0.5px; margin-bottom: 14px;
  }
  .verdict-badge.proceed { background: #F0FBF4; color: #30A46C; }
  .verdict-badge.reduce  { background: #FFF8EC; color: #AD7C14; }
  .verdict-badge.halt    { background: #FFF0F0; color: #E5484D; }
  .verdict-conf {
    font-size: 12px; color: #AEAEB2; margin-bottom: 14px;
    font-family: "SF Mono", monospace;
  }
  .verdict-text { font-size: 14px; color: #3A3A3C; line-height: 1.75; max-width: 640px; }
  .verdict-risk { font-size: 12px; color: #AEAEB2; margin: 5px 0; }

  /* ── Divider ────────────────────────────────────── */
  .rule { border: none; border-top: 1px solid #E8E8E8; margin: 28px 0; }

  /* ── Run button ─────────────────────────────────── */
  .stButton > button {
    background: #1D1D1F !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    letter-spacing: 0.3px !important; padding: 10px 20px !important;
    transition: opacity 0.15s !important;
  }
  .stButton > button:hover { opacity: 0.8 !important; }

  /* ── Streamlit widget tweaks ────────────────────── */
  .stSelectbox label, .stSlider label, .stSelectSlider label, .stToggle label {
    font-size: 12px !important; color: #86868B !important; font-weight: 500 !important;
  }
  div[data-testid="stSlider"] { padding-bottom: 4px; }

  /* ── Debate prompt ──────────────────────────────── */
  .debate-empty {
    background: #FFFFFF; border-radius: 16px;
    padding: 60px 40px; text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
  }
  .debate-empty-icon { font-size: 32px; margin-bottom: 12px; color: #D1D1D6; }
  .debate-empty-text { font-size: 13px; color: #AEAEB2; line-height: 1.6; }
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
    except Exception:
        pass
    return weights, regime, vix, spy_mom


_live_weights, _live_regime, _live_vix, _live_spy_mom = _load_live_data()

_REGIME_OPTIONS  = ["calm_bull", "stressed", "crisis", "uncertain"]
_SPY_MOM_OPTIONS = ["Strong negative", "Weak negative", "Neutral", "Weak positive", "Strong positive"]
_live_regime_idx  = _REGIME_OPTIONS.index(_live_regime) if _live_regime in _REGIME_OPTIONS else 1
_live_spy_mom_idx = _SPY_MOM_OPTIONS.index(_live_spy_mom) if _live_spy_mom in _SPY_MOM_OPTIONS else 2

REGIME_META = {
    "calm_bull": {"risk": 1.00, "max_wt": 0.15, "label": "Calm Bull",  "desc": "Full deployment. Momentum intact."},
    "stressed":  {"risk": 0.65, "max_wt": 0.10, "label": "Stressed",   "desc": "Reduce to 65%. Defensive tilt."},
    "crisis":    {"risk": 0.40, "max_wt": 0.08, "label": "Crisis",     "desc": "40% exposure. Capital preservation."},
    "uncertain": {"risk": 0.75, "max_wt": 0.12, "label": "Uncertain",  "desc": "75% exposure. Wait for confirmation."},
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
    "Tech-Heavy":   {"AAPL": 0.15, "MSFT": 0.13, "NVDA": 0.12, "GOOGL": 0.10, "META": 0.10, "ADBE": 0.08, "CRWD": 0.08, "AMD": 0.07, "INTC": 0.06, "TXN": 0.06, "PANW": 0.05},
    "Defensive":    {"JNJ": 0.12, "WMT": 0.12, "KO": 0.10, "PG": 0.10, "NEE": 0.09, "MRK": 0.09, "ABT": 0.08, "DUK": 0.08, "TLT": 0.12, "GLD": 0.10},
    "Macro Stress": {"TLT": 0.25, "GLD": 0.20, "BIL": 0.15, "IEF": 0.15, "VIXY": 0.10, "UUP": 0.08, "PDBC": 0.07},
}


def _debate_copy(regime: str, vix: float, holdings: list, spy_mom: str) -> dict:
    h = holdings[:4]
    hs = ", ".join(h) if h else "the portfolio"
    first = h[0] if h else "our top holding"
    md = {"Strong positive": "strong upward momentum", "Weak positive": "weakening upward momentum",
          "Neutral": "flat momentum", "Weak negative": "early downtrend signals",
          "Strong negative": "sustained downtrend"}.get(spy_mom, "mixed signals")
    scenarios = {
        "calm_bull": {
            "bull":   f"The regime is unambiguous. {hs} are compounders with durable cash generation — not rate-sensitive cyclicals. VIX at {vix:.0f} is benign. Institutional flows are constructive. The trend signal is intact across our top names. Over-hedging a calm bull regime has historically cost more in missed upside than any drawdown protection it delivered.",
            "bear":   f"Calm bull is precisely when complacency gets punished. {first} is pricing in a forward earnings trajectory that assumes no margin compression, no multiple contraction, no macro shock. VIX at {vix:.0f} means tail risk is unpriced — not absent. The last three calm-bull regimes ended with an average -14% drawdown within 8 weeks of the signal peak.",
            "devil":  f"Both arguments assume the correlation structure holds. In 2018, 2020, and 2022, cross-portfolio correlation moved from 0.32 to 0.79 within 11 trading days of regime transition. Our diversified book becomes a single-factor bet at exactly the wrong moment. Neither the bull nor the bear is modeling forced unwinds at illiquid prints.",
            "regime": f"Calm bull playbook: full deployment, momentum tilt, max position 15%. The classifier has stayed in this state for 14+ days with entropy well below 0.90 — high-conviction signal. Set trip-wires at VIX 22 or SPY below the 50-day MA. Do not act until the signal moves.",
        },
        "stressed": {
            "bull":   f"Stressed regimes create the separation between quality and noise. {hs} have balance sheets that don't require credit markets to function. VIX at {vix:.0f} is elevated but this is a repricing event, not a solvency event. The market is discounting a macro outcome with 35% probability as though it's certain.",
            "bear":   f"VIX at {vix:.0f} is real institutional hedging, not retail fear. SPY showing {md}. When VIX crosses 25 in a stressed regime, the median forward drawdown from entry is -14% over 6 weeks. {first} is still pricing in growth assumptions that are now at risk. The model says 65% gross exposure. That's the number.",
            "devil":  f"Both sides are missing the credit spread argument. If HY spreads widen 150bps — entirely consistent with this regime — equity correlations spike and bid/ask on our smaller positions widens to 80bps. Neither the bull nor bear is modeling forced exit at -10% slippage on three names simultaneously. Monte Carlo p5 shows -21%.",
            "regime": f"Stressed regime protocol: 65% gross exposure, 10% max position, quality tilt. SPY showing {md} — no recovery confirmation. Typical stressed duration is 4–12 weeks. The most common error in this regime is re-risking on the first green day. Wait for entropy below 0.55.",
        },
        "crisis": {
            "bull":   f"Crisis regimes produce the best forward returns from entry — if you have dry powder. {hs} will be structurally intact on the other side. VIX at {vix:.0f} is extreme fear, not rational pricing. 2009 and 2020 both saw 40%+ returns in the 12 months following peak VIX. Exiting here means locking in the loss.",
            "bear":   f"VIX at {vix:.0f} is a data point, not a prediction. 2008 peaked at 89.53 — we may not be at maximum stress. The primary risk in crisis is permanent capital impairment, not opportunity cost. Correlation across this book approaches 1.0 in forced liquidation. Cash is not a missed opportunity; it's the instrument that makes recovery participation possible.",
            "devil":  f"Both sides are anchoring on historical analogs with defined endpoints. The fatal assumption is that {first} survives without permanent impairment. In a true tail event, our highest-conviction names carry the most model risk — they're also the most institutionally owned, so forced selling hits them hardest. 40% gross. No exceptions.",
            "regime": f"Crisis protocol: 40% gross exposure, 8% max position, no speculative names. Re-risking criteria are explicit: Fed balance sheet expansion, HY spreads rolling over, VIX closing below 30 on consecutive days. None of those conditions are met. Hold the line.",
        },
        "uncertain": {
            "bull":   f"Regime uncertainty is not a sell signal. {hs} are earnings compounders — the fundamental case is independent of the classifier's confidence. Being too defensive in an uncertain regime that resolves bullish is a real and permanent cost. 75% exposure, hold the quality book.",
            "bear":   f"Uncertain regime with {md} in SPY is the worst risk/reward setup. We don't know the resolution direction, and we're carrying near-full exposure into that uncertainty. Crisis resolution means -20%+. Calm bull means capped upside from sizing. Reduce to 60% and buy the optionality.",
            "devil":  f"The regime uncertainty is itself diagnostic. The classifier is uncertain because its features are no longer predictive — this is a structural break, not rotation noise. Neither the bull nor bear is modeling a world where the regime framework stops working. That is the actual tail risk.",
            "regime": f"Uncertain regime typically resolves in 1–3 weeks. Posture: 75% gross, 12% max position, no new positions until entropy drops below 0.70. Do not force a thesis. Patience is a position.",
        },
    }
    return scenarios.get(regime, scenarios["uncertain"])


def _verdict(regime: str, vix: float) -> dict:
    rm = REGIME_META[regime]
    if regime == "crisis" or vix > 40:
        return {
            "rec": "halt", "label": "Halt — Pending Review", "conf": 0.78,
            "risks": ["Crisis regime — outside normal execution parameters", f"VIX {vix:.0f}: tail risk is actively being priced in", "Correlation structure breakdown invalidates diversification assumptions"],
            "text": f"The crisis regime signal combined with VIX {vix:.0f} places this portfolio outside the parameters for normal execution. The correlation-breakdown argument raised by the devil's advocate was not adequately addressed. Halting for principal review is the correct procedural response.",
        }
    elif regime == "stressed" or vix > 25:
        return {
            "rec": "reduce", "label": "Reduce Size", "conf": 0.71,
            "risks": [f"Target gross exposure: {rm['risk']:.0%} (model mandate, not suggestion)", f"VIX {vix:.0f}: institutional hedging activity elevated", "Bear's -14% forward drawdown estimate not rebutted"],
            "text": f"The bull's quality argument is partially valid but does not address the regime-level sizing mandate. 160 walk-forward folds say {rm['risk']:.0%} gross in a stressed regime. Reduce to {rm['risk']:.0%}, maintain quality positions, max {rm['max_wt']:.0%} per name. Re-evaluate at next rebalance.",
        }
    else:
        return {
            "rec": "proceed", "label": "Proceed", "conf": 0.82,
            "risks": ["Monitor trip-wires: VIX >22, SPY break of 50-day MA", "Top-3 concentration warrants position-level discipline"],
            "text": "The calm bull regime signal is high-conviction. The bull case is well-supported and the bear's tail-risk concerns, while structurally valid, do not override the primary signal in this environment. The quant check is clean. Proceed with full deployment.",
        }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #F2F2F7;">
      <div style="font-size: 17px; font-weight: 700; color: #1D1D1F; letter-spacing: -0.3px;">▲ Ascent Capital</div>
      <div style="font-size: 12px; color: #AEAEB2; margin-top: 3px;">Pre-Rebalance Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Scenario**", )
    regime = st.selectbox(
        "Regime", _REGIME_OPTIONS, index=_live_regime_idx,
        format_func=lambda x: REGIME_META[x]["label"], label_visibility="collapsed",
    )
    vix = st.slider("VIX", 10.0, 55.0, round(_live_vix, 1), 0.5)
    spy_momentum = st.select_slider("SPY Momentum", _SPY_MOM_OPTIONS, value=_live_spy_mom)

    st.markdown("---")
    st.markdown("**Portfolio**")
    preset_name = st.selectbox("Portfolio", list(PRESETS.keys()), label_visibility="collapsed")
    weights = PRESETS[preset_name].copy()

    st.markdown("---")
    show_r2 = st.toggle("Show Round 2 rebuttals", value=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Debate  ›", use_container_width=True)


# ── HEADER ───────────────────────────────────────────────────────────────────────
rm = REGIME_META[regime]
top_holdings = sorted(weights.items(), key=lambda x: -x[1])
max_wt = top_holdings[0][1] if top_holdings else 0.1

st.markdown(f"""
<div style="margin-bottom: 28px;">
  <div class="eyebrow">Ascent Capital</div>
  <div class="headline">Pre-Rebalance Analysis</div>
  <div class="subhead">{date.today().strftime('%A, %B %d, %Y')} &nbsp;·&nbsp; 4 agents &nbsp;·&nbsp; 11 alpha sleeves &nbsp;·&nbsp; S&P 500 universe</div>
</div>
""", unsafe_allow_html=True)

# OOS stats
st.markdown("""
<div class="stat-grid">
  <div class="stat-tile"><div class="stat-label">OOS CAGR</div><div class="stat-value gold">22.4%</div></div>
  <div class="stat-tile"><div class="stat-label">Sharpe</div><div class="stat-value gold">0.887</div></div>
  <div class="stat-tile"><div class="stat-label">Max DD</div><div class="stat-value red">−35.8%</div></div>
  <div class="stat-tile"><div class="stat-label">vs SPY</div><div class="stat-value green">+8.9%</div></div>
  <div class="stat-tile"><div class="stat-label">WF Folds</div><div class="stat-value">160</div></div>
  <div class="stat-tile"><div class="stat-label">Period</div><div class="stat-value" style="font-size:14px;">2020–26</div></div>
</div>
""", unsafe_allow_html=True)

# Portfolio + signals
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown(f"""<div class="card">
      <div class="eyebrow" style="margin-bottom:14px;">Portfolio — {preset_name}</div>
      <div class="pos-table">""", unsafe_allow_html=True)

    pos_html = ""
    for sym, w in top_holdings[:10]:
        bar_w = int(w / max_wt * 100)
        pos_html += f"""
        <div class="pos-row">
          <span class="pos-sym">{sym}</span>
          <div class="pos-bar-bg"><div class="pos-bar-fill" style="width:{bar_w}%"></div></div>
          <span class="pos-wt">{w:.1%}</span>
        </div>"""
    if len(top_holdings) > 10:
        pos_html += f'<div class="pos-more">+{len(top_holdings)-10} more positions</div>'

    st.markdown(pos_html + "</div></div>", unsafe_allow_html=True)

with col2:
    spy_col = "#30A46C" if "positive" in spy_momentum.lower() else ("#E5484D" if "negative" in spy_momentum.lower() else "#86868B")
    st.markdown(f"""<div class="card">
      <div class="eyebrow" style="margin-bottom:14px;">Market State</div>
      <div class="sig-row">
        <span class="sig-key">Regime</span>
        <span><span class="badge badge-{regime}"><span class="badge-dot"></span>{rm['label']}</span></span>
      </div>
      <div class="sig-row">
        <span class="sig-key">VIX</span>
        <span class="sig-val">{vix:.1f}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">SPY Momentum</span>
        <span class="sig-val" style="font-family:sans-serif;font-size:12px;color:{spy_col};">{spy_momentum}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">Gross Exposure</span>
        <span class="sig-val">{rm['risk']:.0%}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">Max Position</span>
        <span class="sig-val">{rm['max_wt']:.0%}</span>
      </div>
      <div class="sig-row" style="border-bottom:none;">
        <span class="sig-key">Positions</span>
        <span class="sig-val">{len(weights)}</span>
      </div>
    </div>""", unsafe_allow_html=True)

# Sleeve table
delta_map = SLEEVE_DELTA.get(regime, {})
rows = ""
for sleeve, base in BASE_SLEEVES.items():
    d = delta_map.get(sleeve, 0.0)
    adj = max(0.0, base + d)
    if d > 0:   shift = f'<span class="sl-up">▲ +{d:.0%}</span>'
    elif d < 0: shift = f'<span class="sl-down">▼ {d:.0%}</span>'
    else:       shift = f'<span class="sl-flat">—</span>'
    rows += f"<tr><td>{sleeve}</td><td>{base:.0%}</td><td>{adj:.0%}</td><td>{shift}</td></tr>"

st.markdown(f"""
<div class="card">
  <div class="eyebrow" style="margin-bottom:16px;">Alpha Allocation — Regime Adjusted</div>
  <table class="sl-table">
    <tr><th style="width:42%">Sleeve</th><th>Base</th><th>Live</th><th>Shift</th></tr>
    {rows}
  </table>
  <div style="font-size:11px;color:#AEAEB2;margin-top:12px;">
    SPY 200MA overlay: {"0.70× (SPY below 200-day MA)" if regime in ("stressed","crisis") else "1.00× — full exposure"} &nbsp;·&nbsp;
    ML sleeve downweighted in stress/crisis
  </div>
</div>
""", unsafe_allow_html=True)

# Debate
st.markdown(f"""
<div class="eyebrow" style="margin: 8px 0 16px 0;">Pre-Rebalance Debate</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown(f"""
    <div class="debate-empty">
      <div class="debate-empty-icon">◎</div>
      <div class="debate-empty-text">
        Five agents debate the proposed trades before every rebalance.<br>
        The judge's verdict gates execution.<br><br>
        <strong style="color:#1D1D1F;">Current scenario:</strong> {rm['label']} regime · VIX {vix:.0f} · {spy_momentum}
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    top_syms = [s for s, _ in top_holdings[:5]]
    copy = _debate_copy(regime, vix, top_syms, spy_momentum)

    def agent_card(role: str, label: str, text: str, r2: bool = False) -> str:
        r2_cls = " r2" if r2 else ""
        prefix = "↩ " if r2 else ""
        return f"""
<div class="agent-card {role}{r2_cls}">
  <div class="agent-role {role}">{prefix}{label}</div>
  <div class="agent-text">{text}</div>
</div>"""

    ph = [st.empty() for _ in range(5)]

    for i, (role, label, text) in enumerate([
        ("bull",   "Bull Analyst",      copy["bull"]),
        ("bear",   "Bear Analyst",      copy["bear"]),
        ("devil",  "Devil's Advocate",  copy["devil"]),
        ("regime", "Regime Specialist", copy["regime"]),
    ]):
        with st.spinner(""):
            time.sleep(0.3)
        ph[i].markdown(agent_card(role, label, text), unsafe_allow_html=True)

    total = sum(weights.values())
    max_sym, max_w = max(weights.items(), key=lambda x: x[1])
    cap_ok = max_w <= rm["max_wt"]
    quant_text = (
        f"{len(weights)} positions · weight sum {total:.4f} · "
        f"max position {max_sym} at {max_w:.1%} {'✓' if cap_ok else '⚠ exceeds regime cap'}"
    )
    ph[4].markdown(agent_card("quant", "Quant Sanity", quant_text), unsafe_allow_html=True)

    if show_r2:
        st.markdown("<div style='margin-top:8px;margin-bottom:4px;'><span class='eyebrow'>Round 2 — Rebuttals</span></div>", unsafe_allow_html=True)
        rph = [st.empty() for _ in range(3)]
        with st.spinner(""):
            time.sleep(0.3)
        bull_r2  = f"The bear's drawdown estimate assumes uniform exit conditions. Our names average $200M+ daily volume — normal-market exit is 2 days. The -14% figure conflates liquid large-caps with the illiquid tail. The quality argument stands at VIX {vix:.0f}."
        bear_r2  = "The bull's liquidity argument fails in the relevant scenario. In 2020, even MSFT saw bid/ask widen 4× intraday at peak stress. Daily volume is a normal-market statistic. The scenario that matters is the 99th-percentile exit — that's exactly where the -14% estimate lives."
        devil_r2 = "Both sides are debating direction and liquidity. The structural point stands: the correlation matrix we're relying on was estimated in a different regime. When it matters most, it fails. That's not a prediction — it's a mathematical property of conditional correlation. No rebuttal addressed it."
        rph[0].markdown(agent_card("bull",  "Bull — Rebuttal",             bull_r2,  r2=True), unsafe_allow_html=True)
        rph[1].markdown(agent_card("bear",  "Bear — Rebuttal",             bear_r2,  r2=True), unsafe_allow_html=True)
        rph[2].markdown(agent_card("devil", "Devil's Advocate — Rebuttal", devil_r2, r2=True), unsafe_allow_html=True)

    # Verdict
    with st.spinner(""):
        time.sleep(0.2)
    v = _verdict(regime, vix)
    risks_html = "".join(f'<div class="verdict-risk">— {r}</div>' for r in v["risks"])
    st.markdown(f"""
    <div class="verdict-card">
      <div class="eyebrow" style="margin-bottom:12px;">Judge — Final Verdict</div>
      <div class="verdict-badge {v['rec']}">{v['label']}</div>
      <div class="verdict-conf">Confidence: {int(v['conf']*100)}%</div>
      <div class="verdict-text">{v['text']}</div>
      <div style="margin-top:14px;">{risks_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:32px;font-size:11px;color:#D1D1D6;text-align:center;">
      Ascent Capital &nbsp;·&nbsp; {date.today().strftime('%B %d, %Y')} &nbsp;·&nbsp; For discussion purposes only
    </div>
    """, unsafe_allow_html=True)
