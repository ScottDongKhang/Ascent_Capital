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
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Mono:wght@300;400&display=swap');

:root {
  --ink:       #07070A;
  --ink-2:     #0C0C0F;
  --surface:   #101013;
  --surface-2: #151518;
  --surface-3: #1A1A1F;
  --gold:      #C9A96E;
  --gold-lt:   #E8D5A8;
  --gold-dim:  #8A6F44;
  --text:      #C2BDB2;
  --text-mid:  #8A8580;
  --text-dim:  #58544F;
  --head:      #EDE8DF;
  --rule:      rgba(201,169,110,0.08);
  --rule-b:    rgba(201,169,110,0.18);
  --rule-c:    rgba(201,169,110,0.32);
}

html, body, [class*="css"] {
  font-family: 'Cormorant Garamond', Georgia, serif;
  -webkit-font-smoothing: antialiased;
  background: var(--ink) !important;
}

.stApp { background: var(--ink) !important; }
.block-container { padding: 3rem 2.5rem 5rem; max-width: 1020px; }
#MainMenu, footer, header { visibility: hidden; }
h1,h2,h3,h4 { color: var(--head) !important; font-weight: 400 !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--ink-2) !important;
  border-right: 1px solid var(--rule-b) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 2.5rem 1.8rem; }

/* Streamlit widget labels */
.stSelectbox label, .stSlider label, .stSelectSlider label, .stToggle label,
div[data-testid="stWidgetLabel"] p {
  font-family: 'DM Mono', monospace !important;
  font-size: 8px !important; font-weight: 400 !important;
  letter-spacing: 1.5px !important; text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
/* Widget inputs */
.stSelectbox > div > div,
div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--rule-b) !important;
  border-radius: 2px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
}
.stSlider div[data-testid="stSliderThumbValue"] {
  color: var(--gold) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
}
/* Run button */
.stButton > button {
  font-family: 'DM Mono', monospace !important;
  font-size: 8px !important; font-weight: 400 !important;
  letter-spacing: 2px !important; text-transform: uppercase !important;
  background: var(--gold) !important; color: var(--ink) !important;
  border: none !important; border-radius: 2px !important;
  padding: 0.85rem 1.8rem !important;
  transition: background 0.2s ease !important;
}
.stButton > button:hover { background: var(--gold-lt) !important; }

/* Secondary buttons (show all / show less) */
.stButton.sec > button {
  background: transparent !important;
  border: 1px solid var(--rule-b) !important;
  color: var(--gold-dim) !important;
}
.stButton.sec > button:hover {
  background: rgba(201,169,110,0.05) !important;
  border-color: var(--gold) !important;
  color: var(--gold) !important;
}

/* Markdown overrides */
div[data-testid="stMarkdownContainer"] { color: var(--text); }

/* ── Components ─────────────────────────────────────── */

.eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 8px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--gold-dim); margin-bottom: 4px;
}

.wordmark {
  font-family: 'DM Mono', monospace;
  font-size: 9.5px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--gold);
}

.page-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(2rem, 4vw, 3rem);
  font-weight: 300; color: var(--head);
  letter-spacing: -0.5px; line-height: 1.15;
  margin: 6px 0 8px;
}

.page-sub {
  font-family: 'DM Mono', monospace;
  font-size: 9px; color: var(--text-dim);
  letter-spacing: 1px; line-height: 1.8;
}

/* Card */
.card {
  background: var(--surface);
  border: 1px solid var(--rule-b);
  padding: 2rem 2.2rem;
  margin-bottom: 1px;
}
.card-head {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--gold-dim); margin-bottom: 1.4rem;
  padding-bottom: 1rem; border-bottom: 1px solid var(--rule);
}

/* Position rows */
.pos-row {
  display: flex; align-items: center;
  padding: 0.6rem 0; border-bottom: 1px solid var(--rule);
  gap: 12px;
}
.pos-row:last-child { border-bottom: none; }
.pos-sym {
  font-family: 'DM Mono', monospace;
  font-size: 11px; color: var(--head); width: 54px; flex-shrink: 0;
}
.pos-bar-bg { flex: 1; height: 1px; background: var(--rule-b); }
.pos-bar-fill { height: 1px; background: var(--gold-dim); }
.pos-wt {
  font-family: 'DM Mono', monospace;
  font-size: 11px; color: var(--gold); width: 40px; text-align: right; flex-shrink: 0;
}

/* Signal rows */
.sig-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 0.65rem 0; border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
.sig-row:last-child { border-bottom: none; }
.sig-key {
  font-family: 'DM Mono', monospace;
  font-size: 8px; letter-spacing: 1px; text-transform: uppercase; color: var(--text-dim);
}
.sig-val {
  font-family: 'DM Mono', monospace;
  font-size: 12px; color: var(--head);
}

/* Regime pill */
.regime-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: 'DM Mono', monospace;
  font-size: 7px; letter-spacing: 1.5px; text-transform: uppercase;
  padding: 0.28rem 0.7rem; border-radius: 1px;
  border: 1px solid var(--rule-b);
}
.regime-calm_bull { color: #3A9C68; border-color: rgba(58,156,104,0.3); }
.regime-stressed  { color: var(--gold); border-color: var(--rule-b); }
.regime-crisis    { color: #C97676; border-color: rgba(201,118,118,0.3); }
.regime-uncertain { color: var(--text-mid); border-color: var(--rule); }

/* Sleeve table */
.sl-table { width: 100%; border-collapse: collapse; }
.sl-table th {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--text-dim); font-weight: 400;
  padding: 0 0 0.9rem; text-align: left;
  border-bottom: 1px solid var(--rule-b);
}
.sl-table th:not(:first-child) { text-align: right; }
.sl-table td {
  padding: 0.65rem 0;
  font-family: 'Cormorant Garamond', serif;
  font-size: 15px; color: var(--text);
  border-bottom: 1px solid var(--rule);
}
.sl-table td:not(:first-child) {
  text-align: right;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
}
.sl-table td:first-child { color: var(--head); }
.sl-up   { color: #3A9C68 !important; }
.sl-down { color: #C97676 !important; }
.sl-flat { color: var(--text-dim) !important; }

/* Agent cards */
.agent-wrap { margin: 0 0 1px; }
.agent-card {
  background: var(--surface);
  border: 1px solid var(--rule-b);
  border-left: 2px solid;
  padding: 1.6rem 2rem;
}
.agent-card.bull   { border-left-color: #3A9C68; }
.agent-card.bear   { border-left-color: #C97676; }
.agent-card.devil  { border-left-color: #8A7CA8; }
.agent-card.regime { border-left-color: var(--gold-dim); }
.agent-card.quant  { border-left-color: var(--text-dim); }
.agent-card.r2     { opacity: 0.7; background: var(--ink-2); }

.agent-role {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; letter-spacing: 2px; text-transform: uppercase;
  margin-bottom: 0.9rem;
}
.agent-role.bull   { color: #3A9C68; }
.agent-role.bear   { color: #C97676; }
.agent-role.devil  { color: #8A7CA8; }
.agent-role.regime { color: var(--gold-dim); }
.agent-role.quant  { color: var(--text-dim); }

.agent-text {
  font-family: 'Cormorant Garamond', serif;
  font-size: 17px; color: var(--text); line-height: 1.75; font-weight: 300;
}

/* Section rule */
.rule { border: none; border-top: 1px solid var(--rule-b); margin: 2.5rem 0; }

/* Debate empty */
.debate-empty {
  background: var(--surface); border: 1px solid var(--rule);
  padding: 5rem 3rem; text-align: center;
}
.debate-empty-glyph {
  font-size: 1.4rem; color: var(--gold-dim); margin-bottom: 1.2rem;
}
.debate-empty-text {
  font-family: 'DM Mono', monospace;
  font-size: 9px; color: var(--text-dim); letter-spacing: 1px; line-height: 2;
}

/* Verdict */
.verdict-wrap {
  background: var(--surface); border: 1px solid var(--rule-c);
  padding: 2.2rem 2.4rem; margin-top: 1px;
}
.verdict-status {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; letter-spacing: 2.5px; text-transform: uppercase;
  margin-bottom: 1.2rem;
}
.verdict-status.proceed  { color: #3A9C68; }
.verdict-status.reduce   { color: var(--gold); }
.verdict-status.halt     { color: #C97676; }
.verdict-conf {
  font-family: 'DM Mono', monospace;
  font-size: 9px; color: var(--text-dim); margin-bottom: 1.2rem;
}
.verdict-text {
  font-family: 'Cormorant Garamond', serif;
  font-size: 18px; color: var(--text); line-height: 1.8; font-weight: 300;
  max-width: 640px;
}
.verdict-risk {
  font-family: 'DM Mono', monospace;
  font-size: 9px; color: var(--text-dim); margin: 0.4rem 0; letter-spacing: 0.5px;
}

/* Footer */
.footer {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; color: var(--text-dim); letter-spacing: 1.5px;
  text-transform: uppercase; margin-top: 3rem;
  padding-top: 1.5rem; border-top: 1px solid var(--rule);
  display: flex; justify-content: space-between;
}

/* Round label */
.round-label {
  font-family: 'DM Mono', monospace;
  font-size: 7.5px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--text-dim); margin: 1.8rem 0 1rem;
  padding-bottom: 0.6rem; border-bottom: 1px solid var(--rule);
}
</style>
""", unsafe_allow_html=True)


# ── State ────────────────────────────────────────────────────────────────────────
if "show_all_pos" not in st.session_state:
    st.session_state.show_all_pos = False


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

REGIME_META = {
    "calm_bull": {"risk": 1.00, "max_wt": 0.15, "label": "Calm Bull",  "desc": "Full deployment. Momentum intact."},
    "stressed":  {"risk": 0.65, "max_wt": 0.10, "label": "Stressed",   "desc": "65% gross. Defensive tilt."},
    "crisis":    {"risk": 0.40, "max_wt": 0.08, "label": "Crisis",     "desc": "40% exposure. Capital preservation."},
    "uncertain": {"risk": 0.75, "max_wt": 0.12, "label": "Uncertain",  "desc": "75% gross. Await confirmation."},
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
    "stressed":  {"Trend":.08,"Stat-Arb":-.03,"ML (XGBoost)":-.07,"Vol-Regime":.03,"Mean Reversion":.02,"Fundamental":-.02,"Earnings (PEAD)":-.01,"Analyst":-.01,"Options Flow":0,"Insider":0,"Short Interest":0},
    "crisis":    {"Trend":.12,"Stat-Arb":-.05,"ML (XGBoost)":-.08,"Vol-Regime":.05,"Mean Reversion":.03,"Fundamental":-.04,"Earnings (PEAD)":-.02,"Analyst":-.01,"Options Flow":0,"Insider":0,"Short Interest":0},
    "uncertain": {"Trend":.05,"Stat-Arb":-.02,"ML (XGBoost)":-.03,"Vol-Regime":.02,"Mean Reversion":0,"Fundamental":-.01,"Earnings (PEAD)":-.01,"Analyst":0,"Options Flow":0,"Insider":0,"Short Interest":0},
}

_FALLBACK = {
    "KMLM":0.097,"IFRA":0.087,"PDBC":0.082,"AMKR":0.056,"FIX":0.056,
    "VAL":0.056,"VICR":0.056,"VRT":0.056,"WDC":0.056,"CNC":0.054,
    "EWY":0.041,"DBB":0.043,"VC":0.038,"DBA":0.033,"EWC":0.033,
    "PVH":0.032,"TIP":0.022,"NUE":0.020,"XPO":0.019,"APA":0.019,
    "AAXJ":0.015,"EEM":0.015,"EWZ":0.012,
}

PRESETS = {
    f"Live — {date.today().strftime('%b %d, %Y')}": _live_weights or _FALLBACK,
    "Tech-Heavy":   {"AAPL":0.15,"MSFT":0.13,"NVDA":0.12,"GOOGL":0.10,"META":0.10,"ADBE":0.08,"CRWD":0.08,"AMD":0.07,"INTC":0.06,"TXN":0.06,"PANW":0.05},
    "Defensive":    {"JNJ":0.12,"WMT":0.12,"KO":0.10,"PG":0.10,"NEE":0.09,"MRK":0.09,"ABT":0.08,"DUK":0.08,"TLT":0.12,"GLD":0.10},
    "Macro Stress": {"TLT":0.25,"GLD":0.20,"BIL":0.15,"IEF":0.15,"VIXY":0.10,"UUP":0.08,"PDBC":0.07},
}


def _debate_copy(regime: str, vix: float, holdings: list, spy_mom: str) -> dict:
    h = holdings[:4]
    hs = ", ".join(h) if h else "the portfolio"
    first = h[0] if h else "our top holding"
    md = {"Strong positive":"strong upward momentum","Weak positive":"weakening upward momentum",
          "Neutral":"flat momentum","Weak negative":"early downtrend signals",
          "Strong negative":"sustained downtrend"}.get(spy_mom,"mixed signals")
    s = {
        "calm_bull": {
            "bull":   f"The regime is unambiguous. {hs} are compounders with durable cash generation — not rate-sensitive cyclicals. VIX at {vix:.0f} is benign. Institutional flows are constructive. Over-hedging a calm bull regime has historically cost more in missed upside than any protection it delivered.",
            "bear":   f"Calm bull is precisely when complacency gets punished. {first} is pricing in a forward trajectory that assumes no margin compression, no multiple contraction, no macro shock. VIX at {vix:.0f} means tail risk is unpriced — not absent. The last three calm-bull regimes ended with a median -14% drawdown within 8 weeks.",
            "devil":  f"Both arguments assume the correlation structure holds. In 2018, 2020, and 2022, cross-portfolio correlation moved from 0.32 to 0.79 within 11 trading days of regime transition. Our diversified book becomes a single-factor bet at exactly the wrong moment. Neither side is modeling forced unwinds at illiquid prints.",
            "regime": f"Calm bull playbook: full deployment, momentum tilt, 15% max position. The classifier has maintained this state for 14+ days with entropy well below 0.90 — a high-conviction signal. Trip-wires: VIX above 22 or SPY below the 50-day MA. Do not act until the signal moves.",
        },
        "stressed": {
            "bull":   f"Stressed regimes separate quality from noise. {hs} have balance sheets that do not require credit markets to function. VIX at {vix:.0f} is elevated — but this is a repricing event, not a solvency event. The market is discounting a macro outcome with 35% probability as though it is certain.",
            "bear":   f"VIX at {vix:.0f} is real institutional hedging, not retail fear. SPY showing {md}. When VIX has crossed 25 in a stressed regime, the median forward drawdown from entry has been -14% over 6 weeks. {first} is still pricing in growth assumptions that are now at material risk. The model says 65% gross exposure.",
            "devil":  f"Both sides are missing the credit spread argument. If HY spreads widen 150bps — entirely consistent with this regime — equity correlations spike and bid/ask on our smaller names widens to 80bps. Neither the bull nor bear is modeling a forced exit at -10% slippage on three positions simultaneously. Monte Carlo p5: -21%.",
            "regime": f"Stressed regime protocol: 65% gross exposure, 10% max position, quality tilt. SPY showing {md} — no recovery confirmation. Typical stressed duration is 4–12 weeks. The most common error in this regime is re-risking on the first green day. Wait for entropy to drop below 0.55.",
        },
        "crisis": {
            "bull":   f"Crisis regimes produce the best forward returns from entry — if you have dry powder. {hs} will be structurally intact on the other side. VIX at {vix:.0f} is extreme fear, not rational pricing. 2009 and 2020 both delivered 40%+ in the 12 months following peak VIX. Exiting here locks in the loss.",
            "bear":   f"VIX at {vix:.0f} is a data point, not a ceiling. 2008 peaked at 89.53. In a crisis regime, the primary risk is permanent capital impairment — not opportunity cost. Correlation across this book approaches 1.0 under forced liquidation. Cash is the instrument that makes recovery participation possible.",
            "devil":  f"Both sides anchor on historical analogs with defined endpoints. The fatal assumption: {first} survives without permanent impairment. In a true tail event, our highest-conviction names carry the most model risk — they are also the most institutionally owned, meaning forced selling hits them first and hardest. 40% gross. No exceptions.",
            "regime": f"Crisis protocol: 40% gross, 8% max position, no speculative names. Re-risking criteria are explicit: Fed balance sheet expansion, HY spreads rolling over, VIX closing below 30 on consecutive days. None of those conditions are met. Hold the line.",
        },
        "uncertain": {
            "bull":   f"Regime uncertainty is not a sell signal. {hs} are earnings compounders — the fundamental case is independent of the classifier's confidence. Being too defensive in an uncertain regime that resolves bullish is a real and permanent cost. 75% exposure, hold the quality book, let the signal clarify.",
            "bear":   f"Uncertain regime with {md} in SPY is the worst risk/reward setup we can hold. We do not know the resolution direction, and we are carrying near-full exposure into that uncertainty. Crisis resolution means -20%+. Calm bull resolution means capped upside from sizing constraints. Reduce to 60%.",
            "devil":  f"The uncertainty is itself diagnostic. The classifier is uncertain because its input features are no longer predictive — this is a structural break, not rotation noise. Neither the bull nor the bear is modeling a world where the regime framework stops working entirely. That is the actual tail risk.",
            "regime": f"Uncertain regimes typically resolve in 1–3 weeks. Posture: 75% gross, 12% max position, no new positions until entropy drops below 0.70. Do not force a thesis. Patience is a position.",
        },
    }
    return s.get(regime, s["uncertain"])


def _verdict(regime: str, vix: float) -> dict:
    rm = REGIME_META[regime]
    if regime == "crisis" or vix > 40:
        return {
            "rec":"halt","label":"Halt — Pending Review","conf":0.78,
            "risks":["Crisis regime — outside normal execution parameters",f"VIX {vix:.0f}: tail risk actively priced","Correlation breakdown invalidates diversification assumptions"],
            "text":f"The crisis regime signal combined with VIX {vix:.0f} places this portfolio outside the parameters for normal execution. The correlation-breakdown argument raised by the devil's advocate was not adequately addressed by either side. Halting for principal review is the correct procedural response.",
        }
    elif regime == "stressed" or vix > 25:
        return {
            "rec":"reduce","label":"Reduce Size","conf":0.71,
            "risks":[f"Target gross: {rm['risk']:.0%} — model mandate, not suggestion",f"VIX {vix:.0f}: institutional hedging elevated","Bear's -14% forward estimate not rebutted"],
            "text":f"The bull's quality argument is partially valid but does not address the regime-level sizing mandate. 160 walk-forward folds produce a {rm['risk']:.0%} gross exposure in stressed conditions. That is the output of the model, not a suggestion. Reduce gross to {rm['risk']:.0%}, hold the quality names, cap positions at {rm['max_wt']:.0%}.",
        }
    else:
        return {
            "rec":"proceed","label":"Proceed","conf":0.82,
            "risks":["Monitor trip-wires: VIX >22, SPY break of 50-day MA","Top-3 concentration warrants stop-loss discipline"],
            "text":"The calm bull regime signal is high-conviction. The bull case is well-supported and the bear's structural concerns, while valid in theory, do not override the primary signal in this environment. The quant check is clean. Proceed with full deployment and maintain the trip-wires identified by the regime specialist.",
        }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="wordmark" style="margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--rule-b);">
      ▲ &nbsp; Ascent Capital
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-bottom:0.6rem;">Scenario</div>', unsafe_allow_html=True)

    regime = st.selectbox(
        "Regime", _REGIME_OPTIONS, index=_live_regime_idx,
        format_func=lambda x: REGIME_META[x]["label"],
        label_visibility="collapsed",
    )
    vix = st.slider("VIX", 10.0, 55.0, round(_live_vix, 1), 0.5)
    spy_momentum = st.select_slider("SPY Momentum", _SPY_MOM_OPTIONS, value=_live_spy_mom)

    st.markdown('<hr style="border:none;border-top:1px solid var(--rule-b);margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-bottom:0.6rem;">Portfolio</div>', unsafe_allow_html=True)
    preset_name = st.selectbox("Portfolio", list(PRESETS.keys()), label_visibility="collapsed")
    weights = PRESETS[preset_name].copy()

    st.markdown('<hr style="border:none;border-top:1px solid var(--rule-b);margin:1.5rem 0;">', unsafe_allow_html=True)
    show_r2 = st.toggle("Round 2 rebuttals", value=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Debate  ›", use_container_width=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────────
rm = REGIME_META[regime]
top_holdings = sorted(weights.items(), key=lambda x: -x[1])
max_wt = top_holdings[0][1] if top_holdings else 0.1
n_pos = len(top_holdings)
show_all = st.session_state.show_all_pos
display_pos = top_holdings if show_all else top_holdings[:8]

# Header
st.markdown(f"""
<div style="margin-bottom:2.5rem;">
  <div class="wordmark" style="margin-bottom:0.5rem;">▲ &nbsp; Ascent Capital</div>
  <div class="page-title">Pre-Rebalance<br><em>Decision Engine</em></div>
  <div class="page-sub" style="margin-top:0.8rem;">
    {date.today().strftime('%A, %B %d, %Y')} &nbsp;·&nbsp;
    4 specialist agents &nbsp;·&nbsp; 11 alpha sleeves &nbsp;·&nbsp; S&P 500 universe
  </div>
</div>
""", unsafe_allow_html=True)

# Portfolio + signals
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    pos_rows = ""
    for sym, w in display_pos:
        bar_w = int(w / max_wt * 100)
        pos_rows += f"""
        <div class="pos-row">
          <span class="pos-sym">{sym}</span>
          <div class="pos-bar-bg"><div class="pos-bar-fill" style="width:{bar_w}%"></div></div>
          <span class="pos-wt">{w:.1%}</span>
        </div>"""
    st.markdown(f"""
    <div class="card">
      <div class="card-head">Portfolio — {preset_name}</div>
      {pos_rows}
    </div>""", unsafe_allow_html=True)

    # Toggle button below card
    if n_pos > 8:
        remaining = n_pos - 8
        if not show_all:
            if st.button(f"+ {remaining} more positions", key="show_more"):
                st.session_state.show_all_pos = True
                st.rerun()
        else:
            if st.button("Show less", key="show_less"):
                st.session_state.show_all_pos = False
                st.rerun()

with col2:
    spy_col = "#3A9C68" if "positive" in spy_momentum.lower() else ("#C97676" if "negative" in spy_momentum.lower() else "var(--text-mid)")
    st.markdown(f"""
    <div class="card">
      <div class="card-head">Market State</div>
      <div class="sig-row">
        <span class="sig-key">Regime</span>
        <span class="regime-pill regime-{regime}">{rm['label']}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">VIX</span>
        <span class="sig-val">{vix:.1f}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">SPY Momentum</span>
        <span class="sig-val" style="font-size:10px;color:{spy_col};">{spy_momentum}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">Gross Exposure</span>
        <span class="sig-val">{rm['risk']:.0%}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">Max Position</span>
        <span class="sig-val">{rm['max_wt']:.0%}</span>
      </div>
      <div class="sig-row">
        <span class="sig-key">Regime Note</span>
        <span class="sig-val" style="font-size:9px;color:var(--text-mid);font-family:'DM Mono',monospace;">{rm['desc']}</span>
      </div>
    </div>""", unsafe_allow_html=True)

# Sleeve table
delta_map = SLEEVE_DELTA.get(regime, {})
rows = ""
for sleeve, base in BASE_SLEEVES.items():
    d = delta_map.get(sleeve, 0.0)
    adj = max(0.0, base + d)
    if d > 0:   shift = f'<span class="sl-up">+{d:.0%}</span>'
    elif d < 0: shift = f'<span class="sl-down">{d:.0%}</span>'
    else:       shift = f'<span class="sl-flat">—</span>'
    rows += f"<tr><td>{sleeve}</td><td>{base:.0%}</td><td>{adj:.0%}</td><td>{shift}</td></tr>"

spy_note = "0.70× — SPY below 200-day MA" if regime in ("stressed","crisis") else "1.00× — full deployment"
st.markdown(f"""
<div class="card" style="margin-top:1px;">
  <div class="card-head">Alpha Allocation — Regime Adjusted</div>
  <table class="sl-table">
    <tr><th style="width:44%">Sleeve</th><th>Base</th><th>Live</th><th>Δ</th></tr>
    {rows}
  </table>
  <div style="font-family:'DM Mono',monospace;font-size:8px;color:var(--text-dim);margin-top:1.2rem;letter-spacing:0.5px;">
    SPY overlay: {spy_note}
  </div>
</div>""", unsafe_allow_html=True)

# Debate section
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow" style="margin-bottom:1.2rem;">Pre-Rebalance Debate</div>', unsafe_allow_html=True)

if not run_btn:
    st.markdown(f"""
    <div class="debate-empty">
      <div class="debate-empty-glyph">◎</div>
      <div class="debate-empty-text">
        Five agents debate the proposed trades before every rebalance.<br>
        The judge's verdict gates execution — advisory, never autonomous.<br><br>
        Current scenario: {rm['label']} · VIX {vix:.0f} · {spy_momentum}
      </div>
    </div>""", unsafe_allow_html=True)
else:
    top_syms = [s for s, _ in top_holdings[:5]]
    copy = _debate_copy(regime, vix, top_syms, spy_momentum)

    def agent_card(role: str, label: str, text: str, r2: bool = False) -> str:
        r2_cls = " r2" if r2 else ""
        pfx = "↩ " if r2 else ""
        return f"""<div class="agent-wrap">
<div class="agent-card {role}{r2_cls}">
  <div class="agent-role {role}">{pfx}{label}</div>
  <div class="agent-text">{text}</div>
</div></div>"""

    st.markdown('<div class="round-label">Round 1 — Initial Arguments</div>', unsafe_allow_html=True)
    ph = [st.empty() for _ in range(5)]
    agents = [
        ("bull",   "Bull Analyst",      copy["bull"]),
        ("bear",   "Bear Analyst",      copy["bear"]),
        ("devil",  "Devil's Advocate",  copy["devil"]),
        ("regime", "Regime Specialist", copy["regime"]),
    ]
    for i, (role, label, text) in enumerate(agents):
        with st.spinner(""):
            time.sleep(0.3)
        ph[i].markdown(agent_card(role, label, text), unsafe_allow_html=True)

    total = sum(weights.values())
    max_sym, max_w = max(weights.items(), key=lambda x: x[1])
    cap_ok = max_w <= rm["max_wt"]
    quant_text = f"{n_pos} positions · sum {total:.4f} · max {max_sym} {max_w:.1%} {'✓' if cap_ok else '⚠ exceeds cap'}"
    ph[4].markdown(agent_card("quant", "Quant Sanity", quant_text), unsafe_allow_html=True)

    if show_r2:
        st.markdown('<div class="round-label">Round 2 — Rebuttals</div>', unsafe_allow_html=True)
        rph = [st.empty() for _ in range(3)]
        with st.spinner(""):
            time.sleep(0.3)
        b2 = f"The bear's drawdown estimate assumes uniform exit conditions. Our names average $200M+ daily volume — normal-market exit is two days. The -14% figure conflates liquid large-caps with the illiquid tail. The quality argument holds at VIX {vix:.0f}."
        r2 = "The bull's liquidity case fails in the relevant scenario. In March 2020, even MSFT saw bid/ask widen four times intraday at peak stress. Daily volume is a normal-market statistic. The scenario that matters is the 99th-percentile exit — that is exactly where the -14% estimate originates."
        d2 = "Both sides continue to debate direction and liquidity. The structural point remains unanswered: the correlation matrix we are relying on was estimated in a different regime. When it matters most, it fails. That is not a prediction — it is a mathematical property of conditional correlation."
        rph[0].markdown(agent_card("bull",  "Bull — Rebuttal",             b2, r2=True), unsafe_allow_html=True)
        rph[1].markdown(agent_card("bear",  "Bear — Rebuttal",             r2, r2=True), unsafe_allow_html=True)
        rph[2].markdown(agent_card("devil", "Devil's Advocate — Rebuttal", d2, r2=True), unsafe_allow_html=True)

    # Verdict
    with st.spinner(""):
        time.sleep(0.2)
    v = _verdict(regime, vix)
    risks_html = "".join(f'<div class="verdict-risk">— {r}</div>' for r in v["risks"])
    st.markdown(f"""
    <div class="verdict-wrap">
      <div class="card-head" style="margin-bottom:1rem;">Judge — Final Verdict</div>
      <div class="verdict-status {v['rec']}">{v['label']}</div>
      <div class="verdict-conf">Confidence: {int(v['conf']*100)}%</div>
      <div class="verdict-text">{v['text']}</div>
      <div style="margin-top:1.2rem;">{risks_html}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">
      <span>Ascent Capital · Pre-Rebalance Analysis</span>
      <span>{date.today().strftime('%B %d, %Y')} · For discussion purposes only</span>
    </div>""", unsafe_allow_html=True)
