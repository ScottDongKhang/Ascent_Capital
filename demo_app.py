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

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ascent Capital",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0D0D0D; color: #E0E0E0; }
  section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
  .block-container { padding-top: 2rem; max-width: 1100px; }
  #MainMenu, footer, header { visibility: hidden; }
  h1, h2, h3 { color: #C9A84C !important; }

  .ac-card {
    background: #161616; border: 1px solid #252525;
    border-radius: 6px; padding: 18px 20px; margin: 10px 0;
  }
  .ac-card-gold   { border-left: 3px solid #C9A84C; }
  .ac-card-bull   { border-left: 3px solid #27AE60; }
  .ac-card-bear   { border-left: 3px solid #E74C3C; }
  .ac-card-devil  { border-left: 3px solid #9B59B6; }
  .ac-card-regime { border-left: 3px solid #2980B9; }
  .ac-card-quant  { border-left: 3px solid #7F8C8D; }
  .ac-card-r2     { border-left: 3px solid #444; opacity: 0.9; }

  .ac-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 8px;
  }
  .ac-label-gold   { color: #C9A84C; }
  .ac-label-bull   { color: #27AE60; }
  .ac-label-bear   { color: #E74C3C; }
  .ac-label-devil  { color: #9B59B6; }
  .ac-label-regime { color: #2980B9; }
  .ac-label-quant  { color: #7F8C8D; }
  .ac-label-r2     { color: #888; }

  .ac-text  { color: #D8D8D8; line-height: 1.65; font-size: 14px; }
  .ac-muted { color: #666; font-size: 12px; }

  .verdict-proceed {
    background: #1A3A27; border: 1px solid #27AE60; color: #2ECC71;
    padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }
  .verdict-reduce {
    background: #3A2A10; border: 1px solid #F39C12; color: #F39C12;
    padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }
  .verdict-halt {
    background: #3A1010; border: 1px solid #E74C3C; color: #E74C3C;
    padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }

  .metric-row { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }
  .metric-box {
    background: #1A1A1A; border: 1px solid #252525; border-radius: 4px;
    padding: 10px 16px; min-width: 120px;
  }
  .metric-label { font-size: 10px; color: #888; letter-spacing: 1px; text-transform: uppercase; }
  .metric-value { font-size: 20px; font-weight: 600; color: #C9A84C; margin-top: 2px; }

  .ac-sep { border: none; border-top: 1px solid #222; margin: 24px 0; }

  .np-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .np-table th {
    color: #888; font-weight: 600; font-size: 10px; letter-spacing: 1px;
    text-transform: uppercase; padding: 6px 12px; text-align: left;
    border-bottom: 1px solid #222;
  }
  .np-table td { padding: 8px 12px; color: #D8D8D8; border-bottom: 1px solid #1A1A1A; }
  .np-shift-up   { color: #27AE60; }
  .np-shift-down { color: #E74C3C; }
  .np-neutral    { color: #888; }

  .conf-bar-bg   { background: #222; border-radius: 3px; height: 6px; margin-top: 6px; }
  .conf-bar-fill {
    background: linear-gradient(90deg, #C9A84C, #F0C86B);
    border-radius: 3px; height: 6px;
  }

  .sidebar-section {
    color: #C9A84C; font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 20px 0 8px 0; border-top: 1px solid #222; padding-top: 14px;
  }

  .step-label {
    display: inline-block;
    background: #1A1A1A; border: 1px solid #333;
    color: #C9A84C; font-size: 9px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 3px 10px; border-radius: 3px; margin-bottom: 10px;
  }

  .oos-stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
  .oos-stat {
    background: #141414; border: 1px solid #252525; border-radius: 4px;
    padding: 8px 14px; text-align: center;
  }
  .oos-stat-label { font-size: 9px; color: #666; letter-spacing: 1px; text-transform: uppercase; }
  .oos-stat-value { font-size: 17px; font-weight: 600; color: #C9A84C; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Live data loader ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_live_data() -> tuple:
    """Return (weights, regime, vix, spy_momentum) from output files + yfinance."""
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
            if pct > 0.03:       spy_mom = "Strong positive"
            elif pct > 0.01:     spy_mom = "Weak positive"
            elif pct < -0.03:    spy_mom = "Strong negative"
            elif pct < -0.01:    spy_mom = "Weak negative"
            else:                spy_mom = "Neutral"
    except Exception:
        pass

    return weights, regime, vix, spy_mom


_live_weights, _live_regime, _live_vix, _live_spy_mom = _load_live_data()

_REGIME_OPTIONS  = ["calm_bull", "stressed", "crisis", "uncertain"]
_SPY_MOM_OPTIONS = ["Strong negative", "Weak negative", "Neutral", "Weak positive", "Strong positive"]
_live_regime_idx  = _REGIME_OPTIONS.index(_live_regime) if _live_regime in _REGIME_OPTIONS else 1
_live_spy_mom_idx = _SPY_MOM_OPTIONS.index(_live_spy_mom) if _live_spy_mom in _SPY_MOM_OPTIONS else 2


# ── Constants ────────────────────────────────────────────────────────────────────
REGIME_CONFIG = {
    "calm_bull": {
        "risk_multiplier": 1.00, "max_weight": 0.15,
        "description": "Momentum intact. Full deployment appropriate.",
        "color": "#27AE60",
    },
    "stressed": {
        "risk_multiplier": 0.65, "max_weight": 0.10,
        "description": "Elevated drawdown risk. Defensive rebalancing warranted.",
        "color": "#F39C12",
    },
    "crisis": {
        "risk_multiplier": 0.40, "max_weight": 0.08,
        "description": "Acute stress. Preserve capital. Minimum equity exposure.",
        "color": "#E74C3C",
    },
    "uncertain": {
        "risk_multiplier": 0.75, "max_weight": 0.12,
        "description": "Regime signal unclear. Reduce size, wait for confirmation.",
        "color": "#9B59B6",
    },
}

BASE_SLEEVES = {
    "Trend":           0.44,
    "Stat-Arb":        0.15,
    "Mean Rev":        0.05,
    "ML (XGBoost)":    0.10,
    "Vol-Regime":      0.05,
    "Fundamental":     0.05,
    "Earnings (PEAD)": 0.05,
    "Analyst":         0.05,
    "Options Flow":    0.02,
    "Insider":         0.02,
    "Short Interest":  0.02,
}

SLEEVE_ADJUSTMENTS = {
    "calm_bull": {k: 0.0 for k in BASE_SLEEVES},
    "stressed": {
        "Trend": +0.08, "Stat-Arb": -0.03, "Mean Rev": +0.02, "ML (XGBoost)": -0.07,
        "Vol-Regime": +0.03, "Fundamental": -0.02, "Earnings (PEAD)": -0.01,
        "Analyst": -0.01, "Options Flow": 0.0, "Insider": 0.0, "Short Interest": 0.0,
    },
    "crisis": {
        "Trend": +0.12, "Stat-Arb": -0.05, "Mean Rev": +0.03, "ML (XGBoost)": -0.08,
        "Vol-Regime": +0.05, "Fundamental": -0.04, "Earnings (PEAD)": -0.02,
        "Analyst": -0.01, "Options Flow": 0.0, "Insider": 0.0, "Short Interest": 0.0,
    },
    "uncertain": {
        "Trend": +0.05, "Stat-Arb": -0.02, "Mean Rev": 0.0, "ML (XGBoost)": -0.03,
        "Vol-Regime": +0.02, "Fundamental": -0.01, "Earnings (PEAD)": -0.01,
        "Analyst": 0.0, "Options Flow": 0.0, "Insider": 0.0, "Short Interest": 0.0,
    },
}

_FALLBACK_WEIGHTS = {
    "KMLM": 0.097, "IFRA": 0.087, "PDBC": 0.082, "AMKR": 0.056,
    "FIX": 0.056, "VAL": 0.056, "VICR": 0.056, "VRT": 0.056, "WDC": 0.056,
    "CNC": 0.054, "EWY": 0.041, "DBB": 0.043, "VC": 0.038,
    "DBA": 0.033, "EWC": 0.033, "PVH": 0.032, "TIP": 0.022,
    "NUE": 0.020, "XPO": 0.019, "APA": 0.019, "AAXJ": 0.015, "EEM": 0.015, "EWZ": 0.012,
}

PORTFOLIO_PRESETS = {
    f"Live Portfolio ({date.today().strftime('%b %d')})": _live_weights or _FALLBACK_WEIGHTS,
    "Tech-Heavy": {
        "AAPL": 0.15, "MSFT": 0.13, "NVDA": 0.12, "GOOGL": 0.10,
        "META": 0.10, "ADBE": 0.08, "CRWD": 0.08, "AMD": 0.07,
        "INTC": 0.06, "TXN": 0.06, "PANW": 0.05,
    },
    "Defensive": {
        "JNJ": 0.12, "WMT": 0.12, "KO": 0.10, "PG": 0.10,
        "NEE": 0.09, "MRK": 0.09, "ABT": 0.08, "DUK": 0.08,
        "TLT": 0.12, "GLD": 0.10,
    },
    "Macro Stress": {
        "TLT": 0.25, "GLD": 0.20, "BIL": 0.15, "IEF": 0.15,
        "VIXY": 0.10, "UUP": 0.08, "PDBC": 0.07,
    },
}


# ── Demo debate arguments ────────────────────────────────────────────────────────
def get_demo_args(regime: str, vix: float, top_holdings: list, spy_momentum: str) -> dict:
    holdings_str = ", ".join(top_holdings[:4]) if top_holdings else "the portfolio"
    momentum_desc = {
        "Strong positive": "strong upward momentum",
        "Weak positive":   "weakening upward momentum",
        "Neutral":         "flat momentum",
        "Weak negative":   "early downtrend signals",
        "Strong negative": "strong downtrend momentum",
    }.get(spy_momentum, "mixed signals")
    first = top_holdings[0] if top_holdings else "our top holding"

    args = {
        "calm_bull": {
            "bull": f"The regime is clearly supportive. {holdings_str} represent quality compounders with durable earnings power — not speculative names that need macro tailwinds to work. VIX at {vix:.0f} is benign; institutional positioning is constructive. The trend signal is intact across our top holdings. Missing this move by over-hedging is the real risk.",
            "bear": f"Calm bull can flip fast. VIX at {vix:.0f} doesn't mean risk is absent — it means it's unpriced. {first} and peers are priced for perfection at current multiples. Any earnings miss or Fed pivot repricing could gap us down 8–12% before stop-losses trigger. The risk/reward is not in our favor.",
            "devil": f"The assumption both sides are making: correlation stays low. In the last three calm-bull-to-stress transitions (2018, 2020, 2022), inter-portfolio correlation spiked from 0.35 to 0.82 within two weeks. Our {len(top_holdings)}-position portfolio becomes a single factor bet exactly when we need diversification most.",
            "regime": f"Calm bull historically lasts 3–6 months before exhaustion signals appear. Breadth narrowing and vol compression are the canonical warnings. Current posture is textbook-correct. Set trip-wires: VIX above 20 or SPY below 50-day MA triggers defensive review.",
        },
        "stressed": {
            "bull": f"Stressed regimes are when quality separates from noise. {holdings_str} have $10B+ in free cash flow — they don't need credit markets to function. VIX at {vix:.0f} is elevated but not crisis-level. The market is pricing in a macro outcome that won't materialize.",
            "bear": f"VIX at {vix:.0f} reflects real institutional hedging. With {momentum_desc} in SPY, the trend has broken. When VIX crosses 25 in a stressed regime, the median drawdown from entry is -14% over the next 6 weeks. Reduce gross exposure to 65%.",
            "devil": f"Both sides are anchoring on the wrong variable — VIX and momentum. The actual kill switch is credit spreads. If HY spreads widen 150bps from here, equity correlations spike and liquidity in smaller positions disappears. Monte Carlo p5 shows -21% in this scenario.",
            "regime": f"Stressed regime playbook: 65% gross exposure, cap max position at 10%, shift toward quality. With {momentum_desc}, we are not yet confirming recovery. Typical stressed duration is 4–12 weeks. Premature re-risking is the most common mistake in this regime.",
        },
        "crisis": {
            "bull": f"Crisis creates the best entry points. {holdings_str} have balance sheets that can absorb 18–24 months of stress. VIX at {vix:.0f} is extreme fear, not rational risk pricing. Investors who held quality through 2009 and 2020 captured 40%+ in the subsequent 12 months. Cutting here locks in the loss.",
            "bear": f"VIX at {vix:.0f} is the market's honest assessment of tail risk. In a crisis regime, the primary risk is permanent capital impairment. Correlation across this portfolio will be near 1.0 in a liquidation event. Cash and TLT allow us to participate in the recovery from a position of strength.",
            "devil": f"The fatal assumption: this crisis has a defined endpoint. 2008 lasted 18 months. Our conviction in {first} assumes business-as-usual recovery — it ignores counterparty exposure, off-balance-sheet risk, regulatory response. The unknown unknowns are the point. Forty percent gross exposure, maximum.",
            "regime": f"Crisis regime: capital preservation above all. Maximum defensive posture, 40% gross exposure, no speculative positions. Recovery positioning comes after stabilization signals: Fed intervention, credit spread reversal, VIX below 30 on a closing basis. We are not there.",
        },
        "uncertain": {
            "bull": f"Uncertainty is not a reason to underperform. {holdings_str} are earnings compounders, not macro plays. An unclear regime signal doesn't change the fundamental case. The risk of being too defensive in an uncertain regime that resolves bullish is real. Maintain 75% exposure.",
            "bear": f"Uncertain regime with {momentum_desc} is the worst combination. We don't know which direction this resolves. If it resolves to crisis, we face -20%+; if it resolves to calm bull, upside is capped. Reduce to 60% and buy optionality.",
            "devil": f"The regime uncertainty itself is the signal everyone is ignoring. The model is uncertain because the underlying market structure is breaking down — the features that trained the classifier are no longer predictive. This is a structural break, not regime rotation.",
            "regime": f"Uncertain regime historically resolves in 1–3 weeks. Correct posture: 75% gross exposure, max 12% per position, wait for confirmation. The entropy signal has crossed 0.90 — insufficient conviction to justify full deployment in either direction.",
        },
    }
    return args.get(regime, args["uncertain"])


# ── Helpers ──────────────────────────────────────────────────────────────────────
def agent_card(label: str, style: str, text: str, round2: bool = False) -> str:
    card_class = f"ac-card ac-card-{style}" + (" ac-card-r2" if round2 else "")
    prefix = "↩ " if round2 else ""
    return f"""
<div class="{card_class}">
  <div class="ac-label ac-label-{style}">{prefix}{label}</div>
  <div class="ac-text">{text}</div>
</div>"""


def verdict_card(verdict: dict) -> str:
    rec = verdict.get("recommendation", "reduce_size")
    conf = float(verdict.get("confidence", 0.5))
    risks = verdict.get("key_risks", [])
    reasoning = verdict.get("reasoning", "")
    badge_class = {"proceed": "verdict-proceed", "reduce_size": "verdict-reduce", "halt_and_review": "verdict-halt"}.get(rec, "verdict-reduce")
    badge_text  = {"proceed": "▲ PROCEED", "reduce_size": "◆ REDUCE SIZE", "halt_and_review": "■ HALT & REVIEW"}.get(rec, rec.upper())
    conf_pct = int(conf * 100)
    risks_html = "".join(f"<div style='color:#999;font-size:13px;margin:4px 0;'>· {r}</div>" for r in risks[:3])
    return f"""
<div class="ac-card ac-card-gold" style="margin-top:24px;">
  <div class="ac-label ac-label-gold">Judge — Final Verdict</div>
  <div style="margin:12px 0;">
    <span class="{badge_class}">{badge_text}</span>
    <span style="color:#666;font-size:12px;margin-left:12px;">Confidence: {conf_pct}%</span>
  </div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%;"></div></div>
  {f'<div style="margin-top:14px;">{risks_html}</div>' if risks else ''}
  {f'<div class="ac-text" style="margin-top:12px;color:#BBB;">{reasoning}</div>' if reasoning else ''}
</div>"""


def neuroplasticity_card(regime: str, gross_exposure: float) -> str:
    adj = SLEEVE_ADJUSTMENTS.get(regime, SLEEVE_ADJUSTMENTS["uncertain"])
    rc  = REGIME_CONFIG[regime]
    spy_overlay = "0.70× applied" if regime in ("stressed", "crisis") else "1.00× (full)"
    rows = ""
    for sleeve, base in BASE_SLEEVES.items():
        delta    = adj.get(sleeve, 0.0)
        adjusted = max(0.0, base + delta)
        if delta > 0:   shift = f'<span class="np-shift-up">▲ +{delta:.0%}</span>'
        elif delta < 0: shift = f'<span class="np-shift-down">▼ {delta:.0%}</span>'
        else:           shift = '<span class="np-neutral">—</span>'
        rows += f"<tr><td>{sleeve}</td><td>{base:.0%}</td><td>{adjusted:.0%}</td><td>{shift}</td></tr>"
    return f"""
<div class="ac-card ac-card-gold">
  <div class="ac-label ac-label-gold">Step 2 — Regime Adaptation (Neuroplasticity)</div>
  <div class="ac-text" style="margin-bottom:14px;color:#AAA;">
    Every rebalance, the regime engine re-weights all 11 alpha sleeves based on the current
    market state. In a stressed or crisis regime, trend following gets more weight and ML/fundamental
    models get less — they were trained on calmer distributions and lose predictive power in transitions.
    The SPY 200-day MA overlay further scales gross exposure.
  </div>
  <div class="metric-row" style="margin-bottom:16px;">
    <div class="metric-box">
      <div class="metric-label">Regime</div>
      <div class="metric-value" style="font-size:15px;color:{rc['color']}">{regime.replace('_',' ').title()}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Gross Exposure</div>
      <div class="metric-value">{gross_exposure:.0%}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Max Position</div>
      <div class="metric-value">{rc['max_weight']:.0%}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">SPY 200MA</div>
      <div class="metric-value" style="font-size:13px;">{spy_overlay}</div>
    </div>
  </div>
  <table class="np-table">
    <tr><th>Alpha Sleeve</th><th>Base Weight</th><th>Adapted Weight</th><th>Shift</th></tr>
    {rows}
  </table>
</div>"""


def _demo_verdict(regime: str, vix: float) -> dict:
    if regime == "crisis" or vix > 40:
        return {
            "confidence": 0.78, "recommendation": "halt_and_review",
            "key_risks": [
                f"Crisis regime with VIX={vix:.0f} exceeds risk tolerance",
                "Correlation structure breakdown — diversification near zero",
                "Liquidity gap risk in smaller positions under forced selling",
            ],
            "reasoning": f"The regime signal (crisis) combined with VIX={vix:.0f} places us outside the risk envelope for normal execution. The devil's advocate argument on correlation breakdown is the decisive factor — we cannot rely on cross-position diversification. Halting for human review is the correct process response.",
        }
    elif regime == "stressed" or vix > 25:
        return {
            "confidence": 0.71, "recommendation": "reduce_size",
            "key_risks": [
                f"Stressed regime — gross exposure should be {REGIME_CONFIG['stressed']['risk_multiplier']:.0%}",
                f"VIX at {vix:.0f} signals active institutional hedging",
                "Devil's advocate credit-spread argument unresolved",
            ],
            "reasoning": f"The bull case is noted but insufficient against the stressed regime signal. Gross exposure reduction to {REGIME_CONFIG['stressed']['risk_multiplier']:.0%} is warranted. Reduce size, maintain quality positions, re-evaluate at next rebalance.",
        }
    else:
        return {
            "confidence": 0.82, "recommendation": "proceed",
            "key_risks": [
                "Regime transition risk — calm bull can flip on catalyst",
                "Position concentration in top 3 holdings",
            ],
            "reasoning": "The bull case is well-supported by the calm bull regime signal. The bear's concerns are noted but appropriately priced. The quant sanity check is clean. Proceed with full deployment.",
        }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
      <div style="color:#C9A84C;font-size:22px;font-weight:700;letter-spacing:2px;">▲ ASCENT</div>
      <div style="color:#555;font-size:10px;letter-spacing:3px;margin-top:2px;">CAPITAL</div>
    </div>
    <div style="color:#555;font-size:11px;text-align:center;padding-bottom:10px;line-height:1.5;">
      Modular quant platform · 4 agents · 11 alpha sleeves · LLM debate layer
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Scenario Controls</div>', unsafe_allow_html=True)
    st.caption("Change these to explore how the system responds to different market conditions.")

    regime = st.selectbox(
        "Market Regime",
        _REGIME_OPTIONS,
        index=_live_regime_idx,
        format_func=lambda x: x.replace("_", " ").title(),
        help="The HMM regime engine classifies the current market state. This drives sleeve weights, position sizing, and gross exposure.",
    )

    vix = st.slider(
        "VIX Level", min_value=10.0, max_value=55.0,
        value=round(_live_vix, 1), step=0.5,
        help="Used by debate agents to calibrate tail-risk arguments.",
    )

    spy_momentum = st.select_slider(
        "SPY Momentum",
        options=_SPY_MOM_OPTIONS,
        value=_live_spy_mom,
        help="20-day SPY momentum signal.",
    )

    st.markdown('<div class="sidebar-section">Portfolio</div>', unsafe_allow_html=True)
    st.caption("First preset is today's live portfolio. Others are illustrative scenarios.")

    preset_name = st.selectbox("Portfolio Preset", list(PORTFOLIO_PRESETS.keys()))
    weights = PORTFOLIO_PRESETS[preset_name].copy()

    st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)
    show_round2 = st.toggle("Show Round 2 Rebuttals", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run the Debate", use_container_width=True, type="primary")


# ── MAIN PANEL ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:16px;">
  <div style="color:#C9A84C;font-size:11px;letter-spacing:3px;text-transform:uppercase;">
    Ascent Capital — Decision Engine
  </div>
  <div style="color:#888;font-size:12px;margin-top:4px;">
    {date.today().strftime('%B %d, %Y')} · 4 specialist agents · 11 alpha sleeves ·
    S&amp;P 500 universe · Walk-forward validated
  </div>
</div>
""", unsafe_allow_html=True)

# OOS stats strip
st.markdown("""
<div class="oos-stat-row">
  <div class="oos-stat">
    <div class="oos-stat-label">OOS CAGR</div>
    <div class="oos-stat-value">22.4%</div>
  </div>
  <div class="oos-stat">
    <div class="oos-stat-label">OOS Sharpe</div>
    <div class="oos-stat-value">0.887</div>
  </div>
  <div class="oos-stat">
    <div class="oos-stat-label">Max Drawdown</div>
    <div class="oos-stat-value" style="color:#E74C3C;">−35.8%</div>
  </div>
  <div class="oos-stat">
    <div class="oos-stat-label">Alpha vs SPY</div>
    <div class="oos-stat-value">+8.9%</div>
  </div>
  <div class="oos-stat">
    <div class="oos-stat-label">Walk-forward Folds</div>
    <div class="oos-stat-value">160</div>
  </div>
  <div class="oos-stat">
    <div class="oos-stat-label">Methodology</div>
    <div class="oos-stat-value" style="font-size:11px;">No look-ahead</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)

# ── Step 1: Portfolio snapshot ───────────────────────────────────────────────────
st.markdown('<div class="step-label">Step 1 — Today\'s Portfolio &amp; Market State</div>', unsafe_allow_html=True)
st.markdown("""
<div class="ac-text" style="color:#888;margin-bottom:12px;">
  Each day, 4 specialist agents (US Equities, Macro, International, Alternatives) each run the full
  pipeline independently and submit target weights. The orchestrator merges them using skill-score-based
  capital allocation and cross-agent correlation guards.
</div>""", unsafe_allow_html=True)

rc = REGIME_CONFIG[regime]
gross_exposure = rc["risk_multiplier"]
col1, col2 = st.columns([3, 2])

with col1:
    top_holdings = sorted(weights.items(), key=lambda x: -x[1])
    holdings_html = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
        f"border-bottom:1px solid #1E1E1E;font-size:13px;'>"
        f"<span style='color:#D0D0D0;'>{sym}</span>"
        f"<span style='color:#C9A84C;font-weight:600;'>{w:.1%}</span></div>"
        for sym, w in top_holdings[:8]
    )
    if len(top_holdings) > 8:
        holdings_html += f"<div style='color:#555;font-size:11px;margin-top:6px;'>+{len(top_holdings)-8} more positions</div>"

    live_badge = " <span style='color:#27AE60;font-size:9px;'>● live</span>" if _live_weights else ""
    st.markdown(f"""
    <div class="ac-card">
      <div class="ac-label ac-label-gold">Merged Portfolio — {preset_name}{live_badge}</div>
      {holdings_html}
    </div>""", unsafe_allow_html=True)

with col2:
    regime_color = rc["color"]
    st.markdown(f"""
    <div class="metric-row" style="flex-direction:column;gap:10px;">
      <div class="metric-box">
        <div class="metric-label">Regime Signal</div>
        <div class="metric-value" style="color:{regime_color};font-size:16px;">{regime.replace('_',' ').title()}</div>
        <div class="ac-muted" style="margin-top:4px;">{rc['description']}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">VIX</div>
        <div class="metric-value">{vix:.0f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Gross Exposure</div>
        <div class="metric-value">{gross_exposure:.0%}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)

# ── Step 2: Neuroplasticity ──────────────────────────────────────────────────────
st.markdown(neuroplasticity_card(regime, gross_exposure), unsafe_allow_html=True)

st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)

# ── Step 3: Debate intro ─────────────────────────────────────────────────────────
st.markdown('<div class="step-label">Step 3 — Pre-Rebalance Debate</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="ac-text" style="color:#888;margin-bottom:16px;">
  Before every rebalance, five agents debate the proposed trade list. Each takes a distinct role:
  the <span style="color:#27AE60;">Bull</span> makes the strongest case for executing;
  the <span style="color:#E74C3C;">Bear</span> argues for reducing risk;
  the <span style="color:#9B59B6;">Devil's Advocate</span> attacks the most dangerous assumption both sides share;
  the <span style="color:#2980B9;">Regime Specialist</span> gives the sizing playbook for the current regime;
  and the <span style="color:#7F8C8D;">Quant Sanity</span> check verifies the weights are structurally clean.
  After Round 2 rebuttals, the <span style="color:#C9A84C;">Judge</span> synthesizes a verdict:
  <em>proceed</em>, <em>reduce size</em>, or <em>halt and review</em>.
  The verdict gates execution — it does not override the portfolio engine directly.
</div>
<div class="ac-muted" style="margin-bottom:16px;">
  Use the sidebar to explore how the debate changes under different regimes and VIX levels.
  The current scenario: <strong style="color:#E0E0E0;">{regime.replace('_',' ').title()} · VIX {vix:.0f} · SPY {spy_momentum}</strong>
</div>""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:50px 0;color:#333;border:1px dashed #222;border-radius:6px;">
      <div style="font-size:32px;margin-bottom:12px;">▶</div>
      <div style="font-size:14px;color:#555;">Click "Run the Debate" in the sidebar to see the agents argue</div>
      <div style="font-size:11px;color:#333;margin-top:8px;">Try switching the regime to "crisis" first</div>
    </div>
    """, unsafe_allow_html=True)
else:
    top_symbols = [s for s, _ in top_holdings[:5]]
    demo_args   = get_demo_args(regime, vix, top_symbols, spy_momentum)

    st.markdown("""
    <div style="color:#C9A84C;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">
      Round 1 — Initial Arguments
    </div>
    <div class="ac-muted" style="margin-bottom:12px;">Each agent argues independently before seeing the others' positions.</div>
    """, unsafe_allow_html=True)

    bull_ph = st.empty(); bear_ph = st.empty(); devil_ph = st.empty()
    regime_ph = st.empty(); quant_ph = st.empty()

    with st.spinner("Bull analyst making the case..."):
        time.sleep(0.35)
    bull_ph.markdown(agent_card("Bull Analyst", "bull", demo_args["bull"]), unsafe_allow_html=True)

    with st.spinner("Bear analyst making the case..."):
        time.sleep(0.35)
    bear_ph.markdown(agent_card("Bear Analyst", "bear", demo_args["bear"]), unsafe_allow_html=True)

    with st.spinner("Devil's advocate finding the shared blind spot..."):
        time.sleep(0.35)
    devil_ph.markdown(agent_card("Devil's Advocate", "devil", demo_args["devil"]), unsafe_allow_html=True)

    with st.spinner("Regime specialist applying the sizing playbook..."):
        time.sleep(0.25)
    regime_ph.markdown(agent_card("Regime Specialist", "regime", demo_args["regime"]), unsafe_allow_html=True)

    with st.spinner("Quant sanity check..."):
        time.sleep(0.15)
    total = sum(weights.values())
    max_sym, max_w = max(weights.items(), key=lambda x: x[1])
    n = len(weights)
    quant_text = (
        f"✓ {n} positions · weight sum = {total:.4f} · "
        f"max position: {max_sym} at {max_w:.1%} "
        f"{'(within cap)' if max_w <= rc['max_weight'] else '⚠ exceeds regime cap'}"
    )
    quant_ph.markdown(agent_card("Quant Sanity", "quant", quant_text), unsafe_allow_html=True)

    if show_round2:
        st.markdown("""
        <hr class="ac-sep">
        <div style="color:#555;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">
          Round 2 — Rebuttals
        </div>
        <div class="ac-muted" style="margin-bottom:12px;">Agents read each other's Round 1 arguments and respond.</div>
        """, unsafe_allow_html=True)

        bull_r2_ph = st.empty(); bear_r2_ph = st.empty(); devil_r2_ph = st.empty()

        with st.spinner("Agents reading each other's arguments..."):
            time.sleep(0.35)

        bull_r2  = f"The bear's duration concern is real but backward-looking. These positions weren't selected for rate sensitivity — they were selected for pricing power. That doesn't change with VIX at {vix:.0f}."
        bear_r2  = "The bull concedes nothing on correlation risk. In stress transitions I've modeled, even 'quality' names saw 0.78+ cross-correlation at the bottom. The diversification case is theoretical until it's tested."
        devil_r2 = "Both sides are still arguing about direction. The shared blind spot: liquidity. Three positions in this book average under $200M daily volume. In a real exit scenario, we are the market."

        bull_r2_ph.markdown(agent_card("Bull — Rebuttal",            "bull",  bull_r2,  round2=True), unsafe_allow_html=True)
        bear_r2_ph.markdown(agent_card("Bear — Rebuttal",            "bear",  bear_r2,  round2=True), unsafe_allow_html=True)
        devil_r2_ph.markdown(agent_card("Devil's Advocate — Rebuttal","devil", devil_r2, round2=True), unsafe_allow_html=True)

    st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 4 — Judge\'s Verdict</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ac-muted" style="margin-bottom:8px;">
      The judge synthesizes all arguments and issues a binding recommendation.
      <em>Reduce size</em> triggers an automated weight adjustment via Claude Haiku.
      <em>Halt and review</em> persists to a state file and blocks execution until manually cleared.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Judge synthesizing verdict..."):
        time.sleep(0.3)
    verdict = _demo_verdict(regime, vix)
    st.markdown(verdict_card(verdict), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ac-muted" style="margin-top:20px;text-align:right;">
      Ascent Capital · {date.today().strftime('%B %d, %Y')} · Demo mode · Neuroplasticity ON
    </div>""", unsafe_allow_html=True)
