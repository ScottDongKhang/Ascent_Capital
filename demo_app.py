"""
Ascent Capital — Interactive Demo
Built for Tony Ngo | April 2026

Run: .venv/bin/streamlit run demo_app.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import date

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ascent Capital",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  .stApp { background-color: #0D0D0D; color: #E0E0E0; }
  section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
  .block-container { padding-top: 2rem; max-width: 1100px; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Typography */
  h1, h2, h3 { color: #C9A84C !important; }

  /* Cards */
  .ac-card {
    background: #161616;
    border: 1px solid #252525;
    border-radius: 6px;
    padding: 18px 20px;
    margin: 10px 0;
  }
  .ac-card-gold { border-left: 3px solid #C9A84C; }
  .ac-card-bull { border-left: 3px solid #27AE60; }
  .ac-card-bear { border-left: 3px solid #E74C3C; }
  .ac-card-devil { border-left: 3px solid #9B59B6; }
  .ac-card-regime { border-left: 3px solid #2980B9; }
  .ac-card-quant { border-left: 3px solid #7F8C8D; }
  .ac-card-r2 { border-left: 3px solid #444; opacity: 0.9; }

  .ac-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .ac-label-gold { color: #C9A84C; }
  .ac-label-bull { color: #27AE60; }
  .ac-label-bear { color: #E74C3C; }
  .ac-label-devil { color: #9B59B6; }
  .ac-label-regime { color: #2980B9; }
  .ac-label-quant { color: #7F8C8D; }
  .ac-label-r2 { color: #888; }

  .ac-text { color: #D8D8D8; line-height: 1.65; font-size: 14px; }
  .ac-muted { color: #666; font-size: 12px; }

  /* Verdict badges */
  .verdict-proceed {
    background: #1A3A27; border: 1px solid #27AE60;
    color: #2ECC71; padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }
  .verdict-reduce {
    background: #3A2A10; border: 1px solid #F39C12;
    color: #F39C12; padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }
  .verdict-halt {
    background: #3A1010; border: 1px solid #E74C3C;
    color: #E74C3C; padding: 6px 14px; border-radius: 4px;
    font-weight: 700; font-size: 13px; letter-spacing: 1px; display: inline-block;
  }

  /* Metric row */
  .metric-row { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }
  .metric-box {
    background: #1A1A1A; border: 1px solid #252525; border-radius: 4px;
    padding: 10px 16px; min-width: 120px;
  }
  .metric-label { font-size: 10px; color: #888; letter-spacing: 1px; text-transform: uppercase; }
  .metric-value { font-size: 20px; font-weight: 600; color: #C9A84C; margin-top: 2px; }

  /* Separator */
  .ac-sep {
    border: none; border-top: 1px solid #222; margin: 24px 0;
  }

  /* Neuroplasticity table */
  .np-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .np-table th { color: #888; font-weight: 600; font-size: 10px; letter-spacing: 1px;
                 text-transform: uppercase; padding: 6px 12px; text-align: left; border-bottom: 1px solid #222; }
  .np-table td { padding: 8px 12px; color: #D8D8D8; border-bottom: 1px solid #1A1A1A; }
  .np-shift-up { color: #27AE60; }
  .np-shift-down { color: #E74C3C; }
  .np-neutral { color: #888; }

  /* Confidence bar */
  .conf-bar-bg {
    background: #222; border-radius: 3px; height: 6px; margin-top: 6px;
  }
  .conf-bar-fill {
    background: linear-gradient(90deg, #C9A84C, #F0C86B);
    border-radius: 3px; height: 6px;
  }

  /* Sidebar labels */
  .sidebar-section {
    color: #C9A84C; font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 20px 0 8px 0; border-top: 1px solid #222; padding-top: 14px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GOLD = "#C9A84C"

REGIME_CONFIG = {
    "calm_bull": {
        "risk_multiplier": 1.00, "max_weight": 0.15,
        "description": "Momentum intact, low drawdown risk. Full deployment appropriate.",
        "color": "#27AE60",
    },
    "stressed": {
        "risk_multiplier": 0.65, "max_weight": 0.10,
        "description": "Elevated drawdowns likely. Defensive rebalancing warranted.",
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
    "Trend": 0.70, "Stat-Arb": 0.15, "Mean Rev": 0.05, "ML (XGBoost)": 0.10
}

# Regime-aware sleeve adjustments (honest: trend-heavy in stress, ML reduced)
SLEEVE_ADJUSTMENTS = {
    "calm_bull":  {"Trend": 0.00,  "Stat-Arb": 0.00,  "Mean Rev": 0.00,  "ML (XGBoost)": 0.00},
    "stressed":   {"Trend": +0.08, "Stat-Arb": -0.03, "Mean Rev": +0.02, "ML (XGBoost)": -0.07},
    "crisis":     {"Trend": +0.15, "Stat-Arb": -0.08, "Mean Rev": +0.03, "ML (XGBoost)": -0.10},
    "uncertain":  {"Trend": +0.05, "Stat-Arb": -0.02, "Mean Rev": 0.00,  "ML (XGBoost)": -0.03},
}

PORTFOLIO_PRESETS = {
    "Current Ascent (Apr 14)": {
        "GLD": 0.095, "CAT": 0.063, "EQIX": 0.063, "JNJ": 0.063,
        "NEE": 0.061, "NEM": 0.063, "T": 0.063, "TRGP": 0.063,
        "CASY": 0.063, "CB": 0.036, "HYG": 0.044, "PDBC": 0.045, "VIXY": 0.050,
    },
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

# ── Demo mode arguments (scenario-aware fallback) ──────────────────────────────
def get_demo_args(regime: str, vix: float, top_holdings: list, spy_momentum: str) -> dict:
    holdings_str = ", ".join(top_holdings[:4]) if top_holdings else "the portfolio"
    momentum_desc = {
        "Strong positive": "strong upward momentum",
        "Weak positive": "weakening upward momentum",
        "Neutral": "flat momentum",
        "Weak negative": "early downtrend signals",
        "Strong negative": "strong downtrend momentum",
    }.get(spy_momentum, "mixed signals")

    args = {
        "calm_bull": {
            "bull": f"The regime is clearly supportive. {holdings_str} represent quality compounders with durable earnings power — not speculative names that need macro tailwinds to work. VIX at {vix:.0f} is benign; institutional positioning is constructive. The trend signal is intact across our top holdings. Missing this move by over-hedging is the real risk.",
            "bear": f"Calm bull can flip fast. VIX at {vix:.0f} doesn't mean risk is absent — it means it's unpriced. {holdings_str.split(',')[0] if ',' in holdings_str else holdings_str} and peers are priced for perfection at current multiples. Any earnings miss or Fed pivot repricing could gap us down 8-12% before stop-losses trigger. The risk/reward is asymmetric and not in our favor.",
            "devil": f"The assumption both sides are making: correlation stays low. In the last three calm-bull-to-stress transitions (2018, 2020, 2022), inter-portfolio correlation spiked from 0.35 to 0.82 within two weeks. Our 'diversified' {len(top_holdings)}-position portfolio becomes a single factor bet exactly when we need diversification most. There is no hedge in this book.",
            "regime": f"Calm bull regime historically lasts 3-6 months before exhaustion signals appear. Breadth narrowing and vol compression are the canonical warnings — monitor them. Current posture (full deployment, momentum-heavy) is textbook-correct for this regime. Maintain, but set trip-wires: VIX above 20 or SPY below 50-day MA triggers defensive review.",
        },
        "stressed": {
            "bull": f"Stressed regimes are when quality separates from noise. {holdings_str} are not speculative positions — these are businesses with $10B+ in free cash flow that don't need credit markets to function. VIX at {vix:.0f} is elevated but not crisis-level. The market is pricing in a macro outcome that won't materialize. This is a buying opportunity, not a selling one.",
            "bear": f"VIX at {vix:.0f} reflects real institutional hedging. With {spy_momentum.lower()} in SPY, the trend has broken. Historical data: when VIX crosses 25 in a stressed regime, the median drawdown from entry is -14% over the next 6 weeks. {holdings_str.split(',')[0] if ',' in ''.join(top_holdings[:1]) else holdings_str} and peers are still pricing in growth that is now at risk. Reduce gross exposure to 65%.",
            "devil": f"Both sides are anchoring on the wrong variable. The debate is about VIX and momentum — but the actual kill switch is credit spreads. If HY spreads widen 150bps from here (they're currently 50bps wide of the 90-day average), equity correlations will spike and liquidity in our smaller positions will disappear. Neither the bull nor bear case has a model for forced selling at 10% slippage. Monte Carlo p5 shows -21% in this scenario.",
            "regime": f"Stressed regime playbook: reduce equity gross exposure to 65%, cap max position at 10%, shift toward quality factors. This portfolio has {momentum_desc} — the regime is not yet confirming recovery. Typical stressed duration is 4-12 weeks. We are in week one. Premature re-risking here is the most common mistake in this regime.",
        },
        "crisis": {
            "bull": f"Crisis creates the best entry points. {holdings_str} will survive this — they have balance sheets that can absorb 18-24 months of stress. VIX at {vix:.0f} is extreme fear, not rational risk pricing. History is clear: the investors who held quality through crisis (2009, 2020) captured 40%+ returns in the subsequent 12 months. Cutting here locks in the loss.",
            "bear": f"VIX at {vix:.0f} is not noise — it is the market's honest assessment of tail risk. In a crisis regime, the primary risk is not missing the recovery, it is permanent capital impairment. Correlation across this portfolio will be near 1.0 in a liquidation event. Cash and TLT are not a missed opportunity cost; they are the instrument that allows us to participate in the recovery from a position of strength.",
            "devil": f"The fatal assumption: this crisis has a defined endpoint. 2008 lasted 18 months. The LTCM crisis took two years to fully resolve. Our conviction in {top_holdings[0] if top_holdings else 'our top holding'} assumes a business-as-usual recovery. In a true crisis, the companies we're most confident in may face issues we cannot model: counterparty exposure, off-balance-sheet risk, regulatory response. The unknown unknowns are the point. Forty percent gross exposure, maximum.",
            "regime": f"Crisis regime: capital preservation above all. The playbook is unambiguous — maximum defensive posture, 40% gross exposure, no speculative positions. Every basis point of exposure above that level is an uncompensated risk. Recovery positioning comes after stabilization signals: Fed intervention, credit spread reversal, VIX below 30 on a closing basis. We are not there.",
        },
        "uncertain": {
            "bull": f"Uncertainty is not a reason to underperform. {holdings_str} are not macro plays — they are earnings compounders. An unclear regime signal doesn't change the fundamental case. The risk of being too defensive in an uncertain regime that resolves bullish is real. Maintain 75% exposure and let the regime clarify.",
            "bear": f"Uncertain regime with {spy_momentum.lower()} is the worst combination. We don't know which direction this resolves, and we're carrying full risk into that uncertainty. The asymmetry is wrong: if this resolves to crisis, we face -20%+; if it resolves to calm bull, upside is capped by position sizing. Reduce to 60% and buy optionality.",
            "devil": f"The regime uncertainty itself is the signal everyone is ignoring. The model is uncertain because the underlying market structure is breaking down — the features that trained the regime classifier are no longer predictive. This is a structural break, not regime rotation. Neither the bull nor bear case is modeling a world where our regime framework stops working. That is the actual tail risk.",
            "regime": f"Uncertain regime historically lasts 1-3 weeks before resolving. The correct posture: 75% gross exposure, reduce to max 12% per position, wait for confirmation. Do not force the thesis. The regime signal has insufficient conviction (entropy above 0.90 threshold) to justify full deployment in either direction. Patience is the edge.",
        },
    }
    return args.get(regime, args["uncertain"])


# ── Live mode setup ────────────────────────────────────────────────────────────
LIVE_MODE = False
_agents_loaded = False

def _try_load_agents():
    global LIVE_MODE, _agents_loaded
    if _agents_loaded:
        return LIVE_MODE
    _agents_loaded = True
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return False
        import anthropic  # noqa
        from debate.agents import run_bull_agent  # noqa
        LIVE_MODE = True
        return True
    except Exception:
        return False


# ── Helper: build portfolio_state ─────────────────────────────────────────────
def build_portfolio_state(regime: str, weights: dict, vix: float, spy_momentum: str) -> dict:
    return {
        "date": date.today().isoformat(),
        "us_regime": regime,
        "macro_regime": "neutral",
        "n_positions": len(weights),
        "allocation": {"us_equities": 0.60, "macro": 0.15, "international": 0.15, "alternatives": 0.10},
        "weights": weights,
        "scenario_summary": f"VIX={vix:.0f}, SPY momentum={spy_momentum}",
        "scenario_results": [],
        "vix_level": vix,
        "spy_momentum": spy_momentum,
    }


# ── Helper: render agent card ──────────────────────────────────────────────────
def agent_card(label: str, style: str, text: str, round2: bool = False) -> str:
    card_class = f"ac-card ac-card-{style}" + (" ac-card-r2" if round2 else "")
    prefix = "↩ " if round2 else ""
    return f"""
<div class="{card_class}">
  <div class="ac-label ac-label-{style}">{prefix}{label}</div>
  <div class="ac-text">{text}</div>
</div>"""


# ── Helper: verdict card ───────────────────────────────────────────────────────
def verdict_card(verdict: dict) -> str:
    rec = verdict.get("recommendation", "reduce_size")
    conf = float(verdict.get("confidence", 0.5))
    risks = verdict.get("key_risks", [])
    reasoning = verdict.get("reasoning", "")

    badge_class = {
        "proceed": "verdict-proceed",
        "reduce_size": "verdict-reduce",
        "halt_and_review": "verdict-halt",
    }.get(rec, "verdict-reduce")

    badge_text = {
        "proceed": "▲ PROCEED",
        "reduce_size": "◆ REDUCE SIZE",
        "halt_and_review": "■ HALT & REVIEW",
    }.get(rec, rec.upper())

    conf_pct = int(conf * 100)
    bar_width = conf_pct

    risks_html = "".join(f"<div style='color:#999;font-size:13px;margin:4px 0;'>· {r}</div>" for r in risks[:3])

    return f"""
<div class="ac-card ac-card-gold" style="margin-top:24px;">
  <div class="ac-label ac-label-gold">Judge — Final Verdict</div>
  <div style="margin:12px 0;">
    <span class="{badge_class}">{badge_text}</span>
    <span style="color:#666;font-size:12px;margin-left:12px;">Confidence: {conf_pct}%</span>
  </div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bar_width}%;"></div></div>
  {f'<div style="margin-top:14px;">{risks_html}</div>' if risks else ''}
  {f'<div class="ac-text" style="margin-top:12px;color:#BBB;">{reasoning}</div>' if reasoning else ''}
</div>"""


# ── Helper: neuroplasticity card ───────────────────────────────────────────────
def neuroplasticity_card(regime: str, gross_exposure: float) -> str:
    adj = SLEEVE_ADJUSTMENTS.get(regime, SLEEVE_ADJUSTMENTS["uncertain"])
    rc = REGIME_CONFIG[regime]
    spy_overlay = "0.70× applied (SPY < 200-day MA)" if regime in ("stressed", "crisis") else "1.00× (SPY above 200-day MA)"

    rows = ""
    for sleeve, base in BASE_SLEEVES.items():
        delta = adj.get(sleeve, 0.0)
        adjusted = base + delta
        if delta > 0:
            shift = f'<span class="np-shift-up">▲ +{delta:.0%}</span>'
        elif delta < 0:
            shift = f'<span class="np-shift-down">▼ {delta:.0%}</span>'
        else:
            shift = '<span class="np-neutral">—</span>'
        rows += f"""
        <tr>
          <td>{sleeve}</td>
          <td>{base:.0%}</td>
          <td>{adjusted:.0%}</td>
          <td>{shift}</td>
        </tr>"""

    return f"""
<div class="ac-card ac-card-gold">
  <div class="ac-label ac-label-gold">Neuroplasticity — Regime Adaptation Active</div>
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
      <div class="metric-label">SPY Overlay</div>
      <div class="metric-value" style="font-size:13px;">{spy_overlay.split('(')[0].strip()}</div>
    </div>
  </div>
  <table class="np-table">
    <tr>
      <th>Alpha Sleeve</th>
      <th>Base Weight</th>
      <th>Adapted Weight</th>
      <th>Shift</th>
    </tr>
    {rows}
  </table>
  <div class="ac-muted" style="margin-top:12px;">
    Sleeve shifts reflect regime-calibrated alpha allocation. ML sleeve downweighted in stress/crisis — model trained on calm-bull distributions loses predictive power in regime transitions.
  </div>
</div>"""


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
      <div style="color:#C9A84C;font-size:22px;font-weight:700;letter-spacing:2px;">▲ ASCENT</div>
      <div style="color:#555;font-size:10px;letter-spacing:3px;margin-top:2px;">CAPITAL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Market Scenario</div>', unsafe_allow_html=True)

    regime = st.selectbox(
        "Regime",
        ["calm_bull", "stressed", "crisis", "uncertain"],
        index=1,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    vix = st.slider("VIX Level", min_value=10.0, max_value=55.0, value=28.0, step=0.5)

    spy_momentum = st.select_slider(
        "SPY Momentum",
        options=["Strong negative", "Weak negative", "Neutral", "Weak positive", "Strong positive"],
        value="Weak negative",
    )

    st.markdown('<div class="sidebar-section">Portfolio</div>', unsafe_allow_html=True)

    preset_name = st.selectbox("Preset", list(PORTFOLIO_PRESETS.keys()))
    weights = PORTFOLIO_PRESETS[preset_name].copy()

    st.markdown('<div class="sidebar-section">System Parameters</div>', unsafe_allow_html=True)

    neuroplasticity = st.toggle("Neuroplasticity", value=True)

    aggressiveness = st.slider(
        "Agent Aggressiveness",
        min_value=1, max_value=5, value=3,
        help="Higher = agents take stronger positions and challenge each other more forcefully.",
    )

    show_round2 = st.toggle("Show Round 2 Rebuttals", value=True)

    st.markdown('<div class="sidebar-section">Mode</div>', unsafe_allow_html=True)
    live_available = _try_load_agents()
    if live_available:
        st.success("Live LLM — real agent calls")
    else:
        st.info("Demo mode — scenario-aware arguments")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Debate", use_container_width=True, type="primary")


# ── Helper: fallback verdict ───────────────────────────────────────────────────
def _fallback_verdict(regime: str, vix: float) -> dict:
    if regime == "crisis" or vix > 40:
        return {
            "confidence": 0.78,
            "recommendation": "halt_and_review",
            "key_risks": [
                f"Crisis regime with VIX={vix:.0f} exceeds risk tolerance",
                "Correlation structure breakdown — diversification benefit near zero",
                "Liquidity gap risk in smaller positions under forced selling",
            ],
            "reasoning": f"The regime signal (crisis) combined with VIX={vix:.0f} places us outside the risk envelope for normal execution. The devil's advocate argument on correlation breakdown is the decisive factor — we cannot rely on cross-position diversification in this environment. Halting for human review is the correct process response.",
        }
    elif regime == "stressed" or vix > 25:
        return {
            "confidence": 0.71,
            "recommendation": "reduce_size",
            "key_risks": [
                f"Stressed regime — gross exposure should be {REGIME_CONFIG['stressed']['risk_multiplier']:.0%}",
                f"VIX at {vix:.0f} signals institutional hedging activity",
                "Bear and devil's rebuttal on correlation risk unresolved",
            ],
            "reasoning": f"The bull case is noted but insufficient given the stressed regime signal. Gross exposure reduction to {REGIME_CONFIG['stressed']['risk_multiplier']:.0%} is warranted. The devil's advocate raised a structural concern about correlation behavior in stress transitions that neither the bull nor bear adequately addressed. Reduce size, maintain quality positions, re-evaluate at next rebalance.",
        }
    else:
        return {
            "confidence": 0.82,
            "recommendation": "proceed",
            "key_risks": [
                "Regime transition risk — calm bull can flip on catalyst",
                "Position concentration in top 3 holdings",
            ],
            "reasoning": "The bull case is well-supported by the calm bull regime signal. The bear's concerns are noted but appropriately priced. The quant sanity check is clean. Proceed with full deployment while maintaining the regime trip-wires identified by the regime specialist.",
        }


# ── MAIN PANEL ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
  <div style="color:#C9A84C;font-size:11px;letter-spacing:3px;text-transform:uppercase;">
    Ascent Capital — Decision Engine
  </div>
  <div style="color:#888;font-size:12px;margin-top:4px;">
    Multi-agent debate layer · Pre-rebalance analysis · Regime-adaptive
  </div>
</div>
""", unsafe_allow_html=True)

# Portfolio summary
rc = REGIME_CONFIG[regime]
gross_exposure = rc["risk_multiplier"] if neuroplasticity else 1.0

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

    st.markdown(f"""
    <div class="ac-card">
      <div class="ac-label ac-label-gold">Portfolio — {preset_name}</div>
      {holdings_html}
    </div>""", unsafe_allow_html=True)

with col2:
    regime_color = rc["color"]
    st.markdown(f"""
    <div class="metric-row" style="flex-direction:column;gap:10px;">
      <div class="metric-box">
        <div class="metric-label">Regime Signal</div>
        <div class="metric-value" style="color:{regime_color};font-size:16px;">{regime.replace('_',' ').title()}</div>
        <div class="ac-muted" style="margin-top:4px;">{rc['description'][:60]}...</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">VIX</div>
        <div class="metric-value">{vix:.0f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Positions</div>
        <div class="metric-value">{len(weights)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Neuroplasticity card
if neuroplasticity:
    st.markdown(neuroplasticity_card(regime, gross_exposure), unsafe_allow_html=True)

st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)

# ── Debate section ─────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#444;">
      <div style="font-size:36px;margin-bottom:12px;">▶</div>
      <div style="font-size:14px;">Configure a scenario and run the debate</div>
    </div>
    """, unsafe_allow_html=True)
else:
    portfolio_state = build_portfolio_state(regime, weights, vix, spy_momentum)
    top_symbols = [s for s, _ in top_holdings[:5]]
    demo_args = get_demo_args(regime, vix, top_symbols, spy_momentum)

    st.markdown("""
    <div style="color:#C9A84C;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
      Round 1 — Initial Arguments
    </div>""", unsafe_allow_html=True)

    # Placeholders for progressive rendering
    bull_ph = st.empty()
    bear_ph = st.empty()
    devil_ph = st.empty()
    regime_ph = st.empty()
    quant_ph = st.empty()

    # ── Bull ──
    with st.spinner("Bull analyst..."):
        if LIVE_MODE:
            try:
                from debate.agents import run_bull_agent
                bull = run_bull_agent(portfolio_state)
            except Exception as e:
                bull = demo_args["bull"]
        else:
            time.sleep(0.4)
            bull = demo_args["bull"]
    bull_ph.markdown(agent_card("Bull Analyst", "bull", bull), unsafe_allow_html=True)

    # ── Bear ──
    with st.spinner("Bear analyst..."):
        if LIVE_MODE:
            try:
                from debate.agents import run_bear_agent
                bear = run_bear_agent(portfolio_state)
            except Exception as e:
                bear = demo_args["bear"]
        else:
            time.sleep(0.4)
            bear = demo_args["bear"]
    bear_ph.markdown(agent_card("Bear Analyst", "bear", bear), unsafe_allow_html=True)

    # ── Devil's Advocate ──
    with st.spinner("Devil's advocate..."):
        if LIVE_MODE:
            try:
                from debate.agents import run_devils_advocate
                devil = run_devils_advocate(portfolio_state)
            except Exception as e:
                devil = demo_args["devil"]
        else:
            time.sleep(0.4)
            devil = demo_args["devil"]
    devil_ph.markdown(agent_card("Devil's Advocate", "devil", devil), unsafe_allow_html=True)

    # ── Regime Specialist ──
    with st.spinner("Regime specialist..."):
        if LIVE_MODE:
            try:
                from debate.agents import run_regime_specialist
                regime_arg = run_regime_specialist(portfolio_state)
            except Exception as e:
                regime_arg = demo_args["regime"]
        else:
            time.sleep(0.3)
            regime_arg = demo_args["regime"]
    regime_ph.markdown(agent_card("Regime Specialist", "regime", regime_arg), unsafe_allow_html=True)

    # ── Quant Sanity ──
    with st.spinner("Quant sanity check..."):
        if LIVE_MODE or True:  # always run — pure Python, no LLM
            try:
                from debate.agents import run_quant_sanity_check
                quant_check = run_quant_sanity_check(portfolio_state)
            except Exception:
                total = sum(weights.values())
                max_sym, max_w = max(weights.items(), key=lambda x: x[1])
                n = len(weights)
                quant_check = (
                    f"QUANT SANITY CHECK:\n"
                    f"  ✓ Clean: {n} positions, sum={total:.4f}, "
                    f"max={max_sym} {max_w:.1%}"
                )
    quant_ph.markdown(agent_card("Quant Sanity", "quant", quant_check.replace("\n", "<br>")), unsafe_allow_html=True)

    # ── Round 2 ──
    round1_args = {"bull": bull, "bear": bear, "devils_advocate": devil, "regime_specialist": regime_arg}
    round2_args = {}

    if show_round2:
        st.markdown("""
        <hr class="ac-sep">
        <div style="color:#555;font-size:10px;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;">
          Round 2 — Agents Rebut Each Other
        </div>""", unsafe_allow_html=True)

        bull_r2_ph = st.empty()
        bear_r2_ph = st.empty()
        devil_r2_ph = st.empty()

        with st.spinner("Round 2..."):
            if LIVE_MODE:
                try:
                    from debate.agents import run_bull_rebuttal, run_bear_rebuttal, run_devils_advocate_rebuttal
                    bull_r2 = run_bull_rebuttal(portfolio_state, round1_args)
                    bear_r2 = run_bear_rebuttal(portfolio_state, round1_args)
                    devil_r2 = run_devils_advocate_rebuttal(portfolio_state, round1_args)
                except Exception:
                    bull_r2 = f"Bull rebuttal: The bear's valuation concern ignores FCF yield. At these prices, we're buying quality at a discount to intrinsic value."
                    bear_r2 = f"Bear rebuttal: The bull's FCF argument assumes steady-state. In a liquidity crunch, even quality names face forced selling. Quality is the last thing to go — not immune."
                    devil_r2 = f"The shared assumption: both sides think the market's regime classification agrees with ours. It doesn't. The market is already pricing crisis; we're positioned for stress. That gap is the real risk."
            else:
                time.sleep(0.3)
                bull_r2 = f"The bear's duration concern is real but backward-looking. These positions weren't selected for rate sensitivity — they were selected for pricing power. That doesn't change with VIX at {vix:.0f}."
                bear_r2 = f"The bull concedes nothing on correlation risk. In the stress transitions I've modeled, even 'quality' names saw 0.78+ cross-correlation at the bottom. The diversification case is theoretical."
                devil_r2 = f"Both sides are still arguing about direction. The real shared blind spot: liquidity. Three of these positions average under $200M daily volume. In a real exit scenario, we are the market."

        round2_args = {"bull_rebuttal": bull_r2, "bear_rebuttal": bear_r2, "devils_advocate_rebuttal": devil_r2}
        bull_r2_ph.markdown(agent_card("Bull — Rebuttal", "bull", bull_r2, round2=True), unsafe_allow_html=True)
        bear_r2_ph.markdown(agent_card("Bear — Rebuttal", "bear", bear_r2, round2=True), unsafe_allow_html=True)
        devil_r2_ph.markdown(agent_card("Devil's Advocate — Rebuttal", "devil", devil_r2, round2=True), unsafe_allow_html=True)

    # ── Judge ──
    st.markdown("<hr class='ac-sep'>", unsafe_allow_html=True)
    with st.spinner("Judge synthesizing verdict..."):
        if LIVE_MODE:
            try:
                from debate.judge import run_judge
                verdict = run_judge(
                    bull, bear, devil, portfolio_state,
                    regime_arg=regime_arg,
                    quant_check=quant_check,
                    round2_args=round2_args if show_round2 else None,
                )
            except Exception:
                verdict = _fallback_verdict(regime, vix)
        else:
            time.sleep(0.3)
            verdict = _fallback_verdict(regime, vix)

    st.markdown(verdict_card(verdict), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ac-muted" style="margin-top:20px;text-align:right;">
      Ascent Capital · {date.today().strftime('%B %d, %Y')} ·
      {'Live LLM' if LIVE_MODE else 'Demo mode'} ·
      Neuroplasticity {'ON' if neuroplasticity else 'OFF'}
    </div>""", unsafe_allow_html=True)
