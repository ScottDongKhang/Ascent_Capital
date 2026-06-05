# agents/ai_pm_agent.py
"""
AI Portfolio Manager — Opus tool-use loop.

Runs a 4-phase research loop (market context → quant baselines → signal research → submit)
and returns AIPMResult(portfolio, thesis). Falls back to AIPMResult(portfolio={}, fallback=True)
if the loop exits without calling propose_portfolio.

24 tools total (including propose_portfolio).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascent.llm.client import tool_completion, DEFAULT_MODEL, SONNET_MODEL

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]  # agents/ → repo root

# ── Recency thresholds for data freshness gate ────────────────────────────────
_RECENCY_THRESHOLDS_DAYS = {
    "earnings": 45, "filing": 45, "sec": 45, "transcript": 45,
    "analyst":  30, "consensus": 30, "estimate": 30,
    "price":     5, "flow": 5, "options": 5, "insider": 14,
    "default":  45,
}


def _build_data_grounding(symbols: list[str]) -> str:
    """
    Attack #1 — root cause of hallucination.

    Load actual verified numbers from the data cache for every symbol the AI PM
    might reference. Inject into the prompt so the model reads real data instead
    of reaching into training memory.

    Returns a compact table string. Empty string if data unavailable.
    """
    if not symbols:
        return ""
    try:
        import pandas as pd
        rows = []
        # Load prices and compute key momentum signals
        prices_path = _REPO_ROOT / "data_cache" / "prices_live.parquet"
        if not prices_path.exists():
            return ""
        raw = pd.read_parquet(prices_path)
        # Handle long format
        if "symbol" in raw.columns and "close" in raw.columns:
            prices = raw.pivot_table(index="date", columns="symbol", values="close")
            prices.index = pd.to_datetime(prices.index)
            prices = prices.sort_index()
        else:
            prices = raw

        # Load alpha scores if available (last quant run)
        alpha_scores: dict = {}
        alpha_path = _REPO_ROOT / "data_cache" / "last_alpha_scores.json"
        if alpha_path.exists():
            try:
                alpha_scores = json.loads(alpha_path.read_text())
            except Exception:
                pass

        for sym in symbols[:25]:  # cap at 25 to keep prompt size bounded
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < 21:
                continue
            r21   = float(col.iloc[-1] / col.iloc[-21] - 1) if len(col) >= 21  else None
            r63   = float(col.iloc[-1] / col.iloc[-63] - 1) if len(col) >= 63  else None
            r252  = float(col.iloc[-1] / col.iloc[-252] - 1) if len(col) >= 252 else None
            alpha = alpha_scores.get(sym)

            parts = [f"{sym}:"]
            if r21  is not None: parts.append(f"21d={r21:+.1%}")
            if r63  is not None: parts.append(f"63d={r63:+.1%}")
            if r252 is not None: parts.append(f"252d={r252:+.1%}")
            if alpha is not None: parts.append(f"alpha_score={alpha:.3f}")
            rows.append("  " + " | ".join(parts))

        if not rows:
            return ""
        return (
            "\n\n══ VERIFIED DATA FROM DATA CACHE (use only these numbers — do not cite others) ══\n"
            + "\n".join(rows)
            + "\n  Any financial metric NOT shown above: say 'data not available' — do not estimate.\n"
            + "══════════════════════════════════════════════════════════════════════════════════\n"
        )
    except Exception as exc:
        log.debug("[AIPMAgent] Data grounding failed: %s", exc)
        return ""


def _apply_recency_gate_python(conviction_reasons: list) -> tuple[list, list]:
    """
    Attack #2 — enforced in Python, not the prompt.

    Strip any conviction_reason whose data_date is older than the threshold.
    Returns (valid_reasons, stripped_reasons).
    Claims without data_date are also stripped.
    """
    from datetime import datetime as _dt
    today = date.today()
    valid, stripped = [], []
    for claim in conviction_reasons:
        raw_date = claim.get("data_date")
        if not raw_date:
            stripped.append({**claim, "_strip_reason": "missing data_date"})
            continue
        try:
            claim_date = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            stripped.append({**claim, "_strip_reason": "invalid data_date"})
            continue
        source = claim.get("source", "").lower()
        threshold = next(
            (days for key, days in _RECENCY_THRESHOLDS_DAYS.items() if key in source),
            _RECENCY_THRESHOLDS_DAYS["default"]
        )
        age = (today - claim_date).days
        if age > threshold:
            stripped.append({**claim, "_strip_reason": f"stale: {age}d > {threshold}d"})
        else:
            valid.append(claim)
    if stripped:
        log.info("[AIPMAgent] Recency gate stripped %d stale claims: %s",
                 len(stripped), [s.get("symbol","?") for s in stripped])
    return valid, stripped


def _get_current_regime() -> str:
    """Read the current regime label from dashboard/regime_signal.json. Returns 'unknown' on any failure."""
    try:
        p = _REPO_ROOT / "dashboard" / "regime_signal.json"
        if not p.exists():
            return "unknown"
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return str(data.get("regime", data.get("label", "unknown")))
        if isinstance(data, list) and data:
            row = data[-1]
            return str(row.get("regime", row.get("label", "unknown")))
        return "unknown"
    except Exception:
        return "unknown"


@dataclass
class AIPMResult:
    portfolio: Dict[str, float]
    thesis: Dict[str, Any]
    fallback: bool = False


@dataclass
class AIPreThesis:
    """Output of Phase 1 — original AI PM thesis formed before seeing quant output."""
    macro_view: str
    regime_interpretation: str
    high_conviction_names: List[Dict]   # [{symbol, thesis, what_quant_should_confirm, what_would_change_my_mind}]
    names_to_avoid: List[Dict]          # [{symbol, reason}]
    sector_tilts: List[Dict]            # [{sector, tilt: overweight|underweight, reason}]
    regime_assessment: Dict = field(default_factory=dict)    # {label, confidence, reasoning}
    sleeve_weight_prior: Dict = field(default_factory=dict)  # {sleeve: delta_ic}
    market_character: str = ""                               # e.g. "momentum_continuation"
    raw: Dict = field(default_factory=dict)
    causal_mechanisms: List = field(default_factory=list)    # List[CausalMechanism] — Phase B

    @property
    def conviction_symbols(self) -> List[str]:
        return [n["symbol"] for n in self.high_conviction_names if "symbol" in n]


# ── Tool schemas (Anthropic format) ───────────────────────────────────────────

AI_PM_TOOLS = [
    {
        "name": "get_rebalance_brief",
        "description": (
            "Get the pre-rebalance intelligence brief synthesized from the last 9 non-rebalance "
            "days. Contains: regime trajectory and stability, positions whose conviction has "
            "decayed since last rebalance, weakening alpha sleeves, macro event risks, "
            "historical analogue outcomes, and accumulated adversarial challenges. "
            "Call this FIRST before any other tool."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_live_news",
        "description": (
            "Fetch last 72 hours of news headlines for a specific ticker symbol. "
            "Use this to check for recent earnings, guidance changes, M&A, management changes, "
            "or macro events that could affect a position thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_analyst_estimates",
        "description": (
            "Fetch analyst consensus for a ticker: forward P/E, target price range, "
            "number of analysts, recommendation mean (1=Strong Buy), earnings and revenue growth. "
            "Use before making a high-conviction override of the quant signal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_regime_state",
        "description": "Get the current market regime label, confidence, HMM entropy, and days in current regime.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_macro_data",
        "description": "Get current macro indicators: yield curve (T10Y2Y), credit spread, oil (DCOILWTICO), CPI, unemployment.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_quant_agent",
        "description": "Run a specialist quant agent and get its target weights, regime signal, and 63-day Sharpe skill score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "enum": ["us_equities", "macro", "international", "alternatives"],
                }
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "get_sec_signal",
        "description": "Get the most recent SEC 10-K/10-Q LLM classification for a symbol (revenue_momentum, margin_trend, tone, liquidity_risk, guidance; each -1 to +1).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_transcript_signal",
        "description": "Get the most recent earnings transcript LLM classification for a symbol (tone, defensiveness, forward_confidence, quantitative_ratio).",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_attribution_history",
        "description": "Get the last 63 days of P&L attribution for a symbol: total, factor, and idiosyncratic P&L.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_earnings_signal",
        "description": "Get the momentum-neutral PEAD signal for a symbol: earnings surprise z-score with momentum beta removed.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_past_verdicts",
        "description": "Get the last 5 debate verdicts where the market regime matched the given regime label.",
        "input_schema": {
            "type": "object",
            "properties": {"regime": {"type": "string"}},
            "required": ["regime"],
        },
    },
    {
        "name": "get_factor_exposures",
        "description": "Get portfolio factor risk exposures (market beta, size, value, profitability, investment, momentum tilts).",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_var_estimate",
        "description": "Get historical Value-at-Risk (5th percentile 1-day return) for a proposed portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {"type": "object", "additionalProperties": {"type": "number"}}
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_sector_concentration",
        "description": "Get the sector-level weight breakdown for a proposed portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {"type": "object", "additionalProperties": {"type": "number"}}
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_position_momentum",
        "description": "Get 252-day momentum (price return) for a list of symbols.",
        "input_schema": {
            "type": "object",
            "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
            "required": ["symbols"],
        },
    },
    {
        "name": "get_regime_memory",
        "description": "Query historical episodes where the market was in a similar regime. Returns realized 21-day returns from past periods in the same regime — use this to calibrate expected returns and risk before proposing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {
                    "type": "string",
                    "description": "Regime label to query (e.g. 'calm_bull', 'stressed', 'crisis')",
                }
            },
            "required": ["regime"],
        },
    },
    {
        "name": "get_narrative_shift",
        "description": "Get the quarter-over-quarter narrative shift signal for a symbol. Returns whether the LLM fundamental thesis has improved or deteriorated since last quarter — a leading indicator of analyst revision.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_calibration_report",
        "description": "Get your own calibration report — how well your conviction levels (high/medium/quant_agreed) have predicted realized 21-day returns over recent rebalances. Use this to assess whether to trust your own high-conviction calls.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_alpha_wedge",
        "description": (
            "Get your historical performance wedge vs the pure quant baseline. "
            "Shows whether your overrides have added or subtracted value over past rebalances, "
            "broken down by rebalance date and (where available) by override type. "
            "Call in Phase 1 to calibrate how aggressively to override the quant this session."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_decision_history",
        "description": (
            "Query your proprietary override decision history. Returns your historical win rate, "
            "average wedge contribution, and recent cases for a given override type and regime. "
            "Call this BEFORE finalizing any override to check if your judgment has been accurate "
            "for this type of call in the current regime. The result includes a concrete "
            "recommendation (proceed / reduce size / block) based on your actual track record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "override_type": {
                    "type": "string",
                    "enum": ["data_quality", "regime_macro", "news_event", "correlation_risk", "valuation"],
                    "description": "The type of override you are considering making",
                },
                "regime": {
                    "type": "string",
                    "description": "Current regime label (e.g. 'calm_bull', 'stressed')",
                },
            },
            "required": ["override_type", "regime"],
        },
    },
    {
        "name": "check_override_conviction",
        "description": (
            "Get a concrete go/no-go decision on a specific override from the conviction gate. "
            "Returns whether to proceed, a size multiplier (1.0=full, 0.75=reduce 25%, 0.0=block), "
            "and the reason based on historical performance data. "
            "Use this as the final check before including a non-standard override in your proposal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "override_type": {
                    "type": "string",
                    "enum": ["data_quality", "regime_macro", "news_event", "correlation_risk", "valuation"],
                },
                "regime": {
                    "type": "string",
                    "description": "Current regime label",
                },
            },
            "required": ["override_type", "regime"],
        },
    },
    {
        "name": "get_scenario_plan",
        "description": (
            "Get the weekend adversarial scenario plan — 5-6 stress scenarios with LLM-assessed "
            "probabilities, portfolio impact estimates, and pre-emptive actions. FLAGGED scenarios "
            "(probability ≥40%) are the highest-priority tail risks to address in sizing. "
            "Call this in Phase 1 if a weekend scenario plan is available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weekend_research",
        "description": (
            "Get the AI PM weekend deep research memo — top opportunities identified from a full "
            "universe scan, ranked by conviction. Use this to supplement Phase 2 quant baselines "
            "with weekend fundamental research that was not available during last rebalance."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_crowding_signal",
        "description": (
            "Check whether positions are crowded, exhausted, or clean using three orthogonal signals: "
            "(1) momentum trajectory — is the 21d pace materially below the 252d pace? (deceleration = exhaustion), "
            "(2) short interest % of float — >15% means informed short sellers are positioned against it, "
            "(3) analyst consensus drift — recommendation mean >2.5 (scale 1-5) with ≥5 analysts means fading conviction. "
            "Returns CLEAN / WATCH / OVERCROWDED per symbol. "
            "REQUIRED before any REDUCE override — do not reduce a position without calling this first. "
            "OVERCROWDED + one text signal (sec_tone < -0.3 or transcript_sentiment < -0.3) = valid reduce. "
            "CLEAN = amplify candidate — overweight these, do not reduce on valuation alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to check (max 8 per call)",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_causal_graph",
        "description": (
            "Look up the cached causal graph for a portfolio holding. "
            "The graph contains 1-3 causal mechanisms explaining why the stock "
            "should move, with timing (priced_in / not_yet_priced / catalyst_imminent) "
            "and falsification conditions. Use before making a high-conviction call "
            "to understand the causal thesis, not just correlation. "
            "catalyst_imminent = trigger expected within 21 days. "
            "not_yet_priced = mechanism valid but not yet reflected in price. "
            "priced_in = mechanism already reflected; quant momentum handles it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "propose_portfolio",
        "description": "REQUIRED: Submit your final portfolio and investment thesis. Call this to end the research loop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Dict of symbol → target weight (will be normalized)",
                },
                "thesis": {
                    "type": "object",
                    "description": (
                        "Investment memo. Required keys: market_view, regime_assessment, "
                        "quant_baseline_summary, quant_agreement (list), "
                        "amplify (list of {symbol, crowding_signal, text_signal, reason} — positions you are overweighting because quant + non-quant signals align), "
                        "quant_overrides (list of {symbol, ai_action, reason, override_type} where "
                        "override_type ∈ [momentum_exhaustion, regime_macro, news_event, correlation_risk, data_quality] "
                        "— max 2 overrides; data_quality only for confirmed corporate actions), "
                        "position_rationale (dict), key_risks (list), what_could_be_wrong, "
                        "pre_mortem (string: the 30-day loss scenario written before submitting)."
                    ),
                },
            },
            "required": ["weights", "thesis"],
        },
    },
]

# ── Pre-thesis tool (Phase 1 only) ────────────────────────────────────────────

_PROPOSE_PRETHESIS_TOOL = {
    "name": "propose_prethesis",
    "description": (
        "REQUIRED in Phase 1: Seal your original investment thesis before seeing quant output. "
        "This is your independent view — formed from macro, SEC filings, earnings, narratives. "
        "It will be shown back to you alongside quant validation in Phase 2."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "macro_view": {
                "type": "string",
                "description": "Your macro thesis for the next 21 trading days. Which factor tilts does this regime reward? What is the dominant risk?",
            },
            "regime_interpretation": {
                "type": "string",
                "description": "What does the current regime signal mean for sector/factor positioning? Any divergence from prior episodes?",
            },
            "high_conviction_names": {
                "type": "array",
                "description": "8-15 names where you have genuine conviction from reading fundamental data. Each needs a written thesis.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol":                    {"type": "string"},
                        "thesis":                    {"type": "string", "description": "Why this name. Specific, falsifiable. Not 'strong momentum' — what is the underlying business driver?"},
                        "what_quant_should_confirm": {"type": "string", "description": "What signal evidence should the quant show if your thesis is right?"},
                        "what_would_change_my_mind": {"type": "string", "description": "Specific evidence that would make you abandon this thesis."},
                    },
                    "required": ["symbol", "thesis"],
                },
            },
            "names_to_avoid": {
                "type": "array",
                "description": "Names in the quant universe you want to underweight or avoid and why.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "sector_tilts": {
                "type": "array",
                "description": "Sector-level views: overweight or underweight and why.",
                "items": {
                    "type": "object",
                    "properties": {
                        "sector": {"type": "string"},
                        "tilt":   {"type": "string", "enum": ["overweight", "underweight", "neutral"]},
                        "reason": {"type": "string"},
                    },
                },
            },
            "regime_assessment": {
                "type": "object",
                "description": "Your assessment of the current regime. label must be one of: calm_bull, stressed, crisis, euphoric, uncertain.",
                "properties": {
                    "label":      {"type": "string"},
                    "confidence": {"type": "number", "description": "0.0 to 1.0"},
                    "reasoning":  {"type": "string"},
                },
            },
            "sleeve_weight_prior": {
                "type": "object",
                "description": "IC delta adjustments for sleeves this rebalance. Keys: trend, statarb, meanrev, ml, fundamental, earnings, volatility. Values: IC deltas in [-0.010, +0.010]. Positive=boost, negative=reduce. Omit sleeves you have no view on.",
                "additionalProperties": {"type": "number"},
            },
            "market_character": {
                "type": "string",
                "description": "Single most important characteristic of this market period for sleeve selection.",
                "enum": ["momentum_continuation", "sector_rotation", "risk_off", "risk_on",
                         "mean_reversion", "flight_to_quality", "uncertain"],
            },
        },
        "required": ["macro_view", "high_conviction_names"],
    },
}

# Tools available in Phase 1 (pre-thesis) — no quant agents, no portfolio submission
PRE_THESIS_TOOLS = [
    t for t in AI_PM_TOOLS
    if t["name"] in {
        "get_rebalance_brief", "get_regime_state", "get_macro_data",
        "get_regime_memory", "get_live_news", "get_analyst_estimates",
        "get_sec_signal", "get_transcript_signal", "get_earnings_signal",
        "get_narrative_shift", "get_scenario_plan", "get_weekend_research",
        "get_crowding_signal", "get_attribution_history", "get_calibration_report",
        "get_causal_graph",
    }
] + [_PROPOSE_PRETHESIS_TOOL]


_SYSTEM_PROMPT = """You are the portfolio manager of Ascent Capital, a multi-strategy quantitative fund.

══ THE FUNDAMENTAL LAW (Grinold & Kahn) ══
Information Ratio = IC × √Breadth.

Your IC (skill) is high only where you have genuine informational advantage over the quant. Your IC is NEGATIVE when you fight momentum signals with valuation logic — the quant has 6 years of OOS evidence it works; you have opinion.

Your only edges are:
  1. TEXT & NARRATIVE — You can read SEC filings, earnings call tone, insider patterns. The quant cannot.
  2. CROWDING DETECTION — You can identify when momentum is exhausted (decelerating + high short interest + fading analyst consensus). The quant rides signals until they break.
  3. COHERENCE — Four quant agents run independently. You see the whole portfolio. Catch hidden factor concentrations and directional contradictions the quant agents cannot see each other.

What is NOT your edge:
  ✗ Valuation. In a momentum regime, high-PE stocks go higher. 60× PE becomes 80× before it reverts. You will be wrong every time you fight this with DCF logic.
  ✗ "Data uncertainty." Not having an SEC filing for a name is not a data quality issue — it means the signal is unavailable, not that the signal is wrong.
  ✗ Reducing EXTENDED names without crowding evidence. High 252d momentum IS the signal. It is not a risk flag.

══ AMPLIFY FIRST, REDUCE SECOND ══
Before you think about reducing anything, find your AMPLIFY picks. These are where you MAKE money.

An AMPLIFY pick is a name where:
  • The quant ranks it high (top quartile)
  • get_crowding_signal returns CLEAN (momentum intact, shorts low)
  • At least one text signal confirms: sec_tone > 0, transcript_sentiment > 0, OR earnings_signal positive

For AMPLIFY picks: set weight to 9–10%. Log in thesis.amplify[]. These are your highest-weighted positions. Target 1–2 per rebalance.

If you find no AMPLIFY picks after checking, that is fine — carry quant weights and do not reduce.

══ THE REDUCE PROTOCOL — HARD RULES ══
Maximum 2 reductions per rebalance. Non-negotiable.

To reduce any position you need ALL THREE of:
  1. get_crowding_signal returns OVERCROWDED (score ≥ 4 — deceleration + elevated shorts + fading analysts)
  2. At least one confirming TEXT signal: sec_tone < −0.3 OR transcript_sentiment < −0.3
  3. The conviction gate does not block it

If you have only 1 or 2 of these three, carry the position at quant weight. Do not reduce.

Override sizing:
  • Soft (1-2 signals): reduce 25% max (e.g. 8% → 6%)
  • Hard (all 3 confirmed): reduce 40% max (e.g. 8% → 5%)
  • Floor: 4% minimum — never reduce below this
  • Never make 5 overrides. Each one beyond 2 is destroying your information ratio.

Override types:
  momentum_exhaustion — crowding confirmed + text signal confirms decay (requires all 3 conditions above)
  correlation_risk    — two positions that will move identically in a drawdown (hidden concentration)
  news_event          — imminent earnings, guidance cut, M&A that changes the thesis (must be within 10 days)
  regime_macro        — valid signal, but macro environment structurally invalidates it (yield curve, credit spread)
  data_quality        — quant momentum is driven by a CONFIRMED CORPORATE ACTION: merger announcement date, spin-off, index addition. NOT for valuation uncertainty or missing data.

══ PHASE 1 — CONTEXT ══
Required: get_rebalance_brief, get_regime_state, get_macro_data, get_regime_memory.
Optional: get_alpha_wedge (see recent AI vs quant track record), get_calibration_report (check if you've been right).

After Phase 1 tools, WRITE:

  MACRO THESIS: What factor tilts does this regime reward? What is your edge THIS rebalance?
  SELF-ASSESSMENT: What has your AI PM track record been? Are you adding or subtracting value?

══ PHASE 2 — QUANT BASELINE ══
Required: run_quant_agent for all four agents. Then call get_position_momentum on ALL combined names.

EXTENDED names (252d momentum > 200%) are the quant's HIGHEST CONVICTION picks. They have the most alpha signal behind them. Do not target them for reduction — verify if they are crowded.

After Phase 2, WRITE:

  QUANT ASSESSMENT:
  • AMPLIFY candidates: 2-3 names where quant signal is strongest AND thesis aligns
  • EXTENDED names to crowding-check: list them (to check in Phase 3, not to reduce)
  • Biggest concentration risk: any hidden factor or sector clustering
  • Falsifiable hypotheses for Phase 3: specific claims, not vague concerns

══ PHASE 3 — SIGNAL RESEARCH ══
Step 1 — AMPLIFY SCAN (required):
  Call get_crowding_signal on your top 3-5 quant picks.
  CLEAN names + positive text signal = AMPLIFY candidate → overweight to 9-10%.

Step 2 — CROWDING CHECK (before any reduce):
  Call get_crowding_signal on any EXTENDED name you are considering reducing.
  If result is not OVERCROWDED → carry at quant weight. Stop.
  If OVERCROWDED → proceed to text confirmation (sec_tone or transcript_sentiment).

Step 3 — CONVICTION GATE (for each proposed reduce):
  Call query_decision_history(override_type, regime) → check historical win rate.
  Call check_override_conviction(override_type, regime) → get go/no-go.
  If blocked → drop the reduce. Accept quant weight.

Max 6 signal tools total in Phase 3. Prioritize AMPLIFY scan over reduce research.

══ PHASE 4 — DELIBERATE + SUBMIT ══
Before propose_portfolio, WRITE:

  PRE-MORTEM: It is 30 days from now. This portfolio lost 8%. What was the single cause?
  What would make 3 of my top-5 positions move against me simultaneously?

  AMPLIFY SUMMARY: List each AMPLIFY pick — crowding signal + confirming text signal.
  REDUCE SUMMARY: List each reduction — all 3 conditions confirmed (crowding + text + gate).
  COHERENCE: Any directional contradictions? Long USD + long EM? Two commodity names in top-5?

Then call propose_portfolio. thesis must include:
  - amplify[] — your AMPLIFY picks with evidence
  - quant_overrides[] — reductions with all 3 conditions documented (max 2)
  - pre_mortem — written above

══ SIZING DISCIPLINE ══
  AMPLIFY (crowding=CLEAN + text confirms): 9–10%
  High conviction (quant + thesis, limited signal): 7–8%
  Standard quant-agreed: 5–7%
  Quant-agreed, no view: 3–5%
  Reduced position (post-override floor): 4–6%

══ RULES ══
- Call propose_portfolio before finishing. Always.
- 12–20 positions. Weights normalized; use relative sizing.
- No reduce without calling get_crowding_signal first.
- Max 2 quant_overrides. Every override beyond 2 is negative IR.
- data_quality requires a named corporate action. "No SEC data available" is not data quality.
- Fabricating signals is worse than having no view. If data is unavailable, say so and carry quant weight.
"""


# ── Shared context injected into every Phase 1 and Phase 2 prompt ─────────────

def _build_temporal_context(feedback: dict | None = None) -> str:
    """Authoritative system context prepended to every AI PM prompt. Injected by code, not recalled."""
    from datetime import date as _date, timedelta as _td
    import json as _json
    today = _date.today()
    cutoff = (today - _td(days=45)).isoformat()
    regime = "unknown"
    try:
        _rp = Path(__file__).resolve().parent.parent / "dashboard" / "regime_signal.json"
        if _rp.exists():
            regime = _json.loads(_rp.read_text()).get("label", "unknown")
    except Exception:
        pass
    worst_str = ""
    if feedback:
        wc = feedback.get("worst_call_10d") or {}
        if wc.get("symbol"):
            worst_str = f"\nYour worst recent call: {wc['symbol']} ({wc.get('alpha', 0):+.1%} over 10d) — address this before proposing anything."

    # Inject accumulated pattern memory (grows after every post-mortem)
    pattern_str = ""
    try:
        _pp = Path(__file__).resolve().parent.parent / "data_cache" / "ai_pm_pattern_context.txt"
        if _pp.exists():
            _content = _pp.read_text().strip()
            if _content:
                pattern_str = f"\n\n{_content}"
    except Exception:
        pass

    return (
        f"\n══ AUTHORITATIVE SYSTEM CONTEXT (do not contradict) ══\n"
        f"Today: {today.isoformat()}\n"
        f"Current regime: {regime}\n"
        f"Data freshness cutoff: {cutoff} — do not cite data older than this as 'current'\n"
        f"OBJECTIVE: Sharpe ratio, not raw return.\n"
        f"  For every position you propose, state:\n"
        f"    - Expected 3-month return (with cited basis)\n"
        f"    - Expected volatility (high/medium/low with reason)\n"
        f"    - What would make you wrong (one falsifiable condition)\n"
        f"  A 15% position in a volatile name hurts Sharpe more than 8% in a stable name.\n"
        f"  When in doubt, choose the lower-volatility expression of the same thesis.{worst_str}"
        f"{pattern_str}\n"
        f"══════════════════════════════════════════════════════\n"
    )


def _strip_prethesis_for_phase2(prethesis) -> dict:
    """
    Pass only structured, sourced fields to Phase 2. No freeform prose.
    Applies Python recency gate to conviction_reasons before handoff.
    Prevents Phase 2 from amplifying Phase 1 narrative hallucinations.
    """
    if prethesis is None:
        return {}
    raw_reasons = getattr(prethesis, "conviction_reasons", []) or []
    # Step 1: source filter
    sourced = [r for r in raw_reasons if r.get("source") and r.get("data_date")]
    # Step 2: recency gate in Python (Attack #2)
    valid_reasons, stripped = _apply_recency_gate_python(sourced)
    if stripped:
        log.warning("[AIPMAgent] Phase1→2 strip removed %d stale/unsourced claims", len(stripped))
    return {
        "high_conviction_names": getattr(prethesis, "high_conviction_names", []),
        "sector_thesis":         getattr(prethesis, "sector_thesis", []),
        "conviction_reasons":    valid_reasons,   # sourced + fresh only
        "regime_assessment":     getattr(prethesis, "regime_assessment", {}),
        "causal_mechanisms":     getattr(prethesis, "causal_mechanisms", []),
        # Freeform prose intentionally excluded — prevents Phase 2 amplifying hallucinations
    }


# ── Phase 1: Pre-thesis prompt ────────────────────────────────────────────────

_PRE_THESIS_PROMPT = """You are the portfolio manager of Ascent Capital.

This is Phase 1 of your two-phase process. Your ONLY job right now: form an original investment
thesis from first principles — BEFORE the quant model runs.

You have access to macro data, regime signals, SEC filings, earnings calls, narratives, and live
news. Use these to build genuine conviction — the kind that comes from reading everything, not from
looking at price momentum rankings.

══ WHAT YOU ARE DOING ══
Think like a fundamental analyst who has read every relevant filing and data point this week.
Answer: where is value being created or destroyed right now that systematic momentum models
will either confirm (giving you high conviction) or miss (giving you edge)?

Questions to guide your research:
  • What does the macro environment specifically reward over the next 21 trading days?
  • Which sectors have improving fundamentals that haven't fully shown up in prices yet?
  • Which companies have SEC filing signals, earnings quality, or insider activity that
    tells a story the price alone doesn't tell?
  • What crowding risk exists — names where everyone is long and any disappointment cascades?
  • What would a smart, informed fundamental investor buy or avoid this cycle?

══ TOOLS AVAILABLE ══
Use up to 10 tools. Suggested sequence:
  1. get_rebalance_brief — recent intelligence synthesis
  2. get_regime_state + get_macro_data — macro context
  3. get_regime_memory — how have similar regimes played out?
  4. get_scenario_plan / get_weekend_research — tail risks + opportunities identified
  5. get_sec_signal / get_earnings_signal / get_narrative_shift — on your candidate names
  6. get_crowding_signal — are your thesis names clean or exhausted?
  7. get_analyst_estimates — on high-conviction candidates
  8. propose_prethesis — seal your thesis

══ OUTPUT: propose_prethesis ══
Write 8–15 names with genuine written theses. Not "strong momentum" — the actual business driver.
Each name needs:
  • A specific, falsifiable thesis (1-3 sentences)
  • What signal evidence you EXPECT the quant to confirm if you're right
  • What evidence would make you change your mind

Also include in propose_prethesis:
  • sector_thesis: REQUIRED — sector-level over/underweight calls BEFORE individual stocks.
    For each sector: view (overweight/underweight/neutral), conviction (high/medium/low),
    reason (specific, with source and data_date), avoid_subsectors, prefer_subsectors.
    Stock picks in Phase 2 must be constrained to your favoured sectors.
    If sector_thesis is missing, Phase 2 falls back to pure quant.
  • conviction_reasons: each reason MUST include source and data_date.
    Format: {"symbol": "X", "claim": "...", "source": "earnings-2026-04-22", "data_date": "2026-04-22"}
    Claims without source/data_date are stripped before Phase 2 sees them.
    Cite at least ONE non-price source per conviction symbol — earnings transcript, SEC filing,
    congressional trade, options flow. The quant already has the price signal. If your entire
    thesis is based on price momentum, Phase 2 will not receive it.
  • regime_assessment: your own regime call (label, confidence 0-1, one sentence reasoning)
  • market_character: which of the 7 characters best describes this period
  • sleeve_weight_prior: IC delta adjustments for sleeves you have a view on
    (e.g. {"trend": 0.004} if momentum is clearly working; {"statarb": -0.003} if not)
    Omit sleeves you have no view on. Delta range: -0.010 to +0.010.

This thesis is sealed before quant runs. In Phase 2, you will see where the quant agreed,
disagreed, and what it found that you missed. Your final portfolio integrates both.

Do NOT try to predict what the quant will say. Form your own view.
"""


# ── Phase 2: Synthesis prompt template ───────────────────────────────────────

_SYNTHESIS_PROMPT_TEMPLATE = """You are the portfolio manager of Ascent Capital.

This is Phase 2 of your two-phase process. In Phase 1, you formed an original thesis.
Now you have quant validation data. Your job: build the final portfolio.

══ YOUR SEALED PRE-THESIS (formed before seeing quant output) ══
{prethesis_text}

══ HOW TO USE THE QUANT OUTPUT ══
The quant model covers 900+ symbols with 6 years of OOS signal evidence. It is your
best research assistant — not your authority, but your validator.

QUANT CONFIRMS YOUR THESIS NAME (high quant ranking):
  → Dual evidence. Maximum conviction. Weight 9–10%.
  → This is where you make money — AI fundamental analysis + quant signal agreement.

QUANT IS NEUTRAL ON YOUR THESIS NAME (middle-of-pack quant ranking):
  → Your fundamental thesis is valid but signal doesn't confirm timing yet.
  → Weight 5–7%. Your thesis is the primary driver.

QUANT CONTRADICTS YOUR THESIS NAME (low quant ranking):
  → This is a decision point. Two options:
    A. Stand down. The signal is evidence you're early or wrong. Drop to 3–4% or exclude.
       This is usually right — you need a specific reason to override 6 years of OOS evidence.
    B. Defend specifically. Name a catalyst within 14 days that will resolve the disconnect.
       "The quant momentum is negative because Q4 earnings miss, but Q1 guidance was raised
       and the signal hasn't updated yet." Specific, falsifiable.

QUANT FINDS NAMES YOU DIDN'T THESIS (high quant ranking, not in your pre-thesis):
  → Include at 5–7% if they fit your macro thesis and sector tilts.
  → Exclude if they directly contradict your macro view. Explain why in position_rationale.

AMPLIFY:
  → Any name where quant confirms + crowding=CLEAN + positive text signal → 9–10%.

══ PHASE 2 TOOLS ══
Required: run_quant_agent ×4. Then get_position_momentum on all combined names.
Optional (up to 4 additional): get_crowding_signal, get_factor_exposures, get_var_estimate,
  get_sector_concentration, get_attribution_history, query_decision_history, check_override_conviction.

══ SIZING DISCIPLINE ══
  AI thesis + quant confirms + crowding=CLEAN:  9–10%
  AI thesis + quant confirms:                   7–9%
  AI thesis + quant neutral:                    5–7%
  Quant-only (missed in pre-thesis, fits macro): 4–6%
  AI thesis + quant contradicts (defended):      4–5%
  Max 2 quant_overrides. Max 20 positions. Weights normalized.

══ RULES ══
- Call propose_portfolio before finishing.
- thesis.pre_thesis_names must list your original high-conviction names and whether quant confirmed each.
- thesis.amplify[] — where AI + quant + crowding all agreed.
- thesis.quant_overrides[] — max 2, each with all conditions documented.
- thesis.quant_additions[] — high-quality quant picks you didn't pre-thesis but are including.
- pre_mortem: what is the most likely cause if this portfolio loses 8% in 30 days?
- No reduce without get_crowding_signal. No data_quality override without a named corporate action.
"""


def _build_system_prompt(ic: float | None = None) -> str:
    """Return the AI PM system prompt, prepending a calibration warning if IC < 0.05."""
    base = _SYSTEM_PROMPT
    if ic is not None and ic < 0.05:
        warning = (
            "\n\n⚠️  CALIBRATION WARNING: Your recent conviction-vs-outcome IC is "
            f"{ic:.3f} (Uncalibrated, threshold 0.05). Your high-conviction overrides "
            "have not been predictive. Be conservative — prefer the quant baseline "
            "unless you have a clearly non-quantitative thesis (news, events, regime shift). "
            "Do not override quant on momentum or valuation alone this session.\n"
        )
        return warning + base
    return base


def _get_calibration_ic_safe(n_rebalances: int = 10) -> float | None:
    """Return the Spearman IC float from the calibration report, or None on any failure.

    ``get_calibration_report`` returns a plain-text string; we parse the IC value
    out of the known line format:
        "  IC (conviction → 21d return): +0.12  [Weak]"
    Returns None when there is no data yet or when parsing fails.
    """
    import re
    try:
        from ascent.strategy.calibration_tracker import get_calibration_report
        report_str = get_calibration_report(n_rebalances=n_rebalances)
        if not isinstance(report_str, str):
            return None
        m = re.search(r"IC \(conviction.*?\):\s*([+-]?\d+\.\d+)", report_str)
        if m:
            return float(m.group(1))
        return None
    except Exception:
        return None


# ── Tool executor implementations ──────────────────────────────────────────────

def _tool_get_regime_state(_: dict) -> str:
    try:
        p = _REPO_ROOT / "dashboard" / "regime_signal.json"
        if not p.exists():
            return "Regime signal file not found."
        data = json.loads(p.read_text())
        row = data[-1] if isinstance(data, list) and data else data if isinstance(data, dict) else {}
        return (
            f"Current regime: {row.get('regime_label', row.get('label', 'unknown'))}\n"
            f"Confidence: {row.get('confidence', row.get('regime_confidence', 'n/a'))}\n"
            f"HMM entropy: {row.get('entropy', 'n/a')}\n"
            f"Days in regime: {row.get('days_in_regime', 'n/a')}"
        )
    except Exception as exc:
        return f"Could not read regime state: {exc}"


def _tool_get_macro_data(_: dict) -> str:
    try:
        import pandas as pd
        for name in ("macro_live", "macro_simulated"):
            p = _REPO_ROOT / "data_cache" / f"{name}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    latest = df.sort_index().iloc[-1]
                    lines = ["Current macro indicators (latest available):"]
                    for col in list(df.columns)[:10]:
                        val = latest.get(col)
                        if val is not None:
                            lines.append(f"  {col}: {val:.4f}")
                    return "\n".join(lines)
        return "Macro data not found."
    except Exception as exc:
        return f"Could not read macro data: {exc}"


def _tool_run_quant_agent(inputs: dict) -> str:
    agent_id = inputs.get("agent_id", "")
    _AGENT_MAP = {
        "us_equities":   ("agents.us_equities_agent", "run_us_equities_agent"),
        "macro":         ("agents.macro_agent",        "run_macro_agent"),
        "international": ("agents.international_agent","run_international_agent"),
        "alternatives":  ("agents.alternatives_agent", "run_alternatives_agent"),
    }
    if agent_id not in _AGENT_MAP:
        return f"Unknown agent_id: '{agent_id}'. Valid: {list(_AGENT_MAP.keys())}"
    try:
        import importlib
        module_name, fn_name = _AGENT_MAP[agent_id]
        mod = importlib.import_module(module_name)
        result = getattr(mod, fn_name)()
        if result is None:
            return f"Agent {agent_id} returned no result."
        top = sorted(result.target_weights.items(), key=lambda x: -x[1])[:10]
        weight_str = ", ".join(f"{s}={w:.1%}" for s, w in top)
        skill = result.skill_score
        skill_str = f"{skill:.3f}" if isinstance(skill, (int, float)) else str(skill)

        # ── Two-way quant → AI PM: per-sleeve signal quality ──────────────────
        sq = result.metadata.get("signal_quality", {}) if result.metadata else {}
        quality_lines = []
        for sym, _ in top[:8]:
            q = sq.get(sym)
            if not q:
                continue
            conv = q.get("convergence", "?")
            pslv = q.get("primary_sleeve", "?")
            scores = q.get("sleeve_scores", {})
            top_sleeves = sorted(scores.items(), key=lambda x: -abs(x[1]))[:3]
            slv_str = ", ".join(f"{sl}={v:+.2f}" for sl, v in top_sleeves)
            quality_lines.append(
                f"  {sym}: convergence={conv:.0%}  primary={pslv}  [{slv_str}]"
            )
        quality_block = ""
        if quality_lines:
            quality_block = "\nSignal quality (convergence = fraction of sleeves agreeing):\n" + "\n".join(quality_lines)
            quality_block += (
                "\nInterpretation: convergence>70% = high quant confidence (amplify with conviction);"
                " convergence<40% = single-sleeve call (verify with text signals before overriding)."
            )

        return (
            f"Quant agent: {agent_id}\n"
            f"Regime: {result.regime_signal}\n"
            f"Skill score (63d Sharpe): {skill_str}\n"
            f"Top weights: {weight_str}"
            f"{quality_block}"
        )
    except Exception as exc:
        return f"Agent {agent_id} failed: {exc}"


def _tool_get_sec_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = _REPO_ROOT / "data_cache" / "sec_signals.parquet"
        if not p.exists():
            return f"SEC signals not found for {symbol} (run sec_filings.py to populate)."
        df = pd.read_parquet(p)
        if "symbol" not in df.columns:
            return "SEC signals malformed."
        subset = df[df["symbol"] == symbol]
        if subset.empty:
            return f"No SEC signal for {symbol}."
        row = subset.sort_values("date").iloc[-1]
        cols = [c for c in ["revenue_momentum","margin_trend","tone","liquidity_risk","guidance"] if c in row.index]
        lines = [f"SEC 10-K/10-Q signal for {symbol} (as of {row.get('date','?')}):"]
        for c in cols:
            lines.append(f"  {c}: {row[c]:.2f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"SEC signal failed for {symbol}: {exc}"


def _tool_get_transcript_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = _REPO_ROOT / "data_cache" / "transcript_signals.parquet"
        if not p.exists():
            return f"Transcript signals not found for {symbol}."
        df = pd.read_parquet(p)
        if "symbol" not in df.columns:
            return "Transcript signals malformed."
        subset = df[df["symbol"] == symbol]
        if subset.empty:
            return f"No transcript signal for {symbol}."
        row = subset.sort_values("date").iloc[-1]
        cols = [c for c in ["tone","defensiveness","forward_confidence","quantitative_ratio"] if c in row.index]
        lines = [f"Transcript signal for {symbol}:"]
        for c in cols:
            lines.append(f"  {c}: {row[c]:.2f}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Transcript signal failed for {symbol}: {exc}"


def _tool_get_attribution_history(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        p = _REPO_ROOT / "logs" / "attribution_log.jsonl"
        if not p.exists():
            return "Attribution log not found."
        records = []
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("symbol") == symbol:
                        records.append(r)
                except Exception:
                    pass
        records = records[-63:]
        if not records:
            return f"No attribution history for {symbol}."
        total = sum(r.get("pnl", 0) for r in records)
        factor = sum(r.get("factor_pnl", 0) for r in records)
        idio = sum(r.get("idiosyncratic_pnl", 0) for r in records)
        return (
            f"Attribution for {symbol} (last {len(records)} days):\n"
            f"  Total P&L: {total:+.4f}\n"
            f"  Factor P&L: {factor:+.4f}\n"
            f"  Idiosyncratic P&L: {idio:+.4f}"
        )
    except Exception as exc:
        return f"Attribution failed for {symbol}: {exc}"


def _tool_get_earnings_signal(inputs: dict) -> str:
    symbol = inputs.get("symbol", "")
    try:
        import pandas as pd
        p = _REPO_ROOT / "data_cache" / "earnings_cache.parquet"
        if not p.exists():
            return "Earnings cache not found."
        df = pd.read_parquet(p)
        if "symbol" not in df.columns:
            return "Earnings cache malformed."
        subset = df[df["symbol"] == symbol]
        if subset.empty:
            return f"No earnings signal for {symbol}."
        row = subset.sort_values("date").iloc[-1]
        surprise = row.get("surprise_pct", row.get("earnings_surprise", "n/a"))
        return f"PEAD signal for {symbol}:\n  Earnings surprise (momentum-neutral): {surprise}"
    except Exception as exc:
        return f"Earnings signal failed for {symbol}: {exc}"


def _tool_get_past_verdicts(inputs: dict) -> str:
    regime = inputs.get("regime", "")
    try:
        d = _REPO_ROOT / "outputs" / "debate_log"
        if not d.exists():
            return "No debate log found."
        verdicts = []
        for vf in sorted(d.glob("verdict_*.json"), reverse=True):
            try:
                v = json.loads(vf.read_text())
                if not regime or v.get("regime", "") == regime:
                    verdicts.append(v)
                    if len(verdicts) >= 5:
                        break
            except Exception:
                pass
        if not verdicts:
            return f"No past verdicts for regime '{regime}'."
        lines = [f"Last {len(verdicts)} verdicts for regime '{regime}':"]
        for v in verdicts:
            conf = v.get("confidence", 0)
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
            lines.append(f"  {v.get('date','?')}: {v.get('verdict','?')} (conf={conf_str})")
        return "\n".join(lines)
    except Exception as exc:
        return f"Verdict lookup failed: {exc}"


def _tool_get_factor_exposures(inputs: dict) -> str:
    weights = inputs.get("weights", {})
    try:
        import pandas as pd
        from ascent.risk.factor_exposure import format_exposure_context
        w_series = pd.Series(weights, dtype=float)
        return format_exposure_context(w_series, date.today())
    except Exception as exc:
        return f"Factor exposure failed: {exc}"


def _tool_get_regime_memory(inputs: dict) -> str:
    regime = inputs.get("regime", "")
    try:
        from memory.regime_memory import query_episodes
        return query_episodes(regime, n=5)
    except Exception as exc:
        return f"Regime memory unavailable: {exc}"


def _tool_get_narrative_shift(inputs: dict) -> str:
    symbol = inputs.get("symbol", "").upper()
    try:
        from ascent.alpha.narrative_alpha import _get_symbol_analyses, _compute_shift
        fund_cache_path = _REPO_ROOT / "data_cache" / "llm_fundamental_cache.json"
        if not fund_cache_path.exists():
            return f"No LLM fundamental cache found for {symbol}."
        fund_cache = json.loads(fund_cache_path.read_text())
        analyses = _get_symbol_analyses(symbol, fund_cache)
        if len(analyses) < 2:
            return f"Insufficient history for {symbol} — only {len(analyses)} quarter(s) cached."
        current, prior = analyses[0], analyses[1]
        shift = _compute_shift(symbol, current["analysis"], prior["analysis"])
        direction = "improved" if shift > 0.1 else "deteriorated" if shift < -0.1 else "stable"
        return (
            f"Narrative shift for {symbol}: {shift:+.2f} ({direction})\n"
            f"Current Q ({current['date']}): {current['analysis'].get('direction','?')} — {current['analysis'].get('key_trend','?')}\n"
            f"Prior Q ({prior['date']}): {prior['analysis'].get('direction','?')} — {prior['analysis'].get('key_trend','?')}"
        )
    except Exception as exc:
        return f"Narrative shift failed for {symbol}: {exc}"


def _tool_get_calibration_report(_: dict) -> str:
    try:
        from ascent.strategy.calibration_tracker import get_calibration_report
        return get_calibration_report(n_rebalances=10)
    except Exception as exc:
        return f"Calibration report unavailable: {exc}"


def _tool_get_alpha_wedge(_: dict) -> str:
    try:
        from ascent.monitoring.alpha_wedge_tracker import get_wedge_summary
        return get_wedge_summary(n=10)
    except Exception as exc:
        return f"Alpha wedge data unavailable: {exc}"


def _tool_query_decision_history(inputs: dict) -> str:
    try:
        from ascent.memory.decision_memory import format_query_result
        return format_query_result(
            override_type=inputs.get("override_type"),
            regime=inputs.get("regime"),
        )
    except Exception as exc:
        return f"Decision history unavailable: {exc}"


def _tool_check_override_conviction(inputs: dict) -> str:
    try:
        from ascent.strategy.conviction_gate import evaluate, format_gate_result
        ic = _get_calibration_ic_safe()
        result = evaluate(
            override_type=inputs.get("override_type", ""),
            regime=inputs.get("regime", ""),
            calibration_ic=ic,
        )
        return format_gate_result(result)
    except Exception as exc:
        return f"Conviction gate unavailable: {exc}"


def _tool_get_rebalance_brief(_: dict) -> str:
    try:
        from pathlib import Path
        import json
        brief_path = Path("data_cache/rebalance_brief.json")
        if not brief_path.exists():
            return "No pre-rebalance brief available. Proceed with standard research."
        data = json.loads(brief_path.read_text())
        lines = [
            f"=== PRE-REBALANCE INTELLIGENCE BRIEF ({data.get('date', 'N/A')}) ===",
            f"\nSYNTHESIS:\n{data.get('synthesis', 'N/A')}",
            f"\nSTALE POSITIONS (rank decayed ≥10 since rebalance): {data.get('stale_positions') or 'none'}",
            f"WEAKENING ALPHA SLEEVES: {data.get('weakening_sleeves') or 'none'}",
            f"ANALOGUE SIGNAL: {data.get('analogue_signal', 'N/A')}",
            f"TOP MACRO RISKS: {'; '.join(data.get('top_macro_risks', [])) or 'none'}",
            "\nACCUMULATED ADVERSARIAL CHALLENGES (last 3 days):",
        ] + [f"  - {t}" for t in data.get("adversarial_themes", [])]
        return "\n".join(lines)
    except Exception as e:
        return f"Brief unavailable: {e}"


def _tool_get_live_news(inputs: dict) -> str:
    """Fetch last 72h news headlines for a symbol via yfinance."""
    import time as _time
    from datetime import datetime as _dt
    symbol = inputs.get("symbol", "").upper().strip()
    if not symbol:
        return "Error: symbol required"
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news or []
        cutoff = _time.time() - 72 * 3600
        recent = [n for n in news if n.get("providerPublishTime", 0) > cutoff][:5]
        if not recent:
            return f"No news in last 72h for {symbol}."
        lines = [f"{symbol} news (last 72h):"]
        for n in recent:
            ts = _dt.fromtimestamp(n["providerPublishTime"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{ts}] {n.get('title', 'No title')} — {n.get('publisher', '')}")
        return "\n".join(lines)
    except Exception as e:
        return f"News fetch failed for {symbol}: {e}"


def _tool_get_analyst_estimates(inputs: dict) -> str:
    """Fetch forward valuation and analyst consensus for a symbol via yfinance."""
    symbol = inputs.get("symbol", "").upper().strip()
    if not symbol:
        return "Error: symbol required"
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        fields = [
            ("forwardPE",               "Forward P/E"),
            ("priceToBook",             "Price/Book"),
            ("targetMeanPrice",         "Analyst target (mean)"),
            ("targetLowPrice",          "Analyst target (low)"),
            ("targetHighPrice",         "Analyst target (high)"),
            ("numberOfAnalystOpinions", "# analysts covering"),
            ("recommendationMean",      "Rec mean (1=Strong Buy, 5=Strong Sell)"),
            ("earningsGrowth",          "Earnings growth (YoY)"),
            ("revenueGrowth",           "Revenue growth (YoY)"),
        ]
        lines = [f"{symbol} analyst consensus:"]
        for key, label in fields:
            val = info.get(key)
            if val is not None:
                if isinstance(val, float) and key.endswith("Growth"):
                    lines.append(f"  {label}: {val*100:.1f}%")
                else:
                    lines.append(f"  {label}: {val}")
        if len(lines) == 1:
            return f"No analyst data available for {symbol}."
        return "\n".join(lines)
    except Exception as e:
        return f"Analyst data failed for {symbol}: {e}"


def _tool_get_scenario_plan(_: dict) -> str:
    try:
        p = _REPO_ROOT / "data_cache" / "scenario_plan.json"
        if not p.exists():
            return "No weekend scenario plan available."
        data = json.loads(p.read_text())
        as_of = data.get("as_of", "unknown")
        scenarios = data.get("scenarios", [])
        if not scenarios:
            return f"Scenario plan ({as_of}): no scenarios computed."
        lines = [f"=== WEEKEND SCENARIO PLAN (as of {as_of}) ==="]
        flagged = [s for s in scenarios if s.get("flagged")]
        if flagged:
            lines.append(f"\n⚠️  FLAGGED SCENARIOS (prob ≥40%) — address in sizing:")
            for s in flagged:
                prob = s.get("probability", 0)
                impact = s.get("impact", {})
                lines.append(
                    f"  {s['name']}: {prob:.0%} prob | "
                    f"impact {impact.get('total_impact_pct', 0):+.1f}% | "
                    f"action: {s.get('pre_emptive_action', 'n/a')}"
                )
        lines.append("\nAll scenarios:")
        for s in scenarios:
            prob = s.get("probability", 0)
            impact = s.get("impact", {})
            flag = " [FLAGGED]" if s.get("flagged") else ""
            lines.append(
                f"  {s['name']}{flag}: {s.get('description', '')} | "
                f"{prob:.0%} prob | impact {impact.get('total_impact_pct', 0):+.1f}%"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Scenario plan unavailable: {exc}"


def _tool_get_weekend_research(_: dict) -> str:
    try:
        p = _REPO_ROOT / "data_cache" / "weekend_research.json"
        if not p.exists():
            return "No weekend research memo available."
        data = json.loads(p.read_text())
        as_of = data.get("as_of", "unknown")
        opps = data.get("opportunities", [])
        if not opps:
            return f"Weekend research memo ({as_of}): no opportunities identified."
        lines = [f"=== WEEKEND RESEARCH MEMO (as of {as_of}) ==="]
        lines.append(f"\nTop opportunities ({len(opps)} identified):")
        for i, opp in enumerate(opps[:5], 1):
            sym = opp.get("symbol", "?")
            thesis = opp.get("thesis", "")
            conviction = opp.get("conviction", "")
            lines.append(f"  {i}. {sym} [{conviction}]: {thesis[:120]}")
        summary = data.get("summary", "")
        if summary:
            lines.append(f"\nSummary: {summary[:200]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Weekend research unavailable: {exc}"


def _tool_propose_portfolio(inputs: dict, result_store: list) -> str:
    weights = inputs.get("weights", {})
    thesis  = inputs.get("thesis", {})

    # Attack #3 — feedback citation enforcement in code, not the prompt.
    # If a feedback file exists and the model didn't acknowledge it, reject.
    _fb_path = _REPO_ROOT / "data_cache" / "ai_pm_perf_feedback.json"
    if _fb_path.exists():
        acknowledged = thesis.get("feedback_acknowledged", False)
        if not acknowledged:
            log.warning("[AIPMAgent] Phase 2 rejected — feedback_acknowledged missing or false. "
                        "Falling back to pure quant for this rebalance.")
            result_store.append(AIPMResult(portfolio={}, thesis={}, fallback=True))
            return ("REJECTED: You must set feedback_acknowledged=true and include worst_call_response "
                    "before submitting. Read the feedback file and acknowledge your worst recent call.")

    # Attack #4 — conviction inflation cap enforced in code.
    # If >40% of thesis positions are 'high conviction', downgrade excess to 'medium'.
    from ascent.strategy.ai_pm_guardrails import check_conviction_inflation
    conviction_map = {
        sym: info.get("conviction", "medium")
        for sym, info in thesis.get("position_rationale", {}).items()
        if isinstance(info, dict)
    }
    if conviction_map:
        adjusted = check_conviction_inflation(conviction_map)
        inflated = [s for s in conviction_map if conviction_map[s] != adjusted.get(s)]
        if inflated:
            log.info("[AIPMAgent] Conviction inflation: downgraded %d names from high→medium: %s",
                     len(inflated), inflated)
            for sym in inflated:
                if isinstance(thesis.get("position_rationale", {}).get(sym), dict):
                    thesis["position_rationale"][sym]["conviction"] = "medium"

    result_store.append(AIPMResult(portfolio=weights, thesis=thesis))

    # Log for calibration tracking
    try:
        from ascent.strategy.calibration_tracker import log_prediction
        log_prediction(str(date.today()), weights, thesis)
    except Exception:
        pass

    return "Portfolio submitted. Research loop complete."


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _tool_get_crowding_signal(inputs: dict) -> str:
    """
    Crowding / momentum-exhaustion signal.
    Combines momentum trajectory, short interest, and analyst consensus drift.
    This is the gating check before any REDUCE override — do not reduce without it.
    """
    import yfinance as yf

    symbols = inputs.get("symbols", [])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]
    symbols = [s.upper() for s in symbols[:8]]

    results = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info   = ticker.info or {}
            hist   = ticker.history(period="14mo", auto_adjust=True)

            row: dict = {}

            # ── Momentum trajectory ──────────────────────────────────────────
            if len(hist) >= 252:
                c = hist["Close"]
                m252 = (c.iloc[-1] / c.iloc[-252] - 1) * 100
                m63  = (c.iloc[-1] / c.iloc[-63]  - 1) * 100
                m21  = (c.iloc[-1] / c.iloc[-21]  - 1) * 100
                # Expected 21d return if 252d trend held steady
                expected_21d = m252 / 12.0
                decel_ratio  = m21 / (expected_21d + 1e-6)
                row.update(mom_252d=round(m252, 1), mom_63d=round(m63, 1),
                           mom_21d=round(m21, 1), decelerating=decel_ratio < 0.4)
            else:
                row["decelerating"] = None

            # ── Short interest ───────────────────────────────────────────────
            sf = info.get("shortPercentOfFloat")
            row["short_float_pct"] = round(sf * 100, 1) if sf is not None else None

            # ── Analyst consensus ────────────────────────────────────────────
            rec = info.get("recommendationMean")          # 1=Strong Buy, 5=Strong Sell
            n_a = info.get("numberOfAnalystOpinions", 0)
            row["rec_mean"]   = round(rec, 2) if rec else None
            row["n_analysts"] = n_a

            # ── Score & signal ───────────────────────────────────────────────
            score, flags = 0, []

            if row.get("decelerating"):
                score += 2
                m21v = row.get("mom_21d", 0)
                exp  = row.get("mom_252d", 0) / 12
                flags.append(f"momentum decelerating: 21d {m21v:+.0f}% vs expected {exp:+.1f}%/period")

            sf_val = row.get("short_float_pct")
            if sf_val and sf_val > 15:
                score += 2
                flags.append(f"short interest {sf_val}% of float — informed bears present")
            elif sf_val and sf_val > 8:
                score += 1
                flags.append(f"short interest {sf_val}% of float — elevated")

            if rec and n_a >= 5:
                if rec > 2.5:
                    score += 2
                    flags.append(f"analyst consensus fading ({rec:.1f}/5.0, {n_a} analysts)")
                elif rec > 2.0:
                    score += 1
                    flags.append(f"analyst consensus mixed ({rec:.1f}/5.0, {n_a} analysts)")

            signal = "OVERCROWDED" if score >= 4 else ("WATCH" if score >= 2 else "CLEAN")
            row.update(signal=signal, score=score, flags=flags or ["no crowding signals"])
            results[sym] = row

        except Exception as exc:
            results[sym] = {"signal": "UNKNOWN", "error": str(exc)}

    lines = ["CROWDING SIGNAL REPORT", "=" * 44]
    for sym, r in results.items():
        sig = r.get("signal", "UNKNOWN")
        marker = {"OVERCROWDED": "🔴", "WATCH": "🟡", "CLEAN": "🟢"}.get(sig, "⚪")
        lines.append(f"\n{marker} {sym}: {sig} (score {r.get('score', '?')}/6)")
        if "mom_252d" in r:
            lines.append(f"   Momentum  252d={r['mom_252d']:+.1f}%  63d={r['mom_63d']:+.1f}%  21d={r['mom_21d']:+.1f}%")
            lines.append(f"   Decelerating: {r.get('decelerating')}")
        if r.get("short_float_pct") is not None:
            lines.append(f"   Short interest: {r['short_float_pct']}% of float")
        if r.get("rec_mean") is not None:
            lines.append(f"   Analyst rec:    {r['rec_mean']}/5.0  ({r.get('n_analysts', 0)} analysts)")
        for f in r.get("flags", []):
            lines.append(f"   ⚠  {f}")

    lines += [
        "",
        "Signal guide:",
        "  🟢 CLEAN       — momentum intact, shorts low, analysts bullish → AMPLIFY candidate",
        "  🟡 WATCH       — 1-2 mild signals → hold at quant weight, no reduce without text confirmation",
        "  🔴 OVERCROWDED — 2+ signals → valid reduce IF a text signal also confirms (sec_tone or transcript)",
    ]
    return "\n".join(lines)


def _tool_get_causal_graph(inputs: dict) -> str:
    """Return the cached causal graph for a symbol, or a 'not available' message."""
    try:
        from ascent.causal.dag_builder import load_or_build, get_quarter_end
        symbol = inputs.get("symbol", "").upper()
        if not symbol:
            return "Error: symbol required"
        quarter_end = get_quarter_end(symbol)
        graph = load_or_build(symbol, quarter_end)
        if not graph.get("mechanisms"):
            return f"No causal graph available for {symbol}. Build one by running the weekend pipeline."
        lines = [f"Causal graph for {symbol} (quarter_end={quarter_end}):"]
        for i, m in enumerate(graph["mechanisms"], 1):
            lines.append(
                f"\n[Mechanism {i}] {m.get('mechanism', 'N/A')}\n"
                f"  Timing: {m.get('timing', 'N/A')}\n"
                f"  Intervention: {m.get('intervention', 'N/A')}\n"
                f"  Falsification: {m.get('falsification_condition', 'N/A')}\n"
                f"  Horizon: {m.get('horizon_days', 'N/A')} trading days"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"get_causal_graph failed: {exc}"


def _tool_propose_prethesis(inputs: dict, result_store: list) -> str:
    """Seal the pre-thesis. Stores it so run_ai_pm_prethesis() can return it."""
    result_store.append(inputs)
    n = len(inputs.get("high_conviction_names", []))
    avoid = len(inputs.get("names_to_avoid", []))
    log.info("[AIPMAgent] Pre-thesis sealed: %d conviction names, %d to avoid", n, avoid)
    return f"Pre-thesis sealed: {n} conviction names, {avoid} to avoid. Phase 1 complete."


def _make_prethesis_executor(result_store: list):
    """Dispatcher for Phase 1 (pre-thesis) tools — no quant agents, no propose_portfolio."""
    _map = {
        "get_rebalance_brief":      _tool_get_rebalance_brief,
        "get_live_news":            _tool_get_live_news,
        "get_analyst_estimates":    _tool_get_analyst_estimates,
        "get_regime_state":         _tool_get_regime_state,
        "get_macro_data":           _tool_get_macro_data,
        "get_sec_signal":           _tool_get_sec_signal,
        "get_transcript_signal":    _tool_get_transcript_signal,
        "get_attribution_history":  _tool_get_attribution_history,
        "get_earnings_signal":      _tool_get_earnings_signal,
        "get_regime_memory":        _tool_get_regime_memory,
        "get_narrative_shift":      _tool_get_narrative_shift,
        "get_calibration_report":   _tool_get_calibration_report,
        "get_scenario_plan":        _tool_get_scenario_plan,
        "get_weekend_research":     _tool_get_weekend_research,
        "get_crowding_signal":      _tool_get_crowding_signal,
        "get_causal_graph":         _tool_get_causal_graph,
        "propose_prethesis":        lambda i: _tool_propose_prethesis(i, result_store),
    }

    def executor(tool_name: str, tool_inputs: dict) -> str:
        fn = _map.get(tool_name)
        if fn is None:
            return f"Tool '{tool_name}' not available in Phase 1 (pre-thesis). Call propose_prethesis to complete Phase 1."
        try:
            return fn(tool_inputs)
        except Exception as exc:
            log.warning("[AIPMAgent] Pre-thesis tool %s failed: %s", tool_name, exc)
            return f"Tool {tool_name} failed: {exc}"

    return executor


def _make_executor(result_store: list, precomputed: dict | None = None):
    from debate.agent_tools import (
        get_var_estimate, get_sector_concentration, get_position_momentum,
    )

    _cache = precomputed or {}

    def _run_quant_agent_cached(inputs: dict) -> str:
        agent_id = inputs.get("agent_id", "")
        if agent_id in _cache:
            log.info("[AIPMAgent] Using precomputed output for agent '%s'", agent_id)
            return _cache[agent_id]
        return _tool_run_quant_agent(inputs)

    _map = {
        "get_rebalance_brief":      _tool_get_rebalance_brief,
        "get_live_news":            _tool_get_live_news,
        "get_analyst_estimates":    _tool_get_analyst_estimates,
        "get_regime_state":         _tool_get_regime_state,
        "get_macro_data":           _tool_get_macro_data,
        "run_quant_agent":          _run_quant_agent_cached,
        "get_sec_signal":           _tool_get_sec_signal,
        "get_transcript_signal":    _tool_get_transcript_signal,
        "get_attribution_history":  _tool_get_attribution_history,
        "get_earnings_signal":      _tool_get_earnings_signal,
        "get_past_verdicts":        _tool_get_past_verdicts,
        "get_regime_memory":        _tool_get_regime_memory,
        "get_factor_exposures":     _tool_get_factor_exposures,
        "get_var_estimate":         lambda i: get_var_estimate(i),
        "get_sector_concentration": lambda i: get_sector_concentration(i),
        "get_position_momentum":    lambda i: get_position_momentum(i),
        "get_narrative_shift":      _tool_get_narrative_shift,
        "get_calibration_report":       _tool_get_calibration_report,
        "get_alpha_wedge":              _tool_get_alpha_wedge,
        "query_decision_history":       _tool_query_decision_history,
        "check_override_conviction":    _tool_check_override_conviction,
        "get_scenario_plan":            _tool_get_scenario_plan,
        "get_weekend_research":         _tool_get_weekend_research,
        "get_crowding_signal":          _tool_get_crowding_signal,
        "propose_portfolio":            lambda i: _tool_propose_portfolio(i, result_store),
    }

    def executor(tool_name: str, tool_inputs: dict) -> str:
        fn = _map.get(tool_name)
        if fn is None:
            return f"Unknown tool: {tool_name}"
        try:
            return fn(tool_inputs)
        except Exception as exc:
            log.warning("[AIPMAgent] Tool %s failed: %s", tool_name, exc)
            return f"Tool {tool_name} failed: {exc}"

    return executor


# ── Causal helpers (Phase B) ───────────────────────────────────────────────────

def _assemble_causal_mechanisms(
    high_conviction_symbols: list,
    regime: str,
    cache_dir=None,
) -> list:
    """
    After propose_prethesis, assemble CausalMechanism objects for all
    high-conviction symbols. Applies Gate 1 (compatibility) + Gate 2 (priced_in).
    Returns list[CausalMechanism].
    """
    try:
        from ascent.causal.dag_builder import load_or_build, get_quarter_end
        from ascent.causal.compatibility import regime_compatible
        from ascent.config.types import CausalMechanism

        results = []
        for symbol in high_conviction_symbols:
            quarter_end = get_quarter_end(symbol)
            graph = load_or_build(symbol, quarter_end, cache_dir)
            for m in graph.get("mechanisms", []):
                mtype = m.get("mechanism_type", "")
                if not regime_compatible(mtype, regime):
                    continue
                if m.get("timing") == "priced_in":
                    continue
                results.append(CausalMechanism(
                    symbol=symbol,
                    mechanism=m.get("mechanism", ""),
                    intervention=m.get("intervention", ""),
                    falsification_condition=m.get("falsification_condition", ""),
                    horizon_days=int(m.get("horizon_days", 63)),
                    timing=m.get("timing", "not_yet_priced"),
                    velocity=0.0,
                    mechanism_type=mtype,
                    regime_compatible=True,
                ))
        return results
    except Exception as exc:
        log.warning("[AIPMAgent] _assemble_causal_mechanisms failed: %s", exc)
        return []


_TIMING_PRIORITY = {"catalyst_imminent": 2, "not_yet_priced": 1, "priced_in": 0}


def _build_velocity_context(
    symbols: list,
    regime: str,
    cache_dir=None,
) -> list:
    """
    Build a ranked list of causal context lines for injection into Phase 1 prompt.
    Returns list of strings, sorted by timing priority (catalyst_imminent first).
    """
    try:
        from ascent.causal.dag_builder import load_or_build, get_quarter_end
        from ascent.causal.compatibility import regime_compatible

        candidates = []
        for symbol in symbols:
            quarter_end = get_quarter_end(symbol)
            graph = load_or_build(symbol, quarter_end, cache_dir)
            for m in graph.get("mechanisms", []):
                mtype = m.get("mechanism_type", "")
                timing = m.get("timing", "not_yet_priced")
                if not regime_compatible(mtype, regime):
                    continue
                if timing == "priced_in":
                    continue
                priority = _TIMING_PRIORITY.get(timing, 0)
                candidates.append((priority, symbol, m.get("mechanism", ""), timing))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            f"  {sym} [{timing}]: {mechanism}"
            for _, sym, mechanism, timing in candidates
        ]
    except Exception as exc:
        log.warning("[AIPMAgent] _build_velocity_context failed: %s", exc)
        return []


# ── Entry point ────────────────────────────────────────────────────────────────

def run_ai_pm_prethesis() -> Optional[AIPreThesis]:
    """
    Phase 1: AI PM reads broad data and forms original thesis BEFORE quant runs.
    Returns AIPreThesis or None on failure. Should be called before run_quant_agents().
    """
    result_store: list = []
    executor = _make_prethesis_executor(result_store)

    # Inject causal context for current portfolio holdings
    _current_regime = _get_current_regime()
    _portfolio_symbols: list = []
    try:
        mw_path = _REPO_ROOT / "data_cache" / "merged_weights.json"
        if mw_path.exists():
            _portfolio_symbols = list(json.loads(mw_path.read_text()).keys())
    except Exception:
        pass

    _causal_lines = _build_velocity_context(_portfolio_symbols, _current_regime)
    _causal_context = ""
    if _causal_lines:
        _causal_context = (
            "\n\n══ CAUSAL INTELLIGENCE (regime-compatible, catalyst not yet priced) ══\n"
            "Top causal mechanisms for current holdings, ranked by timing priority.\n"
            "Use them to AMPLIFY where mechanism + quant agree. "
            "Call get_causal_graph(symbol) for full falsification conditions.\n"
            + "\n".join(_causal_lines)
        )

    # Load feedback for temporal context injection
    _p1_feedback: dict = {}
    try:
        import json as _j1
        _p1_fp = Path(__file__).resolve().parent.parent / "data_cache" / "ai_pm_perf_feedback.json"
        if _p1_fp.exists():
            _p1_feedback = _j1.loads(_p1_fp.read_text())
    except Exception:
        pass

    # Attack #1 — data grounding for Phase 1.
    # Pre-load verified numbers from cache so model reads real data, not training memory.
    _p1_grounding = _build_data_grounding(
        [n.get("symbol", "") for n in _prethesis_universe[:30]] if _prethesis_universe else []
    ) if "_prethesis_universe" in dir() else _build_data_grounding([])

    try:
        tool_completion(
            system_prompt=_build_temporal_context(feedback=_p1_feedback) + _p1_grounding + _PRE_THESIS_PROMPT,
            user_prompt=(
                f"Today is {date.today()}. Read the available data and form your original "
                "investment thesis for the next rebalance. Call propose_prethesis when ready."
                + _causal_context
            ),
            tools=PRE_THESIS_TOOLS,
            tool_executor=executor,
            model=SONNET_MODEL,   # Sonnet reads + summarises; Opus reserved for synthesis judgment
            max_tokens=6000,
            max_tool_calls=10,
            use_cache=True,
        )
    except Exception as exc:
        log.error("[AIPMAgent] Pre-thesis phase failed: %s", exc)
        return None

    if not result_store:
        log.warning("[AIPMAgent] Pre-thesis: propose_prethesis never called — no thesis")
        return None

    raw = result_store[-1]
    prethesis = AIPreThesis(
        macro_view=raw.get("macro_view", ""),
        regime_interpretation=raw.get("regime_interpretation", ""),
        high_conviction_names=raw.get("high_conviction_names", []),
        names_to_avoid=raw.get("names_to_avoid", []),
        sector_tilts=raw.get("sector_tilts", []),
        regime_assessment=dict(raw.get("regime_assessment") or {}),
        sleeve_weight_prior=dict(raw.get("sleeve_weight_prior") or {}),
        market_character=str(raw.get("market_character") or ""),
        raw=raw,
    )

    # Populate causal_mechanisms with Gate 1 + Gate 2 filtered mechanisms
    try:
        prethesis.causal_mechanisms = _assemble_causal_mechanisms(
            high_conviction_symbols=prethesis.conviction_symbols,
            regime=_current_regime,
        )
        log.info("[AIPMAgent] Pre-thesis: %d causal mechanisms assembled", len(prethesis.causal_mechanisms))
    except Exception as exc:
        log.warning("[AIPMAgent] Causal mechanism assembly failed: %s", exc)

    log.info(
        "[AIPMAgent] Pre-thesis complete: %d conviction names, macro_view=%s...",
        len(prethesis.high_conviction_names),
        prethesis.macro_view[:80],
    )
    return prethesis


def _format_prethesis_for_prompt(prethesis: AIPreThesis) -> str:
    """Render the sealed pre-thesis as readable text for the synthesis prompt."""
    lines = [f"Macro view: {prethesis.macro_view}"]
    if prethesis.regime_interpretation:
        lines.append(f"Regime interpretation: {prethesis.regime_interpretation}")
    if prethesis.sector_tilts:
        lines.append("\nSector tilts:")
        for t in prethesis.sector_tilts:
            lines.append(f"  {t.get('sector','?')}: {t.get('tilt','?')} — {t.get('reason','')}")
    lines.append(f"\nHigh conviction names ({len(prethesis.high_conviction_names)}):")
    for n in prethesis.high_conviction_names:
        lines.append(f"  {n.get('symbol','?')}: {n.get('thesis','')}")
        if n.get("what_quant_should_confirm"):
            lines.append(f"    Expected quant confirmation: {n['what_quant_should_confirm']}")
        if n.get("what_would_change_my_mind"):
            lines.append(f"    Would change my mind if: {n['what_would_change_my_mind']}")
    if prethesis.names_to_avoid:
        lines.append("\nNames to avoid:")
        for a in prethesis.names_to_avoid:
            lines.append(f"  {a.get('symbol','?')}: {a.get('reason','')}")
    return "\n".join(lines)


def run_ai_pm(
    quant_outputs: Optional[list] = None,
    merged_weights: Optional[Dict[str, float]] = None,
    prethesis: Optional[AIPreThesis] = None,
    causal_track_record: Optional[dict] = None,
    model_override: Optional[str] = None,
) -> AIPMResult:
    """
    Run the AI PM agent. Returns AIPMResult(portfolio, thesis).
    Falls back to AIPMResult(portfolio={}, thesis={}, fallback=True) on any failure
    or if propose_portfolio is never called.
    """
    result_store: List[AIPMResult] = []

    # Build precomputed cache from already-run quant outputs — avoids 4 redundant pipeline runs
    precomputed: dict = {}
    if quant_outputs:
        for ao in quant_outputs:
            try:
                all_pos = sorted(ao.target_weights.items(), key=lambda x: -x[1])
                weight_str = ", ".join(f"{s}={w:.1%}" for s, w in all_pos)
                skill = ao.skill_score
                skill_str = f"{skill:.3f}" if isinstance(skill, (int, float)) else str(skill)

                # Two-way: include per-sleeve signal quality in precomputed cache
                sq = (ao.metadata or {}).get("signal_quality", {})
                quality_lines = []
                for sym, _ in all_pos[:8]:
                    q = sq.get(sym)
                    if not q:
                        continue
                    conv  = q.get("convergence", 0.5)
                    pslv  = q.get("primary_sleeve", "trend")
                    scores = q.get("sleeve_scores", {})
                    top_s  = sorted(scores.items(), key=lambda x: -abs(x[1]))[:3]
                    slv_str = ", ".join(f"{sl}={v:+.2f}" for sl, v in top_s)
                    quality_lines.append(f"  {sym}: convergence={conv:.0%} primary={pslv} [{slv_str}]")
                quality_block = ""
                if quality_lines:
                    quality_block = "\nSignal quality:\n" + "\n".join(quality_lines)

                precomputed[ao.agent_id] = (
                    f"Quant agent: {ao.agent_id}\n"
                    f"Regime: {ao.regime_signal}\n"
                    f"Skill score (63d Sharpe): {skill_str}\n"
                    f"All positions ({len(all_pos)}): {weight_str}"
                    f"{quality_block}"
                )
            except Exception as exc:
                log.warning("[AIPMAgent] Could not cache output for agent: %s", exc)

    if precomputed:
        log.info("[AIPMAgent] Preloaded %d agent outputs — skipping redundant pipeline runs", len(precomputed))

    # Load daily feedback for temporal context + worst-call injection
    _feedback: dict = {}
    try:
        import json as _json
        _fp = Path(__file__).resolve().parent.parent / "data_cache" / "ai_pm_perf_feedback.json"
        if _fp.exists():
            _feedback = _json.loads(_fp.read_text())
    except Exception:
        pass

    # Build system prompt — synthesis mode when prethesis is available
    _ic = _get_calibration_ic_safe(n_rebalances=10)
    _temporal_ctx = _build_temporal_context(feedback=_feedback)

    if prethesis is not None:
        # Apply Phase 1→2 context strip: only structured, sourced fields pass through
        _stripped = _strip_prethesis_for_phase2(prethesis)
        prethesis_text = _format_prethesis_for_prompt(prethesis)
        _system = _temporal_ctx + _SYNTHESIS_PROMPT_TEMPLATE.format(prethesis_text=prethesis_text)

        # Append stripped structured fields note for Phase 2
        if _stripped.get("sector_thesis"):
            _system += f"\n\n══ PHASE 1 SECTOR THESIS (sourced, validated) ══\n{_stripped['sector_thesis']}"
        if _stripped.get("conviction_reasons"):
            _system += f"\n\n══ PHASE 1 SOURCED CLAIMS ({len(_stripped['conviction_reasons'])} validated) ══\n" + \
                       "\n".join(f"  {r.get('symbol','?')}: {r.get('claim','')} [{r.get('source','')}]"
                                 for r in _stripped["conviction_reasons"][:10])

        # Prepend calibration warning if IC is poor
        if _ic is not None and _ic < 0.05:
            _system = (
                f"⚠️ CALIBRATION WARNING: Your recent conviction IC is {_ic:.3f} (Uncalibrated). "
                "Prefer quant confirmation over your own overrides. Stand down when quant contradicts "
                "your thesis unless you have a specific dated catalyst.\n\n" + _system
            )
        _causal_track_context = ""
        if causal_track_record and causal_track_record.get("total", 0) >= 3:
            acc = causal_track_record.get("accuracy_pct", 0)
            total = causal_track_record.get("total", 0)
            conf = causal_track_record.get("confirmed", 0)
            fals = causal_track_record.get("falsified", 0)
            verdict = (
                "High accuracy — trust your causal mechanisms when velocity > 0.70."
                if acc >= 60 else
                "Below-target accuracy — only concentrate when mechanism velocity > 0.70 AND timing=catalyst_imminent."
            )
            _causal_track_context = (
                f"\n\n══ CAUSAL THESIS TRACK RECORD ══\n"
                f"{total} resolved: {conf} confirmed, {fals} falsified, accuracy={acc:.1f}%.\n"
                f"{verdict}"
            )
        user_prompt = (
            f"Today is {date.today()}. Your pre-thesis is sealed above. "
            "Now run the quant agents, validate your thesis, and submit your final portfolio."
            + _causal_track_context
        )
        log.info("[AIPMAgent] Using two-phase synthesis mode (%d pre-thesis names)",
                 len(prethesis.high_conviction_names))
    else:
        _system = _temporal_ctx + _build_system_prompt(ic=_ic)
        user_prompt = f"Today is {date.today()}. Please conduct your research and submit your portfolio."
        log.info("[AIPMAgent] No pre-thesis — using standard single-phase mode")

    # Determine model: use override if provided (smart Opus trigger from run_all_agents)
    _phase2_model = model_override or DEFAULT_MODEL

    # Attack #1 — data grounding for Phase 2.
    # Grounding covers all symbols from quant output + prethesis conviction names.
    _p2_symbols = list(set(
        list(merged_weights or {})[:20] +
        [n.get("symbol","") for n in (getattr(prethesis,"high_conviction_names",[]) or [])]
    ))
    _p2_grounding = _build_data_grounding(_p2_symbols)
    _system = _p2_grounding + _system  # prepend grounding before all other context

    try:
        tool_completion(
            system_prompt=_system,
            user_prompt=user_prompt,
            tools=AI_PM_TOOLS,
            tool_executor=_make_executor(result_store, precomputed),
            model=_phase2_model,
            max_tokens=4000,
            max_tool_calls=14,
            use_cache=True,
        )
    except Exception as exc:
        log.error("[AIPMAgent] tool_completion failed: %s", exc)
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    if len(result_store) > 1:
        log.warning("[AIPMAgent] propose_portfolio called %d times; using last submission", len(result_store))

    if not result_store:
        log.warning("[AIPMAgent] No propose_portfolio call — using fallback")
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    # ── Red team adversarial self-play ────────────────────────────────────────
    initial_result = result_store[-1]

    from agents.red_team_agent import run_red_team
    regime_str = _get_current_regime()
    critique = run_red_team(
        initial_result.portfolio, initial_result.thesis, regime_str,
        quant_weights=merged_weights,
    )

    if critique:
        log.info(
            "[AIPMAgent] Red team critique generated (%d chars) — giving AI PM revision pass",
            len(critique),
        )
        result_store_v2: List[AIPMResult] = []
        revision_prompt = (
            f"A red team adversarial analyst has reviewed your portfolio submission and raised the following concerns:\n\n"
            f"{critique}\n\n"
            f"You may revise your portfolio in response to these concerns, or resubmit the same portfolio if you believe it is sound. "
            f"Call propose_portfolio when ready."
        )
        try:
            tool_completion(
                system_prompt=_system,
                user_prompt=revision_prompt,
                tools=AI_PM_TOOLS,
                tool_executor=_make_executor(result_store_v2, precomputed),
                model=DEFAULT_MODEL,
                max_tokens=2000,
                max_tool_calls=6,
                use_cache=True,
            )
        except Exception as exc:
            log.warning("[AIPMAgent] Revision pass failed: %s — using initial proposal", exc)

        if result_store_v2:
            log.info("[AIPMAgent] AI PM revised portfolio after red team critique")
            return result_store_v2[-1]
        else:
            log.info("[AIPMAgent] AI PM did not revise — using initial proposal")

    return initial_result
