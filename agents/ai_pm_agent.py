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

from ascent.llm.client import tool_completion, DEFAULT_MODEL

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]  # agents/ → repo root


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
        return (
            f"Quant agent: {agent_id}\n"
            f"Regime: {result.regime_signal}\n"
            f"Skill score (63d Sharpe): {skill_str}\n"
            f"Top weights: {weight_str}"
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
    thesis = inputs.get("thesis", {})
    result_store.append(AIPMResult(portfolio=weights, thesis=thesis))

    # Log for calibration tracking
    try:
        from ascent.strategy.calibration_tracker import log_prediction
        from datetime import date as _date
        log_prediction(str(_date.today()), weights, thesis)
    except Exception:
        pass  # never block

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


# ── Entry point ────────────────────────────────────────────────────────────────

def run_ai_pm(
    quant_outputs: Optional[list] = None,
    merged_weights: Optional[Dict[str, float]] = None,
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
                precomputed[ao.agent_id] = (
                    f"Quant agent: {ao.agent_id}\n"
                    f"Regime: {ao.regime_signal}\n"
                    f"Skill score (63d Sharpe): {skill_str}\n"
                    f"All positions ({len(all_pos)}): {weight_str}"
                )
            except Exception as exc:
                log.warning("[AIPMAgent] Could not cache output for agent: %s", exc)

    if precomputed:
        log.info("[AIPMAgent] Preloaded %d agent outputs — skipping redundant pipeline runs", len(precomputed))

    # Build system prompt with calibration gate
    _ic = _get_calibration_ic_safe(n_rebalances=10)
    _system = _build_system_prompt(ic=_ic)

    try:
        tool_completion(
            system_prompt=_system,
            user_prompt=f"Today is {date.today()}. Please conduct your research and submit your portfolio.",
            tools=AI_PM_TOOLS,
            tool_executor=_make_executor(result_store, precomputed),
            model=DEFAULT_MODEL,
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
