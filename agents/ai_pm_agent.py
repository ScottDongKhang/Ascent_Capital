# agents/ai_pm_agent.py
"""
AI Portfolio Manager — Opus tool-use loop.

Runs a 4-phase research loop (market context → quant baselines → signal research → submit)
and returns AIPMResult(portfolio, thesis). Falls back to AIPMResult(portfolio={}, fallback=True)
if the loop exits without calling propose_portfolio.
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


@dataclass
class AIPMResult:
    portfolio: Dict[str, float]
    thesis: Dict[str, Any]
    fallback: bool = False


# ── Tool schemas (Anthropic format) ───────────────────────────────────────────

AI_PM_TOOLS = [
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
                        "Investment memo. Keys: market_view, regime_assessment, "
                        "quant_baseline_summary, quant_agreement (list), "
                        "quant_overrides (list of {symbol, ai_action, reason}), "
                        "position_rationale (dict), key_risks (list), what_could_be_wrong."
                    ),
                },
            },
            "required": ["weights", "thesis"],
        },
    },
]

_SYSTEM_PROMPT = """You are the portfolio manager of Ascent Capital, a multi-strategy quantitative fund.

Your job is to construct a portfolio for the next rebalance period using the research tools available.

Work through your research in order:
1. PHASE 1 — Market context: Call get_regime_state and get_macro_data first.
2. PHASE 2 — Quant baseline: Call run_quant_agent for all four agents (us_equities, macro, international, alternatives).
3. PHASE 3 — Signal research: For names you are considering, call up to 6 of the signal tools.
4. PHASE 4 — Submit: Call propose_portfolio with your final weights and investment thesis.

Rules:
- You MUST call propose_portfolio before finishing. The loop ends only when you call it.
- Target 12-20 positions. Weights will be normalized; use relative sizing.
- For every name where you override a quant recommendation, include a specific reason in thesis.quant_overrides referencing the signal data you reviewed.
- If data is unavailable for a symbol, say so — do not fabricate signals.
- The quant models are your research assistants, not your bosses.
"""


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


def _tool_propose_portfolio(inputs: dict, result_store: list) -> str:
    weights = inputs.get("weights", {})
    thesis = inputs.get("thesis", {})
    result_store.append(AIPMResult(portfolio=weights, thesis=thesis))
    return "Portfolio submitted. Research loop complete."


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _make_executor(result_store: list):
    from debate.agent_tools import (
        get_var_estimate, get_sector_concentration, get_position_momentum,
    )

    _map = {
        "get_regime_state":         _tool_get_regime_state,
        "get_macro_data":           _tool_get_macro_data,
        "run_quant_agent":          _tool_run_quant_agent,
        "get_sec_signal":           _tool_get_sec_signal,
        "get_transcript_signal":    _tool_get_transcript_signal,
        "get_attribution_history":  _tool_get_attribution_history,
        "get_earnings_signal":      _tool_get_earnings_signal,
        "get_past_verdicts":        _tool_get_past_verdicts,
        "get_factor_exposures":     _tool_get_factor_exposures,
        "get_var_estimate":         lambda i: get_var_estimate(i),
        "get_sector_concentration": lambda i: get_sector_concentration(i),
        "get_position_momentum":    lambda i: get_position_momentum(i),
        "propose_portfolio":        lambda i: _tool_propose_portfolio(i, result_store),
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

    try:
        tool_completion(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Today is {date.today()}. Please conduct your research and submit your portfolio.",
            tools=AI_PM_TOOLS,
            tool_executor=_make_executor(result_store),
            model=DEFAULT_MODEL,
            max_tokens=4000,
            max_tool_calls=14,
        )
    except Exception as exc:
        log.error("[AIPMAgent] tool_completion failed: %s", exc)
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    if len(result_store) > 1:
        log.warning("[AIPMAgent] propose_portfolio called %d times; using last submission", len(result_store))

    if not result_store:
        log.warning("[AIPMAgent] No propose_portfolio call — using fallback")
        return AIPMResult(portfolio={}, thesis={}, fallback=True)

    return result_store[-1]
