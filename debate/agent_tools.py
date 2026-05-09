"""
debate/agent_tools.py

Tool definitions and pure-Python implementations for tool-capable debate agents.

Tools:
  get_sector_concentration(weights) -- sector breakdown of portfolio
  get_var_estimate(weights)         -- historical 5th-percentile 1-day return
  get_position_momentum(symbols)    -- 252-day momentum for each symbol
  get_regime_conditional_stats(regime) -- historical regime outcome statistics

These are Anthropic tool schema definitions + synchronous implementations.
All implementations must be fast (< 1s) and never raise unhandled exceptions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# -- Anthropic tool schema definitions -----------------------------------------

DEBATE_TOOLS = [
    {
        "name": "get_sector_concentration",
        "description": (
            "Tool: Compute the sector-level weight breakdown for the proposed portfolio. "
            "Use this to identify sector concentration risk before making your argument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Dict mapping symbol -> portfolio weight (float)",
                    "additionalProperties": {"type": "number"},
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_var_estimate",
        "description": (
            "Estimate the portfolio's historical Value-at-Risk (5th percentile 1-day return). "
            "Use this to quantify downside risk in your argument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Dict mapping symbol -> portfolio weight",
                    "additionalProperties": {"type": "number"},
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_position_momentum",
        "description": (
            "Look up 252-day momentum (price return) for a list of symbols. "
            "Use this to verify whether positions are actually in uptrends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_regime_conditional_stats",
        "description": (
            "Get historical statistics for a given regime label: typical duration, "
            "average drawdown, base rate of continued stress, and historical examples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {
                    "type": "string",
                    "description": "Regime label: calm_bull, stressed, crisis, neutral, uncertain",
                }
            },
            "required": ["regime"],
        },
    },
]


# -- Regime statistics (static -- no live data dependency) ---------------------

_REGIME_STATS = {
    "calm_bull": {
        "avg_duration_weeks": 18, "avg_drawdown_pct": -4.2,
        "base_rate_continues_pct": 72, "avg_return_annualized_pct": 14.8,
        "tail_risk_note": "Low tail risk -- watch for euphoria/breadth narrowing as warning sign",
        "historical_examples": "2013-2014, 2017, 2019, 2021 H1",
    },
    "stressed": {
        "avg_duration_weeks": 7, "avg_drawdown_pct": -12.4,
        "base_rate_continues_pct": 38, "avg_return_annualized_pct": -6.2,
        "tail_risk_note": "High -- 38% chance of escalating to crisis; credit spreads are leading indicator",
        "historical_examples": "Q4 2018, Aug 2015, Q1 2020 onset, Q4 2022",
    },
    "crisis": {
        "avg_duration_weeks": 5, "avg_drawdown_pct": -28.7,
        "base_rate_continues_pct": 25, "avg_return_annualized_pct": -42.0,
        "tail_risk_note": "Extreme -- correlation spikes to 0.85+, liquidity gaps appear; capital preservation mode",
        "historical_examples": "Mar 2020, Q4 2008, Q3 2002",
    },
    "neutral": {
        "avg_duration_weeks": 3, "avg_drawdown_pct": -5.8,
        "base_rate_continues_pct": 30, "avg_return_annualized_pct": 4.1,
        "tail_risk_note": "Moderate -- typically transitions quickly in either direction",
        "historical_examples": "Various 2-4 week windows between regimes",
    },
    "uncertain": {
        "avg_duration_weeks": 2, "avg_drawdown_pct": -7.1,
        "base_rate_continues_pct": 20, "avg_return_annualized_pct": 1.2,
        "tail_risk_note": "High uncertainty -- HMM entropy > 0.90, reduce size and wait for clarity",
        "historical_examples": "Regime transition periods, data disruptions",
    },
}


# -- Price cache (shared across tool calls in a single debate session) ----------

_PRICE_CACHE: Dict[str, Any] = {}


def _fetch_prices_cached(symbols: List[str]) -> Dict[str, Any]:
    """Fetch price series for symbols, using in-process cache. Hard 5s timeout."""
    import pandas as pd
    import concurrent.futures
    missing = [s for s in symbols if s not in _PRICE_CACHE]
    if missing:
        try:
            import yfinance as yf

            def _download():
                return yf.download(missing, period="2y", auto_adjust=True, progress=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_download)
                try:
                    raw = future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    log.warning("[AgentTools] Price fetch timed out after 5s")
                    for sym in missing:
                        _PRICE_CACHE[sym] = pd.Series(dtype=float)
                    return {s: _PRICE_CACHE[s] for s in symbols}

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            for sym in missing:
                if sym in close.columns:
                    _PRICE_CACHE[sym] = close[sym].dropna()
                else:
                    _PRICE_CACHE[sym] = pd.Series(dtype=float)
        except Exception as exc:
            log.warning("[AgentTools] Price fetch failed: %s", exc)
            for sym in missing:
                _PRICE_CACHE[sym] = pd.Series(dtype=float)
    return {s: _PRICE_CACHE[s] for s in symbols}


# -- Tool implementations ------------------------------------------------------

def get_sector_concentration(inputs: dict) -> str:
    """Return sector breakdown of portfolio weights as plain text."""
    weights = inputs.get("weights", {})
    if not weights:
        return "No weights provided."

    sector_weights: Dict[str, float] = {}
    unknown_syms = []
    try:
        from pathlib import Path
        import pandas as pd
        profiles_path = Path("data_cache/profiles.parquet")
        if profiles_path.exists():
            df = pd.read_parquet(profiles_path)
            if "symbol" in df.columns and "sector" in df.columns:
                sector_map = dict(zip(df["symbol"], df["sector"]))
            else:
                sector_map = {}
        else:
            sector_map = {}
    except Exception:
        sector_map = {}

    # Add ETF buckets
    sector_map.update({
        "TLT": "rates", "IEF": "rates", "LQD": "rates", "BIL": "cash",
        "HYG": "credit", "UUP": "fx", "GLD": "commodities",
        "PDBC": "commodities", "DBA": "commodities", "DBB": "commodities",
        "VNQ": "reits", "IFRA": "infrastructure", "VIXY": "volatility",
        "EEM": "em_equity", "VWO": "em_equity", "EWT": "em_equity",
        "EWZ": "em_equity", "EWY": "em_equity", "INDA": "em_equity",
        "EWJ": "developed_intl", "EWG": "developed_intl",
        "EWU": "developed_intl", "EFA": "developed_intl",
    })

    for sym, w in weights.items():
        sector = sector_map.get(sym, "unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(w)
        if sector == "unknown":
            unknown_syms.append(sym)

    lines = ["Sector concentration:"]
    for sector, sw in sorted(sector_weights.items(), key=lambda x: -x[1]):
        lines.append(f"  {sector}: {sw:.1%}")
    if unknown_syms:
        lines.append(f"  (unknown sector for: {', '.join(unknown_syms[:5])})")

    max_sector = max(sector_weights.items(), key=lambda x: x[1])
    lines.append(f"\nLargest sector: {max_sector[0]} at {max_sector[1]:.1%}")
    return "\n".join(lines)


def get_var_estimate(inputs: dict) -> str:
    """Estimate portfolio historical VaR (5th percentile 1-day return) from 1-year of data."""
    weights = inputs.get("weights", {})
    if not weights:
        return "No weights provided for VaR estimate."

    try:
        import numpy as np
        import pandas as pd
        syms  = list(weights.keys())
        wvals = list(weights.values())
        total = sum(wvals)
        if total > 0:
            wvals = [w / total for w in wvals]

        prices = _fetch_prices_cached(syms)

        rets_list = []
        w_used    = []
        for sym, w in zip(syms, wvals):
            s = prices.get(sym, pd.Series(dtype=float))
            if len(s) > 50:
                rets_list.append(s.pct_change().dropna().values[-252:])
                w_used.append(w)

        if not rets_list:
            return "Insufficient price data for VaR estimate."

        min_len   = min(len(r) for r in rets_list)
        rets_list = [r[-min_len:] for r in rets_list]
        w_arr     = np.array(w_used) / sum(w_used)
        portfolio_rets = np.sum(np.column_stack(rets_list) * w_arr, axis=1)

        var_5  = float(np.percentile(portfolio_rets, 5))
        var_1  = float(np.percentile(portfolio_rets, 1))
        avg    = float(np.mean(portfolio_rets))
        vol    = float(np.std(portfolio_rets))

        return (
            f"Portfolio VaR estimate (1-year history, {min_len} days):\n"
            f"  5th percentile (daily VaR-95): {var_5:+.2%}\n"
            f"  1st percentile (daily VaR-99): {var_1:+.2%}\n"
            f"  Mean daily return: {avg:+.3%}\n"
            f"  Daily volatility:  {vol:.3%} ({vol * 16:.1%} annualized)"
        )
    except Exception as exc:
        return f"VaR estimate failed: {exc}"


def get_position_momentum(inputs: dict) -> str:
    """Return 252-day momentum for a list of symbols."""
    symbols = inputs.get("symbols", [])
    if not symbols:
        return "No symbols provided."

    try:
        prices = _fetch_prices_cached(symbols)
        lines  = ["252-day momentum (price return):"]
        for sym in symbols:
            s = prices.get(sym)
            if s is None or len(s) < 21:
                lines.append(f"  {sym}: insufficient data")
                continue
            if len(s) >= 252:
                mom_252 = float(s.iloc[-1] / s.iloc[-252] - 1)
                lines.append(f"  {sym}: {mom_252:+.1%} (252d)")
            else:
                mom_avail = float(s.iloc[-1] / s.iloc[0] - 1)
                lines.append(f"  {sym}: {mom_avail:+.1%} ({len(s)}d, <252d available)")
        return "\n".join(lines)
    except Exception as exc:
        return f"Momentum fetch failed: {exc}"


def get_regime_conditional_stats(inputs: dict) -> str:
    """Return historical regime statistics as plain text."""
    regime = str(inputs.get("regime", "unknown")).lower()
    stats  = _REGIME_STATS.get(regime)
    if not stats:
        return (
            f"No historical statistics for regime '{regime}'. "
            f"Valid regimes: {', '.join(_REGIME_STATS.keys())}"
        )
    return (
        f"Historical statistics for {regime.upper()} regime:\n"
        f"  Typical duration: {stats['avg_duration_weeks']} weeks\n"
        f"  Average drawdown: {stats['avg_drawdown_pct']:.1f}%\n"
        f"  Base rate of continuation: {stats['base_rate_continues_pct']}%\n"
        f"  Average annualized return: {stats['avg_return_annualized_pct']:+.1f}%\n"
        f"  Tail risk: {stats['tail_risk_note']}\n"
        f"  Historical examples: {stats['historical_examples']}"
    )


# -- Tool dispatcher -----------------------------------------------------------

def execute_tool(tool_name: str, tool_inputs: dict) -> str:
    """Dispatch a tool call by name. Returns plain-text result string."""
    _TOOL_MAP = {
        "get_sector_concentration":     get_sector_concentration,
        "get_var_estimate":             get_var_estimate,
        "get_position_momentum":        get_position_momentum,
        "get_regime_conditional_stats": get_regime_conditional_stats,
    }
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    try:
        return fn(tool_inputs)
    except Exception as exc:
        log.warning("[AgentTools] Tool %s failed: %s", tool_name, exc)
        return f"Tool {tool_name} failed: {exc}"
