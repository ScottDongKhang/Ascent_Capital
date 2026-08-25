"""ascent/execution/capacity_model.py

Strategy capacity model — max AUM before signal decay from market impact.
Informational only: does not block trades or modify weights.
Output logged weekly to logs/capacity_log.jsonl.

Not a duplicate of ascent/execution/cost_model.py or
ascent/backtest/costs.py::liquidity_scaled_cost_model, despite also containing
an Almgren-Chriss-style impact formula (estimate_market_impact(), its own eta
and exponent). Those two answer "what does this trade cost"; this module
answers a different question — "at what AUM does a sleeve's signal IC stop
covering its own impact cost" (compute_signal_breakeven_adv /
compute_strategy_capacity) — a strategic capacity ceiling, not a per-trade
cost estimate, and it has no live callers outside its own test. Left as its
own formula rather than force-merged.
"""
from __future__ import annotations
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_CAPACITY_LOG = Path("logs/capacity_log.jsonl")

# Almgren-Chriss temporary impact coefficient (η)
_ETA = 0.142

# Default signal IC assumptions by sleeve (conservative estimates)
_DEFAULT_SLEEVE_IC = {
    "trend":           0.04,
    "statarb":         0.03,
    "ml":              0.03,
    "fundamental":     0.02,
    "llm_fundamental": 0.02,
    "earnings":        0.03,
    "meanrev":         0.02,
    "volatility":      0.02,
    "analyst":         0.02,
    "altdata":         0.02,
    "options_flow":    0.02,
    "insider":         0.02,
    "short_interest":  0.02,
}


def estimate_market_impact(
    trade_size_pct_adv: float,
    volatility: float,
    participation_rate: float = 0.1,
) -> float:
    """
    Almgren-Chriss temporary market impact in basis points.
    impact = η · σ · (trade_size / ADV)^0.6 · 10000
    """
    if trade_size_pct_adv <= 0 or volatility <= 0:
        return 0.0
    sigma_daily = volatility / math.sqrt(252)
    impact_bps  = _ETA * sigma_daily * (trade_size_pct_adv ** 0.6) * 10_000
    return round(impact_bps, 2)


def compute_signal_breakeven_adv(
    signal_ic: float,
    signal_ir: float = 0.5,
    holding_period_days: int = 5,
) -> float:
    """
    Minimum ADV fraction at which signal's gross alpha > market impact cost.
    Returns fraction (e.g. 0.02 = 2% of ADV).
    """
    if signal_ic <= 0:
        return float("inf")

    sigma_daily = 0.015  # ~24% annual vol → 1.5% daily
    # Grinold: expected alpha per holding period = IC * σ * sqrt(T) * sqrt(2/π)
    alpha_per_period = signal_ic * sigma_daily * math.sqrt(holding_period_days) * math.sqrt(2 / math.pi)
    alpha_bps = alpha_per_period * 10_000

    # Breakeven: eta * sigma * x^0.6 = alpha_bps  →  x = (alpha_bps / (eta * sigma))^(1/0.6)
    denominator = _ETA * sigma_daily * 10_000
    if denominator <= 0 or alpha_bps <= 0:
        return float("inf")

    breakeven_adv_frac = (alpha_bps / denominator) ** (1 / 0.6)
    return round(float(breakeven_adv_frac), 6)


def compute_strategy_capacity(
    portfolio_weights: dict[str, float],
    signal_ic_by_sleeve: Optional[dict[str, float]] = None,
    universe_avg_adv_dollars: float = 50_000_000,
) -> dict:
    """
    Estimate maximum NAV at which each sleeve's signal survives its own market impact.
    Returns {"by_sleeve": {sleeve: max_nav}, "binding_constraint": str, "overall_capacity": float}.
    """
    if signal_ic_by_sleeve is None:
        signal_ic_by_sleeve = _DEFAULT_SLEEVE_IC.copy()

    by_sleeve = {}
    for sleeve, ic in signal_ic_by_sleeve.items():
        breakeven_adv_frac = compute_signal_breakeven_adv(ic)
        if breakeven_adv_frac == float("inf") or breakeven_adv_frac <= 0:
            by_sleeve[sleeve] = None
            continue
        # Max position size = breakeven_adv_frac * ADV
        max_position_dollars = breakeven_adv_frac * universe_avg_adv_dollars
        # Max NAV = max_position / typical_weight (use 10% as conservative single-name weight)
        typical_weight = 0.10
        max_nav = max_position_dollars / typical_weight
        by_sleeve[sleeve] = round(max_nav, 0)

    valid_caps = {s: v for s, v in by_sleeve.items() if v is not None and v > 0}
    if not valid_caps:
        return {"by_sleeve": by_sleeve, "binding_constraint": "unknown", "overall_capacity": None}

    binding = min(valid_caps, key=valid_caps.get)
    overall = valid_caps[binding]

    return {
        "by_sleeve":          by_sleeve,
        "binding_constraint": binding,
        "overall_capacity":   overall,
    }


def capacity_report(
    prices_df: pd.DataFrame,
    portfolio_weights: dict[str, float],
    signal_ic_by_sleeve: Optional[dict[str, float]] = None,
) -> dict:
    """
    Run capacity model against current holdings. Logs to logs/capacity_log.jsonl.
    Informational only.
    """
    # Estimate universe average ADV from price data
    avg_adv_dollars = 50_000_000  # fallback
    if not prices_df.empty:
        try:
            recent = prices_df.tail(21)
            # Rough ADV: median daily range * price (not actual volume, but order-of-magnitude)
            last_prices = prices_df.iloc[-1]
            avg_price = float(last_prices.median())
            # Assume $50M ADV for median large-cap — override when real volume data available
            avg_adv_dollars = max(avg_price * 1_000_000, 10_000_000)
        except Exception:
            pass

    result = compute_strategy_capacity(portfolio_weights, signal_ic_by_sleeve, avg_adv_dollars)
    result["report_date"] = date.today().isoformat()
    result["avg_adv_dollars"] = avg_adv_dollars

    _CAPACITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_CAPACITY_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    log.info(
        "[CapacityModel] Overall capacity: $%.1fM | Binding: %s",
        (result.get("overall_capacity") or 0) / 1e6,
        result.get("binding_constraint"),
    )
    return result
