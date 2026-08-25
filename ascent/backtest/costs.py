"""
Ascent Capital — Transaction Cost Model
Realistic cost estimation for backtesting.

Cost components:
1. Spread cost: half bid-ask spread per side
2. Market impact: proportional to sqrt(participation rate) × volatility
3. Commission: per-share fee (often $0 for modern brokers)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import math


def estimate_trade_cost(
    price: float,
    shares: int,
    daily_volume: float,
    volatility: float,
    spread_bps: float = 5.0,
    impact_mult: float = 0.1,
    commission_per_share: float = 0.0,
) -> dict:
    """
    Estimate total cost for a single trade.

    Args:
        price: Current price
        shares: Number of shares to trade
        daily_volume: Average daily volume
        volatility: Daily return volatility
        spread_bps: Half-spread in basis points
        impact_mult: Market impact multiplier
        commission_per_share: Commission per share

    Returns:
        Dict with cost breakdown
    """
    if shares == 0 or price <= 0:
        return {"total_cost": 0.0, "spread_cost": 0.0, "impact_cost": 0.0, "commission": 0.0}

    notional = abs(shares) * price

    # Spread cost
    spread_cost = notional * (spread_bps / 10_000)

    # Market impact (simplified Almgren-Chriss)
    if daily_volume > 0:
        participation = abs(shares) / daily_volume
        impact_cost = notional * impact_mult * volatility * math.sqrt(participation)
    else:
        impact_cost = notional * 0.001  # fallback: 10bps

    # Commission
    commission = abs(shares) * commission_per_share

    total = spread_cost + impact_cost + commission

    return {
        "total_cost": total,
        "total_cost_bps": (total / notional * 10_000) if notional > 0 else 0,
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "commission": commission,
        "notional": notional,
    }


def estimate_rebalance_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    volumes: pd.Series,
    volatilities: pd.Series,
    spread_bps: float = 5.0,
    impact_mult: float = 0.1,
) -> dict:
    """
    Estimate total cost of rebalancing from old weights to new weights.
    """
    all_symbols = old_weights.index.union(new_weights.index)
    total_cost = 0.0
    trade_details = []

    for sym in all_symbols:
        old_w = old_weights.get(sym, 0.0)
        new_w = new_weights.get(sym, 0.0)
        delta_w = new_w - old_w

        if abs(delta_w) < 1e-6:
            continue

        trade_notional = abs(delta_w) * portfolio_value
        price = prices.get(sym, 100)
        vol = volumes.get(sym, 1e6)
        daily_vol = volatilities.get(sym, 0.02)

        shares = int(trade_notional / price) if price > 0 else 0

        cost = estimate_trade_cost(price, shares, vol, daily_vol, spread_bps, impact_mult)
        total_cost += cost["total_cost"]

        trade_details.append({
            "symbol": sym,
            "old_weight": old_w,
            "new_weight": new_w,
            "delta_weight": delta_w,
            "shares": shares,
            **cost,
        })

    return {
        "total_cost": total_cost,
        "total_cost_pct": total_cost / portfolio_value if portfolio_value > 0 else 0,
        "n_trades": len(trade_details),
        "trades": trade_details,
    }


def flat_cost_model(turnover_fraction: float, cost_bps: float = 10.0) -> float:
    """
    Simple flat cost model: total cost = turnover × cost_per_unit.
    Returns cost as a fraction of portfolio.
    """
    return turnover_fraction * (cost_bps / 10_000)


def liquidity_scaled_cost_model(
    delta_weights: pd.Series,
    portfolio_value: float,
    adv_dollar: "pd.Series | None",
    spread_bps: float = 5.0,
    impact_bps: float = 5.0,
    impact_floor_mult: float = 0.1,
    impact_ceil_mult: float = 10.0,
) -> float:
    """
    Per-symbol, ADV-scaled generalization of flat_cost_model.

    Relationship to ascent/execution/cost_model.py::estimate() (Almgren-Chriss,
    live pre-trade order sizing/blocking): these are deliberately two separate
    formulas, not a bug. This function is a fast, vectorized, whole-book
    backtest approximation — it runs once per rebalance across every symbol at
    once inside BacktestEngine.run(), and only needs a spread/impact bps rate
    and an ADV series (no volatility series is currently plumbed through the
    engine or the walk-forward runners that call it). cost_model.estimate()
    is a precise, single-order, live pre-trade estimate that additionally
    requires a per-symbol annualized-volatility feature and a permanent-impact
    term, and is used to actually block/flag orders in order_engine.py before
    they hit the broker. Consolidating them for real would mean plumbing a
    volatility panel through BacktestEngine.run() and every walk-forward
    caller — out of scope here. Because of the different inputs (this model
    has no volatility term) the two produce different bps estimates for the
    same hypothetical trade; a scratch comparison across 0.2%-15% participation
    (2026-08-24) showed this backtest model consistently *understating* cost
    vs. the live Almgren-Chriss estimate — roughly 5.5 vs 6.6bps round-trip at
    0.2% participation, widening to ~6.9 vs ~19.6bps at 15% participation.
    That gap is a real, measured limitation of the backtest cost estimate
    (it will not fully capture impact cost at high participation), not a
    silent one — see ascent/execution/cost_model.py's module docstring for
    the reverse cross-reference.

    Spread cost stays flat per CLAUDE.md guidance (less size-dependent).
    Impact cost scales with sqrt(trade_notional / adv_dollar) — a standard
    square-root impact model — per symbol, then the whole book's cost is the
    turnover-weighted average, exactly like flat_cost_model:

        cost_rate = sum_sym(|delta_w_sym| * (spread_bps + impact_bps_eff_sym)) / 2 / 10_000

    which collapses to `flat_cost_model(turnover, spread_bps + impact_bps)`
    when impact_bps_eff_sym == impact_bps for every symbol — i.e. when
    `adv_dollar` is None, or a symbol's ADV entry is missing/NaN/<=0, that
    symbol falls back to the flat impact_bps untouched. This is what keeps
    callers that don't pass volume data byte-identical to today's behavior.

    Args:
        delta_weights: new_weight - old_weight per symbol for this rebalance.
        portfolio_value: portfolio value in dollars, to size the trade notional.
        adv_dollar: average daily dollar volume per symbol (already lagged /
            point-in-time by the caller — this function does no shifting).
            None disables scaling entirely (flat behavior).
        spread_bps: half-spread, applied flat (unscaled).
        impact_bps: base impact rate; scaled per symbol by sqrt(participation).
        impact_floor_mult / impact_ceil_mult: clip the *effective* impact_bps
            to [impact_bps * floor, impact_bps * ceil] so a near-zero ADV
            can't blow the cost up to infinity, and a tiny trade can't drive
            it to (numerically) zero.

    Returns:
        Cost as a fraction of portfolio value (same units as flat_cost_model).
    """
    abs_delta = delta_weights.abs()

    if adv_dollar is None:
        total_bps = spread_bps + impact_bps
        return float(abs_delta.sum() / 2 * (total_bps / 10_000))

    adv = adv_dollar.reindex(delta_weights.index)
    trade_notional = abs_delta * float(portfolio_value)

    no_adv = adv.isna() | (adv <= 0)
    # Avoid 0/0 and negative-ADV warnings; overwritten by the flat fallback below.
    safe_adv = adv.where(~no_adv, 1.0)
    participation = (trade_notional / safe_adv).clip(lower=0.0)
    impact_eff = impact_bps * np.sqrt(participation)
    impact_eff = impact_eff.clip(
        lower=impact_bps * impact_floor_mult,
        upper=impact_bps * impact_ceil_mult,
    )
    impact_eff = impact_eff.where(~no_adv, impact_bps)  # flat fallback per symbol

    cost_bps_per_symbol = spread_bps + impact_eff
    cost_rate = float((abs_delta * cost_bps_per_symbol).sum() / 2 / 10_000)
    return cost_rate
