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
