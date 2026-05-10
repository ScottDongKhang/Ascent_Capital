"""ascent/execution/twap_executor.py

TWAP execution engine.
Splits large orders into equally-spaced child limit orders.
TWAP_ENABLED = False by default — paper-mode testing first.
"""
from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# Kill switch — enable after paper-mode validation
TWAP_ENABLED = False

# Almgren-Chriss temporary impact coefficient
_ETA   = 0.142
_GAMMA = 0.05   # permanent impact coefficient

# Execution window bounds (minutes)
_MIN_WINDOW_URGENT  = 5
_MAX_WINDOW_NORMAL  = 90
_MAX_WINDOW_HIGH    = 15
_MAX_WINDOW_LOW     = 90

# 5% of ADV threshold for TWAP routing
TWAP_ADV_THRESHOLD = 0.05


def should_use_twap(
    trade_size_dollars: float,
    adv_dollars: float,
    threshold_pct: float = TWAP_ADV_THRESHOLD,
) -> bool:
    """True when trade_size_dollars > threshold_pct * adv_dollars."""
    if adv_dollars <= 0:
        return False
    return (trade_size_dollars / adv_dollars) > threshold_pct


def compute_twap_window(
    adv: float,
    trade_size_shares: float,
    urgency: str = "normal",
) -> int:
    """
    Compute optimal TWAP execution window in minutes using Almgren-Chriss.
    T* = sqrt(trade_size / (2 * eta * sigma^2)) — bounded by urgency limits.
    """
    if adv <= 0 or trade_size_shares <= 0:
        return _MIN_WINDOW_URGENT

    try:
        # Simplified: participation_rate = trade_size / adv (shares as fraction)
        sigma_daily = 0.015  # ~24% annual vol → 1.5% daily
        participation = trade_size_shares / max(adv, 1.0)
        # Almgren-Chriss optimal time (in "periods", scaled to minutes)
        t_star = math.sqrt(participation / (2 * _ETA * sigma_daily ** 2))
        window_minutes = int(min(t_star * 30, _MAX_WINDOW_NORMAL))  # scale to minutes
        window_minutes = max(window_minutes, _MIN_WINDOW_URGENT)
    except Exception:
        window_minutes = 30

    urgency_lower = str(urgency).lower()
    if urgency_lower == "high":
        window_minutes = min(window_minutes, _MAX_WINDOW_HIGH)
    elif urgency_lower == "low":
        window_minutes = min(window_minutes, _MAX_WINDOW_LOW)
    else:
        window_minutes = min(window_minutes, _MAX_WINDOW_NORMAL)

    return window_minutes


def build_twap_schedule(
    symbol: str,
    total_shares: int,
    side: str,
    start_time: datetime,
    window_minutes: int = 60,
    n_slices: int = 6,
) -> list[dict]:
    """
    Build a TWAP schedule: equally-spaced child orders over window_minutes.
    Minimum slice size is 1 share.
    Returns list of {"symbol", "shares", "side", "submit_at"} dicts.
    """
    total_shares = max(1, int(total_shares))
    # If n_slices would produce <1 share per slice, reduce to one-share-per-slice
    if total_shares < n_slices:
        n_slices = total_shares

    base_shares  = total_shares // n_slices
    remainder    = total_shares - base_shares * n_slices
    interval_min = window_minutes / n_slices if n_slices > 0 else window_minutes

    schedule = []
    for i in range(n_slices):
        submit_at = start_time + timedelta(minutes=i * interval_min)
        # Add remainder to the last slice
        shares = base_shares + (remainder if i == n_slices - 1 else 0)
        schedule.append({
            "symbol":    symbol,
            "shares":    shares,
            "side":      side,
            "submit_at": submit_at,
        })
    return schedule


def execute_twap(
    symbol: str,
    total_dollars: float,
    side: str,
    adv: float,
    price: float,
    urgency: str = "normal",
    _alpaca_api=None,
) -> list[dict]:
    """
    Execute a TWAP order by submitting child limit orders.
    When TWAP_ENABLED=False, logs the plan but submits nothing.
    Returns list of fill dicts.
    """
    if price <= 0:
        log.warning("[TWAP] %s: invalid price %.2f — skipping", symbol, price)
        return []

    total_shares = max(1, int(total_dollars / price))
    n_slices     = 6
    window       = compute_twap_window(adv / price if price > 0 else 0, total_shares, urgency)
    schedule     = build_twap_schedule(symbol, total_shares, side, datetime.now(), window, n_slices)

    if not TWAP_ENABLED:
        log.info(
            "[TWAP] TWAP_ENABLED=False — logging plan only: %s %s %d shares "
            "over %d min in %d slices",
            side, symbol, total_shares, window, len(schedule),
        )
        return [{"status": "disabled", "symbol": symbol, "total_shares": total_shares}]

    fills = []
    remaining = total_shares
    for i, slice_order in enumerate(schedule):
        if remaining <= 0:
            break
        shares = min(slice_order["shares"], remaining)
        try:
            if _alpaca_api is None:
                import alpaca_trade_api as tradeapi
                from ascent.config.settings import APIKeys
                keys = APIKeys.from_env()
                _alpaca_api = tradeapi.REST(
                    keys.alpaca_key, keys.alpaca_secret,
                    base_url="https://paper-api.alpaca.markets",
                )
            quote = _alpaca_api.get_latest_quote(symbol)
            if side == "buy":
                limit_price = round(float(quote.ap) + 0.01, 2)
            else:
                limit_price = round(float(quote.bp) - 0.01, 2)

            order = _alpaca_api.submit_order(
                symbol=symbol, qty=shares, side=side,
                type="limit", time_in_force="day",
                limit_price=str(limit_price),
            )
            fills.append({
                "status":      "submitted",
                "slice":       i,
                "order_id":    order.id,
                "symbol":      symbol,
                "shares":      shares,
                "limit_price": limit_price,
                "fill_price":  None,
            })
            remaining -= shares
        except Exception as e:
            log.warning("[TWAP] slice %d for %s failed: %s — rolling to next slice", i, symbol, e)

    return fills
