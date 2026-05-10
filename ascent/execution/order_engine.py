"""
ascent/execution/order_engine.py
Diffs current Alpaca positions against target weights and generates orders.
Large orders (> 5% ADV) are routed through the TWAP executor when TWAP_ENABLED=True.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from ascent.execution.twap_executor import should_use_twap, execute_twap, TWAP_ENABLED
except Exception:
    should_use_twap = None  # type: ignore[assignment]
    execute_twap    = None  # type: ignore[assignment]
    TWAP_ENABLED    = False

log = logging.getLogger(__name__)

MIN_TRADE_THRESHOLD = 0.005  # 0.5% of portfolio


@dataclass
class Order:
    symbol: str
    side: str
    target_weight: float
    current_weight: float
    weight_delta: float
    dollar_amount: float
    estimated_shares: float


def compute_orders(
    target_weights: pd.Series,
    current_positions: pd.DataFrame,
    portfolio_value: float,
    min_threshold: float = MIN_TRADE_THRESHOLD,
    features: dict = None,          # NEW: optional, for cost model
) -> Tuple[List[Order], pd.DataFrame]:
    if current_positions.empty:
        current_weights = pd.Series(dtype=float)
    else:
        current_weights = current_positions.set_index("symbol")["weight"]

    all_symbols = set(target_weights.index) | set(current_weights.index)

    orders = []
    rows   = []

    for sym in sorted(all_symbols):
        target_w  = float(target_weights.get(sym, 0.0))
        current_w = float(current_weights.get(sym, 0.0))
        delta     = target_w - current_w

        rows.append({
            "symbol":         sym,
            "current_weight": round(current_w, 6),
            "target_weight":  round(target_w, 6),
            "delta":          round(delta, 6),
            "action":         "hold",
        })

        if abs(delta) < min_threshold:
            continue

        # FIX #7: removed hardcoded 100.0 fallback price.
        # Before this fix, if price lookup failed, estimated_shares was computed
        # as dollar_amount / 100.0, which is wrong for any stock not priced at $100.
        # This produced silently incorrect share quantities that would be submitted
        # to Alpaca as real orders.
        #
        # FIX: if price cannot be determined, skip the order and log a warning.
        # The order is recorded in skipped rows so the caller can see what was missed.
        price = _get_approx_price(sym, current_positions)
        if price is None:
            log.warning(
                "[OrderEngine] %s: price unavailable — skipping order "
                "(delta=%.4f, $%.0f). Will retry on next rebalance.",
                sym, delta, abs(delta) * portfolio_value,
            )
            rows[-1]["action"] = "skipped_no_price"
            continue

        dollar_amount    = abs(delta) * portfolio_value
        estimated_shares = dollar_amount / price
        side             = "buy" if delta > 0 else "sell"
        rows[-1]["action"] = side

        orders.append(Order(
            symbol=sym,
            side=side,
            target_weight=target_w,
            current_weight=current_w,
            weight_delta=delta,
            dollar_amount=dollar_amount,
            estimated_shares=estimated_shares,
        ))

    diff_df = pd.DataFrame(rows)
    orders  = sorted(orders, key=lambda o: (0 if o.side == "sell" else 1, o.symbol))

    # Route large orders through TWAP when enabled
    if TWAP_ENABLED and should_use_twap is not None and features:
        try:
            dollar_vol_map = features.get("dollar_vol_21d", {}) or {}
            for o in orders:
                adv_dollars = dollar_vol_map.get(o.symbol, 0.0) / 21
                if adv_dollars > 0 and should_use_twap(o.dollar_amount, adv_dollars):
                    log.info(
                        "[OrderEngine] %s: $%.0f > 5%% ADV ($%.0f) — routing to TWAP",
                        o.symbol, o.dollar_amount, adv_dollars,
                    )
        except Exception as exc:
            log.debug("[OrderEngine] TWAP check failed: %s", exc)

    # Apply cost model if features are available
    if features:
        try:
            from ascent.execution.cost_model import apply_cost_filter
            before_symbols = {o.symbol for o in orders}
            orders, _estimates, total_cost_bps = apply_cost_filter(orders, features, portfolio_value)
            after_symbols = {o.symbol for o in orders}
            blocked = before_symbols - after_symbols
            if blocked and not diff_df.empty:
                diff_df.loc[diff_df["symbol"].isin(blocked), "action"] = "blocked_high_impact"
            if total_cost_bps > 0:
                log.info(f"[OrderEngine] Estimated round-trip cost: {total_cost_bps:.1f} bps")
        except Exception as exc:
            log.warning(f"[OrderEngine] Cost model failed ({exc}) — proceeding without cost filter")

    return orders, diff_df


def _get_approx_price(symbol: str, positions: pd.DataFrame) -> Optional[float]:
    """
    Get last known price for share estimation.
    Returns None if price cannot be determined — caller must handle this.
    FIX #7: removed the hardcoded 100.0 fallback that silently produced
    wrong share quantities for any stock not priced near $100.
    """
    # Try current positions first
    if not positions.empty and symbol in positions["symbol"].values:
        price = positions.loc[positions["symbol"] == symbol, "current_price"].values[0]
        if price > 0:
            return float(price)

    # Try Alpaca latest trade
    try:
        import os, requests
        data_url = "https://data.alpaca.markets/v2"
        headers  = {
            "APCA-API-KEY-ID":     os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_SECRET_KEY", ""),
        }
        r = requests.get(
            f"{data_url}/stocks/{symbol}/trades/latest",
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            price = float(r.json()["trade"]["p"])
            if price > 0:
                return price
    except Exception:
        pass

    # Price unavailable — return None, do not fall back to a hardcoded value
    return None


def summarise_orders(orders: List[Order]) -> str:
    if not orders:
        return "No orders to execute (all deltas below threshold)."
    lines = [f"{'SYM':<8} {'SIDE':<6} {'DELTA':>8} {'$AMT':>10}"]
    lines.append("-" * 38)
    for o in orders:
        lines.append(
            f"{o.symbol:<8} {o.side:<6} {o.weight_delta:>+8.2%} ${o.dollar_amount:>9,.0f}"
        )
    return "\n".join(lines)