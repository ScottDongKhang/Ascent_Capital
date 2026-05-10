"""ascent/execution/implementation_shortfall.py

Implementation shortfall (IS) decomposition.

Three components:
  delay_cost      = (arrival_price - decision_price) * direction
  market_impact   = (fill_price - arrival_price) * direction
  opportunity_cost= shares_unfilled * (eod_price - decision_price) * direction

All costs reported in basis points relative to decision price.
"""
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SLIPPAGE_LOG = Path("logs/slippage_log.jsonl")

# In-memory store for intra-day decision/arrival prices
_DECISION_PRICES: dict[str, dict] = {}
_ARRIVAL_PRICES:  dict[str, dict] = {}


def record_decision_price(symbol: str, decision_price: float, decision_time=None) -> None:
    """Record the price at signal generation time (before orders are built)."""
    from datetime import datetime
    _DECISION_PRICES[symbol] = {
        "price": float(decision_price),
        "time":  (decision_time or datetime.now()).isoformat(),
    }


def record_arrival_price(symbol: str, arrival_price: float, arrival_time=None) -> None:
    """Record the price when the first child order is submitted."""
    from datetime import datetime
    _ARRIVAL_PRICES[symbol] = {
        "price": float(arrival_price),
        "time":  (arrival_time or datetime.now()).isoformat(),
    }


def compute_is(
    symbol: str,
    fill_records: list[dict],
    eod_price: float,
    decision_price: Optional[float] = None,
    arrival_price: Optional[float] = None,
    side: str = "buy",
) -> dict:
    """
    Compute IS decomposition for a symbol.

    Args:
        symbol:         Ticker symbol
        fill_records:   List of {"fill_price": float, "shares": int} dicts
        eod_price:      End-of-day price for opportunity cost
        decision_price: Price at signal generation (overrides in-memory store)
        arrival_price:  Price at order submission (overrides in-memory store)
        side:           "buy" or "sell"

    Returns dict with IS components in basis points.
    """
    # Resolve prices
    if decision_price is None:
        decision_price = _DECISION_PRICES.get(symbol, {}).get("price")
    if arrival_price is None:
        arrival_price = _ARRIVAL_PRICES.get(symbol, {}).get("price")

    if decision_price is None or decision_price == 0:
        return {
            "symbol": symbol, "decision_price": None, "arrival_price": None,
            "avg_fill_price": None, "eod_price": eod_price,
            "delay_cost_bps": None, "market_impact_bps": None,
            "opportunity_cost_bps": None, "total_is_bps": None,
        }

    if arrival_price is None:
        arrival_price = decision_price  # no delay if no arrival recorded

    # Compute average fill price
    total_shares = sum(r.get("shares", 1) for r in fill_records)
    if total_shares == 0 or not fill_records:
        avg_fill_price = arrival_price
        filled_shares  = 0
    else:
        avg_fill_price = sum(r["fill_price"] * r.get("shares", 1) for r in fill_records) / total_shares
        filled_shares  = total_shares

    direction = 1.0 if side == "buy" else -1.0

    # IS components in basis points relative to decision_price
    delay_bps    = (arrival_price - decision_price) / decision_price * 10_000 * direction
    impact_bps   = (avg_fill_price - arrival_price) / decision_price * 10_000 * direction
    opp_bps      = 0.0  # simplified: no unfilled shares tracked here
    total_is_bps = delay_bps + impact_bps + opp_bps

    return {
        "symbol":               symbol,
        "decision_price":       round(decision_price, 4),
        "arrival_price":        round(arrival_price, 4),
        "avg_fill_price":       round(avg_fill_price, 4),
        "eod_price":            round(float(eod_price), 4),
        "delay_cost_bps":       round(delay_bps, 2),
        "market_impact_bps":    round(impact_bps, 2),
        "opportunity_cost_bps": round(opp_bps, 2),
        "total_is_bps":         round(total_is_bps, 2),
    }


def log_is_record(is_dict: dict, rebalance_date: date) -> None:
    """Append IS record to slippage_log.jsonl with is_breakdown sub-dict. Backwards-compatible."""
    SLIPPAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_date":    rebalance_date.isoformat(),
        "symbol":      is_dict.get("symbol"),
        "is_breakdown": is_dict,
        "total_is_bps": is_dict.get("total_is_bps"),
    }
    with open(SLIPPAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def is_summary(lookback_days: int = 63) -> dict:
    """
    Compute mean and std of each IS component over trailing lookback_days.
    Reads logs/slippage_log.jsonl.
    """
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=lookback_days)
    records = []
    if SLIPPAGE_LOG.exists():
        with open(SLIPPAGE_LOG) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "is_breakdown" not in row:
                        continue
                    if row.get("run_date", "") < cutoff.isoformat():
                        continue
                    records.append(row["is_breakdown"])
                except Exception:
                    pass

    if not records:
        return {"n_records": 0, "mean_total_is_bps": None, "mean_delay_cost_bps": None,
                "mean_market_impact_bps": None, "mean_opportunity_cost_bps": None}

    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "n_records":                   len(records),
        "mean_total_is_bps":           _mean("total_is_bps"),
        "mean_delay_cost_bps":         _mean("delay_cost_bps"),
        "mean_market_impact_bps":      _mean("market_impact_bps"),
        "mean_opportunity_cost_bps":   _mean("opportunity_cost_bps"),
    }
