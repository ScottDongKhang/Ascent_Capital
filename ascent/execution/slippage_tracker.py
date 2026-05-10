"""
ascent/execution/slippage_tracker.py
Post-execution slippage measurement.

Queries Alpaca for filled orders from today's session,
compares fill prices against the signal close price from the prior day,
and writes per-trade slippage records to logs/slippage_log.jsonl.

Called by eod_runner.py after order submission on rebalance days.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ascent.execution.implementation_shortfall import compute_is, _DECISION_PRICES
except Exception:
    compute_is       = None  # type: ignore[assignment]
    _DECISION_PRICES = {}    # type: ignore[assignment]

SLIPPAGE_LOG_PATH = Path("logs/slippage_log.jsonl")


# ── Alpaca order query ────────────────────────────────────────────────────────

def _get_todays_fills(after_date: Optional[str] = None) -> List[dict]:
    """
    Query Alpaca for today's filled orders.

    Uses the same ALPACA_BASE_URL and _headers() pattern as alpaca_broker.py.
    Imports from alpaca_broker to avoid duplicating auth logic.

    Returns list of filled order dicts, each with:
        symbol, filled_avg_price, filled_qty, side, filled_at
    """
    try:
        from ascent.execution.alpaca_broker import ALPACA_BASE_URL, _headers
        import requests

        # Default to today's date if not specified
        after = after_date or datetime.now().strftime("%Y-%m-%dT00:00:00Z")

        r = requests.get(
            f"{ALPACA_BASE_URL}/orders",
            headers=_headers(),
            params={
                "status": "filled",
                "after":  after,
                "limit":  100,
                "direction": "desc",
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[Slippage] Alpaca orders query returned {r.status_code} — skipping")
            return []
        return r.json() if isinstance(r.json(), list) else []

    except Exception as e:
        print(f"[Slippage] Failed to query Alpaca orders ({e}) — skipping slippage tracking")
        return []


# ── Core computation ──────────────────────────────────────────────────────────

def compute_slippage(
    fills: List[dict],
    signal_prices: Dict[str, float],
    run_date: date,
) -> List[dict]:
    """
    Compare fill prices against signal prices and compute slippage in bps.

    Slippage convention:
        buys:  positive bps = paid more than signal price (bad)
        sells: positive bps = received less than signal price (bad)

    Args:
        fills:         List of Alpaca filled order dicts
        signal_prices: Dict of {symbol: close_price} from the signal date (prior day)
        run_date:      The EOD run date

    Returns:
        List of slippage records, each a dict with:
            run_date, symbol, side, qty, signal_price, fill_price, slippage_bps, filled_at
    """
    records = []
    for fill in fills:
        sym = fill.get("symbol")
        if not sym:
            continue

        fill_price_raw = fill.get("filled_avg_price")
        if fill_price_raw is None:
            continue
        fill_price = float(fill_price_raw)
        if fill_price == 0:
            continue

        side = fill.get("side", "unknown")
        qty  = float(fill.get("filled_qty", fill.get("qty", 0)))

        if sym not in signal_prices:
            continue

        signal_price = signal_prices[sym]
        if signal_price == 0:
            continue

        # Positive slippage = worse execution than signal price
        if side == "buy":
            slippage_bps = (fill_price - signal_price) / signal_price * 10_000
        else:
            slippage_bps = (signal_price - fill_price) / signal_price * 10_000

        record = {
            "run_date":     run_date.isoformat(),
            "symbol":       sym,
            "side":         side,
            "qty":          qty,
            "signal_price": round(signal_price, 4),
            "fill_price":   round(fill_price, 4),
            "slippage_bps": round(slippage_bps, 2),
            "filled_at":    fill.get("filled_at"),
        }
        # Add IS decomposition if decision price was recorded
        if compute_is is not None and sym in _DECISION_PRICES:
            try:
                is_result = compute_is(
                    symbol=sym,
                    fill_records=[{"fill_price": fill_price, "shares": int(qty)}],
                    eod_price=fill_price,
                    side=side,
                )
                record["is_breakdown"] = is_result
            except Exception:
                pass
        records.append(record)

    return records


# ── Logging ───────────────────────────────────────────────────────────────────

def log_slippage(records: List[dict]) -> None:
    """Append slippage records to the JSONL log."""
    os.makedirs(SLIPPAGE_LOG_PATH.parent, exist_ok=True)
    with open(SLIPPAGE_LOG_PATH, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[Slippage] Logged {len(records)} trades to {SLIPPAGE_LOG_PATH}")


def aggregate_slippage(records: List[dict]) -> dict:
    """
    Compute aggregate slippage stats for a single EOD run.
    Returns dict with: mean_bps, median_bps, max_bps, n_trades
    """
    if not records:
        return {"mean_bps": 0.0, "median_bps": 0.0, "max_bps": 0.0, "n_trades": 0}

    bps_list = sorted(r["slippage_bps"] for r in records)
    n = len(bps_list)

    return {
        "mean_bps":   round(sum(bps_list) / n, 2),
        "median_bps": round(bps_list[n // 2], 2),
        "max_bps":    round(max(bps_list), 2),
        "n_trades":   n,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def track_slippage(
    signal_prices: Dict[str, float],
    run_date: Optional[date] = None,
):
    """
    Main entry point — called by eod_runner.py after order fills.

    Args:
        signal_prices: Dict of {symbol: close_price} from the signal date.
                       The EOD runner has this from the pipeline's price data.
        run_date:      Override for testing. Defaults to today.
    """
    run_date = run_date or date.today()

    # Query orders filled today
    after_str = datetime.combine(run_date, datetime.min.time()).strftime("%Y-%m-%dT00:00:00Z")
    fills = _get_todays_fills(after_date=after_str)

    if not fills:
        print("[Slippage] No fills found for today — skipping slippage tracking")
        return

    records = compute_slippage(fills, signal_prices, run_date)
    if records:
        log_slippage(records)
        agg = aggregate_slippage(records)
        print(
            f"[Slippage] {agg['n_trades']} trades | "
            f"mean={agg['mean_bps']}bps | "
            f"median={agg['median_bps']}bps | "
            f"max={agg['max_bps']}bps"
        )
    else:
        print("[Slippage] No matched fills against signal prices")


def fill_quality_report(lookback_days: int = 63) -> dict:
    """
    Summarize fill quality over trailing lookback_days.
    Returns mean IS components, TWAP vs market order comparison, and by-sleeve breakdown.
    """
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=lookback_days)).date().isoformat()

    records = []
    if SLIPPAGE_LOG_PATH.exists():
        with open(SLIPPAGE_LOG_PATH) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("run_date", "") >= cutoff:
                        records.append(row)
                except Exception:
                    pass

    if not records:
        return {
            "n_trades": 0, "mean_is_bps": None, "mean_delay_cost_bps": None,
            "mean_market_impact_bps": None, "mean_opportunity_cost_bps": None,
            "twap_vs_market_order_comparison": {}, "by_sleeve": {},
        }

    def _safe_mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 2) if v else None

    bps_vals      = [r.get("slippage_bps") for r in records]
    is_total      = [r.get("is_breakdown", {}).get("total_is_bps") for r in records]
    is_delay      = [r.get("is_breakdown", {}).get("delay_cost_bps") for r in records]
    is_impact     = [r.get("is_breakdown", {}).get("market_impact_bps") for r in records]
    is_opp        = [r.get("is_breakdown", {}).get("opportunity_cost_bps") for r in records]

    return {
        "n_trades":                      len(records),
        "mean_slippage_bps":             _safe_mean(bps_vals),
        "mean_is_bps":                   _safe_mean(is_total),
        "mean_delay_cost_bps":           _safe_mean(is_delay),
        "mean_market_impact_bps":        _safe_mean(is_impact),
        "mean_opportunity_cost_bps":     _safe_mean(is_opp),
        "twap_vs_market_order_comparison": {},  # populated when TWAP live
        "by_sleeve":                     {},    # populated when sleeve tags added to log
    }
