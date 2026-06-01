"""ascent/causal/tracker.py

Causal predictions log writer and outcome tracker.

write_predictions() — called on rebalance day, writes one record per mechanism.
check_outcomes()    — called weekly, marks outcome for past-horizon predictions.
check_early_exits() — called daily (non-rebalance), returns symbols to cut early.
get_track_record()  — called at Phase 2 start, returns accuracy stats.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/causal_predictions.jsonl")

_EARLY_EXIT_CATALYST_THRESHOLD = -0.08   # -8% for catalyst_imminent
_EARLY_EXIT_NOTYET_THRESHOLD   = -0.05   # -5% + >70% horizon elapsed
_OUTCOME_CONFIRMED_THRESHOLD   = 0.05    # +5% → confirmed
_OUTCOME_FALSIFIED_THRESHOLD   = -0.05   # -5% → falsified


def write_predictions(
    mechanisms: list,
    rebalance_date: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Append one prediction record per CausalMechanism to the log file.

    Args:
        mechanisms: list of CausalMechanism objects
        rebalance_date: ISO date string for this rebalance
        log_path: path to jsonl log (default: DEFAULT_LOG_PATH)
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        for m in mechanisms:
            record = {
                "symbol": m.symbol,
                "mechanism": m.mechanism,
                "intervention": m.intervention,
                "falsification_condition": m.falsification_condition,
                "horizon_days": m.horizon_days,
                "rebalance_date": rebalance_date,
                "timing": m.timing,
                "velocity": m.velocity,
                "regime_compatible": m.regime_compatible,
                "outcome": "pending",
                "early_exit": False,
                "checked_date": None,
            }
            f.write(json.dumps(record) + "\n")

    log.info("[CausalTracker] Wrote %d predictions for rebalance %s", len(mechanisms), rebalance_date)


def _read_records(log_path: Path) -> list:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text().strip().split("\n"):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _write_records(records: list, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=log_path.parent, suffix=".tmp"))
    tmp.write_text("\n".join(json.dumps(r) for r in records))
    os.replace(tmp, log_path)


def _get_price_return(symbol: str, from_date: str) -> float:
    """Compute price return for symbol from from_date to today. Returns 0.0 if unavailable."""
    try:
        import pandas as pd
        pp = Path("data_cache/prices_live.parquet")
        if not pp.exists():
            return 0.0
        prices = pd.read_parquet(pp)
        if "symbol" in prices.columns:
            sym_prices = prices[prices["symbol"] == symbol].copy()
            sym_prices["date"] = pd.to_datetime(sym_prices["date"])
            sym_prices = sym_prices.sort_values("date")
        else:
            cols = [c for c in prices.columns if symbol in str(c)]
            if not cols:
                return 0.0
            sym_prices = prices[cols].copy()

        from_dt = pd.to_datetime(from_date)
        if "date" in sym_prices.columns:
            after = sym_prices[sym_prices["date"] >= from_dt]
        else:
            after = sym_prices

        if len(after) < 2:
            return 0.0

        close_col = "close" if "close" in after.columns else after.columns[-1]
        start_price = float(after.iloc[0][close_col])
        end_price   = float(after.iloc[-1][close_col])
        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price
    except Exception as exc:
        log.debug("[CausalTracker] Price fetch failed for %s: %s", symbol, exc)
        return 0.0


def check_early_exits(log_path: Optional[Path] = None) -> List[str]:
    """
    Check all pending predictions for early exit conditions.
    Updates early_exit flag in the log file in-place.

    Returns:
        List of symbols with active early_exit flags.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    today = date.today()
    early_exit_symbols = []
    changed = False

    for r in records:
        if r.get("outcome") != "pending":
            continue
        if r.get("early_exit"):
            early_exit_symbols.append(r["symbol"])
            continue

        symbol = r["symbol"]
        timing = r.get("timing", "not_yet_priced")
        rebalance_date = r.get("rebalance_date", str(today))
        horizon_days = int(r.get("horizon_days", 63))

        price_return = _get_price_return(symbol, rebalance_date)

        should_exit = False
        if timing == "catalyst_imminent":
            should_exit = price_return < _EARLY_EXIT_CATALYST_THRESHOLD
        elif timing == "not_yet_priced":
            rebalance_dt = date.fromisoformat(rebalance_date)
            elapsed_days = (today - rebalance_dt).days
            elapsed_fraction = elapsed_days / max(horizon_days, 1)
            should_exit = (elapsed_fraction > 0.70 and price_return < _EARLY_EXIT_NOTYET_THRESHOLD)

        if should_exit:
            r["early_exit"] = True
            r["checked_date"] = str(today)
            changed = True
            early_exit_symbols.append(symbol)
            log.info(
                "[CausalTracker] Early exit flagged: %s (timing=%s, return=%.1f%%)",
                symbol, timing, price_return * 100,
            )

    if changed:
        _write_records(records, Path(log_path))

    return list(set(early_exit_symbols))


def check_outcomes(log_path: Optional[Path] = None) -> None:
    """
    For all past-horizon pending predictions, classify outcome as
    'confirmed', 'falsified', or 'neutral' based on realized price return.
    Updates the log file in place.
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    today = date.today()
    changed = False

    for r in records:
        if r.get("outcome") != "pending":
            continue

        rebalance_date = r.get("rebalance_date", str(today))
        horizon_days = int(r.get("horizon_days", 63))
        rebalance_dt = date.fromisoformat(rebalance_date)
        if (today - rebalance_dt).days < horizon_days + 5:
            continue

        symbol = r["symbol"]
        price_return = _get_price_return(symbol, rebalance_date)
        if price_return >= _OUTCOME_CONFIRMED_THRESHOLD:
            r["outcome"] = "confirmed"
        elif price_return <= _OUTCOME_FALSIFIED_THRESHOLD:
            r["outcome"] = "falsified"
        else:
            r["outcome"] = "neutral"
        r["checked_date"] = str(today)
        changed = True
        log.info(
            "[CausalTracker] Outcome for %s: %s (return=%.1f%%)",
            symbol, r["outcome"], price_return * 100,
        )

    if changed:
        _write_records(records, Path(log_path))


def get_track_record(log_path: Optional[Path] = None) -> dict:
    """
    Return accuracy statistics for all resolved predictions.

    Returns:
        {total, confirmed, falsified, neutral, pending, accuracy_pct}
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    records = _read_records(Path(log_path))
    counts: dict = {"confirmed": 0, "falsified": 0, "neutral": 0, "pending": 0}
    for r in records:
        outcome = r.get("outcome", "pending")
        counts[outcome] = counts.get(outcome, 0) + 1

    resolved = counts["confirmed"] + counts["falsified"] + counts["neutral"]
    accuracy_pct = (
        round(counts["confirmed"] / (counts["confirmed"] + counts["falsified"]) * 100, 1)
        if (counts["confirmed"] + counts["falsified"]) > 0 else 0.0
    )
    return {
        "total": resolved,
        "confirmed": counts["confirmed"],
        "falsified": counts["falsified"],
        "neutral": counts["neutral"],
        "pending": counts["pending"],
        "accuracy_pct": accuracy_pct,
    }
