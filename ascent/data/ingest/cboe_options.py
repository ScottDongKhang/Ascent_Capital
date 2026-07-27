# ascent/data/ingest/cboe_options.py
"""
Historical CBOE options data — IV skew, put/call ratio, ATM IV.
Extends the existing options_flow cache with historical data.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from ascent.integrations.openbb_client import get_options_snapshot

log = logging.getLogger(__name__)

_REPO_ROOT   = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "options_flow.parquet"

# This is an enrichment step: it must never be able to stall the trading run.
# A single provider call can take ~14s and still return "unavailable", so an
# unbounded loop over the ~900-symbol universe is multiple hours of no-op work.
# The caller wraps this in try/except, but a slow loop raises nothing, so the
# bounds have to live here.
_DEFAULT_TIME_BUDGET_S      = 120.0
_DEFAULT_MAX_CONSEC_FAILURES = 8


def fetch_cboe_options_row(symbol: str, fetch_date: str) -> Optional[dict]:
    """
    Fetch a single CBOE options row for symbol on fetch_date.
    Returns dict with columns matching options_flow schema, or None if unavailable.
    """
    snapshot = get_options_snapshot([symbol])
    entry = snapshot.get(symbol.upper(), {})

    if entry.get("unavailable"):
        log.debug("[CBOEOptions] %s unavailable", symbol)
        return None

    return {
        "symbol":         symbol.upper(),
        "date":           fetch_date,
        "put_call_ratio": entry.get("put_call_ratio"),
        "atm_iv":         entry.get("atm_iv"),
        "iv_skew":        entry.get("iv_skew"),
        "iv_rank_52w":    entry.get("iv_rank_52w"),
        "source":         "cboe",
    }


def update_options_cache(
    symbols: list[str],
    fetch_date: str,
    cache_path: Path = _DEFAULT_CACHE,
    time_budget_s: float = _DEFAULT_TIME_BUDGET_S,
    max_consecutive_failures: int = _DEFAULT_MAX_CONSEC_FAILURES,
) -> int:
    """
    Fetch options data for each symbol and append new rows to the cache.
    Deduplicates on (symbol, date) — existing rows for the same date are not overwritten.
    Returns count of newly added rows.

    Bounded on two axes so a degraded provider degrades this step instead of the
    whole run: it stops after `time_budget_s` wall-clock, and trips a circuit
    breaker after `max_consecutive_failures` symbols return nothing. Whatever was
    collected before stopping is still written.
    """
    existing: pd.DataFrame = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if "date" in existing.columns:
                existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        except Exception as exc:
            log.warning("[CBOEOptions] Could not read cache: %s", exc)

    new_rows = []
    started       = time.monotonic()
    consec_fail   = 0
    n_attempted   = 0
    for sym in symbols:
        if not existing.empty and "symbol" in existing.columns and "date" in existing.columns:
            dup = existing[(existing["symbol"] == sym.upper()) & (existing["date"] == fetch_date)]
            if not dup.empty:
                log.debug("[CBOEOptions] %s @ %s already in cache, skipping", sym, fetch_date)
                continue

        elapsed = time.monotonic() - started
        if elapsed > time_budget_s:
            log.warning("[CBOEOptions] time budget %.0fs exhausted after %d symbols "
                        "(%d rows); skipping the remainder",
                        time_budget_s, n_attempted, len(new_rows))
            break

        n_attempted += 1
        try:
            row = fetch_cboe_options_row(sym, fetch_date)
        except Exception as exc:
            log.debug("[CBOEOptions] %s fetch failed: %s", sym, exc)
            row = None

        if row:
            new_rows.append(row)
            consec_fail = 0
        else:
            consec_fail += 1
            if consec_fail >= max_consecutive_failures:
                log.warning("[CBOEOptions] provider returned nothing for %d consecutive "
                            "symbols; assuming unavailable and skipping the remainder",
                            consec_fail)
                break

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if existing.empty:
        combined = new_df
    else:
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("[CBOEOptions] Added %d new rows to %s", len(new_rows), cache_path.name)
    return len(new_rows)
