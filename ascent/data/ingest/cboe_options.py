# ascent/data/ingest/cboe_options.py
"""
Historical CBOE options data — IV skew, put/call ratio, ATM IV.
Extends the existing options_flow cache with historical data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ascent.integrations.openbb_client import get_options_snapshot

log = logging.getLogger(__name__)

_REPO_ROOT   = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "options_flow.parquet"


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
) -> int:
    """
    Fetch options data for each symbol and append new rows to the cache.
    Deduplicates on (symbol, date) — existing rows for the same date are not overwritten.
    Returns count of newly added rows.
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
    for sym in symbols:
        if not existing.empty and "symbol" in existing.columns and "date" in existing.columns:
            dup = existing[(existing["symbol"] == sym.upper()) & (existing["date"] == fetch_date)]
            if not dup.empty:
                log.debug("[CBOEOptions] %s @ %s already in cache, skipping", sym, fetch_date)
                continue

        row = fetch_cboe_options_row(sym, fetch_date)
        if row:
            new_rows.append(row)

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
