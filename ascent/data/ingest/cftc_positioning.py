# ascent/data/ingest/cftc_positioning.py
"""
CFTC Commitments of Traders — S&P 500 e-mini speculator positioning.
Fetches weekly COT report via OpenBB CFTC provider.
Cache: data_cache/cftc_positioning.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ascent.integrations.openbb_client import get_cot_snapshot

log = logging.getLogger(__name__)

_REPO_ROOT     = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "cftc_positioning.parquet"


def fetch_cot_row() -> Optional[dict]:
    """Fetch latest COT row via openbb_client. Returns None on failure."""
    return get_cot_snapshot()


def update_cot_cache(cache_path: Path = _DEFAULT_CACHE) -> bool:
    """
    Fetch latest COT report and append to cache if not already present.
    Deduplicates on as_of_date. Returns True if new row added.
    """
    row = fetch_cot_row()
    if row is None:
        log.warning("[COT] Fetch failed — cache not updated")
        return False

    existing: pd.DataFrame = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
        except Exception as exc:
            log.warning("[COT] Could not read cache: %s", exc)

    as_of = row.get("as_of_date", "")
    if not existing.empty and "as_of_date" in existing.columns:
        if as_of in existing["as_of_date"].astype(str).values:
            log.debug("[COT] %s already in cache", as_of)
            return False

    new_df = pd.DataFrame([row])
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("[COT] Added row for %s", as_of)
    return True


def get_latest_cot(cache_path: Path = _DEFAULT_CACHE) -> Optional[dict]:
    """
    Read latest COT row from cache. Returns dict or None if cache absent.
    Used by AI PM tool executor — does NOT fetch live.
    """
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None
        latest = df.sort_values("as_of_date").iloc[-1]
        return latest.to_dict()
    except Exception as exc:
        log.warning("[COT] get_latest_cot failed: %s", exc)
        return None
