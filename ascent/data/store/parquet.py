"""
Ascent Capital — Parquet Store
Handles saving and loading DataFrames to/from the data_cache directory.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

log = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parents[3]
DATA_DIR = _ROOT / "data_cache"


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _force_refresh() -> bool:
    val = os.environ.get("ASCENT_FORCE_REFRESH", "0").strip().lower()
    return val in ("1", "true", "yes")


def has_data(name: str) -> bool:
    if _force_refresh():
        log.info("[cache] ASCENT_FORCE_REFRESH active — treating %s as missing", name)
        return False
    return _cache_path(name).exists()


def save_parquet(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    if path.exists():
        old = pd.read_parquet(path)
        # FIX #3: include series_id in preferred id columns alongside symbol.
        # Before: only "series" was checked, which doesn't exist in either
        # simulated.py or fred.py output — those both write "series_id".
        # This meant macro deduplication never matched on the right column,
        # so every save appended duplicate rows instead of replacing them.
        id_cols = [
            c for c in ["symbol", "date", "series_id", "series"]
            if c in old.columns
        ]
        if not id_cols:
            id_cols = [c for c in old.columns if c not in ("known_time", "source")]
        combined = pd.concat([old, df], ignore_index=True)
        # Build the dedup key with any datetime `date` column normalised to its
        # CALENDAR DAY. Different fetches store the same bar at different intraday
        # timestamps / tz offsets (the cache keeps `date` at 19:00 ET), so a raw
        # (symbol, date) match never fired and every save appended duplicate rows
        # (audit 2026-06-22: prices_live had ~59% duplicates). Normalising the key
        # collapses those without mutating the stored `date` values.
        key = combined[id_cols].copy()
        if "date" in key.columns and pd.api.types.is_datetime64_any_dtype(key["date"]):
            _d = key["date"].dt.normalize()
            if isinstance(key["date"].dtype, pd.DatetimeTZDtype):
                _d = _d.dt.tz_localize(None)
            key["date"] = _d
        df = combined[~key.duplicated(keep="last")]
    df.to_parquet(path, index=False)
    log.info("[cache] saved %s  rows=%d", name, len(df))


def load_parquet(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_parquet(path)


def validate_cache(
    name: str,
    required_start: Optional[str] = None,
    required_end: Optional[str] = None,
    required_symbols: Optional[List[str]] = None,
    stale_days: int = 3,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> tuple[bool, str]:
    if _force_refresh():
        return False, "ASCENT_FORCE_REFRESH is active"

    path = _cache_path(name)
    if not path.exists():
        return False, f"cache file missing: {path}"

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return False, f"cache unreadable: {exc}"

    if date_col not in df.columns:
        return True, ""

    dates       = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return False, "cache has no valid dates"

    cache_start = dates.min().normalize().tz_localize(None)
    cache_end   = dates.max().normalize().tz_localize(None)

    if required_start is not None:
        req_s = pd.Timestamp(required_start).normalize()
        if cache_start > req_s:
            return False, (
                f"cache starts {cache_start.date()} but "
                f"required_start={req_s.date()}"
            )

    if required_end is not None:
        req_e = pd.Timestamp(required_end).normalize()
        if cache_end < req_e:
            return False, (
                f"cache ends {cache_end.date()} but "
                f"required_end={req_e.date()}"
            )

    if stale_days > 0:
        today = pd.Timestamp.today().normalize()
        lag   = (today - cache_end).days
        if lag > stale_days:
            return False, (
                f"cache latest date {cache_end.date()} is "
                f"{lag} calendar days old (limit={stale_days})"
            )

    if required_symbols and symbol_col in df.columns:
        cached_syms = set(df[symbol_col].unique())
        missing     = set(required_symbols) - cached_syms
        if missing:
            sample = sorted(missing)[:5]
            return False, (
                f"{len(missing)} required symbols missing from cache "
                f"(sample: {sample})"
            )

    return True, ""