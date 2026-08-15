"""
Ascent Capital — Parquet Store
Handles saving and loading DataFrames to/from the data_cache directory.
"""
from __future__ import annotations

import logging
import os
from datetime import time
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


# A bar timestamped at/after this local hour is treated as belonging to the
# NEXT trading day, not its nominal date. See _calendar_day_key for the
# rationale (Defect 2, 2026-07-28 audit).
_EVENING_ROLLOVER_HOUR = 17


def _calendar_day_key(s: pd.Series) -> pd.Series:
    """Collapse a `date` column to a comparable per-row TRADING day for dedup.

    Handles all three shapes a blended cache produces:
      - tz-aware datetime64  (the hub stores bars at 19:00 NY)
      - tz-naive  datetime64  (other ingest sources)
      - object dtype holding a MIX of tz-aware and tz-naive Timestamps — what
        pd.concat yields when two fetches disagree on tz. is_datetime64_any_dtype
        is False for this, so the old code skipped normalization entirely and
        dedup never fired (the 3-generation blend that bloated prices_live ~3×).

    Evening-rollover rule (Defect 2, 2026-07-28 audit): a bar stamped
    >=17:00 local (i.e. after the US cash-market close at 16:00 ET) does not
    belong to its own nominal date — it is that trading day's already-closed
    bar being fetched and stamped relative to the NEXT calendar day, which one
    provider (yfinance_hub) stores as e.g. "2020-01-01 19:00 NY" for what is
    really 2020-01-02's close (2020-01-01 is a market holiday, so a row dated
    on it cannot be a real session; the audit's own duplicate-count evidence —
    1,999 keys for ~1,625 real trading days — is explained by exactly this: the
    same trading day fetched once late (stamped on the day before, evening) and
    once promptly (stamped at 00:00 on the correct day) produced two different
    un-rolled keys that never collided). Rolling the evening stamp forward by
    one calendar day makes both fetches of the same trading day key identically.

    This is intentionally conservative: it only merges a late-evening stamp on
    day D with a same-day-or-later fetch actually dated D+1. It never merges
    two stamps that are already on different real calendar days without going
    through that exact D-evening / D+1-midnight relationship (e.g. a Friday
    evening bar rolls to Saturday, which will NOT collide with the following
    Monday's real trading day — under-collapsing across a weekend/holiday gap
    rather than ever wrongly merging two distinct trading days).

    Strips tz (keeps wall-clock day) and zeroes the time, so the same trading
    day matches regardless of intraday timestamp / tz / dtype. Stored values
    are not mutated — only the dedup key.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        hour = s.dt.hour
        d = s.dt.normalize()
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            d = d.dt.tz_localize(None)
        rollover = (hour >= _EVENING_ROLLOVER_HOUR).astype("int64")
        return d + pd.to_timedelta(rollover, unit="D")

    def _one(x):
        if x is None or (not isinstance(x, pd.Timestamp) and pd.isna(x)):
            return pd.NaT
        ts = x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
        hour = ts.hour
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        day = ts.normalize()
        if hour >= _EVENING_ROLLOVER_HOUR:
            day = day + pd.Timedelta(days=1)
        return day

    return s.map(_one)


_ID_LIKE_COLS = {"symbol", "date", "series_id", "series"}


def save_parquet(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    # Wide-format caches (prices_macro / prices_international / prices_alternatives:
    # one column per symbol, dates live ONLY in a DatetimeIndex, no id column) used
    # to lose their index entirely: pd.concat(..., ignore_index=True) below discards
    # it on every append, and to_parquet(..., index=False) discards it on every
    # write. With no `date`/`symbol`/etc column, id_cols also fell back to ALL
    # columns (the price values themselves), so dedup keyed on price equality
    # instead of date. Converting the DatetimeIndex into a real `date` column up
    # front lets the existing id_cols / calendar-day-dedup logic below (already
    # correct for long-format caches) handle wide-format caches too, with no
    # further special-casing needed anywhere else in this function.
    if isinstance(df.index, pd.DatetimeIndex):
        collision = _ID_LIKE_COLS & set(df.columns)
        if collision:
            # Ambiguous shape: a DatetimeIndex AND an id-like column. Converting
            # would either clobber an existing `date` column or invent a second
            # date axis, so the conversion is skipped deliberately — but this
            # shape is not produced by any known writer, so make it visible
            # rather than silently writing a frame whose index is dropped.
            log.warning(
                "[cache] %s: DatetimeIndex present alongside id column(s) %s — "
                "skipping wide-format date conversion; the index will NOT be persisted",
                name, sorted(collision),
            )
        else:
            idx_col = df.index.name or "index"
            df = df.reset_index().rename(columns={idx_col: "date"})
    # FIX #3: include series_id in preferred id columns alongside symbol.
    # Before: only "series" was checked, which doesn't exist in either
    # simulated.py or fred.py output — those both write "series_id".
    # This meant macro deduplication never matched on the right column,
    # so every save appended duplicate rows instead of replacing them.
    #
    # Defect 1 fix (2026-07-28): dedup used to run ONLY inside `if
    # path.exists()`, so the very FIRST save_parquet call for a cache name
    # wrote the incoming `df` straight through with no dedup at all. If that
    # first incoming frame already contained exact-duplicate rows (e.g. a hub
    # blend of multiple providers concatenated before ever calling
    # save_parquet), those duplicates were persisted permanently — every
    # later append only ever compared new rows against that already-corrupt
    # baseline, never revisiting it. The dedup logic below now always runs,
    # against `df` alone when there is no existing cache and against
    # `old + df` when there is, so a self-duplicated first write is caught
    # too.
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    id_cols = [
        c for c in ["symbol", "date", "series_id", "series"]
        if c in combined.columns
    ]
    if not id_cols:
        id_cols = [c for c in combined.columns if c not in ("known_time", "source")]
    # Build the dedup key with any datetime `date` column normalised to its
    # CALENDAR DAY (with evening-rollover, see _calendar_day_key). Different
    # fetches store the same bar at different intraday timestamps / tz offsets
    # (the cache keeps `date` at 19:00 ET), so a raw (symbol, date) match never
    # fired and every save appended duplicate rows (audit 2026-06-22:
    # prices_live had ~59% duplicates). Normalising the key collapses those
    # without mutating the stored `date` values.
    key = combined[id_cols].copy()
    if "date" in key.columns:
        key["date"] = _calendar_day_key(key["date"])
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

    # Phantom-row corruption signature (2026-08-15 audit): a bar written with a
    # non-midnight time-of-day (e.g. 19:00 from an un-rolled hub stamp) that
    # slipped past save_parquet's dedup. Gated on date_col actually being
    # datetime-like so this never fires on caches where date_col holds
    # something else, and gated with zero tolerance (not "a handful") because
    # after the Task 2 repair prices_live has none — any count here means new
    # corruption, not a legitimate tz-shifted row elsewhere in the codebase
    # (those are handled by _calendar_day_key's rollover, not by writing
    # non-midnight `date` values to disk).
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        naive_dates = dates
        if isinstance(naive_dates.dtype, pd.DatetimeTZDtype):
            naive_dates = naive_dates.dt.tz_localize(None)
        non_midnight = int((naive_dates.dt.time != time(0, 0)).sum())
        if non_midnight > 0:
            return False, (
                f"{non_midnight} rows have non-midnight time-of-day "
                "(phantom-row corruption)"
            )

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