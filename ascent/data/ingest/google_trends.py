"""ascent/data/ingest/google_trends.py

Google Trends search velocity signal.
Rate limit: 1 request per 5 seconds. For 901 symbols, full refresh ≈ 75 min.
Schedule: weekly (Sunday 7 AM). 1-day lag.
Freshness gate: skip symbols updated within 7 days. Priority queue puts portfolio
symbols first so the most important signals are always fresh.
"""
from __future__ import annotations
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CACHE_PATH = Path("data_cache/altdata_trends.parquet")
_REQUEST_DELAY = 5.1  # seconds between requests (rate limit: 1/5s)
_CACHE_FRESHNESS_DAYS = 7  # skip symbols with valid data newer than this


def fetch_trends(symbol: str, lookback_months: int = 12) -> pd.Series:
    """
    Fetch Google Trends relative search interest for a symbol.
    Returns a daily-frequency Series (0–1 normalized). Empty on failure.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=300, timeout=(10, 25))
        pytrends.build_payload([symbol], timeframe=f"today {lookback_months}-m")
        df = pytrends.interest_over_time()
        if df.empty or symbol not in df.columns:
            return pd.Series(dtype=float)
        series = df[symbol].astype(float)
        # Normalize to 0–1
        max_val = series.max()
        if max_val > 0:
            series = series / max_val
        # Strip timezone for consistency
        series.index = pd.DatetimeIndex([ts.date() for ts in series.index])
        return series
    except Exception as e:
        log.debug("[GoogleTrends] fetch_trends failed for %s: %s", symbol, e)
        return pd.Series(dtype=float)


def _stale_symbols(
    all_symbols: list[str],
    existing: pd.DataFrame,
    max_age_days: int = _CACHE_FRESHNESS_DAYS,
) -> list[str]:
    """Return symbols that are absent or have no valid data within max_age_days."""
    if existing.empty:
        return list(all_symbols)
    cutoff = pd.Timestamp(date.today() - timedelta(days=max_age_days))
    stale = []
    for sym in all_symbols:
        if sym not in existing.columns:
            stale.append(sym)
            continue
        col = existing[sym].dropna()
        if col.empty or col.index.max() < cutoff:
            stale.append(sym)
    return stale


def build_trends_panel(
    symbols: list[str],
    lookback_months: int = 12,
    max_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fetch trends for all symbols one at a time (rate-limited).
    Returns wide panel (dates × symbols), values 0–1.
    Stops early if max_seconds wall-clock time is exceeded.
    """
    panels = {}
    start_time = time.monotonic()
    for i, sym in enumerate(symbols):
        if max_seconds is not None and time.monotonic() - start_time > max_seconds:
            log.info(
                "[GoogleTrends] Time cap reached after %d/%d symbols (%.0fs)",
                i, len(symbols), max_seconds,
            )
            break
        series = fetch_trends(sym, lookback_months=lookback_months)
        if not series.empty:
            panels[sym] = series
        if i < len(symbols) - 1:
            time.sleep(_REQUEST_DELAY)

    if not panels:
        return pd.DataFrame()

    wide = pd.DataFrame(panels)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    wide.index.name = "date"
    # Apply 1-day lag (previous day's search velocity predicts today's price)
    wide = wide.shift(1)
    return wide


def compute_trends_signal(trends_panel: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Search velocity: (current − rolling mean) / rolling mean.
    Cross-sectional z-score per date.
    """
    if trends_panel.empty:
        return pd.DataFrame()

    rolling_mean = trends_panel.rolling(window=lookback, min_periods=5).mean()
    # Velocity = (current - mean) / (mean + epsilon)
    velocity = (trends_panel - rolling_mean) / (rolling_mean + 1e-6)

    # Cross-sectional z-score per date
    def cs_zscore(row: pd.Series) -> pd.Series:
        vals = row.dropna()
        if len(vals) < 2 or vals.std() == 0:
            return row * 0
        return ((row - vals.mean()) / vals.std()).clip(-3, 3)

    signal = velocity.apply(cs_zscore, axis=1)
    signal.index.name = "date"
    return signal


def update_trends_signals(
    symbols: list[str],
    lookback_months: int = 12,
    priority_symbols: Optional[list[str]] = None,
    max_minutes: float = 30.0,
) -> pd.DataFrame:
    """Incremental update with freshness gate and priority queue.

    priority_symbols are always fetched first (portfolio holdings).
    Symbols with valid data within _CACHE_FRESHNESS_DAYS are skipped.
    Stops after max_minutes wall-clock time to bound total runtime.
    """
    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    # Filter to stale symbols only
    to_fetch = _stale_symbols(symbols, existing)
    if not to_fetch:
        log.info("[GoogleTrends] All %d symbols fresh — skipping fetch", len(symbols))
        return existing

    # Priority queue: portfolio symbols first, rest in original order
    if priority_symbols:
        priority_set = set(priority_symbols)
        ordered = [s for s in to_fetch if s in priority_set] + \
                  [s for s in to_fetch if s not in priority_set]
    else:
        ordered = to_fetch

    est_min = len(ordered) * _REQUEST_DELAY / 60
    log.info(
        "[GoogleTrends] Fetching %d/%d stale symbols (est. %.0f min, cap %.0f min)",
        len(ordered), len(symbols), est_min, max_minutes,
    )
    raw_panel = build_trends_panel(
        ordered,
        lookback_months=lookback_months,
        max_seconds=max_minutes * 60,
    )
    if raw_panel.empty:
        return existing

    signal_panel = compute_trends_signal(raw_panel)
    if signal_panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, signal_panel], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = signal_panel

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CACHE_PATH)
    log.info("[GoogleTrends] Updated trends panel: %s", combined.shape)
    return combined


def load_trends_signals() -> pd.DataFrame:
    if _CACHE_PATH.exists():
        try:
            return pd.read_parquet(_CACHE_PATH)
        except Exception as e:
            log.warning("[GoogleTrends] load failed: %s", e)
    return pd.DataFrame()
