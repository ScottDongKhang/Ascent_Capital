"""ascent/data/ingest/google_trends.py

Google Trends search velocity signal.
Rate limit: 1 request per 5 seconds. For 901 symbols, full refresh ≈ 75 min.
Schedule: weekly (Sunday 7 AM). 1-day lag.
"""
from __future__ import annotations
import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_CACHE_PATH = Path("data_cache/altdata_trends.parquet")
_REQUEST_DELAY = 5.1  # seconds between requests (rate limit: 1/5s)


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


def build_trends_panel(
    symbols: list[str],
    lookback_months: int = 12,
) -> pd.DataFrame:
    """
    Fetch trends for all symbols one at a time (rate-limited).
    Returns wide panel (dates × symbols), values 0–1.
    """
    panels = {}
    for i, sym in enumerate(symbols):
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


def update_trends_signals(symbols: list[str], lookback_months: int = 12) -> pd.DataFrame:
    """Incremental weekly update."""
    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
        except Exception:
            pass

    log.info("[GoogleTrends] Fetching trends for %d symbols (est. %.0f min)",
             len(symbols), len(symbols) * _REQUEST_DELAY / 60)
    raw_panel = build_trends_panel(symbols, lookback_months=lookback_months)
    if raw_panel.empty:
        return existing

    signal_panel = compute_trends_signal(raw_panel)
    if signal_panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, signal_panel]).groupby(level=0).last()
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
