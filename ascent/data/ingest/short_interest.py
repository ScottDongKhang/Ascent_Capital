"""ascent/data/ingest/short_interest.py

Fetch short interest metrics from yfinance ticker.info.
Records shortPercentOfFloat and shortRatio (days-to-cover) per symbol per day.

Data refresh: bi-weekly (FINRA publishes short interest twice per month).
Appending daily creates a smooth time series via forward-fill in the feature panel.
"""
from __future__ import annotations
import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from ascent.data.store.parquet import save_parquet, load_parquet

log = logging.getLogger(__name__)
CACHE_NAME = "short_interest"


def _fetch_one(sym: str) -> dict | None:
    """Fetch current short interest metrics for one symbol from yfinance info."""
    try:
        info        = yf.Ticker(sym).info
        short_pct   = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")

        # Skip if both are missing
        if short_pct is None and short_ratio is None:
            return None

        return {
            "short_pct_float": float(short_pct) if short_pct is not None else np.nan,
            "short_ratio":     float(short_ratio) if short_ratio is not None else np.nan,
        }
    except Exception as exc:
        log.debug("[ShortInterest] %s failed: %s", sym, exc)
        return None


def fetch_short_interest(symbols: list, delay_s: float = 0.15) -> pd.DataFrame:
    """
    Fetch today's short interest snapshot for a list of symbols.
    Returns long-format DataFrame: symbol, date, short_pct_float, short_ratio.
    """
    today = pd.Timestamp(date.today())
    rows  = []
    for i, sym in enumerate(symbols):
        if i and i % 100 == 0:
            log.info("[ShortInterest] %d / %d fetched", i, len(symbols))
        metrics = _fetch_one(sym)
        if metrics:
            rows.append({"symbol": sym, "date": today, **metrics})
        time.sleep(delay_s)

    df = pd.DataFrame(rows)
    log.info("[ShortInterest] Fetched %d / %d symbols", len(df), len(symbols))
    return df


def save_short_interest(df: pd.DataFrame) -> None:
    """Append to short_interest parquet, dedup by (symbol, date)."""
    if df is None or df.empty:
        return
    existing = load_short_interest()
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
        combined = combined.sort_values(["symbol", "date"])
    else:
        combined = df.copy()
        combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    save_parquet(combined, CACHE_NAME)
    log.info("[ShortInterest] Saved %d rows to %s", len(combined), CACHE_NAME)


def load_short_interest() -> pd.DataFrame:
    """Load short interest cache. Returns empty DataFrame if not present."""
    try:
        df = load_parquet(CACHE_NAME)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    from ascent.config.settings import get_config
    cfg = get_config()
    df  = fetch_short_interest(cfg.universe.symbols)
    if not df.empty:
        save_short_interest(df)
        print(f"Seeded short_interest: {len(df)} rows")
