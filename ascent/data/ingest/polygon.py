"""
Ascent Capital — Polygon.io Data Ingest
Primary source for OHLCV price data.
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import requests
from ascent.config.settings import get_config


POLYGON_BASE = "https://api.polygon.io"


def fetch_daily_bars(
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Polygon.io.
    Uses unadjusted data by default to avoid look-ahead from future splits.
    """
    cfg = get_config()
    key = cfg.keys.polygon
    if not key:
        raise ValueError("POLYGON_API_KEY not set")

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": key,
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        time.sleep(12)  # free tier rate limit
        resp = requests.get(url, params=params, timeout=30)

    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.rename(columns={
        "t": "timestamp_ms", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "transactions",
    })

    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df["symbol"] = symbol
    df["event_time"] = df["date"]
    df["known_time"] = df["date"]  # EOD data known same day after close
    df["source"] = "polygon"

    cols = ["symbol", "date", "open", "high", "low", "close", "volume",
            "vwap", "transactions", "event_time", "known_time", "source"]
    available = [c for c in cols if c in df.columns]
    df = df[available]

    return df


def fetch_universe_daily(
    symbols: List[str],
    start_date: str,
    end_date: str,
    delay_sec: float = 0.25,
) -> pd.DataFrame:
    """Fetch daily bars for entire universe with rate limiting."""
    frames = []
    for i, sym in enumerate(symbols):
        try:
            df = fetch_daily_bars(sym, start_date, end_date)
            if not df.empty:
                frames.append(df)
                print(f"  [{i+1}/{len(symbols)}] {sym}: {len(df)} bars")
            else:
                print(f"  [{i+1}/{len(symbols)}] {sym}: no data")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym}: ERROR {e}")

        if delay_sec > 0:
            time.sleep(delay_sec)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
