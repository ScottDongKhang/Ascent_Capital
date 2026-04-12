"""
Ascent Capital — Tiingo Data Ingest
Backup/validation source for EOD prices.
"""
from __future__ import annotations
import time
from typing import List
import pandas as pd
import requests
from ascent.config.settings import get_config


TIINGO_BASE = "https://api.tiingo.com"


def fetch_daily_bars(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily EOD from Tiingo."""
    cfg = get_config()
    key = cfg.keys.tiingo
    if not key:
        raise ValueError("TIINGO_API_KEY not set")

    url = f"{TIINGO_BASE}/tiingo/daily/{symbol}/prices"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {key}"}
    params = {"startDate": start_date, "endDate": end_date}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = symbol

    col_map = {
        "adjOpen": "adj_open", "adjHigh": "adj_high",
        "adjLow": "adj_low", "adjClose": "adj_close",
        "adjVolume": "adj_volume",
    }
    df = df.rename(columns=col_map)

    df["event_time"] = df["date"]
    df["known_time"] = df["date"]
    df["source"] = "tiingo"

    cols = ["symbol", "date", "close", "high", "low", "open", "volume",
            "adj_close", "adj_open", "adj_high", "adj_low",
            "event_time", "known_time", "source"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def fetch_universe_daily(
    symbols: List[str],
    start_date: str,
    end_date: str,
    delay_sec: float = 0.5,
) -> pd.DataFrame:
    frames = []
    for i, sym in enumerate(symbols):
        try:
            df = fetch_daily_bars(sym, start_date, end_date)
            if not df.empty:
                frames.append(df)
                print(f"  [Tiingo {i+1}/{len(symbols)}] {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  [Tiingo {i+1}/{len(symbols)}] {sym}: ERROR {e}")
        if delay_sec > 0:
            time.sleep(delay_sec)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
