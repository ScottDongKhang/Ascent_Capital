"""
Ascent Capital — FRED Data Ingest
Federal Reserve Economic Data for macro regime features.
"""
from __future__ import annotations
import time
from typing import List, Optional
import pandas as pd
import requests
from ascent.config.settings import get_config


FRED_BASE = "https://api.stlouisfed.org/fred"

# Key macro series
DEFAULT_SERIES = {
    "DFF": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "T10Y2Y": "yield_spread_10y2y",
    "VIXCLS": "vix",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "DCOILWTICO": "oil_wti",
    "DEXUSEU": "usd_eur",
    "BAMLH0A0HYM2": "hy_spread",
}


def fetch_series(
    series_id: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> pd.DataFrame:
    """Fetch a single FRED series."""
    cfg = get_config()
    key = cfg.keys.fred
    if not key:
        raise ValueError("FRED_API_KEY not set")

    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    if end_date:
        params["observation_end"] = end_date

    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff_base ** attempt
                time.sleep(wait)
    else:
        raise last_exc
    data = resp.json()

    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # FRED data has publication lag — realtime_start is when data was first available
    df["series_id"] = series_id
    df["event_time"] = df["date"]
    # known_time: use realtime_start if available, else date + typical lag
    if "realtime_start" in df.columns:
        df["known_time"] = pd.to_datetime(df["realtime_start"])
    else:
        df["known_time"] = df["date"] + pd.Timedelta(days=1)  # conservative: available next day
    df["source"] = "fred"

    return df[["series_id", "date", "value", "event_time", "known_time", "source"]]


def fetch_all_macro(
    series_map: dict[str, str] | None = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    delay_sec: float = 0.3,
) -> pd.DataFrame:
    """Fetch all default macro series from FRED."""
    if series_map is None:
        series_map = DEFAULT_SERIES

    frames = []
    for series_id, name in series_map.items():
        try:
            df = fetch_series(series_id, start_date, end_date)
            if not df.empty:
                df["name"] = name
                frames.append(df)
                print(f"  [FRED] {series_id} ({name}): {len(df)} observations")
        except Exception as e:
            print(f"  [FRED] {series_id}: ERROR {e}")
        time.sleep(delay_sec)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
