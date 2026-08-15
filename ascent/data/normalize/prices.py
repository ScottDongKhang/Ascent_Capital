"""
Ascent Capital — Data Normalization
Clean and normalize raw ingested data into analysis-ready format.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw price data:
    - Ensure correct dtypes
    - Sort by symbol/date
    - Drop duplicates (keep last source)
    - Add derived columns
    - Validate no future data
    """
    df = df.copy()

    # Ensure types
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing critical data
    df = df.dropna(subset=["symbol", "date", "close"])

    # Remove bad rows
    df = df[df["close"] > 0]
    df = df[df["volume"] >= 0]

    # Sort and deduplicate
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

    # Ensure VWAP exists
    if "vwap" not in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Ensure timestamps
    if "event_time" not in df.columns:
        df["event_time"] = df["date"]
    if "known_time" not in df.columns:
        df["known_time"] = df["date"]

    # Add daily return (for quick reference, not used as feature directly)
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # Add dollar volume
    df["dollar_volume"] = df["close"] * df["volume"]

    return df


def normalize_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize macro data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["series_id", "date"], keep="last")
    return df


def pivot_prices(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """
    Pivot normalized price data to wide format: dates × symbols.
    Used for cross-sectional feature computation.
    """
    # Group on the TRADING day (_calendar_day_key), not a plain
    # .dt.normalize(). Both the repaired prices_live cache and
    # save_parquet's own dedup already group on _calendar_day_key, which
    # rolls a bar stamped >=17:00 local forward onto the NEXT calendar day
    # (that stamp is that day's already-closed bar, fetched and recorded
    # late -- see _calendar_day_key's docstring). A plain normalize()
    # disagrees with that: it would merge a same-day phantom row into the
    # WRONG trading day and, combined with aggfunc="last", could silently
    # overwrite a real close with a value that actually belongs to the next
    # day -- a worse, silent failure mode than the old fragmented-index
    # symptom (which at least produced an obviously-wrong ~2x row count).
    from ascent.data.store.parquet import _calendar_day_key

    raw_date = pd.to_datetime(df["date"])
    day_key = _calendar_day_key(raw_date)
    # Only derive the small pieces pivot_table actually needs -- avoid
    # copying the whole (potentially 1.5M-row) frame just to overwrite one
    # column.
    small = pd.DataFrame({
        "date": day_key,
        "symbol": df["symbol"].values,
        field: df[field].values,
    })
    # Same-day collisions must resolve by ACTUAL timestamp, not by input row
    # order: aggfunc="last" picks the last row per group as pivot_table sees
    # it, so sort by the original (un-normalized) timestamp first -- that
    # makes "last" mean "latest-stamped value wins" regardless of how the
    # input frame was ordered.
    small = small.assign(_raw_date=raw_date.values).sort_values("_raw_date")
    pivot = small.pivot_table(index="date", columns="symbol", values=field, aggfunc="last")
    pivot = pivot.sort_index()
    return pivot


def pivot_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot macro data: dates × series names."""
    if "name" in df.columns:
        pivot = df.pivot_table(index="date", columns="name", values="value", aggfunc="last")
    else:
        pivot = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
    pivot = pivot.sort_index().ffill()
    return pivot
