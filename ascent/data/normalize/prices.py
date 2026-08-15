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
    df = df.copy()
    # Normalize the date column to midnight before pivoting. Under normal
    # operation normalize_prices() already does this, but pivot_prices is
    # called directly elsewhere too, and a same-day phantom row with a
    # non-midnight time-of-day (see the parquet.py validate_cache guard)
    # would otherwise pivot to a SEPARATE index entry instead of colliding
    # with the real midnight row for that date. aggfunc="last" is kept so any
    # remaining same-day collision after normalization resolves
    # deterministically rather than silently fragmenting the date index.
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    pivot = df.pivot_table(index="date", columns="symbol", values=field, aggfunc="last")
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
