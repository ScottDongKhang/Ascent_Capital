"""
Ascent Capital — Data Storage
Parquet-based storage with schema validation and point-in-time metadata.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ascent.config.settings import get_config


PRICE_SCHEMA_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap",
                     "event_time", "known_time", "source"]

MACRO_SCHEMA_COLS = ["series_id", "date", "value", "event_time", "known_time", "source"]


def _cache_path(name: str) -> Path:
    cfg = get_config()
    return cfg.data_dir / f"{name}.parquet"


def save_parquet(df: pd.DataFrame, name: str, partition_cols: Optional[List[str]] = None) -> Path:
    """Save DataFrame to Parquet. Append-safe: existing data preserved."""
    path = _cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=[c for c in df.columns if c not in ("known_time", "source")],
            keep="last"
        )

    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_parquet(name: str) -> Optional[pd.DataFrame]:
    """Load Parquet file. Returns None if doesn't exist."""
    path = _cache_path(name)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def has_data(name: str) -> bool:
    return _cache_path(name).exists()


def validate_price_schema(df: pd.DataFrame) -> bool:
    """Check that price DataFrame has required columns."""
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    return required.issubset(set(df.columns))


def validate_macro_schema(df: pd.DataFrame) -> bool:
    required = {"series_id", "date", "value"}
    return required.issubset(set(df.columns))
