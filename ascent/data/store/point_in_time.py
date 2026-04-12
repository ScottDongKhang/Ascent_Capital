"""
Ascent Capital — Point-in-Time Joins
Ensures no look-ahead bias: at time t, only data with known_time <= t is visible.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def as_of_join(
    base_dates: pd.DatetimeIndex,
    data: pd.DataFrame,
    known_time_col: str = "known_time",
    event_time_col: str = "date",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each date in base_dates, return the latest record from `data`
    where known_time <= date.

    This is the fundamental point-in-time operation.
    For daily prices, event_time == known_time (same day).
    For fundamentals, known_time is the filing/release date.
    """
    data = data.copy()

    if known_time_col not in data.columns:
        # Fallback: treat event_time as known_time (safe for prices)
        data[known_time_col] = pd.to_datetime(data[event_time_col])
    else:
        data[known_time_col] = pd.to_datetime(data[known_time_col])

    data = data.sort_values(known_time_col)

    results = []
    for dt in base_dates:
        mask = data[known_time_col] <= dt
        valid = data[mask]
        if valid.empty:
            continue
        latest = valid.iloc[-1]
        row = {"as_of_date": dt}
        if value_cols:
            for c in value_cols:
                row[c] = latest.get(c)
        else:
            row.update(latest.to_dict())
        results.append(row)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def as_of_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_date_col: str = "date",
    right_known_col: str = "known_time",
    symbol_col: str = "symbol",
    right_value_cols: list[str] | None = None,
    tolerance: str = "365D",
) -> pd.DataFrame:
    """
    Merge right into left using as-of join per symbol.
    For each (symbol, date) in left, finds the latest record in right
    where right.known_time <= left.date.
    """
    left = left.copy()
    right = right.copy()

    left[left_date_col] = pd.to_datetime(left[left_date_col])
    right[right_known_col] = pd.to_datetime(right[right_known_col])

    left = left.sort_values([symbol_col, left_date_col])
    right = right.sort_values([symbol_col, right_known_col])

    result = pd.merge_asof(
        left,
        right,
        left_on=left_date_col,
        right_on=right_known_col,
        by=symbol_col,
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    return result
