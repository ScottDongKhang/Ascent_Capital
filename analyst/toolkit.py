"""The surface generated code is allowed to use.

Keeping data access behind one narrow helper does two jobs. It stops the LLM
guessing at yfinance's MultiIndex column layout, which is the single most common
way generated loaders fail. And it means every LOAD task in the system reaches
real data by exactly one path, so a data problem is one bug, not five.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "analyst_cache"


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV for one ticker, DatetimeIndex, flat single-level columns.

    Columns: Open, High, Low, Close, Adj Close, Volume.

    Cached on disk purely so repeated runs do not re-hit Yahoo during
    development. This is a fetch cache, not the PAT value cache -- it stores
    raw downloads, never computed intermediates.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{ticker.replace('=', '_').replace('-', '_').replace('.', '_')}_{start}_{end}.parquet"
    path = _CACHE / key
    if path.exists():
        df = pd.read_parquet(path)
        # attrs are not persisted by parquet, so a cache hit must re-stamp
        # identity exactly like a fresh fetch does. The cache key already
        # encodes the ticker, so this is a re-derivation, not new trust.
        df.attrs["ticker"] = ticker
        return df

    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df is None or df.empty:
        raise ValueError(f"no data returned for ticker {ticker!r} between {start} and {end}")

    # yfinance returns a (Price, Ticker) MultiIndex even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df = df.sort_index()
    df.index.name = "Date"

    df.to_parquet(path)
    # Stamp which ticker this data actually is, so a downstream consumer --
    # in particular the execution loop's postcondition check -- can verify
    # identity, not just shape. See NodeResult / validate_output.
    df.attrs["ticker"] = ticker
    return df


def pct_move(series: pd.Series, anchor: pd.Timestamp, horizon_days: int) -> float:
    """Percent move from the last close at or before `anchor`, `horizon_days`
    trading sessions later. NaN when the window runs off the end of the data.
    """
    s = series.dropna().sort_index()
    prior = s.loc[:anchor]
    if prior.empty:
        return float("nan")
    base_pos = s.index.get_loc(prior.index[-1])
    target_pos = base_pos + horizon_days
    if target_pos >= len(s):
        return float("nan")
    base = float(s.iloc[base_pos])
    if base == 0:
        return float("nan")
    return (float(s.iloc[target_pos]) / base - 1.0) * 100.0
