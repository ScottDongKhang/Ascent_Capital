"""
Ascent Capital — Simulated Data Generator
Generates realistic synthetic price/macro data for testing without API keys.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from typing import List


def _stable_seed(symbol: str) -> int:
    """
    FIX #4: return a stable integer seed for a given symbol.

    Before: abs(hash(sym)) % (2**31)
    Python's built-in hash() is randomized across interpreter sessions by
    PYTHONHASHSEED, so simulated backtests produced different results every
    run even with the same symbol list. This made reproducibility impossible.

    After: MD5 of the symbol bytes, truncated to 31 bits.
    MD5 is deterministic regardless of session or platform.
    """
    return int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2 ** 31)


def generate_price_data(
    symbols: List[str],
    start_date: str = "2019-01-01",
    end_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic daily OHLCV data.
    Each symbol gets a different return/vol profile based on its stable hash.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    frames = []
    for sym in symbols:
        sym_rng = np.random.default_rng(_stable_seed(sym))  # FIX #4

        annual_return = sym_rng.uniform(-0.05, 0.20)
        annual_vol    = sym_rng.uniform(0.15, 0.50)
        daily_mu      = annual_return / 252
        daily_sigma   = annual_vol / np.sqrt(252)

        raw_rets = sym_rng.normal(daily_mu, daily_sigma, n_days)
        ar_coef  = sym_rng.uniform(-0.05, 0.02)
        for i in range(1, n_days):
            raw_rets[i] += ar_coef * raw_rets[i - 1]

        start_price   = sym_rng.uniform(20, 500)
        close         = start_price * np.exp(np.cumsum(raw_rets))
        intraday_range = daily_sigma * close
        open_         = close * (1 + sym_rng.normal(0, daily_sigma * 0.3, n_days))
        high          = np.maximum(open_, close) + np.abs(sym_rng.normal(0, 1, n_days)) * intraday_range * 0.5
        low           = np.minimum(open_, close) - np.abs(sym_rng.normal(0, 1, n_days)) * intraday_range * 0.5
        low           = np.maximum(low, 0.01)

        base_vol = sym_rng.uniform(1e6, 50e6)
        vol_mult = 1 + 3 * np.abs(raw_rets) / daily_sigma
        volume   = (base_vol * vol_mult * sym_rng.lognormal(0, 0.3, n_days)).astype(int)
        vwap     = (high + low + close) / 3.0

        df = pd.DataFrame({
            "symbol":     sym,
            "date":       dates[:n_days],
            "open":       np.round(open_, 2),
            "high":       np.round(high, 2),
            "low":        np.round(low, 2),
            "close":      np.round(close, 2),
            "volume":     volume,
            "vwap":       np.round(vwap, 2),
            "event_time": dates[:n_days],
            "known_time": dates[:n_days],
            "source":     "simulated",
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def generate_macro_data(
    start_date: str = "2019-01-01",
    end_date: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic macro data for testing."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n     = len(dates)

    series = {
        "fed_funds_rate": (2.5, 0.02, 0.0, 6.0),
        "treasury_10y":   (3.0, 0.03, 0.5, 6.0),
        "treasury_2y":    (2.5, 0.04, 0.0, 6.0),
        "vix":            (20.0, 2.0, 9.0, 80.0),
        "cpi":            (260.0, 0.1, 200.0, 320.0),
        "unemployment":   (4.0, 0.05, 2.0, 15.0),
    }

    frames = []
    for name, (start_val, vol, low, high) in series.items():
        values = [start_val]
        for i in range(1, n):
            change = rng.normal(0, vol) - 0.01 * (values[-1] - start_val)
            values.append(np.clip(values[-1] + change, low, high))

        df = pd.DataFrame({
            "series_id":  name,   # FIX #3: consistent column name
            "name":       name,
            "date":       dates[:n],
            "value":      np.round(values[:n], 4),
            "event_time": dates[:n],
            "known_time": dates[:n] + pd.Timedelta(days=1),
            "source":     "simulated",
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)