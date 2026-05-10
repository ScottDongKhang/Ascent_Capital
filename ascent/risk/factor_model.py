"""ascent/risk/factor_model.py

Rolling OLS factor loading computation (Barra-style).

For each date t, regresses each stock's trailing 252-day excess return series
on [Mkt-RF, SMB, HML, RMW, CMA, UMD] via vectorised numpy lstsq.

Cache: data_cache/factor_loadings.parquet — MultiIndex (date, symbol).
Incremental: only computes dates not already in cache.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ascent.risk.factor_data import get_factor_returns

log = logging.getLogger(__name__)

LOADINGS_PATH = Path("data_cache/factor_loadings.parquet")
BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "beta_umd"]
_FACTOR_SRC = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]


def _load_prices_wide() -> pd.DataFrame:
    """Return wide close-price DataFrame (dates × symbols) from parquet cache."""
    try:
        from ascent.data.store.parquet import has_data, load_parquet
        if not has_data("prices_live"):
            return pd.DataFrame()
        df = load_parquet("prices_live")
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].sort_index()
        if "close" in df.columns and "symbol" in df.columns:
            return df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        if "symbol" in df.columns:
            val_col = next((c for c in ("close", "Close", "adj_close") if c in df.columns), None)
            if val_col:
                return df.pivot_table(index="date", columns="symbol", values=val_col).sort_index()
        return df.sort_index()
    except Exception as exc:
        log.warning("[FactorModel] Price load failed: %s", exc)
        return pd.DataFrame()


def _load_cache() -> pd.DataFrame:
    if LOADINGS_PATH.exists():
        try:
            return pd.read_parquet(LOADINGS_PATH)
        except Exception as exc:
            log.warning("[FactorModel] Cache read failed: %s", exc)
    return pd.DataFrame()


def _save_cache(df: pd.DataFrame) -> None:
    LOADINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOADINGS_PATH)


def compute_factor_loadings(
    prices_df: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 252,
    min_obs: int = 126,
) -> pd.DataFrame:
    """
    Compute rolling factor loadings for all symbols in prices_df.

    Returns DataFrame with MultiIndex (date, symbol) and columns BETA_COLS.
    Symbols with < min_obs non-NaN returns in the window get NaN loadings.
    """
    if prices_df.empty or factor_returns.empty:
        return pd.DataFrame(columns=BETA_COLS)

    # Align indices
    common_idx = prices_df.index.intersection(factor_returns.index)
    if len(common_idx) < lookback + 5:
        log.warning("[FactorModel] Insufficient overlapping data (%d rows)", len(common_idx))
        return pd.DataFrame(columns=BETA_COLS)

    prices_a = prices_df.reindex(common_idx)
    factors_a = factor_returns.reindex(common_idx)

    stock_rets = prices_a.pct_change()
    rf = factors_a["RF"].fillna(0)
    excess = stock_rets.sub(rf, axis=0)                   # dates × symbols
    F_src = factors_a[_FACTOR_SRC].values                 # dates × 6

    symbols = list(excess.columns)
    R_all = excess.values                                  # dates × N
    eligible_dates = common_idx[lookback:]

    records = []
    for dt in eligible_dates:
        iloc_pos = common_idx.get_loc(dt)
        start = max(0, iloc_pos - lookback)
        R_w = R_all[start:iloc_pos]          # T × N
        F_w = F_src[start:iloc_pos]          # T × 6

        T = F_w.shape[0]
        if T < min_obs:
            continue

        n_obs = np.sum(~np.isnan(R_w), axis=0)           # N
        R_clean = np.where(np.isnan(R_w), 0.0, R_w)      # fill NaN with 0

        F_aug = np.column_stack([F_w, np.ones(T)])        # T × 7
        try:
            B, _, _, _ = np.linalg.lstsq(F_aug, R_clean, rcond=None)  # 7 × N
        except np.linalg.LinAlgError:
            continue

        betas = B[:6, :].copy()                           # 6 × N
        betas[:, n_obs < min_obs] = np.nan

        for j, sym in enumerate(symbols):
            if not np.all(np.isnan(betas[:, j])):
                records.append((dt, sym, *betas[:, j]))

    if not records:
        return pd.DataFrame(columns=BETA_COLS)

    df = pd.DataFrame(records, columns=["date", "symbol"] + BETA_COLS)
    df = df.set_index(["date", "symbol"])
    return df


def update_factor_loadings(lookback: int = 252, min_obs: int = 126) -> pd.DataFrame:
    """
    Incrementally compute and cache factor loadings.
    Only computes dates not already in the cache.
    Returns the full cached loadings DataFrame.
    """
    cached = _load_cache()
    prices = _load_prices_wide()
    factors = get_factor_returns()

    if prices.empty or factors.empty:
        log.warning("[FactorModel] Cannot update loadings — missing prices or factors")
        return cached

    # Determine which dates need computation
    if not cached.empty:
        last_cached = cached.index.get_level_values("date").max()
        # Only recompute recent dates (last 5 bdays) to catch any updates
        cutoff = last_cached - pd.Timedelta(days=5)
        prices_new = prices[prices.index > cutoff]
    else:
        prices_new = prices

    if prices_new.empty:
        log.info("[FactorModel] Loadings up to date")
        return cached

    # Compute on full prices (new window needs full lookback history)
    log.info("[FactorModel] Computing loadings for %d dates...", len(prices_new.index))
    new_loadings = compute_factor_loadings(prices, factors, lookback, min_obs)

    if new_loadings.empty:
        return cached

    # Merge with cache — new dates overwrite
    if cached.empty:
        combined = new_loadings
    else:
        new_dates = new_loadings.index.get_level_values("date").unique()
        cached_trimmed = cached[~cached.index.get_level_values("date").isin(new_dates)]
        combined = pd.concat([cached_trimmed, new_loadings]).sort_index()

    _save_cache(combined)
    latest = combined.index.get_level_values("date").max()
    log.info("[FactorModel] Loadings updated through %s", latest.date())
    return combined


def get_factor_loadings(as_of_date) -> pd.DataFrame:
    """
    Return factor loadings for as_of_date as a DataFrame indexed by symbol.
    Falls back to the most recent available date if as_of_date is not in cache.
    Returns empty DataFrame if cache is missing.
    """
    cached = _load_cache()
    if cached.empty:
        return pd.DataFrame(columns=BETA_COLS)

    as_of = pd.Timestamp(as_of_date)
    dates = cached.index.get_level_values("date").unique().sort_values()
    available = dates[dates <= as_of]
    if available.empty:
        available = dates
    best_date = available[-1]

    df = cached.xs(best_date, level="date")
    return df[BETA_COLS]
