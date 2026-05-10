"""ascent/risk/covariance_model.py

Factor-structured covariance matrix: Σ = B·F·B' + D

B — factor loadings (N × 6)
F — factor return covariance (6 × 6)
D — diagonal residual variance (N × N)

Ledoit-Wolf shrinkage applied to the residual covariance before diagonalising.
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from ascent.risk.factor_data import get_factor_returns
from ascent.risk.factor_model import get_factor_loadings, BETA_COLS

log = logging.getLogger(__name__)

_FACTOR_SRC = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]


def compute_factor_covariance(factor_returns: pd.DataFrame, lookback: int = 252) -> np.ndarray:
    """
    6×6 factor return covariance over trailing lookback days (annualised).
    Returns identity * 0.04 if insufficient data.
    """
    src = factor_returns[_FACTOR_SRC].dropna()
    window = src.tail(lookback)
    if len(window) < 20:
        log.warning("[CovModel] Insufficient factor data — using identity proxy")
        return np.eye(6) * 0.04
    return window.cov().values * 252


def compute_residual_variances(
    prices_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 252,
) -> np.ndarray:
    """
    Compute per-symbol residual variance using Ledoit-Wolf on residuals,
    then take the diagonal (N,).

    prices_df — wide close prices (dates × symbols)
    loadings_df — (symbols × 6) loadings for the as-of date
    factor_returns — factor return DataFrame

    Returns 1-D array of length N (residual variances, annualised).
    """
    syms = list(loadings_df.index)
    common_idx = prices_df.index.intersection(factor_returns.index)
    window_returns = prices_df[syms].reindex(common_idx).pct_change().tail(lookback).dropna(how="all")
    factors_window = factor_returns[_FACTOR_SRC].reindex(window_returns.index).dropna()
    window_returns = window_returns.reindex(factors_window.index)

    B = loadings_df.reindex(syms)[BETA_COLS].values  # N × 6

    residuals = []
    for i, sym in enumerate(syms):
        r = window_returns[sym].fillna(0).values
        f = factors_window.values
        fitted = f @ B[i]
        residuals.append(r - fitted)

    resid_matrix = np.column_stack(residuals)  # T × N

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(resid_matrix)
        D = np.diag(lw.covariance_) * 252
    except Exception:
        D = np.nanvar(resid_matrix, axis=0) * 252

    return np.maximum(D, 1e-6)  # floor at tiny positive


def build_factor_covariance_matrix(
    weights_series: pd.Series,
    as_of_date,
    lookback: int = 252,
) -> dict:
    """
    Build full factor-structured covariance for the given portfolio.

    Returns dict:
      full      — N×N covariance matrix (numpy)
      factor    — 6×6 factor covariance (numpy)
      loadings  — N×6 loadings (numpy, same order as weights_series)
      idiosyncractic — (N,) residual variances
      symbols   — list of symbols
    """
    syms = list(weights_series.index)
    N = len(syms)

    loadings_df = get_factor_loadings(as_of_date).reindex(syms).dropna()
    valid_syms = list(loadings_df.index)

    factor_returns = get_factor_returns()

    # Graceful degradation when factor data unavailable
    if loadings_df.empty or factor_returns.empty:
        log.warning("[CovModel] Factor data unavailable — using diagonal proxy")
        sigma = np.eye(N) * 0.04
        return {
            "full": sigma, "factor": np.eye(6) * 0.04,
            "loadings": np.zeros((N, 6)), "idiosyncratic": np.full(N, 0.04),
            "symbols": syms,
        }

    B_valid = loadings_df[BETA_COLS].values              # M × 6 (M ≤ N)
    F = compute_factor_covariance(factor_returns, lookback)  # 6 × 6

    # Prices for residual computation
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("prices_live"):
            raw = load_parquet("prices_live")
            if "symbol" in raw.columns and "close" in raw.columns:
                prices_wide = raw.pivot_table(index="date", columns="symbol", values="close").sort_index()
            elif isinstance(raw.columns, pd.MultiIndex):
                prices_wide = raw["Close"].sort_index()
            else:
                prices_wide = raw.sort_index()
        else:
            prices_wide = pd.DataFrame()
    except Exception:
        prices_wide = pd.DataFrame()

    if not prices_wide.empty and set(valid_syms).issubset(set(prices_wide.columns)):
        D_valid = compute_residual_variances(prices_wide, loadings_df, factor_returns, lookback)
    else:
        D_valid = np.full(len(valid_syms), 0.04)

    # Build full N×N matrix: map valid symbols back to full index
    B_full = np.zeros((N, 6))
    D_full = np.full(N, 0.04)
    for i, sym in enumerate(syms):
        if sym in valid_syms:
            j = valid_syms.index(sym)
            B_full[i] = B_valid[j]
            D_full[i] = D_valid[j]

    Sigma = B_full @ F @ B_full.T + np.diag(D_full)

    return {
        "full":          Sigma,
        "factor":        F,
        "loadings":      B_full,
        "idiosyncratic": D_full,
        "symbols":       syms,
    }


def portfolio_variance(weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
    """Scalar portfolio variance: w'Σw."""
    w = np.asarray(weights, dtype=float)
    return float(w @ covariance_matrix @ w)


def factor_variance_decomposition(
    weights: np.ndarray,
    loadings: np.ndarray,
    factor_cov: np.ndarray,
    idiosyncratic: np.ndarray,
) -> dict:
    """
    Split portfolio variance into factor-explained and idiosyncratic.

    Returns {"factor_explained": float, "idiosyncratic": float, "total": float}.
    """
    w = np.asarray(weights, dtype=float)
    factor_var = float(w @ loadings @ factor_cov @ loadings.T @ w)
    idio_var = float(w @ np.diag(idiosyncratic) @ w)
    total = factor_var + idio_var
    return {
        "factor_explained": round(factor_var, 8),
        "idiosyncratic":    round(idio_var, 8),
        "total":            round(total, 8),
    }
