"""
ascent/research/factor_discovery/pysr_engine.py

Symbolic regression via PySR for factor discovery.

PySR evolves human-readable mathematical expressions (e.g. "sqrt(vol_21d) /
(mom_252d + 1e-6)") using genetic programming on pre-computed features.
Output is a formula string, not Python code — safe by construction.

Fallback: if PySR unavailable, returns empty list.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_UNARY_OPERATORS  = ["sqrt", "log", "abs", "neg"]
_BINARY_OPERATORS = ["+", "-", "*", "/"]


def _compute_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional feature panel from price DataFrame."""
    n    = len(prices_df)
    cols = prices_df.columns
    rows = {}

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    rets = prices_df.pct_change()

    if n >= 22:
        rows["mom_21d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-21] - 1)
    if n >= 64:
        rows["mom_63d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-63] - 1)
    if n >= 127:
        rows["mom_126d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-126] - 1)
    if n >= 253:
        rows["mom_252d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-252] - 1)
    if n >= 6:
        rows["rev_5d"] = _zscore(-(prices_df.iloc[-1] / prices_df.iloc[-5] - 1))
    if n >= 11:
        rows["rev_10d"] = _zscore(-(prices_df.iloc[-1] / prices_df.iloc[-10] - 1))
    if n >= 22:
        rows["vol_21d"] = _zscore(-rets.tail(21).std())
    if n >= 64:
        rows["vol_63d"] = _zscore(-rets.tail(63).std())
    if n >= 22:
        roll_mean = rets.tail(252).rolling(21).mean()
        roll_std  = rets.tail(252).rolling(21).std()
        rows["zscore_21d"] = _zscore(roll_mean.iloc[-1] / (roll_std.iloc[-1] + 1e-8))
    if n >= 253:
        peak = prices_df.tail(252).max()
        rows["high_52w_pct"] = _zscore(prices_df.iloc[-1] / (peak + 1e-8) - 1)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, index=cols).dropna(how="all")


def _run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_iterations: int = 40,
    population_size: int = 30,
) -> Optional[Tuple[str, Callable]]:
    try:
        from pysr import PySRRegressor
        model = PySRRegressor(
            niterations=n_iterations,
            population_size=population_size,
            binary_operators=_BINARY_OPERATORS,
            unary_operators=_UNARY_OPERATORS,
            maxsize=10,
            verbosity=0,
            progress=False,
            random_state=42,
        )
        model.fit(X, y, variable_names=feature_names)
        best     = model.get_best()
        expr_str = str(best["equation"])
        fn       = model.predict
        return expr_str, fn
    except ImportError:
        log.info("[PySR] pysr not installed — skipping symbolic regression path")
        return None
    except Exception as exc:
        log.warning("[PySR] Run failed: %s", exc)
        return None


def discover_via_pysr(
    prices_df: pd.DataFrame,
    n_periods: int = 5,
    lookback_days: int = 252,
    n_iterations: int = 40,
) -> List[Dict]:
    """
    Run PySR on rolling cross-sectional dataset to discover symbolic factors.
    Returns list of {name, expression, description, source, fn}.
    Empty list if PySR unavailable or data insufficient.
    """
    if len(prices_df) < lookback_days + n_periods + 50:
        return []

    X_rows, y_rows = [], []
    dates = prices_df.index[lookback_days:-n_periods:5]
    feats = pd.DataFrame()

    for dt in dates:
        try:
            iloc_pos = prices_df.index.get_loc(dt)
            window   = prices_df.iloc[max(0, iloc_pos - lookback_days): iloc_pos + 1]
            feats    = _compute_features(window)
            if feats.empty or len(feats) < 5:
                continue
            fwd_rets = prices_df.iloc[iloc_pos + n_periods] / prices_df.iloc[iloc_pos] - 1
            common   = feats.index.intersection(fwd_rets.index)
            X_rows.append(feats.reindex(common).values)
            y_rows.append(fwd_rets.reindex(common).values)
        except Exception:
            continue

    if not X_rows or len(X_rows) < 5:
        return []

    try:
        X = np.vstack(X_rows)
        y = np.concatenate(y_rows)
        available_features = list(feats.columns)
    except Exception as exc:
        log.warning("[PySR] Stack failed: %s", exc)
        return []

    result = _run_pysr(X, y, available_features, n_iterations=n_iterations)
    if result is None:
        return []

    expr_str, pysr_predict_fn = result
    feature_names_used = available_features

    def _factor_fn(df: pd.DataFrame, _expr=expr_str, _feats=feature_names_used,
                   _predict=pysr_predict_fn) -> pd.Series:
        feats_df = _compute_features(df)
        if feats_df.empty:
            return pd.Series(0.0, index=df.columns)
        X_eval = feats_df.reindex(columns=_feats, fill_value=0.0).values
        try:
            scores = _predict(X_eval)
            s   = pd.Series(scores, index=feats_df.index)
            std = s.std()
            if std < 1e-8:
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std
        except Exception:
            return pd.Series(0.0, index=feats_df.index)

    return [{
        "name":        f"factor_pysr_{abs(hash(expr_str)) % 10000:04d}",
        "expression":  expr_str,
        "description": f"PySR-discovered symbolic expression: {expr_str}",
        "source":      "pysr",
        "fn":          _factor_fn,
    }]
