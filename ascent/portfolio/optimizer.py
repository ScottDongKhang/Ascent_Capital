"""
Ascent Capital — Portfolio Optimizer
Converts alpha scores into portfolio target weights.

This is PORTFOLIO-NATIVE: we think in weights across the universe,
not individual BUY/SELL decisions.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


class SectorDataError(RuntimeError):
    """Raised when sector coverage is below the 80% threshold required for safe portfolio construction."""
    pass


def top_n_equal_weight(
    alpha: pd.DataFrame,
    n: int = 10,
    max_weight: float = 0.15,
    min_weight: float = 0.02,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Simple top-N portfolio: select top N stocks by alpha, equal weight.

    Args:
        alpha: DataFrame(dates × symbols) with alpha scores
        n: Number of positions
        max_weight: Maximum weight per position
        min_weight: Minimum weight per position
        long_only: If True, only long positions

    Returns:
        DataFrame(dates × symbols) with target weights (sum ≈ 1.0)
    """
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < n:
            continue

        if long_only:
            top = row.nlargest(n)
        else:
            top = row.nlargest(n)

        w = 1.0 / len(top)
        w = np.clip(w, min_weight, max_weight)

        for sym in top.index:
            weights.loc[dt, sym] = w

        # Renormalize to sum to ~1
        row_sum = weights.loc[dt].sum()
        if row_sum > 0:
            weights.loc[dt] = weights.loc[dt] / row_sum

    return weights


def rank_weighted(
    alpha: pd.DataFrame,
    n: int = 10,
    max_weight: float = 0.15,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Rank-weighted portfolio: top N stocks weighted proportional to alpha rank.
    Higher alpha → higher weight.
    """
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < n:
            continue

        top = row.nlargest(n)

        # Rank-proportional weights (shift to be positive)
        scores = top - top.min() + 1e-8
        raw_w = scores / scores.sum()

        # Apply max weight constraint (iterate until stable)
        for _ in range(10):
            raw_w = raw_w.clip(upper=max_weight)
            raw_w = raw_w / raw_w.sum()
            if raw_w.max() <= max_weight + 1e-9:
                break

        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]

    return weights


def apply_turnover_penalty(
    new_weights: pd.DataFrame,
    prev_weights: pd.Series,
    turnover_penalty: float = 0.5,
) -> pd.Series:
    """
    Blend new target weights toward previous weights to reduce turnover.
    penalty = 0: full rebalance. penalty = 1: no change.
    """
    if prev_weights is None or prev_weights.empty:
        return new_weights
    return (1 - turnover_penalty) * new_weights + turnover_penalty * prev_weights


def enforce_constraints(
    weights: pd.Series,
    max_weight: float = 0.15,
    min_weight: float = 0.01,
    max_gross: float = 1.0,
) -> pd.Series:
    """Enforce portfolio constraints on a single-date weight vector."""
    w = weights.copy()

    # Remove tiny positions
    w[w.abs() < min_weight] = 0.0

    # Cap individual weights
    w = w.clip(-max_weight, max_weight)

    # Normalize if needed
    gross = w.abs().sum()
    if gross > max_gross and gross > 0:
        w = w * (max_gross / gross)

    return w


def sector_constrained_weighted(alpha, n=10, max_weight=0.15, max_per_sector=1, sector_map=None):
    import pandas as pd
    import numpy as np
    if sector_map is None:
        sector_map = {}
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    for dt in alpha.index:
        row = alpha.loc[dt].dropna()
        if len(row) < n:
            continue
        ranked = row.sort_values(ascending=False)

        # Check sector coverage: how many of the candidate symbols have known sectors
        known_sectors = sum(1 for sym in ranked.index if sym in sector_map)
        coverage = known_sectors / len(ranked) if len(ranked) > 0 else 0.0

        # Hard fail if coverage is below 80%
        if coverage < 0.80:
            raise SectorDataError(
                f"sector_constrained_weighted(): sector coverage {coverage:.0%} < 80% "
                f"on {dt.date()}. This should have been caught by validate_sector_data() "
                f"at startup. Regenerate profiles.parquet or pass --skip-sector-check."
            )

        selected = []
        sector_count = {}
        for sym in ranked.index:
            sec = sector_map.get(sym, "Unknown")
            if sector_count.get(sec, 0) < max_per_sector:
                selected.append(sym)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= n:
                break
        if not selected:
            continue
        scores = ranked[selected]
        scores = scores - scores.min() + 1e-8
        raw_w = scores / scores.sum()
        for _ in range(10):
            raw_w = raw_w.clip(upper=max_weight)
            raw_w = raw_w / raw_w.sum()
            if raw_w.max() <= max_weight + 1e-9:
                break
        for sym in raw_w.index:
            weights.loc[dt, sym] = raw_w[sym]
    return weights
