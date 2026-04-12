"""
Ascent Capital — Alpha Stack
Combines multiple alpha signals into a composite score.
The stack determines the final ranking used for portfolio construction.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ascent.alpha.trend import trend_alpha
from ascent.alpha.meanrev import meanrev_alpha


def build_alpha_stack(
    features: dict[str, pd.DataFrame],
    alpha_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build composite alpha by combining individual alpha signals.

    Args:
        features: Dict of feature DataFrames (dates × symbols)
        alpha_weights: Optional override for alpha signal weights

    Returns:
        DataFrame(dates × symbols) with composite alpha scores
    """
    if alpha_weights is None:
        alpha_weights = {
            "trend": 0.60,
            "meanrev": 0.25,
            "volatility": 0.15,
        }

    alphas = {}

    # Trend alpha
    trend = trend_alpha(features)
    if not trend.empty:
        alphas["trend"] = trend

    # Mean reversion alpha
    mr = meanrev_alpha(features)
    if not mr.empty:
        alphas["meanrev"] = mr

    # Volatility alpha: prefer lower volatility (risk-adjusted quality)
    if "vol_21d" in features:
        vol_alpha = -_cs_normalize(features["vol_21d"].copy())  # prefer low vol
        alphas["volatility"] = vol_alpha

    if not alphas:
        raise ValueError("No alpha signals could be computed")

    # Weight and combine
    total_w = sum(alpha_weights.get(k, 0) for k in alphas)
    if total_w == 0:
        total_w = 1.0

    composite = None
    for name, alpha_df in alphas.items():
        w = alpha_weights.get(name, 0) / total_w
        if composite is None:
            composite = alpha_df * w
        else:
            # Align indexes
            common_idx = composite.index.intersection(alpha_df.index)
            common_cols = composite.columns.intersection(alpha_df.columns)
            composite = composite.reindex(index=common_idx, columns=common_cols)
            alpha_df = alpha_df.reindex(index=common_idx, columns=common_cols)
            composite = composite + alpha_df * w

    return composite


def alpha_to_ranks(alpha: pd.DataFrame) -> pd.DataFrame:
    """Convert alpha scores to cross-sectional ranks (0 to 1)."""
    return alpha.rank(axis=1, pct=True)


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)
