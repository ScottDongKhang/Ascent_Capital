"""
Ascent Capital — Target Definitions
Forward returns used as prediction targets.

CRITICAL: Targets are forward-looking by definition.
They must NEVER be used as features.
They are used only for model training and evaluation.
The walk-forward split logic ensures proper temporal separation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward return over `horizon` days.
    At date t, this is close[t+horizon] / close[t] - 1.

    This IS the future and must only be used as a TARGET, never a feature.
    """
    return close.shift(-horizon) / close - 1


def forward_return_open(close: pd.DataFrame, open_prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    More realistic target: return from next-open execution to horizon close.
    At date t: close[t+horizon] / open[t+1] - 1

    This aligns with execution model: signal at t close, execute at t+1 open.
    """
    entry_price = open_prices.shift(-1)  # next day's open
    exit_price = close.shift(-horizon)
    return exit_price / entry_price - 1


def forward_return_rank(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cross-sectional rank of forward return (0 to 1). Used for rank-based alpha."""
    fwd = forward_return(close, horizon)
    return fwd.rank(axis=1, pct=True)


def build_targets(
    close: pd.DataFrame,
    open_prices: pd.DataFrame,
    horizons: list[int] = [1, 5, 21],
) -> dict[str, pd.DataFrame]:
    """
    Build all target variables.
    Returns dict of {target_name: DataFrame(dates × symbols)}.
    """
    targets = {}
    for h in horizons:
        targets[f"fwd_ret_{h}d"] = forward_return(close, h)
        targets[f"fwd_ret_open_{h}d"] = forward_return_open(close, open_prices, h)
        targets[f"fwd_ret_rank_{h}d"] = forward_return_rank(close, h)
    return targets
