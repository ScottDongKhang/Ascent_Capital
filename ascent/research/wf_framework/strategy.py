"""
Strategy Interface
==================
BaseStrategy is the ABC every user strategy must subclass.

Implementing a strategy
-----------------------
1. Subclass BaseStrategy.
2. Define `param_grid` property — dict of param_name → list of candidate values.
3. Implement `generate_signals(data)` — accepts OHLCV DataFrame, returns
   a pd.Series of {-1, 0, +1} aligned to the input index.

The engine instantiates your strategy class with **params at optimisation time
and at OOS evaluation time. generate_signals must be strictly causal: signal
at index i may only use data at indices <= i.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base for all walk-forward strategies."""

    def __init__(self, **params):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    @property
    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return {param_name: [candidate_values]} for grid search."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute entry/exit signals from OHLCV data.

        Parameters
        ----------
        data : DataFrame with columns [open, high, low, close, volume].
               Must be sorted ascending by date index. No future data allowed.

        Returns
        -------
        pd.Series of int: +1 (long), -1 (short), 0 (flat).
        Index must match data.index exactly.
        """


class MACrossStrategy(BaseStrategy):
    """
    Simple moving-average crossover.

    Long  when fast SMA > slow SMA.
    Short when fast SMA < slow SMA.
    Flat  during warmup (first `slow` bars).

    Parameters
    ----------
    fast : int — fast SMA lookback (default 10)
    slow : int — slow SMA lookback (default 50)
    """

    def __init__(self, fast: int = 10, slow: int = 50):
        super().__init__(fast=fast, slow=slow)

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [5, 10, 20],
            "slow": [30, 50, 100, 200],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()

        signal = pd.Series(0, index=data.index, dtype=int)
        valid  = fast_ma.notna() & slow_ma.notna()
        signal.loc[valid & (fast_ma > slow_ma)] =  1
        signal.loc[valid & (fast_ma < slow_ma)] = -1
        return signal
