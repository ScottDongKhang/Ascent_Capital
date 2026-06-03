"""
Portfolio Strategy Interface
============================
PortfolioBaseStrategy is the ABC for multi-asset walk-forward strategies.

Implementing a portfolio strategy
----------------------------------
1. Subclass PortfolioBaseStrategy.
2. Define `param_grid` property — dict of param_name → list of candidate values.
3. Implement `generate_signals(data)` — accepts long-format OHLCV DataFrame,
   returns pd.DataFrame (dates × symbols) of portfolio weights.
4. Optionally override `clear_cache()` if the strategy caches intermediate results.

Boundary defense: `generate_signals` must be strictly causal — weight at date t
may only use data at dates <= t. The engine enforces this by passing only the
[IS_start → OOS_end] context window.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class PortfolioBaseStrategy(ABC):
    """Abstract base for multi-asset portfolio strategies."""

    def __init__(self, **params):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    @property
    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """Return {param_name: [candidate_values]} for grid search."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : long-format OHLCV DataFrame with columns
               [date, symbol, open, high, low, close, volume].
               Sorted ascending by date. No future data allowed.

        Returns
        -------
        pd.DataFrame indexed by date, columns = symbol tickers.
        Values = portfolio weights in [0, max_weight].
        Each row sums to <= 1.0 (cash allowed).
        """

    def clear_cache(self) -> None:
        """Clear any internal caches. Engine calls this between folds."""
        pass
