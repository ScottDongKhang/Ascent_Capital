"""
Parameter Optimizer
===================
Grid-searches a strategy's param_grid on in-sample data only.

Boundary defense
----------------
`optimize(is_data)` receives only the IS date slice. The WalkForwardEngine
is responsible for slicing before calling — the optimizer has NO knowledge
of the full dataset and cannot access OOS data even accidentally.

Objective
---------
Default: annualised Sharpe ratio.
"calmar": Calmar ratio (CAGR / |max drawdown|).
Invalid combos receive -inf.

Constraints
-----------
Pass `constraint_fn` to filter invalid combos (e.g., fast >= slow).
"""
from __future__ import annotations
from itertools import product
from typing import Callable, Type
import inspect
import numpy as np
import pandas as pd

from .strategy import BaseStrategy
from .execution import ExecutionModel


def _sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    if ann_vol < 1e-10:
        return -np.inf
    return (ann_ret - rf_annual) / ann_vol


def _calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns).prod() ** (252 / max(len(returns), 1)) - 1
    cum     = (1 + returns).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    if abs(mdd) < 1e-10:
        return -np.inf
    return ann_ret / abs(mdd)


_OBJECTIVES = {"sharpe": _sharpe, "calmar": _calmar}


class ParameterOptimizer:
    """
    Grid-search optimizer for BaseStrategy subclasses.

    Parameters
    ----------
    strategy_cls   : Subclass of BaseStrategy to optimise.
    execution_model: ExecutionModel instance for friction.
    objective      : "sharpe" (default) or "calmar".
    constraint_fn  : Optional callable(params_dict) → bool.
    rf_annual      : Risk-free rate for Sharpe calculation.
    """

    def __init__(
        self,
        strategy_cls:    Type[BaseStrategy],
        execution_model: ExecutionModel,
        objective:       str = "sharpe",
        constraint_fn:   Callable[[dict], bool] | None = None,
        rf_annual:       float = 0.0,
    ):
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {list(_OBJECTIVES)}")
        self.strategy_cls    = strategy_cls
        self.execution_model = execution_model
        self.objective_fn    = _OBJECTIVES[objective]
        self.constraint_fn   = constraint_fn
        self.rf_annual       = rf_annual

    def optimize(self, is_data: pd.DataFrame) -> tuple[dict, float]:
        """
        Search param_grid on IS data only. Returns (best_params, best_score).
        """
        template     = self.strategy_cls()
        grid         = template.param_grid
        param_names  = list(grid.keys())
        param_values = list(grid.values())

        best_params: dict  = {}
        best_score:  float = -np.inf

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            if self.constraint_fn is not None and not self.constraint_fn(params):
                continue

            try:
                strategy = self.strategy_cls(**params)
                signals  = strategy.generate_signals(is_data)
                returns  = self.execution_model.compute_returns(is_data, signals)
                sig = inspect.signature(self.objective_fn)
                if "rf_annual" in sig.parameters:
                    score = self.objective_fn(returns, self.rf_annual)
                else:
                    score = self.objective_fn(returns)
            except Exception:
                score = -np.inf

            if np.isfinite(score) and score > best_score:
                best_score  = score
                best_params = params

        if not best_params:
            best_params = {k: v[0] for k, v in grid.items()}

        return best_params, best_score
