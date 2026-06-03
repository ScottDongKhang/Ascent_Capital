"""
Execution Friction Model
========================
Converts a signal series into a net-of-friction daily return series.

Three friction layers applied in order on each bar:
  1. Slippage  — cost on the open of the execution bar (bar after signal).
  2. Commission — flat percentage of notional traded.
  3. Borrow    — daily overnight financing cost on short positions.

Slippage models
---------------
"atr"       : slippage = atr_multiplier × ATR(14). Scales with volatility.
"fixed_pct" : slippage = fixed_pct × execution_price. Constant percentage.

Signal convention
-----------------
+1 = long, -1 = short, 0 = flat.

Boundary defense: signals are shifted by execution_delay bars before
computing returns so signal at date t is executed at open of date t+1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    slippage_model:     Literal["atr", "fixed_pct"] = "atr"
    atr_multiplier:     float = 0.10
    fixed_pct:          float = 0.0005
    atr_window:         int   = 14
    commission_pct:     float = 0.0005
    borrow_rate_annual: float = 0.0
    execution_delay:    int   = 1


class ExecutionModel:
    def __init__(self, config: ExecutionConfig | None = None):
        self.cfg = config or ExecutionConfig()

    def compute_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        cfg   = self.cfg
        delay = cfg.execution_delay

        position = signals.shift(delay).fillna(0)
        raw_ret  = data["close"].pct_change().fillna(0)
        gross_ret = position * raw_ret

        pos_change = position.diff().fillna(position)
        is_trade   = pos_change.abs() > 0

        if cfg.slippage_model == "atr":
            atr = self._atr(data, cfg.atr_window)
            exec_price = data["open"].shift(-delay).clip(lower=1e-8)
            slip_cost  = cfg.atr_multiplier * atr / exec_price
        else:
            slip_cost = pd.Series(cfg.fixed_pct, index=data.index)

        slip_cost = slip_cost.fillna(0)
        slip_deduction       = is_trade * pos_change.abs() * slip_cost
        commission_deduction = is_trade * cfg.commission_pct
        daily_borrow_rate    = cfg.borrow_rate_annual / 252
        borrow_deduction     = (position < 0).astype(float) * daily_borrow_rate

        net_ret = gross_ret - slip_deduction - commission_deduction - borrow_deduction
        return net_ret.rename("net_return")

    @staticmethod
    def _atr(data: pd.DataFrame, window: int) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False).mean()
