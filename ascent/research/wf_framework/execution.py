"""
Execution Friction Model
========================
Converts a signal (single-asset Series or portfolio weight DataFrame) into a
net-of-friction daily return series.

Single-asset path  : signals is pd.Series of {-1, 0, +1}
Portfolio path     : signals is pd.DataFrame (dates × symbols) of weights

Three friction layers (both paths):
  1. Slippage  — ATR-based or fixed-pct, applied on position-change bars.
  2. Commission — flat pct of notional traded (turnover).
  3. Borrow    — overnight financing for shorts (single-asset path only).

Boundary defense: 1-day execution delay — signal/weight at date t is active
from date t+1 onward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
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

    # ------------------------------------------------------------------
    # Public API — type-dispatches on signals type
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        data:    pd.DataFrame,
        signals: Union[pd.Series, pd.DataFrame],
    ) -> pd.Series:
        if isinstance(signals, pd.DataFrame):
            return self._portfolio_returns(data, signals)
        return self._single_asset_returns(data, signals)

    # ------------------------------------------------------------------
    # Single-asset path (unchanged from original)
    # ------------------------------------------------------------------

    def _single_asset_returns(
        self, data: pd.DataFrame, signals: pd.Series
    ) -> pd.Series:
        cfg   = self.cfg
        delay = cfg.execution_delay

        position  = signals.shift(delay).fillna(0)
        raw_ret   = data["close"].pct_change().fillna(0)
        gross_ret = position * raw_ret

        pos_change = position.diff().fillna(position)
        is_trade   = pos_change.abs() > 0

        if cfg.slippage_model == "atr":
            atr        = self._atr_single(data, cfg.atr_window)
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

    # ------------------------------------------------------------------
    # Portfolio path (new)
    # ------------------------------------------------------------------

    def _portfolio_returns(
        self, data: pd.DataFrame, weight_df: pd.DataFrame
    ) -> pd.Series:
        """
        data      : long-format OHLCV with 'date' and 'symbol' columns.
        weight_df : pd.DataFrame (dates × symbols) from PortfolioBaseStrategy.
        """
        cfg   = self.cfg
        delay = cfg.execution_delay

        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        close_w = data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        open_w  = data.pivot_table(index="date", columns="symbol", values="open",  aggfunc="last").sort_index()
        high_w  = data.pivot_table(index="date", columns="symbol", values="high",  aggfunc="last").sort_index()
        low_w   = data.pivot_table(index="date", columns="symbol", values="low",   aggfunc="last").sort_index()

        # Align weights and prices to common dates and symbols
        common_dates = weight_df.index.intersection(close_w.index)
        all_syms     = weight_df.columns.intersection(close_w.columns)

        weights = weight_df.reindex(index=common_dates, columns=all_syms, fill_value=0.0)
        close   = close_w.reindex(index=common_dates,  columns=all_syms)
        open_   = open_w.reindex(index=common_dates,   columns=all_syms)
        high_   = high_w.reindex(index=common_dates,   columns=all_syms)
        low_    = low_w.reindex(index=common_dates,    columns=all_syms)

        # Daily returns per symbol
        sym_ret = close.pct_change().fillna(0)

        # Portfolio return: delayed weights × symbol returns
        delayed_w = weights.shift(delay).fillna(0)
        gross_ret = (delayed_w * sym_ret).sum(axis=1)

        # Turnover: sum of abs weight changes per bar
        w_diff = weights.diff()
        w_diff.iloc[0] = weights.iloc[0].abs()
        turnover = w_diff.abs().sum(axis=1)

        # Commission on turnover
        commission_deduction = turnover * cfg.commission_pct

        # Slippage on turnover
        if cfg.slippage_model == "atr":
            prev_c  = close.shift(1)
            tr_vals = np.maximum(
                np.maximum(
                    (high_ - low_).fillna(0).values,
                    (high_ - prev_c).abs().fillna(0).values,
                ),
                (low_ - prev_c).abs().fillna(0).values,
            )
            tr  = pd.DataFrame(tr_vals, index=common_dates, columns=all_syms)
            atr = tr.ewm(alpha=1 / cfg.atr_window, adjust=False).mean()
            slip_per_sym   = cfg.atr_multiplier * atr / open_.clip(lower=1e-8)
            slip_deduction = (w_diff.abs() * slip_per_sym.fillna(0)).sum(axis=1)
        else:
            slip_deduction = turnover * cfg.fixed_pct

        net_ret = gross_ret - commission_deduction - slip_deduction
        return net_ret.rename("net_return")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atr_single(data: pd.DataFrame, window: int) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False).mean()
