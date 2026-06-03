"""
Walk-Forward Engine
===================
Top-level orchestrator. Call `engine.run(data, benchmark)` to get the
stitched OOS equity curve and full performance report.

Boundary defenses enforced here
--------------------------------
1. Optimizer receives `data.loc[w.slice_is(dates)]` only — physically
   cannot see any OOS data (slice_is excludes purge tail).

2. Strategy's generate_signals called on context from IS start → OOS end
   (for warmup), but only OOS-date signals are harvested and evaluated.

3. `last_windows_` stored for post-run inspection and testing.
"""
from __future__ import annotations
from typing import Callable, Type
import numpy as np
import pandas as pd

from .windows import WindowGenerator, SplitWindow
from .strategy import BaseStrategy
from .execution import ExecutionModel, ExecutionConfig
from .optimizer import ParameterOptimizer
from .metrics import PerformanceAnalyzer, FoldResult


class WalkForwardEngine:

    def __init__(
        self,
        strategy_cls:     Type[BaseStrategy],
        window_generator: WindowGenerator | None = None,
        exec_config:      ExecutionConfig | None = None,
        objective:        str = "sharpe",
        constraint_fn:    Callable[[dict], bool] | None = None,
        rf_annual:        float = 0.0,
        verbose:          bool = True,
    ):
        self.strategy_cls     = strategy_cls
        self.wg               = window_generator or WindowGenerator()
        self.exec_model       = ExecutionModel(exec_config or ExecutionConfig())
        self.optimizer        = ParameterOptimizer(
            strategy_cls, self.exec_model, objective, constraint_fn, rf_annual
        )
        self.analyser         = PerformanceAnalyzer(rf_annual)
        self.verbose          = verbose
        self.last_windows_: list[SplitWindow] = []

    def run(
        self,
        data:      pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> tuple[pd.Series, dict]:
        dates   = data.index.drop_duplicates().sort_values()
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  WALK-FORWARD ENGINE — {len(windows)} folds")
            print(f"  IS: {self.wg.is_days}d | OOS: {self.wg.oos_days}d | "
                  f"Purge: {self.wg.purge_days}d | Embargo: {self.wg.embargo_days}d")
            print(f"{'='*60}")

        fold_results:        list[FoldResult]  = []
        oos_return_chunks:   list[pd.Series]   = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            # BOUNDARY DEFENSE: optimizer receives IS slice only
            is_data = data.loc[is_dates]
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            # Strategy sees IS-start → OOS-end for warmup; harvest OOS signals only
            full_context_dates = dates[
                (dates >= w.is_start) & (dates <= w.oos_end)
            ]
            full_context_data  = data.loc[full_context_dates]
            strategy           = self.strategy_cls(**best_params)
            all_signals        = strategy.generate_signals(full_context_data)
            oos_signals        = all_signals.loc[oos_dates]
            oos_data           = data.loc[oos_dates]
            oos_returns        = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                oos_sharpe = self.analyser.sharpe(oos_returns)
                print(
                    f"  Fold {w.fold_id}: "
                    f"IS [{w.is_start.date()} → {w.purge_start.date()}) "
                    f"OOS [{w.oos_start.date()} → {w.oos_end.date()}] "
                    f"params={best_params} IS_Sh={is_sharpe:.2f} OOS_Sh={oos_sharpe:.2f}"
                )

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        if not oos_return_chunks:
            raise RuntimeError("All folds skipped — no OOS returns produced.")

        stitched_returns = pd.concat(oos_return_chunks).sort_index()
        stitched_returns = stitched_returns[~stitched_returns.index.duplicated(keep="first")]

        equity_curve = (1 + stitched_returns).cumprod()
        equity_curve = equity_curve / equity_curve.iloc[0]

        bm_aligned = None
        if benchmark is not None:
            bm_aligned = benchmark.reindex(stitched_returns.index).fillna(0)

        report = self.analyser.full_report(stitched_returns, bm_aligned, fold_results)

        if self.verbose:
            print()
            self.analyser.print_report(report)

        return equity_curve, report
