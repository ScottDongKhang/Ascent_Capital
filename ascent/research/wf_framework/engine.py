"""
Walk-Forward Engine
===================
Top-level orchestrator. Supports both:
  - Single-asset strategies (BaseStrategy → pd.Series signals)
  - Portfolio strategies  (PortfolioBaseStrategy → pd.DataFrame weights)

Call `engine.run(data, benchmark)` to get the stitched OOS equity curve
and full performance report.

Boundary defenses
-----------------
1. IS optimizer receives IS-slice data only — cannot see OOS dates.
   Single-asset: `data.loc[is_dates]`
   Portfolio:    `data[data["date"].isin(is_dates)]`

2. Strategy's generate_signals called on [IS_start → OOS_end] for warmup;
   only OOS signals/weights harvested for evaluation.

3. For portfolio strategies: `strategy_cls.clear_cache()` called before each
   fold's IS optimization to prevent cross-fold cache contamination.

4. `last_windows_` stored for post-run inspection and testing.
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


def _is_portfolio(strategy_cls) -> bool:
    """Return True if strategy_cls is a PortfolioBaseStrategy subclass."""
    try:
        from .portfolio_strategy import PortfolioBaseStrategy
        return issubclass(strategy_cls, PortfolioBaseStrategy)
    except ImportError:
        return False


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
        self._portfolio_mode  = _is_portfolio(strategy_cls)

    def run(
        self,
        data:      pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> tuple[pd.Series, dict]:
        if self._portfolio_mode:
            return self._run_portfolio(data, benchmark)
        return self._run_single_asset(data, benchmark)

    # ------------------------------------------------------------------
    # Single-asset path (original, unchanged)
    # ------------------------------------------------------------------

    def _run_single_asset(
        self, data: pd.DataFrame, benchmark: pd.Series | None
    ) -> tuple[pd.Series, dict]:
        dates   = data.index.drop_duplicates().sort_values()
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            self._print_header(len(windows))

        fold_results:      list[FoldResult] = []
        oos_return_chunks: list[pd.Series]  = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            is_data        = data.loc[is_dates]
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            full_context_dates = dates[(dates >= w.is_start) & (dates <= w.oos_end)]
            full_context_data  = data.loc[full_context_dates]
            strategy           = self.strategy_cls(**best_params)
            all_signals        = strategy.generate_signals(full_context_data)
            oos_signals        = all_signals.loc[oos_dates]
            oos_data           = data.loc[oos_dates]
            oos_returns        = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                self._print_fold(w, best_params, is_sharpe,
                                 self.analyser.sharpe(oos_returns))

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        return self._finalise(oos_return_chunks, fold_results, benchmark)

    # ------------------------------------------------------------------
    # Portfolio path (new)
    # ------------------------------------------------------------------

    def _run_portfolio(
        self, data: pd.DataFrame, benchmark: pd.Series | None
    ) -> tuple[pd.Series, dict]:
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        # Strip tz so dates from engine and generate_signals are both tz-naive
        if data["date"].dt.tz is not None:
            data["date"] = data["date"].dt.tz_localize(None)

        # Extract unique trading dates from the long-format 'date' column
        dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        windows = self.wg.generate(dates)
        self.last_windows_ = windows

        if not windows:
            raise ValueError("No valid walk-forward windows — dataset too short.")

        if self.verbose:
            self._print_header(len(windows))

        fold_results:      list[FoldResult] = []
        oos_return_chunks: list[pd.Series]  = []

        for w in windows:
            is_dates  = w.slice_is(dates)
            oos_dates = w.slice_oos(dates)

            if len(is_dates) < 30 or len(oos_dates) < 5:
                if self.verbose:
                    print(f"  Fold {w.fold_id}: SKIPPED — insufficient data")
                continue

            # BOUNDARY DEFENSE: clear cache, then give optimizer IS slice only
            self.strategy_cls.clear_cache()
            is_dates_set           = set(is_dates)
            is_data                = data[data["date"].isin(is_dates_set)]
            best_params, is_sharpe = self.optimizer.optimize(is_data)

            # Full context: IS_start → OOS_end for warmup
            full_context_set  = set(dates[(dates >= w.is_start) & (dates <= w.oos_end)])
            full_context_data = data[data["date"].isin(full_context_set)]

            strategy    = self.strategy_cls(**best_params)
            all_signals = strategy.generate_signals(full_context_data)  # DataFrame

            oos_dates_set = set(oos_dates)
            oos_signals   = all_signals.loc[all_signals.index.isin(oos_dates_set)]
            oos_data      = data[data["date"].isin(oos_dates_set)]
            oos_returns   = self.exec_model.compute_returns(oos_data, oos_signals)

            if self.verbose:
                self._print_fold(w, best_params, is_sharpe,
                                 self.analyser.sharpe(oos_returns))

            fold_results.append(FoldResult(w.fold_id, is_sharpe, oos_returns))
            oos_return_chunks.append(oos_returns)

        return self._finalise(oos_return_chunks, fold_results, benchmark)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        oos_return_chunks: list[pd.Series],
        fold_results:      list[FoldResult],
        benchmark:         pd.Series | None,
    ) -> tuple[pd.Series, dict]:
        if not oos_return_chunks:
            raise RuntimeError("All folds skipped — no OOS returns produced.")

        stitched = pd.concat(oos_return_chunks).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]

        equity_curve = (1 + stitched).cumprod()
        equity_curve = equity_curve / equity_curve.iloc[0]

        bm_aligned = None
        if benchmark is not None:
            bm_aligned = benchmark.reindex(stitched.index).fillna(0)

        report = self.analyser.full_report(stitched, bm_aligned, fold_results)

        if self.verbose:
            print()
            self.analyser.print_report(report)

        return equity_curve, report

    def _print_header(self, n_folds: int) -> None:
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD ENGINE — {n_folds} folds  "
              f"({'portfolio' if self._portfolio_mode else 'single-asset'})")
        print(f"  IS: {self.wg.is_days}d | OOS: {self.wg.oos_days}d | "
              f"Purge: {self.wg.purge_days}d | Embargo: {self.wg.embargo_days}d")
        print(f"{'='*60}")

    def _print_fold(
        self, w: SplitWindow, params: dict, is_sharpe: float, oos_sharpe: float
    ) -> None:
        print(
            f"  Fold {w.fold_id}: "
            f"IS [{w.is_start.date()} → {w.purge_start.date()}) "
            f"OOS [{w.oos_start.date()} → {w.oos_end.date()}] "
            f"IS_Sh={is_sharpe:.2f} OOS_Sh={oos_sharpe:.2f} "
            f"params={params}"
        )
