"""
Performance Analyser
====================
Computes all standard metrics on OOS data only, plus Walk-Forward Efficiency.

All metrics are computed on the stitched OOS equity curve — never on IS data.

WFE interpretation
------------------
> 1.0 : OOS beats IS — unusual, check for data leakage
0.5–1.0: Normal degradation — strategy is tradeable
< 0.5 : Significant overfitting — do not trade live
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    """Per-fold IS Sharpe and OOS returns for WFE computation."""
    fold_id:     int
    is_sharpe:   float
    oos_returns: pd.Series


class PerformanceAnalyzer:

    def __init__(self, rf_annual: float = 0.0, periods_per_year: int = 252):
        self.rf_annual        = rf_annual
        self.periods_per_year = periods_per_year

    def cagr(self, returns: pd.Series) -> float:
        n   = len(returns)
        tot = (1 + returns).prod()
        if n == 0 or tot <= 0:
            return 0.0
        return tot ** (self.periods_per_year / n) - 1

    def volatility(self, returns: pd.Series) -> float:
        return returns.std() * np.sqrt(self.periods_per_year)

    def sharpe(self, returns: pd.Series) -> float:
        vol = self.volatility(returns)
        if vol < 1e-10:
            excess = self.cagr(returns) - self.rf_annual
            if excess > 0:
                return np.inf
            elif excess < 0:
                return -np.inf
            return 0.0
        return (self.cagr(returns) - self.rf_annual) / vol

    def sortino(self, returns: pd.Series) -> float:
        threshold = self.rf_annual / self.periods_per_year
        downside  = returns[returns < threshold]
        dv = downside.std() * np.sqrt(self.periods_per_year)
        if not np.isfinite(dv) or dv < 1e-10:
            return np.inf if self.cagr(returns) > self.rf_annual else 0.0
        return (self.cagr(returns) - self.rf_annual) / dv

    def max_drawdown(self, returns: pd.Series) -> float:
        cum  = (1 + returns).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        return float(dd.min())

    def win_rate(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).mean())

    def alpha_beta(self, returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
        common = returns.index.intersection(benchmark.index)
        r = returns.reindex(common)
        b = benchmark.reindex(common)
        if b.var() < 1e-12:
            return 0.0, 0.0
        beta  = float(r.cov(b) / b.var())
        alpha = self.cagr(r) - beta * self.cagr(b)
        return alpha, beta

    def walk_forward_efficiency(self, fold_results: list[FoldResult]) -> float:
        """
        WFE = mean(OOS_Sharpe_fold / IS_Sharpe_fold) across folds where IS_Sharpe > 0.
        Returns NaN if no valid folds.

        Infinite OOS Sharpe (constant positive returns, zero variance) is capped at
        3.0 so the ratio stays finite and comparable across folds.
        """
        _INF_CAP = 3.0
        ratios = []
        for fold in fold_results:
            if fold.is_sharpe <= 0:
                continue
            oos_sharpe = self.sharpe(fold.oos_returns)
            if oos_sharpe == np.inf:
                oos_sharpe = _INF_CAP
            elif oos_sharpe == -np.inf:
                oos_sharpe = -_INF_CAP
            if np.isfinite(oos_sharpe):
                ratios.append(oos_sharpe / fold.is_sharpe)
        return float(np.mean(ratios)) if ratios else float("nan")

    def full_report(
        self,
        oos_returns:  pd.Series,
        benchmark:    pd.Series | None,
        fold_results: list[FoldResult],
    ) -> dict:
        report = {
            "cagr":         self.cagr(oos_returns),
            "volatility":   self.volatility(oos_returns),
            "sharpe":       self.sharpe(oos_returns),
            "sortino":      self.sortino(oos_returns),
            "max_drawdown": self.max_drawdown(oos_returns),
            "win_rate":     self.win_rate(oos_returns),
            "wfe":          self.walk_forward_efficiency(fold_results),
            "n_folds":      len(fold_results),
            "n_oos_days":   len(oos_returns),
        }
        if benchmark is not None:
            alpha, beta = self.alpha_beta(oos_returns, benchmark)
            report["alpha"] = alpha
            report["beta"]  = beta
        else:
            report["alpha"] = float("nan")
            report["beta"]  = float("nan")
        return report

    def print_report(self, report: dict) -> None:
        print("=" * 55)
        print("  WALK-FORWARD OOS PERFORMANCE REPORT")
        print("=" * 55)
        print(f"  OOS Trading Days  : {report['n_oos_days']}")
        print(f"  Folds             : {report['n_folds']}")
        print(f"  CAGR              : {report['cagr']*100:+.2f}%")
        print(f"  Volatility        : {report['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio      : {report['sharpe']:.3f}")
        print(f"  Sortino Ratio     : {report['sortino']:.3f}")
        print(f"  Max Drawdown      : {report['max_drawdown']*100:.2f}%")
        print(f"  Win Rate          : {report['win_rate']*100:.1f}%")
        print("-" * 55)
        print(f"  Alpha vs BM       : {report['alpha']*100:+.2f}%")
        print(f"  Beta              : {report['beta']:.3f}")
        print("-" * 55)
        wfe = report['wfe']
        label = 'acceptable' if np.isfinite(wfe) and wfe >= 0.5 else 'OVERFIT'
        print(f"  Walk-Forward Eff. : {wfe:.3f}  ({label})")
        print("=" * 55)
