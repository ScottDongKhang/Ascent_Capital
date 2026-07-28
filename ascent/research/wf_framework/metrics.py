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
        # Use calendar span (years) when index is datetime so NaN-fold gaps
        # don't compress the denominator and inflate the annualised return.
        if hasattr(returns.index, 'min') and hasattr(returns.index, 'max'):
            try:
                span_days = (returns.index.max() - returns.index.min()).days
                years = span_days / 365.25 if span_days > 0 else n / self.periods_per_year
            except Exception:
                years = n / self.periods_per_year
        else:
            years = n / self.periods_per_year
        return tot ** (1.0 / years) - 1

    def volatility(self, returns: pd.Series) -> float:
        return returns.std() * np.sqrt(self.periods_per_year)

    def sharpe(self, returns: pd.Series) -> float:
        """
        Arithmetic Sharpe: sqrt(252) * mean(daily_excess) / std(daily_excess).
        Industry-standard convention. Unlike CAGR-based Sharpe, this does not
        penalise high-vol strategies with an additional variance-drag term.
        """
        if len(returns) == 0:
            return 0.0
        rf_daily  = self.rf_annual / self.periods_per_year
        excess    = returns - rf_daily
        std       = excess.std()
        if std < 1e-10:
            m = excess.mean()
            return np.inf if m > 0 else (-np.inf if m < 0 else 0.0)
        return float(np.sqrt(self.periods_per_year) * excess.mean() / std)

    def sortino(self, returns: pd.Series) -> float:
        """Arithmetic Sortino using downside deviation below rf."""
        if len(returns) == 0:
            return 0.0
        rf_daily = self.rf_annual / self.periods_per_year
        excess   = returns - rf_daily
        downside = excess[excess < 0]
        if len(downside) < 2:
            return np.inf if excess.mean() > 0 else 0.0
        # dv stays on the PERIOD scale; annualisation is applied once, in the
        # numerator below. Annualising dv here too divided the result by
        # sqrt(252) and is what produced the long-standing bogus ~0.042
        # readings in every wf_report_*.json written before 2026-07-28.
        dv = downside.std()
        if dv < 1e-10:
            return np.inf if excess.mean() > 0 else 0.0
        return float(np.sqrt(self.periods_per_year) * excess.mean() / dv)

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
        beta = float(r.cov(b) / b.var())
        # Jensen's alpha: excess return above CAPM prediction, both net of rf
        alpha = (self.cagr(r) - self.rf_annual) - beta * (self.cagr(b) - self.rf_annual)
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
