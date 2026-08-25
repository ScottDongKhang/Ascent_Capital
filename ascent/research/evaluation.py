"""
Ascent Capital — Evaluation Metrics
Performance metrics for strategy evaluation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    total = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    if total <= 0:
        # total <= 0 means the compounded portfolio value hit zero or went
        # negative -- i.e. at least one period had a return of exactly -100%
        # (total wipeout / delisting-style event). That is the worst
        # possible outcome, not a neutral "no return": clamping to 0.0 here
        # made a wipeout indistinguishable from (and in downstream ratios
        # like calmar_ratio, rank ABOVE) a genuinely flat, zero-return
        # series. Return -1.0 (-100% annualized) instead, since the
        # portfolio's value floor is 0 and cannot recover within this
        # window -- this is the mathematically honest annualized return for
        # a fully wiped-out position, and it propagates correctly through
        # sharpe_ratio / sortino_ratio / calmar_ratio without any special
        # casing needed in those functions.
        return -1.0
    return total ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    ret = annualized_return(returns, periods_per_year)
    return (ret - rf_annual) / vol


def lo_adjusted_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    q: int = None,
) -> float:
    """Autocorrelation-adjusted (Lo, 2002) annualized Sharpe ratio.

    ADDITIONAL metric — never a replacement for `sharpe_ratio()`. Lo's "The
    Statistics of Sharpe Ratios" (Financial Analysts Journal, 2002) shows that
    naively annualizing a Sharpe ratio computed on serially correlated returns
    (sqrt(periods_per_year) scaling of the per-period Sharpe) can overstate the
    true annualized Sharpe by as much as ~65% under positive autocorrelation.
    This strategy rebalances every `rebalance_freq_days` (10) trading days with
    weights forward-filled daily in between, which mechanically induces
    positive serial correlation in the daily return series (the same position
    repeats for ~10 days), so the correction is directly relevant here.

    Formula:
        sharpe_naive = sharpe_ratio(returns, periods_per_year=periods_per_year)
        rho_k = sample autocorrelation of `returns` at lag k, k = 1..q-1
        sharpe_corrected = sharpe_naive / sqrt(1 + 2 * sum_{k=1}^{q-1} (1 - k/q) * rho_k)

    This is the standard Lo (2002) correction for the case of a q-period
    (here: q trading day) return-generating/holding horizon under an assumed
    stationary AR-type return process; it is applied here to correct the
    *annualization* of a Sharpe computed on daily returns whose serial
    dependence arises from a q-day rebalance cadence.

    Args:
        returns: per-period (daily) returns, NOT annualized.
        periods_per_year: trading periods per year, passed through to the
            underlying naive Sharpe computation.
        q: number of lags to include in the correction (rho_1..rho_{q-1}).
            Callers SHOULD pass `cfg.backtest.rebalance_freq_days` explicitly,
            since the caller knows the true rebalance cadence driving the
            autocorrelation. If omitted, defaults to 10 as a holding-period
            proxy matching this strategy's current `rebalance_freq_days`
            default — this default will silently become wrong if the
            rebalance cadence changes, so passing `q` explicitly is strongly
            preferred.

    Edge cases:
        - Zero-variance returns: falls through to `sharpe_ratio()`'s own
          `vol == 0 -> 0.0` handling (no divide-by-zero).
        - Too few observations for the requested lag count (`len(returns) <=
          q`, or `q < 2` so there is nothing to correct): returns the naive
          annualized Sharpe unadjusted (documented fallback, not NaN) — a
          short window shouldn't make the metric disappear, it just means the
          correction term is untrustworthy and is skipped.
        - If the correction factor `1 + 2*sum(...)` would be <= 0 (possible
          with strong negative autocorrelation estimates on a short/noisy
          sample), falls back to the naive Sharpe rather than taking sqrt of a
          negative number.
    """
    if q is None:
        q = 10

    sharpe_naive = sharpe_ratio(returns, periods_per_year=periods_per_year)

    n = len(returns)
    if q < 2 or n <= q:
        return sharpe_naive
    if returns.std() == 0:
        return sharpe_naive

    r = returns.values
    correction_sum = 0.0
    for k in range(1, q):
        rho_k = pd.Series(r).autocorr(lag=k)
        if rho_k is None or np.isnan(rho_k):
            continue
        correction_sum += (1 - k / q) * rho_k

    factor = 1 + 2 * correction_sum
    if factor <= 0:
        return sharpe_naive

    return sharpe_naive / np.sqrt(factor)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Sortino ratio using downside deviation."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if annualized_return(returns) > 0 else 0.0
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    ret = annualized_return(returns, periods_per_year)
    return (ret - rf_annual) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as negative fraction)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods_per_year) / mdd


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover: sum of absolute weight changes."""
    return weights.diff().abs().sum(axis=1)


def average_turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover."""
    t = turnover(weights)
    return t.mean() / 2  # one-way


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive return days."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """Sum of gains / sum of losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute comprehensive performance metrics."""
    metrics = {
        "total_return": (1 + returns).prod() - 1,
        "cagr": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sharpe_lo_adjusted": lo_adjusted_sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_days": len(returns),
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "avg_daily_return": returns.mean(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align
        common = returns.index.intersection(benchmark_returns.index)
        r = returns.reindex(common)
        b = benchmark_returns.reindex(common)
        excess = r - b

        metrics["benchmark_return"] = annualized_return(b)
        metrics["alpha"] = annualized_return(r) - annualized_return(b)
        metrics["excess_sharpe"] = sharpe_ratio(excess) if len(excess) > 1 else 0.0

        # Beta
        if b.var() > 0:
            metrics["beta"] = r.cov(b) / b.var()
        else:
            metrics["beta"] = 0.0

    if weights is not None:
        metrics["avg_turnover"] = average_turnover(weights)
        metrics["avg_positions"] = (weights > 0.001).sum(axis=1).mean()

    return metrics


def format_metrics(metrics: dict) -> str:
    """Pretty-print metrics."""
    lines = [
        "=" * 60,
        "  PERFORMANCE REPORT",
        "=" * 60,
        f"  Total Return:     {metrics.get('total_return', 0) * 100:+.2f}%",
        f"  CAGR:             {metrics.get('cagr', 0) * 100:+.2f}%",
        f"  Volatility:       {metrics.get('volatility', 0) * 100:.2f}%",
        f"  Sharpe Ratio:     {metrics.get('sharpe', 0):.3f}",
        f"  Sortino Ratio:    {metrics.get('sortino', 0):.3f}",
        f"  Max Drawdown:     {metrics.get('max_drawdown', 0) * 100:.2f}%",
        f"  Calmar Ratio:     {metrics.get('calmar', 0):.3f}",
        f"  Hit Rate:         {metrics.get('hit_rate', 0) * 100:.1f}%",
        f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}",
        "-" * 60,
        f"  Trading Days:     {metrics.get('n_days', 0)}",
        f"  Best Day:         {metrics.get('best_day', 0) * 100:+.2f}%",
        f"  Worst Day:        {metrics.get('worst_day', 0) * 100:+.2f}%",
    ]

    if "benchmark_return" in metrics:
        lines += [
            "-" * 60,
            f"  Benchmark CAGR:   {metrics['benchmark_return'] * 100:+.2f}%",
            f"  Alpha:            {metrics.get('alpha', 0) * 100:+.2f}%",
            f"  Beta:             {metrics.get('beta', 0):.3f}",
            f"  Excess Sharpe:    {metrics.get('excess_sharpe', 0):.3f}",
        ]

    if "avg_turnover" in metrics:
        lines += [
            "-" * 60,
            f"  Avg Turnover:     {metrics['avg_turnover'] * 100:.2f}% per day",
            f"  Avg Positions:    {metrics.get('avg_positions', 0):.1f}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)
