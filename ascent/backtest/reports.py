"""
Ascent Capital — Backtest Reports
Generate formatted performance reports.
"""
from __future__ import annotations
import pandas as pd
from ascent.backtest.engine import BacktestResult
from ascent.research.evaluation import compute_all_metrics, format_metrics


def print_report(result: BacktestResult, title: str = "ASCENT CAPITAL BACKTEST"):
    """Print formatted backtest report."""
    metrics = result.summary()

    print(f"\n{'#' * 70}")
    print(f"  {title}")
    print(f"{'#' * 70}")
    print(format_metrics(metrics))

    # Cost summary
    print(f"\n  Total Transaction Costs: ${result.total_cost * result.initial_capital:,.2f}")
    print(f"  Avg Daily Turnover:     {result.avg_daily_turnover * 100:.3f}%")
    print(f"  Total Cost Drag:        {result.total_cost * 100:.3f}%")

    # Equity curve summary
    eq = result.equity_curve
    if len(eq) > 0:
        print(f"\n  Start Value:  ${eq.iloc[0]:,.2f}")
        print(f"  End Value:    ${eq.iloc[-1]:,.2f}")
        print(f"  Peak Value:   ${eq.max():,.2f}")
        print(f"  Trough Value: ${eq.min():,.2f}")

    # Drawdown
    dd = result.drawdown_series()
    if len(dd) > 0:
        worst_dd_date = dd.idxmin()
        print(f"\n  Worst Drawdown: {dd.min() * 100:.2f}% on {worst_dd_date.date()}")

    # Position stats
    hw = result.held_weights
    if not hw.empty:
        avg_pos = (hw > 0.001).sum(axis=1).mean()
        max_weight = hw.max().max()
        print(f"\n  Avg Positions:  {avg_pos:.1f}")
        print(f"  Max Weight:     {max_weight * 100:.1f}%")

    print(f"{'#' * 70}\n")


def results_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert results to a DataFrame for analysis."""
    df = pd.DataFrame({
        "portfolio_return": result.portfolio_returns,
        "equity": result.equity_curve,
        "drawdown": result.drawdown_series(),
        "turnover": result.turnover,
        "cost": result.costs,
    })
    if result.benchmark_returns is not None:
        df["benchmark_return"] = result.benchmark_returns
        df["benchmark_equity"] = result.initial_capital * (1 + result.benchmark_returns).cumprod()
    return df
