"""
Ascent Capital — Backtest Engine
Portfolio-native backtest with realistic costs and execution delay.

Key design decisions:
- Signal computed at date t close
- Execution at date t+1 open (1-day delay)
- Costs modeled per rebalance
- Weights held constant between rebalances
- Returns computed from actual price movements
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from ascent.backtest.costs import flat_cost_model


class BacktestEngine:
    """
    Vectorized portfolio backtest engine.

    The engine takes pre-computed target weights and simulates portfolio
    performance including execution delay, costs, and benchmarking.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 5.0,
        rebalance_freq_days: int = 21,
        execution_delay: int = 1,
    ):
        self.initial_capital = initial_capital
        self.cost_bps = spread_bps + impact_bps  # total one-way cost
        self.rebalance_freq_days = rebalance_freq_days
        self.execution_delay = execution_delay

    def run(
        self,
        target_weights: pd.DataFrame,
        close_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
        benchmark_prices: pd.Series | None = None,
    ) -> "BacktestResult":
        """
        Run the backtest.

        Args:
            target_weights: DataFrame(dates × symbols) — desired portfolio weights
            close_prices: DataFrame(dates × symbols) — close prices
            open_prices: DataFrame(dates × symbols) — open prices
            benchmark_prices: Series(dates) — benchmark close prices (e.g., SPY)

        Returns:
            BacktestResult with all portfolio metrics and time series
        """
        # Align all data
        common_dates = target_weights.index.intersection(close_prices.index)
        common_dates = common_dates.intersection(open_prices.index)
        common_dates = common_dates.sort_values()

        symbols = target_weights.columns.intersection(close_prices.columns)

        tw = target_weights.reindex(index=common_dates, columns=symbols).fillna(0)
        close = close_prices.reindex(index=common_dates, columns=symbols)
        open_ = open_prices.reindex(index=common_dates, columns=symbols)

        # Daily returns (close-to-close)
        daily_returns = close.pct_change().fillna(0)

        # Determine rebalance dates
        rebal_dates = self._get_rebalance_dates(common_dates)

        # Simulate portfolio
        n_dates = len(common_dates)
        portfolio_returns = pd.Series(0.0, index=common_dates)
        held_weights = pd.DataFrame(0.0, index=common_dates, columns=symbols)
        turnover_series = pd.Series(0.0, index=common_dates)
        cost_series = pd.Series(0.0, index=common_dates)

        # Exact ledgers
        daily_rows = []
        holdings_rows = []

        prev_weights = pd.Series(0.0, index=symbols)
        prev_end_value = float(self.initial_capital)

        for i in range(n_dates):
            dt = common_dates[i]

            # Starting value for the day
            start_value = prev_end_value

            # Current held weights (from drift)
            if i > 0:
                ret = daily_returns.loc[dt]
                drifted = prev_weights * (1 + ret)
                total = drifted.sum()
                if total > 0:
                    current_weights = drifted / total
                else:
                    current_weights = prev_weights.copy()
            else:
                current_weights = prev_weights.copy()

            # Defaults for ledger
            signal_date = pd.NaT
            is_rebalance = dt in rebal_dates
            turn = 0.0
            cost_rate = 0.0

            # Check if rebalance day
            if is_rebalance:
                # Apply execution delay: use weights signaled `delay` days ago
                delay_idx = i - self.execution_delay
                if delay_idx >= 0:
                    signal_date = common_dates[delay_idx]
                    new_target = tw.loc[signal_date]
                else:
                    signal_date = dt
                    new_target = tw.loc[dt]

                # Calculate turnover
                turn = float((new_target - current_weights).abs().sum() / 2)
                turnover_series.loc[dt] = turn

                # Calculate cost rate
                cost_rate = float(flat_cost_model(turn, self.cost_bps))
                cost_series.loc[dt] = cost_rate

                # Update weights to new target
                current_weights = new_target.copy()

            # Portfolio return for this day
            asset_ret = daily_returns.loc[dt]
            gross_return = float((current_weights * asset_ret).sum())
            net_return = gross_return - cost_rate

            portfolio_returns.loc[dt] = net_return
            held_weights.loc[dt] = current_weights
            prev_weights = current_weights.copy()

            # Dollar accounting
            gross_pnl = start_value * gross_return
            cost_dollars = start_value * cost_rate
            net_pnl = start_value * net_return
            end_value = start_value + net_pnl
            prev_end_value = end_value

            positions = int((current_weights.abs() > 1e-12).sum())

            # Daily ledger row
            daily_rows.append({
                "date": dt,
                "start_value": start_value,
                "gross_return": gross_return,
                "gross_pnl": gross_pnl,
                "turnover": turn,
                "cost_rate": cost_rate,
                "cost_dollars": cost_dollars,
                "net_return": net_return,
                "net_pnl": net_pnl,
                "end_value": end_value,
                "positions": positions,
                "is_rebalance": is_rebalance,
                "signal_date": signal_date,
            })

            # Holdings ledger rows
            for sym in symbols:
                w = float(current_weights.get(sym, 0.0))
                if abs(w) <= 1e-12:
                    continue

                r = float(asset_ret.get(sym, 0.0))
                return_contribution = w * r
                pnl_contribution = start_value * return_contribution

                holdings_rows.append({
                    "date": dt,
                    "symbol": sym,
                    "weight": w,
                    "asset_return": r,
                    "return_contribution": return_contribution,
                    "start_value": start_value,
                    "pnl_contribution": pnl_contribution,
                    "is_rebalance": is_rebalance,
                    "signal_date": signal_date,
                })

        # Benchmark returns
        benchmark_returns = None
        if benchmark_prices is not None:
            bm = benchmark_prices.reindex(common_dates)
            benchmark_returns = bm.pct_change().fillna(0)

        # Equity curve
        equity = self.initial_capital * (1 + portfolio_returns).cumprod()

        # Exact ledgers as DataFrames
        daily_ledger = pd.DataFrame(daily_rows).set_index("date")
        holdings_ledger = pd.DataFrame(holdings_rows)

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            equity_curve=equity,
            held_weights=held_weights,
            turnover=turnover_series,
            costs=cost_series,
            benchmark_returns=benchmark_returns,
            initial_capital=self.initial_capital,
            daily_ledger=daily_ledger,
            holdings_ledger=holdings_ledger,
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Get dates when rebalancing occurs."""
        rebal = set()
        count = 0
        for dt in dates:
            if count % self.rebalance_freq_days == 0:
                rebal.add(dt)
            count += 1
        return rebal


class BacktestResult:
    """Container for backtest results."""

    def __init__(
        self,
        portfolio_returns: pd.Series,
        equity_curve: pd.Series,
        held_weights: pd.DataFrame,
        turnover: pd.Series,
        costs: pd.Series,
        benchmark_returns: pd.Series | None,
        initial_capital: float,
        daily_ledger: pd.DataFrame | None = None,
        holdings_ledger: pd.DataFrame | None = None,
    ):
        self.portfolio_returns = portfolio_returns
        self.equity_curve = equity_curve
        self.held_weights = held_weights
        self.turnover = turnover
        self.costs = costs
        self.benchmark_returns = benchmark_returns
        self.initial_capital = initial_capital
        self.daily_ledger = daily_ledger
        self.holdings_ledger = holdings_ledger

    @property
    def total_return(self) -> float:
        return (1 + self.portfolio_returns).prod() - 1

    @property
    def total_cost(self) -> float:
        return self.costs.sum()

    @property
    def avg_daily_turnover(self) -> float:
        return self.turnover.mean()

    def gross_returns(self) -> pd.Series:
        """Returns before costs."""
        return self.portfolio_returns + self.costs

    def drawdown_series(self) -> pd.Series:
        """Drawdown from peak."""
        cum = (1 + self.portfolio_returns).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    def summary(self) -> dict:
        """Quick summary stats."""
        from ascent.research.evaluation import compute_all_metrics
        return compute_all_metrics(
            self.portfolio_returns,
            benchmark_returns=self.benchmark_returns,
            weights=self.held_weights,
        )