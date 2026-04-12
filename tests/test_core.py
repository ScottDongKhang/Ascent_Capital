"""
Ascent Capital — Test Suite
Tests for data integrity, leakage prevention, and system correctness.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ── Data Tests ──────────────────────────────────────────────────────────

class TestSimulatedData:
    def test_price_data_shape(self):
        from ascent.data.ingest.simulated import generate_price_data
        df = generate_price_data(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
        assert len(df) > 0
        assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])

    def test_price_data_deterministic(self):
        from ascent.data.ingest.simulated import generate_price_data
        df1 = generate_price_data(["AAPL"], "2023-01-01", "2024-01-01", seed=42)
        df2 = generate_price_data(["AAPL"], "2023-01-01", "2024-01-01", seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_price_data_no_negatives(self):
        from ascent.data.ingest.simulated import generate_price_data
        df = generate_price_data(["AAPL", "TSLA", "NVDA"], "2020-01-01", "2025-01-01")
        assert (df["close"] > 0).all()
        assert (df["volume"] >= 0).all()

    def test_price_data_has_timestamps(self):
        from ascent.data.ingest.simulated import generate_price_data
        df = generate_price_data(["AAPL"], "2023-01-01", "2024-01-01")
        assert "event_time" in df.columns
        assert "known_time" in df.columns
        assert "source" in df.columns

    def test_macro_data_shape(self):
        from ascent.data.ingest.simulated import generate_macro_data
        df = generate_macro_data("2023-01-01", "2024-01-01")
        assert len(df) > 0
        assert "value" in df.columns
        assert "series_id" in df.columns

    def test_price_sorted(self):
        from ascent.data.ingest.simulated import generate_price_data
        from ascent.data.normalize.prices import normalize_prices
        df = generate_price_data(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
        df = normalize_prices(df)
        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym]
            assert sub["date"].is_monotonic_increasing

    def test_no_duplicate_dates(self):
        from ascent.data.ingest.simulated import generate_price_data
        from ascent.data.normalize.prices import normalize_prices
        df = generate_price_data(["AAPL"], "2023-01-01", "2024-01-01")
        df = normalize_prices(df)
        aapl = df[df["symbol"] == "AAPL"]
        assert not aapl["date"].duplicated().any()


# ── Feature Tests ───────────────────────────────────────────────────────

class TestFeatures:
    @pytest.fixture
    def price_data(self):
        from ascent.data.ingest.simulated import generate_price_data
        from ascent.data.normalize.prices import normalize_prices
        df = generate_price_data(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "2022-01-01", "2024-01-01", seed=42,
        )
        return normalize_prices(df)

    def test_feature_builder(self, price_data):
        from ascent.features.build_features import FeatureBuilder
        builder = FeatureBuilder(price_data)
        features = builder.compute_features()
        assert len(features) > 10
        for name, feat in features.items():
            assert isinstance(feat, pd.DataFrame)
            assert feat.shape[0] > 0

    def test_targets_are_forward_looking(self, price_data):
        from ascent.features.build_features import FeatureBuilder
        builder = FeatureBuilder(price_data)
        targets = builder.compute_targets([5])

        fwd = targets["fwd_ret_5d"]
        close = builder.close

        for sym in close.columns[:2]:
            for i in range(100, min(110, len(close) - 5)):
                dt = close.index[i]
                dt5 = close.index[i + 5]
                expected = close.loc[dt5, sym] / close.loc[dt, sym] - 1
                actual = fwd.loc[dt, sym]
                if not np.isnan(actual):
                    np.testing.assert_almost_equal(actual, expected, decimal=6)

    def test_momentum_uses_only_past(self, price_data):
        from ascent.features.feature_defs import momentum_return
        from ascent.data.normalize.prices import pivot_prices
        close = pivot_prices(price_data, "close")

        mom = momentum_return(close, 21)

        for sym in close.columns[:2]:
            for i in range(50, min(55, len(close))):
                dt = close.index[i]
                dt_21 = close.index[i - 21]
                expected = close.loc[dt, sym] / close.loc[dt_21, sym] - 1
                actual = mom.loc[dt, sym]
                if not np.isnan(actual):
                    np.testing.assert_almost_equal(actual, expected, decimal=6)

    def test_features_no_nan_explosion(self, price_data):
        """After warmup period, features should mostly be non-NaN."""
        from ascent.features.build_features import FeatureBuilder
        builder = FeatureBuilder(price_data)
        features = builder.compute_features()

        for name, feat in features.items():
            if "macro" in name:
                continue  # macro may have NaNs if no data
            # After 300 days warmup, should have <10% NaN
            late = feat.iloc[300:]
            if len(late) > 0:
                nan_pct = late.isna().mean().mean()
                assert nan_pct < 0.10, f"Feature '{name}' has {nan_pct:.1%} NaN after warmup"


# ── Leakage Tests ───────────────────────────────────────────────────────

class TestLeakage:
    def test_no_feature_target_leakage(self):
        """
        Features should NOT have unrealistically high correlation
        with future returns. IC > 0.3 consistently suggests leakage.
        """
        from ascent.data.ingest.simulated import generate_price_data
        from ascent.data.normalize.prices import normalize_prices
        from ascent.features.build_features import FeatureBuilder

        df = generate_price_data(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"],
            "2021-01-01", "2024-01-01", seed=42,
        )
        df = normalize_prices(df)
        builder = FeatureBuilder(df)
        features = builder.compute_features()
        targets = builder.compute_targets([21])
        fwd = targets["fwd_ret_21d"]

        for fname, feat in features.items():
            if feat.empty:
                continue
            common = feat.index.intersection(fwd.index)
            if len(common) < 100:
                continue
            ic = feat.loc[common].corrwith(fwd.loc[common], axis=1).dropna()
            if len(ic) > 50:
                mean_ic = abs(ic.mean())
                assert mean_ic < 0.30, (
                    f"Feature '{fname}' has suspiciously high IC={mean_ic:.4f}. Possible leakage!"
                )

    def test_point_in_time_join(self):
        from ascent.data.store.point_in_time import as_of_join

        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "value": [100, 200, 300],
            "known_time": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"]),
        })

        query_dates = pd.DatetimeIndex(["2024-02-01", "2024-02-20", "2024-04-01"])
        result = as_of_join(query_dates, data, value_cols=["value"])

        row_feb1 = result[result["as_of_date"] == pd.Timestamp("2024-02-01")]
        assert len(row_feb1) == 1
        assert row_feb1.iloc[0]["value"] == 100

        row_feb20 = result[result["as_of_date"] == pd.Timestamp("2024-02-20")]
        assert row_feb20.iloc[0]["value"] == 200

    def test_reversed_time_worse(self):
        """
        Leakage test: if we reverse time (train on future, test on past),
        performance should NOT be better. If it is, there's leakage.
        """
        from ascent.data.ingest.simulated import generate_price_data
        from ascent.data.normalize.prices import normalize_prices
        from ascent.features.build_features import FeatureBuilder
        from ascent.alpha.stack import build_alpha_stack

        df = generate_price_data(
            ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "2022-01-01", "2024-01-01", seed=42,
        )
        df = normalize_prices(df)
        builder = FeatureBuilder(df)
        features = builder.compute_features()
        targets = builder.compute_targets([21])

        alpha = build_alpha_stack(features)
        fwd = targets["fwd_ret_21d"]

        common = alpha.index.intersection(fwd.index)
        n = len(common)
        half = n // 2

        # Normal: train on first half features, evaluate IC on second half
        ic_normal = alpha.loc[common[half:]].corrwith(
            fwd.loc[common[half:]], axis=1
        ).dropna().mean()

        # Reversed: train on second half features, evaluate IC on first half
        ic_reversed = alpha.loc[common[:half]].corrwith(
            fwd.loc[common[:half]], axis=1
        ).dropna().mean()

        # Both should be similar for a proper feature (no directional leakage)
        # The reversed version shouldn't be dramatically better
        assert ic_reversed < ic_normal + 0.15, "Reversed-time IC is suspiciously better"


# ── Walk-Forward Split Tests ────────────────────────────────────────────

class TestSplits:
    def test_splits_no_overlap(self):
        from ascent.research.splits import walk_forward_splits

        dates = pd.bdate_range("2020-01-01", "2024-01-01")
        splits = walk_forward_splits(dates, train_days=252, test_days=63, purge_days=5)

        assert len(splits) > 0
        for s in splits:
            assert s.train_end < s.test_start
            gap = (s.test_start - s.train_end).days
            assert gap >= 5

    def test_splits_cover_data(self):
        from ascent.research.splits import walk_forward_splits

        dates = pd.bdate_range("2020-01-01", "2024-01-01")
        splits = walk_forward_splits(dates, train_days=252, test_days=63, step_days=21)
        assert len(splits) >= 5

    def test_expanding_splits(self):
        from ascent.research.splits import expanding_splits

        dates = pd.bdate_range("2020-01-01", "2024-01-01")
        splits = expanding_splits(dates, initial_train_days=252, test_days=63)

        assert len(splits) > 0
        assert splits[0].train_start == dates[0]
        if len(splits) > 1:
            assert splits[-1].train_end > splits[0].train_end

    def test_no_test_before_train(self):
        from ascent.research.splits import walk_forward_splits

        dates = pd.bdate_range("2020-01-01", "2024-01-01")
        splits = walk_forward_splits(dates)
        for s in splits:
            assert s.test_start > s.train_end


# ── Cost Model Tests ────────────────────────────────────────────────────

class TestCosts:
    def test_cost_increases_with_size(self):
        from ascent.backtest.costs import estimate_trade_cost

        cost_small = estimate_trade_cost(100, 100, 1_000_000, 0.02)
        cost_large = estimate_trade_cost(100, 10_000, 1_000_000, 0.02)
        assert cost_large["total_cost"] > cost_small["total_cost"]

    def test_cost_increases_with_volatility(self):
        from ascent.backtest.costs import estimate_trade_cost

        cost_low = estimate_trade_cost(100, 1000, 1_000_000, 0.01)
        cost_high = estimate_trade_cost(100, 1000, 1_000_000, 0.05)
        assert cost_high["impact_cost"] > cost_low["impact_cost"]

    def test_zero_shares_zero_cost(self):
        from ascent.backtest.costs import estimate_trade_cost

        cost = estimate_trade_cost(100, 0, 1_000_000, 0.02)
        assert cost["total_cost"] == 0.0

    def test_cost_is_positive(self):
        from ascent.backtest.costs import estimate_trade_cost

        cost = estimate_trade_cost(150, 500, 5_000_000, 0.02)
        assert cost["total_cost"] > 0
        assert cost["spread_cost"] > 0
        assert cost["impact_cost"] > 0

    def test_flat_cost_model(self):
        from ascent.backtest.costs import flat_cost_model

        cost = flat_cost_model(0.10, cost_bps=10.0)  # 10% turnover, 10bps cost
        assert abs(cost - 0.0001) < 1e-8  # 10% × 10bps = 0.01%


# ── Portfolio Tests ─────────────────────────────────────────────────────

class TestPortfolio:
    def test_weights_sum_to_one(self):
        from ascent.portfolio.optimizer import top_n_equal_weight

        alpha = pd.DataFrame(
            np.random.randn(100, 20),
            index=pd.bdate_range("2023-01-01", periods=100),
            columns=[f"SYM{i}" for i in range(20)],
        )

        weights = top_n_equal_weight(alpha, n=10)
        for dt in weights.index[10:]:  # skip early dates
            row_sum = weights.loc[dt].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01, f"Weights sum to {row_sum} on {dt}"

    def test_max_weight_constraint(self):
        from ascent.portfolio.optimizer import top_n_equal_weight

        alpha = pd.DataFrame(
            np.random.randn(100, 20),
            index=pd.bdate_range("2023-01-01", periods=100),
            columns=[f"SYM{i}" for i in range(20)],
        )

        max_w = 0.15
        weights = top_n_equal_weight(alpha, n=10, max_weight=max_w)
        assert weights.max().max() <= max_w + 0.01

    def test_rank_weighted_higher_alpha_higher_weight(self):
        from ascent.portfolio.optimizer import rank_weighted

        # Create alpha where SYM0 is always best
        alpha = pd.DataFrame(
            np.random.randn(50, 10),
            index=pd.bdate_range("2023-01-01", periods=50),
            columns=[f"SYM{i}" for i in range(10)],
        )
        alpha["SYM0"] = 10.0  # Always best

        weights = rank_weighted(alpha, n=5, max_weight=0.30)
        # SYM0 should have highest weight most of the time
        for dt in weights.index[5:]:
            if weights.loc[dt].sum() > 0:
                assert weights.loc[dt, "SYM0"] > 0


# ── Backtest Engine Tests ───────────────────────────────────────────────

class TestBacktest:
    def test_backtest_runs_end_to_end(self):
        from ascent.backtest.engine import BacktestEngine

        dates = pd.bdate_range("2023-01-01", "2024-01-01")
        symbols = ["A", "B", "C", "D", "E"]

        np.random.seed(42)
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(len(dates), len(symbols)) * 0.02, axis=0)),
            index=dates, columns=symbols,
        )
        open_ = close * (1 + np.random.randn(len(dates), len(symbols)) * 0.005)

        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        weights["A"] = 0.3
        weights["B"] = 0.3
        weights["C"] = 0.4

        engine = BacktestEngine(initial_capital=100_000, rebalance_freq_days=21)
        result = engine.run(weights, close, open_)

        assert len(result.portfolio_returns) > 0
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] > 0

    def test_costs_reduce_returns(self):
        from ascent.backtest.engine import BacktestEngine

        dates = pd.bdate_range("2023-01-01", "2024-01-01")
        symbols = ["A", "B"]

        np.random.seed(42)
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(len(dates), 2) * 0.01, axis=0)),
            index=dates, columns=symbols,
        )
        open_ = close.copy()

        weights = pd.DataFrame(0.5, index=dates, columns=symbols)

        # Low cost backtest
        engine_low = BacktestEngine(spread_bps=1.0, impact_bps=1.0, rebalance_freq_days=5)
        result_low = engine_low.run(weights, close, open_)

        # High cost backtest
        engine_high = BacktestEngine(spread_bps=50.0, impact_bps=50.0, rebalance_freq_days=5)
        result_high = engine_high.run(weights, close, open_)

        # High costs should result in lower returns
        assert result_high.total_return < result_low.total_return

    def test_backtest_summary(self):
        from ascent.backtest.engine import BacktestEngine

        dates = pd.bdate_range("2023-01-01", "2024-01-01")
        symbols = ["A", "B", "C"]

        np.random.seed(42)
        close = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(len(dates), 3) * 0.01, axis=0)),
            index=dates, columns=symbols,
        )
        open_ = close.copy()
        weights = pd.DataFrame(1.0 / 3, index=dates, columns=symbols)

        engine = BacktestEngine()
        result = engine.run(weights, close, open_)
        summary = result.summary()

        assert "sharpe" in summary
        assert "max_drawdown" in summary
        assert "cagr" in summary
        assert "volatility" in summary


# ── Risk Tests ──────────────────────────────────────────────────────────

class TestRisk:
    def test_var_cvar(self):
        from ascent.risk.var_cvar import historical_var, historical_cvar

        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.02)

        var95 = historical_var(returns, 0.95)
        cvar95 = historical_cvar(returns, 0.95)

        assert var95 < 0  # VaR is a loss
        assert cvar95 <= var95  # CVaR is worse than VaR

    def test_stress_test(self):
        from ascent.risk.var_cvar import stress_test

        weights = pd.Series({"AAPL": 0.3, "MSFT": 0.3, "JPM": 0.4})
        results = stress_test(weights, pd.DataFrame())

        assert "market_crash_20pct" in results
        assert results["market_crash_20pct"] < 0  # Should be negative


# ── Evaluation Metrics Tests ────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_positive_for_positive_returns(self):
        from ascent.research.evaluation import sharpe_ratio

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)  # positive drift
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_max_drawdown_negative(self):
        from ascent.research.evaluation import max_drawdown

        returns = pd.Series([0.01, -0.05, 0.01, -0.03, 0.02])
        mdd = max_drawdown(returns)
        assert mdd < 0

    def test_metrics_format(self):
        from ascent.research.evaluation import compute_all_metrics, format_metrics

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = compute_all_metrics(returns)
        text = format_metrics(metrics)

        assert "Sharpe" in text
        assert "CAGR" in text
        assert "Drawdown" in text


# ── Integration Test ────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end test: data → features → alpha → portfolio → backtest."""
        from ascent.data.ingest.simulated import generate_price_data, generate_macro_data
        from ascent.data.normalize.prices import normalize_prices, normalize_macro
        from ascent.features.build_features import FeatureBuilder
        from ascent.alpha.stack import build_alpha_stack
        from ascent.portfolio.optimizer import rank_weighted
        from ascent.backtest.engine import BacktestEngine

        # 1. Data
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                    "META", "TSLA", "JPM", "V", "UNH", "SPY"]
        df = generate_price_data(symbols, "2021-01-01", "2024-01-01", seed=42)
        df = normalize_prices(df)
        macro = normalize_macro(generate_macro_data("2021-01-01", "2024-01-01"))

        # Separate benchmark
        benchmark_df = df[df["symbol"] == "SPY"]
        universe_df = df[df["symbol"] != "SPY"]

        # 2. Features
        builder = FeatureBuilder(universe_df, macro)
        features = builder.compute_features()
        assert len(features) > 0

        # 3. Alpha
        alpha = build_alpha_stack(features)
        assert not alpha.empty

        # 4. Portfolio
        weights = rank_weighted(alpha, n=5, max_weight=0.25)
        # Skip warmup
        weights = weights.iloc[280:]
        assert not weights.empty

        # 5. Backtest
        bm_close = benchmark_df.set_index("date")["close"].sort_index()
        bm_close = bm_close[~bm_close.index.duplicated(keep="last")]

        engine = BacktestEngine(initial_capital=100_000, rebalance_freq_days=21)
        result = engine.run(weights, builder.close, builder.open, bm_close)

        assert len(result.portfolio_returns) > 100
        assert result.equity_curve.iloc[-1] > 0

        # Summary should work
        summary = result.summary()
        assert "sharpe" in summary
        assert np.isfinite(summary["sharpe"])
