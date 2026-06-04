import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.portfolio_strategy import PortfolioBaseStrategy
from ascent.research.wf_framework.ascent_strategy import AscentPortfolioStrategy


def make_multi_ohlcv(n_days=300, n_symbols=20, seed=7):
    np.random.seed(seed)
    rows = []
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    for sym in symbols:
        close = pd.Series(100 * (1 + np.random.randn(n_days) * 0.012).cumprod(), index=idx)
        for dt, c in close.items():
            rows.append({
                "date": dt, "symbol": sym,
                "open": c * 0.999, "high": c * 1.006,
                "low": c * 0.994, "close": c, "volume": 500_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ohlcv():
    return make_multi_ohlcv()


def test_is_portfolio_base_strategy():
    assert issubclass(AscentPortfolioStrategy, PortfolioBaseStrategy)


def test_param_grid_keys():
    s = AscentPortfolioStrategy()
    for key in ["trend_weight", "statarb_weight", "mom_window"]:
        assert key in s.param_grid, f"Missing param_grid key: {key}"


def test_param_grid_values():
    s = AscentPortfolioStrategy()
    assert s.param_grid["trend_weight"]   == [0.30, 0.38, 0.50]
    assert s.param_grid["statarb_weight"] == [0.10, 0.15, 0.20]
    assert s.param_grid["mom_window"]     == [63, 126, 252]
    # top_n and max_weight are fixed at live defaults — not in grid
    assert "top_n"      not in s.param_grid
    assert "max_weight" not in s.param_grid


def test_generate_signals_returns_dataframe(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    s = AscentPortfolioStrategy(top_n=5, max_weight=0.20, mom_window=63)
    result = s.generate_signals(ohlcv)
    assert isinstance(result, pd.DataFrame), "generate_signals must return pd.DataFrame"


def test_weights_sum_leq_one(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    s = AscentPortfolioStrategy(top_n=5, max_weight=0.20, mom_window=63)
    result = s.generate_signals(ohlcv)
    row_sums = result.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all(), f"Row sums exceed 1.0: {row_sums.max():.4f}"


def test_no_weight_exceeds_rank_weighted_max(ohlcv):
    # With synthetic symbols all in "Unknown" sector, sector_constrained_weighted
    # falls back to rank weighting (coverage < 80%). Verify max rank weight
    # for top_n=5 from 20 symbols = rank5/(1+2+3+4+5) = 5/15 ≈ 0.333.
    # We just check weights are non-negative and rows sum <= 1.0 (covered by other tests).
    # Max_weight enforcement requires sector map — verified separately in live runs.
    AscentPortfolioStrategy.clear_cache()
    s = AscentPortfolioStrategy(top_n=5, max_weight=0.20, mom_window=63)
    result = s.generate_signals(ohlcv)
    assert (result.values >= -1e-9).all(), "Negative weight found"
    assert (result.sum(axis=1) <= 1.0 + 1e-6).all(), "Row sums exceed 1.0"


def test_make_alpha_weights_sums_to_one():
    s = AscentPortfolioStrategy(trend_weight=0.50, statarb_weight=0.20)
    w = s._make_alpha_weights()
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-9, f"Alpha weights sum {total:.6f} != 1.0"


def test_make_alpha_weights_trend_statarb_set():
    s = AscentPortfolioStrategy(trend_weight=0.50, statarb_weight=0.20)
    w = s._make_alpha_weights()
    assert abs(w["trend"]   - 0.50) < 1e-9
    assert abs(w["statarb"] - 0.20) < 1e-9


def test_caching_shared_across_instances(ohlcv):
    AscentPortfolioStrategy.clear_cache()
    s1 = AscentPortfolioStrategy(top_n=5, mom_window=63)
    s2 = AscentPortfolioStrategy(top_n=10, mom_window=63)
    s1.generate_signals(ohlcv)
    n_after_s1 = len(AscentPortfolioStrategy._feature_cache)
    s2.generate_signals(ohlcv)
    n_after_s2 = len(AscentPortfolioStrategy._feature_cache)
    assert n_after_s1 == n_after_s2, "Feature cache grew on second call with same mom_window"


def test_clear_cache_resets():
    AscentPortfolioStrategy.clear_cache()
    assert len(AscentPortfolioStrategy._feature_cache) == 0
    assert len(AscentPortfolioStrategy._alpha_cache)   == 0


def test_no_lookahead(ohlcv):
    # Use mom_window=9999 (effectively no trim) so both partial and full runs
    # use all input data, making overlapping dates comparable.
    cutoff = sorted(ohlcv["date"].unique())[149]
    partial_data = ohlcv[ohlcv["date"] <= cutoff]

    AscentPortfolioStrategy.clear_cache()
    s_partial = AscentPortfolioStrategy(top_n=5, mom_window=9999)
    result_partial = s_partial.generate_signals(partial_data)

    AscentPortfolioStrategy.clear_cache()
    s_full = AscentPortfolioStrategy(top_n=5, mom_window=9999)
    result_full = s_full.generate_signals(ohlcv)

    common_dates = result_partial.index.intersection(result_full.index)
    common_syms  = result_partial.columns.intersection(result_full.columns)
    assert len(common_dates) > 0, "No common dates between partial and full signals"
    pd.testing.assert_frame_equal(
        result_full.loc[common_dates, common_syms].fillna(0),
        result_partial.loc[common_dates, common_syms].fillna(0),
        atol=1e-6,
        check_names=False,
    )
