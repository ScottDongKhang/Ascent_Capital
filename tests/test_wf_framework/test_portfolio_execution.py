import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.portfolio_strategy import PortfolioBaseStrategy
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig


def make_multi_symbol_ohlcv(n_days=100, symbols=("A", "B", "C"), seed=42):
    np.random.seed(seed)
    rows = []
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    for sym in symbols:
        close = pd.Series(100 * (1 + np.random.randn(n_days) * 0.01).cumprod(), index=idx)
        for dt, c in close.items():
            rows.append({
                "date": dt, "symbol": sym,
                "open": c * 0.999, "high": c * 1.005,
                "low": c * 0.995, "close": c, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_ohlcv():
    return make_multi_symbol_ohlcv()


@pytest.fixture
def equal_weights(multi_ohlcv):
    dates   = sorted(multi_ohlcv["date"].unique())
    symbols = sorted(multi_ohlcv["symbol"].unique())
    w = 1.0 / len(symbols)
    return pd.DataFrame(w, index=pd.DatetimeIndex(dates), columns=symbols)


def test_cannot_instantiate_portfolio_base():
    with pytest.raises(TypeError):
        PortfolioBaseStrategy()


def test_portfolio_returns_series(multi_ohlcv, equal_weights):
    em = ExecutionModel(ExecutionConfig(commission_pct=0.0,
                                        slippage_model="fixed_pct", fixed_pct=0.0))
    returns = em.compute_returns(multi_ohlcv, equal_weights)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(equal_weights)


def test_portfolio_zero_friction_matches_manual(multi_ohlcv, equal_weights):
    em = ExecutionModel(ExecutionConfig(commission_pct=0.0,
                                        slippage_model="fixed_pct", fixed_pct=0.0))
    returns = em.compute_returns(multi_ohlcv, equal_weights)
    close_wide = multi_ohlcv.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    sym_ret = close_wide.pct_change().fillna(0)
    delayed_w = equal_weights.shift(1).fillna(0).reindex(
        index=sym_ret.index, columns=sym_ret.columns, fill_value=0
    )
    expected = (delayed_w * sym_ret).sum(axis=1).rename("net_return")
    pd.testing.assert_series_equal(returns, expected, atol=1e-6, check_names=False)


def test_portfolio_commission_reduces_returns(multi_ohlcv, equal_weights):
    cfg_no  = ExecutionConfig(commission_pct=0.0, slippage_model="fixed_pct", fixed_pct=0.0)
    cfg_yes = ExecutionConfig(commission_pct=0.001, slippage_model="fixed_pct", fixed_pct=0.0)
    r_no  = ExecutionModel(cfg_no).compute_returns(multi_ohlcv, equal_weights)
    r_yes = ExecutionModel(cfg_yes).compute_returns(multi_ohlcv, equal_weights)
    assert r_yes.sum() < r_no.sum()


def test_portfolio_atr_slippage_reduces_returns(multi_ohlcv, equal_weights):
    cfg_no  = ExecutionConfig(slippage_model="atr", atr_multiplier=0.0, commission_pct=0.0)
    cfg_yes = ExecutionConfig(slippage_model="atr", atr_multiplier=0.1, commission_pct=0.0)
    r_no  = ExecutionModel(cfg_no).compute_returns(multi_ohlcv, equal_weights)
    r_yes = ExecutionModel(cfg_yes).compute_returns(multi_ohlcv, equal_weights)
    assert r_yes.sum() < r_no.sum()


def test_single_asset_path_unchanged(multi_ohlcv):
    ohlcv_single = multi_ohlcv[multi_ohlcv["symbol"] == "A"].copy()
    ohlcv_single = ohlcv_single.set_index("date").drop(columns=["symbol"])
    signals = pd.Series(1, index=ohlcv_single.index)
    em = ExecutionModel(ExecutionConfig())
    returns = em.compute_returns(ohlcv_single, signals)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(ohlcv_single)
