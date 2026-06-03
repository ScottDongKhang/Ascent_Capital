# tests/test_wf_framework/test_execution.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig

@pytest.fixture
def ohlcv():
    np.random.seed(0)
    n = 100
    idx = pd.bdate_range("2021-01-04", periods=n)
    close = pd.Series(100 * (1 + np.random.randn(n) * 0.01).cumprod(), index=idx)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": 1_000_000,
    })

@pytest.fixture
def long_signals(ohlcv):
    return pd.Series(1, index=ohlcv.index, dtype=int)

def test_zero_friction_gross_return(ohlcv):
    cfg = ExecutionConfig(slippage_model="fixed_pct", fixed_pct=0.0,
                          commission_pct=0.0, borrow_rate_annual=0.0)
    em = ExecutionModel(cfg)
    signals = pd.Series(1, index=ohlcv.index)
    returns = em.compute_returns(ohlcv, signals)
    raw = ohlcv["close"].pct_change().fillna(0)
    pd.testing.assert_series_equal(returns, raw, atol=1e-6, check_names=False)

def test_commission_reduces_returns(ohlcv, long_signals):
    cfg_no  = ExecutionConfig(commission_pct=0.0)
    cfg_yes = ExecutionConfig(commission_pct=0.001)
    r_no  = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_yes = ExecutionModel(cfg_yes).compute_returns(ohlcv, long_signals)
    assert r_yes.sum() < r_no.sum()

def test_atr_slippage_reduces_returns(ohlcv, long_signals):
    cfg_no  = ExecutionConfig(slippage_model="atr", atr_multiplier=0.0)
    cfg_yes = ExecutionConfig(slippage_model="atr", atr_multiplier=0.1)
    r_no  = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_yes = ExecutionModel(cfg_yes).compute_returns(ohlcv, long_signals)
    assert r_yes.sum() < r_no.sum()

def test_borrow_cost_on_shorts(ohlcv):
    cfg = ExecutionConfig(borrow_rate_annual=0.03, commission_pct=0.0,
                          slippage_model="fixed_pct", fixed_pct=0.0)
    signals = pd.Series(-1, index=ohlcv.index)
    returns = ExecutionModel(cfg).compute_returns(ohlcv, signals)
    raw     = -ohlcv["close"].pct_change().fillna(0)
    assert returns.sum() < raw.sum()

def test_no_borrow_on_longs(ohlcv, long_signals):
    cfg_no     = ExecutionConfig(borrow_rate_annual=0.0,  commission_pct=0.0,
                                 slippage_model="fixed_pct", fixed_pct=0.0)
    cfg_borrow = ExecutionConfig(borrow_rate_annual=0.50, commission_pct=0.0,
                                 slippage_model="fixed_pct", fixed_pct=0.0)
    r_no     = ExecutionModel(cfg_no).compute_returns(ohlcv, long_signals)
    r_borrow = ExecutionModel(cfg_borrow).compute_returns(ohlcv, long_signals)
    pd.testing.assert_series_equal(r_no, r_borrow, atol=1e-10, check_names=False)

def test_output_index_matches_input(ohlcv, long_signals):
    em = ExecutionModel(ExecutionConfig())
    returns = em.compute_returns(ohlcv, long_signals)
    assert returns.index.equals(ohlcv.index)
