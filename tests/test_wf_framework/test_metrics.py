# tests/test_wf_framework/test_metrics.py
import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.metrics import PerformanceAnalyzer, FoldResult

@pytest.fixture
def positive_returns():
    np.random.seed(1)
    idx = pd.bdate_range("2021-01-04", periods=252)
    return pd.Series(np.random.randn(252) * 0.008 + 0.0003, index=idx)

@pytest.fixture
def benchmark_returns(positive_returns):
    np.random.seed(2)
    return pd.Series(np.random.randn(len(positive_returns)) * 0.007 + 0.0002,
                     index=positive_returns.index)

@pytest.fixture
def fold_results():
    return [
        FoldResult(fold_id=0, is_sharpe=1.2, oos_returns=pd.Series([0.001]*63)),
        FoldResult(fold_id=1, is_sharpe=0.9, oos_returns=pd.Series([0.0005]*63)),
        FoldResult(fold_id=2, is_sharpe=1.5, oos_returns=pd.Series([-0.0002]*63)),
    ]

def test_sharpe_positive_for_positive_returns(positive_returns):
    pa = PerformanceAnalyzer()
    assert pa.sharpe(positive_returns) > 0

def test_sharpe_uses_rf(positive_returns):
    pa = PerformanceAnalyzer(rf_annual=0.05)
    sharpe_high_rf = pa.sharpe(positive_returns)
    pa_no_rf = PerformanceAnalyzer(rf_annual=0.0)
    sharpe_no_rf = pa_no_rf.sharpe(positive_returns)
    assert sharpe_no_rf > sharpe_high_rf

def test_max_drawdown_negative(positive_returns):
    pa = PerformanceAnalyzer()
    mdd = pa.max_drawdown(positive_returns)
    assert mdd <= 0

def test_sortino_geq_sharpe_positive(positive_returns):
    pa = PerformanceAnalyzer()
    assert pa.sortino(positive_returns) >= pa.sharpe(positive_returns)

def test_win_rate_range(positive_returns):
    pa = PerformanceAnalyzer()
    wr = pa.win_rate(positive_returns)
    assert 0.0 <= wr <= 1.0

def test_wfe_computed(fold_results):
    pa = PerformanceAnalyzer()
    wfe = pa.walk_forward_efficiency(fold_results)
    assert np.isfinite(wfe)
    assert wfe > 0

def test_sortino_all_positive_returns():
    pa = PerformanceAnalyzer()
    all_positive = pd.Series([0.001] * 100)
    result = pa.sortino(all_positive)
    assert np.isfinite(result) or result == np.inf, \
        "sortino on all-positive returns must not be NaN"

def test_full_report_keys(positive_returns, benchmark_returns, fold_results):
    pa = PerformanceAnalyzer()
    report = pa.full_report(positive_returns, benchmark_returns, fold_results)
    for key in ["cagr", "sharpe", "sortino", "max_drawdown", "win_rate",
                "wfe", "alpha", "beta", "n_folds"]:
        assert key in report, f"Missing key: {key}"
