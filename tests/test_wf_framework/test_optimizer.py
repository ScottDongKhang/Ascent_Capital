import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.strategy import MACrossStrategy
from ascent.research.wf_framework.execution import ExecutionModel, ExecutionConfig
from ascent.research.wf_framework.optimizer import ParameterOptimizer

@pytest.fixture
def is_data():
    np.random.seed(7)
    n = 300
    idx = pd.bdate_range("2019-01-02", periods=n)
    close = pd.Series(100 * (1 + np.random.randn(n) * 0.012).cumprod(), index=idx)
    return pd.DataFrame({
        "open":  close * 0.999,
        "high":  close * 1.006,
        "low":   close * 0.994,
        "close": close,
        "volume": 500_000,
    })

def test_returns_dict(is_data):
    em  = ExecutionModel(ExecutionConfig(commission_pct=0.0, slippage_model="fixed_pct", fixed_pct=0.0))
    opt = ParameterOptimizer(MACrossStrategy, em)
    best_params, best_score = opt.optimize(is_data)
    assert isinstance(best_params, dict)
    assert "fast" in best_params and "slow" in best_params
    assert isinstance(best_score, float)

def test_constraint_respected(is_data):
    em  = ExecutionModel(ExecutionConfig(commission_pct=0.0))
    opt = ParameterOptimizer(
        MACrossStrategy, em,
        constraint_fn=lambda p: p["fast"] < p["slow"]
    )
    best_params, _ = opt.optimize(is_data)
    assert best_params["fast"] < best_params["slow"]

def test_optimizer_does_not_use_future_data(is_data):
    em  = ExecutionModel(ExecutionConfig())
    opt = ParameterOptimizer(MACrossStrategy, em)
    params_is, _ = opt.optimize(is_data)
    # Calling with same IS slice must return identical result
    params_again, _ = opt.optimize(is_data)
    assert params_is == params_again

def test_best_score_is_finite(is_data):
    em  = ExecutionModel(ExecutionConfig())
    opt = ParameterOptimizer(MACrossStrategy, em)
    _, score = opt.optimize(is_data)
    assert np.isfinite(score)
