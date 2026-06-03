import numpy as np
import pandas as pd
import pytest
from ascent.research.wf_framework.strategy import BaseStrategy, MACrossStrategy

@pytest.fixture
def price_data():
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-02", periods=300)
    prices = pd.Series(100 * (1 + np.random.randn(300) * 0.01).cumprod(), index=idx)
    return pd.DataFrame({"close": prices, "high": prices * 1.005,
                         "low": prices * 0.995, "volume": 1_000_000})

def test_ma_cross_is_base_strategy():
    assert issubclass(MACrossStrategy, BaseStrategy)

def test_ma_cross_param_grid():
    s = MACrossStrategy(fast=10, slow=50)
    assert "fast" in s.param_grid
    assert "slow" in s.param_grid

def test_ma_cross_signals_shape(price_data):
    s = MACrossStrategy(fast=10, slow=50)
    signals = s.generate_signals(price_data)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(price_data)

def test_ma_cross_signals_values(price_data):
    s = MACrossStrategy(fast=10, slow=50)
    signals = s.generate_signals(price_data)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})

def test_ma_cross_no_future_look(price_data):
    s = MACrossStrategy(fast=5, slow=20)
    sig_full = s.generate_signals(price_data)
    sig_partial = s.generate_signals(price_data.iloc[:150])
    common_idx = sig_partial.index
    pd.testing.assert_series_equal(
        sig_full.loc[common_idx], sig_partial,
        check_names=False,
    )

def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseStrategy()
