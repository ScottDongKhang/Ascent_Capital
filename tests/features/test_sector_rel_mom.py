import pytest
import pandas as pd
import numpy as np
from ascent.features.feature_defs import sector_relative_momentum


def _make_close():
    dates = pd.bdate_range("2022-01-03", periods=300)
    np.random.seed(42)
    data = {"AAPL": np.cumprod(1 + np.random.normal(0.001, 0.02, 300)),
            "MSFT": np.cumprod(1 + np.random.normal(0.0005, 0.02, 300)),
            "JPM":  np.cumprod(1 + np.random.normal(0.0008, 0.02, 300))}
    return pd.DataFrame(data, index=dates)


def test_sector_rel_mom_shape():
    close = _make_close()
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    result = sector_relative_momentum(close, sector_map)
    assert result.shape == close.shape
    assert list(result.columns) == list(close.columns)


def test_sector_rel_mom_tech_sum_near_zero():
    """For a two-stock sector, sector-relative scores sum to ~0."""
    close = _make_close()
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    result = sector_relative_momentum(close, sector_map)
    # Drop early NaN rows (252d lookback)
    valid = result.dropna()
    tech_sum = (valid["AAPL"] + valid["MSFT"]).abs().mean()
    assert tech_sum < 0.01  # within-sector scores cancel


def test_sector_rel_mom_no_sector_map():
    """With no sector map, returns raw momentum (fallback)."""
    close = _make_close()
    result = sector_relative_momentum(close, {})
    assert result.shape == close.shape
