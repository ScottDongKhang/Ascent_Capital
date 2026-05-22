import pytest
import pandas as pd
import numpy as np
from ascent.features.feature_defs import hy_spread_direction


def _make_macro_pivot():
    dates = pd.bdate_range("2023-01-03", periods=100)
    # Spread widens then tightens
    hy = pd.Series(
        [400.0] * 40 + [450.0] * 20 + [380.0] * 40,  # bps
        index=dates
    )
    return pd.DataFrame({"hy_spread": hy})


def test_hy_spread_dir_shape():
    macro = _make_macro_pivot()
    close_cols = ["AAPL", "MSFT"]
    close = pd.DataFrame(
        np.ones((100, 2)),
        index=macro.index,
        columns=close_cols,
    )
    result = hy_spread_direction(macro, close)
    assert result.shape == (100, 2)
    assert list(result.columns) == close_cols


def test_hy_spread_dir_values():
    """Widening spreads produce -1, tightening produce +1."""
    macro = _make_macro_pivot()
    close_cols = ["AAPL"]
    close = pd.DataFrame(np.ones((100, 1)), index=macro.index, columns=close_cols)
    result = hy_spread_direction(macro, close)
    # After 20-day window, widening period → -1
    # Spread goes from 400→450 at index 40; diff(20) turns positive at index 59
    val_widening = result.iloc[55]["AAPL"]  # in widening period
    assert val_widening == -1.0
    # Tightening period → +1
    # Spread drops from 450→380 at index 60; diff(20) turns negative (tightening) at index 60–79
    val_tightening = result.iloc[70]["AAPL"]
    assert val_tightening == 1.0


def test_hy_spread_dir_missing_column():
    """Returns zeros if hy_spread column missing."""
    macro = pd.DataFrame({"vix": [20.0] * 50}, index=pd.bdate_range("2023-01-03", periods=50))
    close = pd.DataFrame(np.ones((50, 1)), index=macro.index, columns=["AAPL"])
    result = hy_spread_direction(macro, close)
    assert (result == 0.0).all().all()
