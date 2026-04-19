# tests/test_regime_features.py
import pytest
import pandas as pd
import numpy as np
from datetime import date


def _spy_prices(n=300):
    idx = pd.bdate_range(end="2026-04-18", periods=n)
    np.random.seed(42)
    px = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, len(idx))), index=idx)
    return px


def _market_prices(n=300):
    """Simulate HYG, LQD, TLT, IEF prices."""
    idx = pd.bdate_range(end="2026-04-18", periods=n)
    np.random.seed(7)
    data = {}
    for sym, drift, vol in [("HYG", 0.0002, 0.005), ("LQD", 0.0002, 0.003),
                             ("TLT", 0.0001, 0.010), ("IEF", 0.0001, 0.006)]:
        data[sym] = 100 * np.cumprod(1 + np.random.normal(drift, vol, len(idx)))
    return pd.DataFrame(data, index=idx)


def test_credit_spread_feature_built_when_hyg_lqd_present():
    """When HYG and LQD are in market_prices, credit_spread_chg_21d must appear in features."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices()
    mkt = _market_prices()

    builder = RegimeFeatureBuilder(spy_prices=spy, market_prices=mkt)
    panel = builder.build()

    assert "credit_spread_chg_21d" in panel.columns, \
        "credit_spread_chg_21d must be in regime feature panel"
    assert "credit_spread_level" in panel.columns, \
        "credit_spread_level must be in regime feature panel"


def test_yield_curve_feature_built_when_tlt_ief_present():
    """When TLT and IEF are in market_prices, yield_curve_slope must appear."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices()
    mkt = _market_prices()

    builder = RegimeFeatureBuilder(spy_prices=spy, market_prices=mkt)
    panel = builder.build()

    assert "yield_curve_slope" in panel.columns, \
        "yield_curve_slope must be in regime feature panel"
    assert "yield_curve_chg" in panel.columns


def test_regime_features_graceful_when_market_prices_none():
    """RegimeFeatureBuilder must work exactly as before when market_prices is None."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices()

    builder = RegimeFeatureBuilder(spy_prices=spy)
    panel = builder.build()

    assert any(c.startswith("spy_") for c in panel.columns)
    assert "credit_spread_chg_21d" not in panel.columns
    assert "yield_curve_slope" not in panel.columns


def test_credit_spread_values_are_finite():
    """Credit spread features must not be all NaN after warmup period."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices(300)
    mkt = _market_prices(300)

    builder = RegimeFeatureBuilder(spy_prices=spy, market_prices=mkt)
    panel = builder.build()

    valid = panel["credit_spread_chg_21d"].dropna()
    assert len(valid) > 200, "credit spread should have >200 valid rows with 300 days of data"
    assert np.isfinite(valid.values).all(), "no inf values in credit spread"


def test_regime_engine_fit_accepts_market_prices():
    """RegimeEngine.fit() must accept and pass through market_prices without error."""
    import pandas as pd
    import numpy as np
    from ascent.regime.engine import RegimeEngine

    idx = pd.bdate_range(end="2026-04-18", periods=300)
    n = len(idx)
    np.random.seed(0)
    spy = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, n)), index=idx)
    univ = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, (n, 5)), axis=0),
        index=idx, columns=["A", "B", "C", "D", "E"]
    )
    mkt = pd.DataFrame({
        "HYG": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.005, n)),
        "LQD": 100 * np.cumprod(1 + np.random.normal(0.0002, 0.003, n)),
        "TLT": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.010, n)),
        "IEF": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.006, n)),
    }, index=idx)

    engine = RegimeEngine()
    engine.fit(spy_prices=spy, universe_prices=univ, market_prices=mkt,
               run_model_selection=False)
    assert engine.best_k >= 2
