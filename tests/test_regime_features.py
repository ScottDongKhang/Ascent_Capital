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


def _macro_panel(n=300):
    """Simulate a wide FRED macro panel keyed by series_id (as main.py pivots it)."""
    idx = pd.bdate_range(end="2026-04-18", periods=n)
    n = len(idx)  # bdate_range(end=, periods=) can return n-1 — match the actual index
    np.random.seed(11)
    return pd.DataFrame({
        "DGS10":        np.random.uniform(3.5, 4.5, n),
        "DFF":          np.random.uniform(4.5, 5.5, n),
        "T10Y2Y":       np.random.uniform(-0.5, 0.5, n),
        "BAMLH0A0HYM2": np.random.uniform(3.0, 5.0, n),  # HY OAS
        "BAMLC0A0CM":   np.random.uniform(0.8, 1.5, n),  # IG OAS
    }, index=idx)


def test_ig_hy_differential_features_present():
    """IG-HY differential features must appear when both OAS series are in macro."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices()
    mkt = _market_prices()
    macro = _macro_panel()

    builder = RegimeFeatureBuilder(spy_prices=spy, market_prices=mkt, macro_df=macro)
    panel = builder.build()

    for col in ("ig_spread_chg_21d", "hy_ig_differential", "hy_ig_diff_chg_21d"):
        assert col in panel.columns, f"missing {col}"

    # differential must equal HY - IG on aligned non-NaN rows
    macro_aligned = macro.reindex(spy.index, method="ffill")
    expected = (macro_aligned["BAMLH0A0HYM2"] - macro_aligned["BAMLC0A0CM"]).dropna()
    actual = panel["hy_ig_differential"].dropna()
    common = expected.index.intersection(actual.index)
    assert len(common) > 0
    assert np.allclose(expected.loc[common].values, actual.loc[common].values)


def test_ig_hy_differential_absent_without_macro():
    """No macro → no IG-HY features, but build must still succeed."""
    from ascent.regime.features import RegimeFeatureBuilder
    spy = _spy_prices()
    mkt = _market_prices()

    builder = RegimeFeatureBuilder(spy_prices=spy, market_prices=mkt)
    panel = builder.build()

    assert "hy_ig_differential" not in panel.columns
    assert len(panel) > 0


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
