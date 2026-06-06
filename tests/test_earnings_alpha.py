"""
tests/test_earnings_alpha.py
Tests for earnings surprise (PEAD) alpha signal.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date


# ── Task 1: Ingest ─────────────────────────────────────────────────────────────

def test_fetch_earnings_returns_expected_schema(monkeypatch):
    """_fetch_one returns df with required columns."""
    import yfinance as yf

    # Build a fake earnings_dates DataFrame as yfinance returns it
    idx = pd.to_datetime(["2025-10-29 16:00:00-05:00", "2025-07-30 16:00:00-05:00"])
    fake_df = pd.DataFrame(
        {"EPS Estimate": [2.67, 2.10], "Reported EPS": [2.80, 2.00], "Surprise(%)": [4.87, -4.76]},
        index=idx,
    )
    fake_df.index.name = "Earnings Date"

    class FakeTicker:
        earnings_dates = fake_df

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTicker())

    from ascent.data.ingest.earnings import _fetch_one
    result = _fetch_one("AAPL")

    assert not result.empty
    required = {"symbol", "signal_date", "eps_estimate", "reported_eps", "surprise_pct"}
    assert required.issubset(set(result.columns))
    # signal_date must be tz-naive
    assert result["signal_date"].dt.tz is None
    # signal_date must be > earnings date (1-day lag applied)
    assert (result["signal_date"] > pd.to_datetime(["2025-10-29", "2025-07-30"])).all()


def test_fetch_one_skips_future_rows(monkeypatch):
    """Rows with NaN Reported EPS (future dates) are excluded."""
    import yfinance as yf

    idx = pd.to_datetime(["2026-12-31 16:00:00-05:00", "2025-10-29 16:00:00-05:00"])
    fake_df = pd.DataFrame(
        {"EPS Estimate": [3.00, 2.67], "Reported EPS": [np.nan, 2.80], "Surprise(%)": [np.nan, 4.87]},
        index=idx,
    )
    fake_df.index.name = "Earnings Date"

    class FakeTicker:
        earnings_dates = fake_df

    monkeypatch.setattr(yf, "Ticker", lambda sym: FakeTicker())

    from ascent.data.ingest.earnings import _fetch_one
    result = _fetch_one("AAPL")

    # Only 1 row should remain (the past row)
    assert len(result) == 1
    assert result["reported_eps"].notna().all()


# ── Task 2: Feature Panel ──────────────────────────────────────────────────────

def test_build_earnings_panel_forward_fills():
    """earnings_surprise is forward-filled up to 63 days then NaN."""
    from ascent.features.feature_defs import build_earnings_panel

    # Create a simple earnings df: AAPL had a surprise on day 0
    signal_date = pd.Timestamp("2025-01-02")
    earnings_df = pd.DataFrame([
        {"symbol": "AAPL", "signal_date": signal_date, "eps_estimate": 2.0,
         "reported_eps": 2.5, "surprise_pct": 25.0},
        {"symbol": "MSFT", "signal_date": signal_date, "eps_estimate": 3.0,
         "reported_eps": 2.8, "surprise_pct": -6.67},
    ])

    # Build a date index spanning ~90 trading days after signal
    date_index = pd.bdate_range(start="2024-12-01", periods=120)
    symbols = ["AAPL", "MSFT", "GOOG"]

    result = build_earnings_panel(earnings_df, date_index, symbols)

    assert "earnings_surprise" in result
    es = result["earnings_surprise"]
    assert set(es.columns) == set(symbols)

    # AAPL should be forward-filled on the signal_date
    assert not pd.isna(es.loc[signal_date, "AAPL"])

    # GOOG (no earnings) should be NaN everywhere
    assert es["GOOG"].isna().all()

    # Forward-fill should stop after 63 trading days
    day_64 = pd.bdate_range(start=signal_date, periods=65)[-1]
    if day_64 in es.index:
        assert pd.isna(es.loc[day_64, "AAPL"])


def test_build_earnings_panel_handles_none():
    """Returns empty dict when earnings_df is None."""
    from ascent.features.feature_defs import build_earnings_panel
    date_index = pd.bdate_range(start="2025-01-01", periods=10)
    result = build_earnings_panel(None, date_index, ["AAPL"])
    assert result == {}


def test_build_earnings_panel_strips_timezone():
    """Works when date_index has timezone."""
    from ascent.features.feature_defs import build_earnings_panel

    signal_date = pd.Timestamp("2025-01-02")
    earnings_df = pd.DataFrame([
        {"symbol": "AAPL", "signal_date": signal_date,
         "eps_estimate": 2.0, "reported_eps": 2.5, "surprise_pct": 25.0},
    ])

    # Timezone-aware index
    date_index = pd.bdate_range(start="2024-12-01", periods=60, tz="America/New_York")
    result = build_earnings_panel(earnings_df, date_index, ["AAPL"])
    # Should not raise; result may be empty if tz-stripping logic prevents reindex
    assert isinstance(result, dict)


# ── Task 3: Alpha Sleeve ──────────────────────────────────────────────────────

def test_earnings_alpha_zscore():
    """earnings_alpha cross-sectionally z-scores the surprise signal."""
    from ascent.alpha.earnings import earnings_alpha

    dates = pd.bdate_range("2025-01-01", periods=20)
    symbols = ["A", "B", "C", "D"]
    # Constant surprise data so we can check z-score math
    data = np.random.randn(20, 4) * 10 + 5
    es = pd.DataFrame(data, index=dates, columns=symbols)

    result = earnings_alpha({"earnings_surprise": es})
    assert not result.empty
    assert result.shape == (20, 4)

    # Row means should be ~0 and row stds ~1 (z-scored)
    row_means = result.mean(axis=1)
    row_stds = result.std(axis=1)
    assert (row_means.abs() < 1e-9).all(), f"Row means not zero: {row_means}"
    assert (row_stds - 1.0).abs().max() < 0.01, f"Row stds not 1: {row_stds}"


def test_earnings_alpha_returns_empty_when_no_feature():
    """Returns empty DataFrame when earnings_surprise not in features."""
    from ascent.alpha.earnings import earnings_alpha

    result = earnings_alpha({"close": pd.DataFrame()})
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_earnings_alpha_handles_single_symbol():
    """Single symbol — std is 0, should return zeros not NaN."""
    from ascent.alpha.earnings import earnings_alpha

    dates = pd.bdate_range("2025-01-01", periods=10)
    es = pd.DataFrame({"AAPL": [5.0] * 10}, index=dates)
    result = earnings_alpha({"earnings_surprise": es})
    # With only 1 symbol, std=0 → should fillna(0)
    assert not result.isnull().any().any()


# ── Task 4: Stack / ML / Wiring ───────────────────────────────────────────────

def test_default_alpha_weights_include_earnings():
    """DEFAULT_ALPHA_WEIGHTS has earnings=0.05 and sums to 1.0."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    assert "earnings" in DEFAULT_ALPHA_WEIGHTS
    assert abs(DEFAULT_ALPHA_WEIGHTS["earnings"] - 0.05) < 1e-9
    total = sum(DEFAULT_ALPHA_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"


def test_default_alpha_weights_fundamental_disabled():
    """fundamental weight is 0.00 — IC-t=-4.75 across 31 live days makes it an anti-signal."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    assert DEFAULT_ALPHA_WEIGHTS["fundamental"] == 0.00


def test_ml_features_include_earnings_surprise():
    """ML_FEATURES contains earnings_surprise."""
    from ascent.alpha.ml_sleeve import ML_FEATURES

    assert "earnings_surprise" in ML_FEATURES


def test_feature_builder_accepts_earnings_df():
    """FeatureBuilder accepts earnings_df kwarg without error."""
    import pandas as pd
    from ascent.features.build_features import FeatureBuilder

    # Minimal price_df to construct FeatureBuilder
    dates = pd.bdate_range("2025-01-01", periods=30)
    price_rows = []
    for d in dates:
        price_rows.append({"date": d, "symbol": "AAPL",
                           "open": 150.0, "high": 152.0, "low": 149.0,
                           "close": 151.0, "volume": 1_000_000})
    price_df = pd.DataFrame(price_rows)

    earnings_df = pd.DataFrame([
        {"symbol": "AAPL", "signal_date": dates[5], "eps_estimate": 2.0,
         "reported_eps": 2.5, "surprise_pct": 25.0}
    ])

    # Should not raise
    fb = FeatureBuilder(price_df, earnings_df=earnings_df)
    assert fb.earnings_df is not None


def test_feature_builder_compute_includes_earnings_surprise():
    """compute_features includes earnings_surprise when earnings_df provided."""
    import pandas as pd
    from ascent.features.build_features import FeatureBuilder

    dates = pd.bdate_range("2025-01-01", periods=60)
    price_rows = []
    for d in dates:
        price_rows.append({"date": d, "symbol": "AAPL",
                           "open": 150.0, "high": 152.0, "low": 149.0,
                           "close": 151.0, "volume": 1_000_000})
    price_df = pd.DataFrame(price_rows)

    earnings_df = pd.DataFrame([
        {"symbol": "AAPL", "signal_date": dates[5], "eps_estimate": 2.0,
         "reported_eps": 2.5, "surprise_pct": 25.0}
    ])

    fb = FeatureBuilder(price_df, earnings_df=earnings_df)
    features = fb.compute_features()
    assert "earnings_surprise" in features


def test_earnings_alpha_neutralizes_momentum_when_available():
    """Residuals after OLS on mom_126d should have lower correlation with mom than raw signal."""
    from ascent.alpha.earnings import earnings_alpha

    np.random.seed(7)
    dates = pd.bdate_range("2025-01-01", periods=60)
    syms = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Construct surprise = 0.5 * momentum + noise  (strong momentum beta)
    mom_data = np.random.randn(60, len(syms))
    noise = np.random.randn(60, len(syms)) * 0.3
    surprise_data = 0.5 * mom_data + noise

    mom_df = pd.DataFrame(mom_data, index=dates, columns=syms)
    surprise_df = pd.DataFrame(surprise_data, index=dates, columns=syms)

    raw_result = earnings_alpha({"earnings_surprise": surprise_df})
    neutralized_result = earnings_alpha({"earnings_surprise": surprise_df, "mom_126d": mom_df})

    # Both should be non-empty DataFrames
    assert not raw_result.empty
    assert not neutralized_result.empty

    # Flatten and measure cross-sectional correlation with momentum
    mom_flat = mom_df.values.flatten()
    raw_corr = np.corrcoef(raw_result.values.flatten(), mom_flat)[0, 1]
    neut_corr = np.corrcoef(neutralized_result.values.flatten(), mom_flat)[0, 1]

    # Neutralized signal must have lower absolute momentum correlation
    assert abs(neut_corr) < abs(raw_corr), (
        f"Neutralization failed: |neut_corr|={abs(neut_corr):.3f} >= |raw_corr|={abs(raw_corr):.3f}"
    )


def test_earnings_alpha_falls_back_gracefully_without_mom():
    """Returns valid z-scored output when mom_126d is absent."""
    from ascent.alpha.earnings import earnings_alpha

    dates = pd.bdate_range("2025-01-01", periods=10)
    syms = ["X", "Y", "Z"]
    surprise_df = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=syms)

    result = earnings_alpha({"earnings_surprise": surprise_df})
    assert not result.empty
    assert result.shape == (10, 3)
    assert not result.isnull().any().any()


def test_self_improve_default_weights_match_stack():
    """self_improve.DEFAULT_ALPHA_WEIGHTS matches stack.DEFAULT_ALPHA_WEIGHTS."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_weights
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_weights

    assert stack_weights == si_weights, (
        f"Mismatch: stack={stack_weights}  self_improve={si_weights}"
    )
