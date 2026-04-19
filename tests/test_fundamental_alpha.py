# tests/test_fundamental_alpha.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def _fake_income_stmt():
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Gross Profit": 5e9, "Net Income": 2e9},
        periods[1]: {"Gross Profit": 4.8e9, "Net Income": 1.9e9},
        periods[2]: {"Gross Profit": 4.5e9, "Net Income": 1.7e9},
        periods[3]: {"Gross Profit": 4.2e9, "Net Income": 1.6e9},
    })

def _fake_balance_sheet():
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Total Assets": 50e9},
        periods[1]: {"Total Assets": 48e9},
        periods[2]: {"Total Assets": 46e9},
        periods[3]: {"Total Assets": 44e9},
    })

def _fake_cashflow():
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Operating Cash Flow": 3e9},
        periods[1]: {"Operating Cash Flow": 2.8e9},
        periods[2]: {"Operating Cash Flow": 2.6e9},
        periods[3]: {"Operating Cash Flow": 2.4e9},
    })

def _mock_ticker(sym):
    t = MagicMock()
    t.quarterly_income_stmt   = _fake_income_stmt()
    t.quarterly_balance_sheet = _fake_balance_sheet()
    t.quarterly_cashflow      = _fake_cashflow()
    return t


def test_fetch_fundamentals_returns_required_columns():
    with patch("yfinance.Ticker", side_effect=_mock_ticker):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["AAPL", "MSFT"], delay_s=0)
    assert not df.empty
    for col in ["symbol", "date", "gross_profit", "total_assets", "net_income", "op_cashflow"]:
        assert col in df.columns, f"missing column: {col}"


def test_fetch_fundamentals_applies_45_day_lag():
    with patch("yfinance.Ticker", side_effect=_mock_ticker):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["AAPL"], delay_s=0)
    aapl = df[df["symbol"] == "AAPL"]
    assert not aapl.empty
    latest = aapl["date"].max()
    assert latest >= pd.Timestamp("2026-02-14"), \
        f"Expected filing date >= 2026-02-14 (Dec 31 + 45d), got {latest}"


def test_fetch_fundamentals_graceful_on_missing_symbol():
    def mock_bad(sym):
        t = MagicMock()
        t.quarterly_income_stmt   = pd.DataFrame()
        t.quarterly_balance_sheet = pd.DataFrame()
        t.quarterly_cashflow      = pd.DataFrame()
        return t
    with patch("yfinance.Ticker", side_effect=mock_bad):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["BADTICKER"], delay_s=0)
    assert isinstance(df, pd.DataFrame)


# ── Task 2 tests ───────────────────────────────────────────────────────────────

def _make_fundamentals_df(syms=("AAPL", "MSFT", "GOOGL"), n_quarters=6):
    rows = []
    np.random.seed(42)
    for sym in syms:
        for q in range(n_quarters):
            period = pd.Timestamp("2024-01-01") + pd.DateOffset(months=3 * q)
            rows.append({
                "symbol":       sym,
                "period_end":   period,
                "date":         period + pd.Timedelta(days=45),
                "gross_profit": np.random.uniform(1e9, 5e9),
                "total_assets": np.random.uniform(20e9, 60e9),
                "net_income":   np.random.uniform(0.5e9, 3e9),
                "op_cashflow":  np.random.uniform(1e9, 4e9),
            })
    return pd.DataFrame(rows)


def _make_close(syms=("AAPL", "MSFT", "GOOGL"), n=300):
    idx = pd.bdate_range(end="2026-04-19", periods=n)
    np.random.seed(0)
    return pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, (len(idx), len(syms))), axis=0),
        index=idx, columns=list(syms)
    )


def test_build_fundamental_panel_produces_three_metrics():
    from ascent.features.feature_defs import build_fundamental_panel
    syms = ["AAPL", "MSFT", "GOOGL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms)
    result = build_fundamental_panel(fund_df, close.index, syms)
    assert "gross_profitability" in result
    assert "accruals" in result
    assert "asset_growth" in result


def test_build_fundamental_panel_forward_fills():
    from ascent.features.feature_defs import build_fundamental_panel
    syms = ["AAPL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms, n_quarters=4)
    result = build_fundamental_panel(fund_df, close.index, syms)
    gp = result.get("gross_profitability")
    assert gp is not None
    valid = gp["AAPL"].dropna()
    assert len(valid) > 50


def test_high_52w_pct_feature():
    from ascent.features.feature_defs import high_52w_pct
    close = _make_close()
    result = high_52w_pct(close)
    valid = result.iloc[252:].dropna(how="all")
    assert not valid.empty
    finite = valid.values[np.isfinite(valid.values)]
    assert (finite <= 1.0).all()
    assert (finite > 0).all()


def test_fundamental_alpha_builds_composite():
    from ascent.alpha.fundamental import fundamental_alpha
    from ascent.features.feature_defs import build_fundamental_panel, high_52w_pct
    syms = ["AAPL", "MSFT", "GOOGL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms)
    panel = build_fundamental_panel(fund_df, close.index, syms)
    features = {"close": close, "high_52w_pct": high_52w_pct(close)}
    features.update(panel)
    result = fundamental_alpha(features)
    assert not result.empty
    assert set(result.columns) == set(syms)


def test_fundamental_alpha_works_without_fundamentals():
    from ascent.alpha.fundamental import fundamental_alpha
    close = _make_close()
    result = fundamental_alpha({"close": close})
    assert not result.empty
