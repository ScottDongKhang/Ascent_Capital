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
