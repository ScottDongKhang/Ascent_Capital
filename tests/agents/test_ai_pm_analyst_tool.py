import pytest
from unittest.mock import patch, MagicMock


def _mock_info():
    return {
        "forwardPE": 28.5,
        "targetMeanPrice": 225.0,
        "numberOfAnalystOpinions": 42,
        "recommendationMean": 1.8,
        "earningsGrowth": 0.15,
        "revenueGrowth": 0.08,
    }


def test_analyst_estimates_returns_data(monkeypatch):
    from agents.ai_pm_agent import _tool_get_analyst_estimates

    mock_ticker = MagicMock()
    mock_ticker.info = _mock_info()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = _tool_get_analyst_estimates({"symbol": "AAPL"})

    assert "28.5" in result           # forward PE
    assert "42" in result             # analyst count
    assert "225.0" in result          # target price


def test_analyst_estimates_no_symbol():
    from agents.ai_pm_agent import _tool_get_analyst_estimates
    result = _tool_get_analyst_estimates({})
    assert "Error" in result or "symbol" in result.lower()


def test_analyst_estimates_handles_failure(monkeypatch):
    from agents.ai_pm_agent import _tool_get_analyst_estimates
    with patch("yfinance.Ticker", side_effect=Exception("timeout")):
        result = _tool_get_analyst_estimates({"symbol": "AAPL"})
    assert "failed" in result.lower() or "error" in result.lower()
