import pytest
from unittest.mock import patch, MagicMock
import time


def _make_news_item(title: str, age_hours: float = 12.0) -> dict:
    return {
        "title": title,
        "providerPublishTime": int(time.time() - age_hours * 3600),
        "publisher": "Reuters",
    }


def test_get_live_news_returns_headlines(monkeypatch):
    from agents.ai_pm_agent import _tool_get_live_news

    mock_ticker = MagicMock()
    mock_ticker.news = [
        _make_news_item("AAPL beats earnings estimates by 12%", 10),
        _make_news_item("Apple expands AI features to iPhone 16", 20),
        _make_news_item("Old news", 100),  # > 72h — should be filtered
    ]
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = _tool_get_live_news({"symbol": "AAPL"})

    assert "AAPL beats earnings" in result
    assert "Old news" not in result


def test_get_live_news_no_symbol():
    from agents.ai_pm_agent import _tool_get_live_news
    result = _tool_get_live_news({})
    assert "Error" in result or "symbol" in result.lower()


def test_get_live_news_handles_yf_failure(monkeypatch):
    from agents.ai_pm_agent import _tool_get_live_news
    with patch("yfinance.Ticker", side_effect=Exception("network error")):
        result = _tool_get_live_news({"symbol": "AAPL"})
    assert "failed" in result.lower() or "error" in result.lower()
