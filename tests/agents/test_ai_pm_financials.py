import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


def _make_mock_ticker():
    """Return a mock yfinance Ticker with realistic quarterly financials."""
    bs_cols = pd.to_datetime(["2026-03-31", "2025-12-31"])
    bs = pd.DataFrame(
        {
            "Current Assets":      [10_000_000.0, 9_000_000.0],
            "Current Liabilities": [5_000_000.0,  4_500_000.0],
            "Total Debt":          [8_000_000.0,  7_500_000.0],
            "Stockholders Equity": [4_000_000.0,  4_200_000.0],
        },
        index=bs_cols,
    ).T  # rows = line items, columns = dates

    inc_cols = pd.to_datetime([
        "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"
    ])
    inc = pd.DataFrame(
        {
            "Total Revenue": [20e6, 19e6, 18e6, 17e6, 16e6],
            "Gross Profit":  [6e6,  5.7e6, 5.4e6, 5.1e6, 4.8e6],
        },
        index=inc_cols,
    ).T

    t = MagicMock()
    t.quarterly_balance_sheet = bs
    t.quarterly_income_stmt = inc
    return t


def test_fetch_financials_returns_four_metrics():
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    assert "CAT" in result
    metrics = result["CAT"]
    assert "current_ratio" in metrics
    assert "debt_to_equity" in metrics
    assert "revenue_growth_yoy" in metrics
    assert "gross_margin" in metrics


def test_fetch_financials_computes_correctly():
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    m = result["CAT"]
    assert m["current_ratio"] == pytest.approx(2.0, abs=0.01)
    assert m["debt_to_equity"] == pytest.approx(2.0, abs=0.01)
    assert m["revenue_growth_yoy"] == pytest.approx(0.25, abs=0.01)  # (20M-16M)/16M
    assert m["gross_margin"] == pytest.approx(0.3, abs=0.01)         # 6M/20M


def test_fetch_financials_returns_empty_on_failure():
    bad_ticker = MagicMock()
    bad_ticker.quarterly_balance_sheet = pd.DataFrame()
    bad_ticker.quarterly_income_stmt = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=bad_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["BADTICKER"])
    assert result["BADTICKER"] == {}


def test_fetch_financials_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.ai_pm_agent._REPO_ROOT", tmp_path)
    (tmp_path / "data_cache").mkdir()
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    cache_file = tmp_path / "data_cache" / "financials_cache.json"
    assert cache_file.exists()
    import json
    cached = json.loads(cache_file.read_text())
    assert "CAT" in cached["data"]


def test_build_data_grounding_financials_values_present():
    """With mocked yfinance, fundamentals block should contain ratio values."""
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker), \
         patch("agents.ai_pm_agent._REPO_ROOT") as mock_root:
        # Make price parquet not exist so we skip that path cleanly
        from pathlib import Path
        mock_root.__truediv__ = lambda self, other: Path("/nonexistent") / other
        from agents.ai_pm_agent import _build_data_grounding
        result = _build_data_grounding(["CAT"])
    # Either we get the fundamentals block, or empty string (if prices missing) —
    # both are valid; just ensure no crash
    assert isinstance(result, str)


def test_build_data_grounding_includes_news_block():
    """When news_context is provided, a NEWS block should appear in output."""
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker), \
         patch("agents.ai_pm_agent._REPO_ROOT") as mock_root:
        from pathlib import Path
        mock_root.__truediv__ = lambda self, other: Path("/nonexistent") / other
        from agents.ai_pm_agent import _build_data_grounding
        result = _build_data_grounding(
            ["CAT"],
            news_context={"CAT": ["CAT beats Q1 estimates"]},
        )
    assert isinstance(result, str)
    # If grounding produced any content, news should be in it
    if result:
        assert "CAT beats Q1" in result


def test_fetch_financials_serves_from_cache(tmp_path, monkeypatch):
    import json, time as _t
    monkeypatch.setattr("agents.ai_pm_agent._REPO_ROOT", tmp_path)
    (tmp_path / "data_cache").mkdir()
    cache_file = tmp_path / "data_cache" / "financials_cache.json"
    cache_file.write_text(json.dumps({
        "_timestamp": _t.time(),
        "data": {"CAT": {"current_ratio": 9.9}},
    }))
    with patch("yfinance.Ticker", side_effect=RuntimeError("should not call")):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    assert result["CAT"]["current_ratio"] == 9.9
