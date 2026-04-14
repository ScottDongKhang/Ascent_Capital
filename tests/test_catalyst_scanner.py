# tests/test_catalyst_scanner.py
from unittest.mock import patch, MagicMock
from datetime import date
import pytest

from ascent.reporting.catalyst_scanner import (
    scan_catalysts,
    _days_until,
    FOMC_DATES_2026,
)


def test_days_until_future():
    d = date(2026, 4, 20)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == 8


def test_days_until_past():
    d = date(2026, 4, 10)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == -2


def test_days_until_today():
    d = date(2026, 4, 12)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == 0


def test_fomc_dates_present():
    # 2026 FOMC has at least 8 meetings
    assert len(FOMC_DATES_2026) >= 8
    assert all(isinstance(d, date) for d in FOMC_DATES_2026)


def test_scan_catalysts_returns_dict():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts(["AAPL", "MSFT"], as_of=date(2026, 4, 12), window_days=30)
    assert isinstance(result, dict)
    assert "upcoming_events" in result
    assert "catalyst_text" in result


def test_scan_catalysts_detects_earnings():
    earnings_date = date(2026, 4, 20)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = lambda sym: {
            "earnings_date": earnings_date if sym == "AAPL" else None,
            "ex_div_date": None,
        }
        result = scan_catalysts(["AAPL", "MSFT"], as_of=date(2026, 4, 12), window_days=30)
    assert any("AAPL" in ev["symbol"] and ev["type"] == "earnings" for ev in result["upcoming_events"])


def test_scan_catalysts_detects_ex_div():
    ex_div = date(2026, 4, 15)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = lambda sym: {
            "earnings_date": None,
            "ex_div_date": ex_div if sym == "MRK" else None,
        }
        result = scan_catalysts(["MRK", "WMT"], as_of=date(2026, 4, 12), window_days=30)
    assert any("MRK" in ev["symbol"] and ev["type"] == "ex_div" for ev in result["upcoming_events"])


def test_scan_catalysts_detects_fomc():
    # FOMC on 2026-05-06 — within 30 days of 2026-04-12
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12), window_days=30)
    fomc_events = [ev for ev in result["upcoming_events"] if ev["type"] == "fomc"]
    assert len(fomc_events) >= 1


def test_scan_catalysts_filters_by_window():
    far_future = date(2026, 8, 1)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": far_future, "ex_div_date": None}
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12), window_days=14)
    # 111 days away, outside the 14-day window
    assert not any(ev["type"] == "earnings" for ev in result["upcoming_events"])


def test_scan_catalysts_empty_symbols():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts([], as_of=date(2026, 4, 12))
    assert result["upcoming_events"] == []
    assert "no upcoming" in result["catalyst_text"].lower()


def test_yfinance_failure_does_not_crash():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = Exception("network error")
        # Should not raise — returns empty events
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12))
    assert result["upcoming_events"] == []


# ── Integration: debate_runner injects catalyst context ───────────────────────

def test_debate_runner_injects_catalyst_context():
    """debate_runner calls scan_catalysts and injects result into portfolio_state."""
    from unittest.mock import patch
    from datetime import date

    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    fake_catalyst = {
        "upcoming_events": [{"symbol": "AAPL", "type": "earnings", "date": "2026-04-20", "days_away": 8}],
        "catalyst_text": "AAPL Earnings: 2026-04-20 (8 days away)",
    }

    import debate.debate_runner as dr
    with patch.object(dr, "scan_catalysts", return_value=fake_catalyst), \
         patch.object(dr, "score_pending_verdicts", return_value=0), \
         patch.object(dr, "run_pending_debriefs", return_value=0), \
         patch.object(dr, "detect_blind_spots"), \
         patch.object(dr, "load_blind_spot_context", return_value=""), \
         patch.object(dr, "run_all_scenarios", return_value=[]), \
         patch.object(dr, "run_bull_agent", return_value="bull"), \
         patch.object(dr, "run_bear_agent", return_value="bear"), \
         patch.object(dr, "run_devils_advocate", return_value="devil"), \
         patch.object(dr, "run_regime_specialist", return_value="regime"), \
         patch.object(dr, "run_quant_sanity_check", return_value="quant"), \
         patch.object(dr, "run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        dr.run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    assert "catalyst_context" in portfolio_state
    assert portfolio_state["catalyst_context"]["catalyst_text"] == fake_catalyst["catalyst_text"]
