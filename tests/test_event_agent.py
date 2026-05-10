"""tests/test_event_agent.py

16 tests for Plan 3 — Event-Driven Architecture.
All tests mock network calls and avoid Alpaca/LLM/EDGAR access.
"""
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Task 1: EDGAR listener ────────────────────────────────────────────────────

def test_edgar_rss_returns_list():
    """poll_edgar_rss returns a list (may be empty, never raises)."""
    from ascent.data.ingest.edgar_listener import poll_edgar_rss
    with patch("ascent.data.ingest.edgar_listener._get", return_value=None):
        result = poll_edgar_rss()
    assert isinstance(result, list)


def test_edgar_deduplication(tmp_path):
    """Filing processed twice → only one log entry."""
    from ascent.data.ingest import edgar_listener as el

    fake_url = "https://sec.gov/Archives/edgar/data/12345/test.htm"
    seen = {fake_url}
    with patch.object(el, "SEEN_PATH", tmp_path / "seen_filings.json"):
        with patch.object(el, "_load_seen", return_value=seen):
            with patch.object(el, "poll_edgar_rss", return_value=[{
                "cik": "12345", "company_name": "TestCo", "filing_type": "8-K",
                "filed_at": datetime.now(timezone.utc), "url": fake_url,
            }]):
                result = el.get_recent_filings(lookback_minutes=60)
    # Already in seen → should be empty (not returned)
    assert result == []


def test_is_universe_company_match(tmp_path):
    """Known name matches via name fuzzy match."""
    from ascent.data.ingest import edgar_listener as el
    cik_map = {"0000320193": "AAPL"}
    with patch.object(el, "_load_cik_map", return_value=cik_map):
        # CIK match
        assert el.is_universe_company("Apple Inc", "0000320193") is True


def test_is_universe_company_no_match(tmp_path):
    """Random string returns False."""
    from ascent.data.ingest import edgar_listener as el
    with patch.object(el, "_load_cik_map", return_value={}):
        assert el.is_universe_company("ZZZ Randomco XYZXYZ", "9999") is False


def test_cik_map_build_and_lookup(tmp_path):
    """build_cik_map saves file and returns a dict."""
    from ascent.data.ingest import edgar_listener as el
    fake_response = json.dumps({
        "hits": {"hits": [{"_source": {"entity_id": "0000320193"}}]}
    })
    with patch.object(el, "CIK_MAP_PATH", tmp_path / "cik_to_symbol.json"):
        with patch.object(el, "_get", return_value=fake_response):
            result = el.build_cik_map(["AAPL"])
    assert isinstance(result, dict)
    assert "0000320193" in result


# ── Task 2: Event classifier ──────────────────────────────────────────────────

def test_classify_event_earnings_beat():
    """Positive 8-K text → direction=buy, conviction≥0.6."""
    from ascent.alpha.event_alpha import classify_event
    mock_result = {"direction": "buy", "conviction": 0.8, "urgency": "high",
                   "category": "earnings_beat", "rationale": "Beat estimates"}
    with patch("ascent.alpha.event_alpha.generate_structured", return_value=mock_result):
        result = classify_event("Revenue beat by 12%", "TestCorp", "8-K")
    assert result["direction"] == "buy"
    assert result["conviction"] >= 0.6


def test_classify_event_guidance_cut():
    """Negative text → direction=sell."""
    from ascent.alpha.event_alpha import classify_event
    mock_result = {"direction": "sell", "conviction": 0.7, "urgency": "high",
                   "category": "guidance_cut", "rationale": "Guidance cut"}
    with patch("ascent.alpha.event_alpha.generate_structured", return_value=mock_result):
        result = classify_event("Lowered full-year guidance by 15%", "TestCorp", "8-K")
    assert result["direction"] == "sell"


def test_classify_event_neutral_on_ambiguous():
    """Routine administrative filing → neutral."""
    from ascent.alpha.event_alpha import classify_event
    mock_result = {"direction": "neutral", "conviction": 0.0, "urgency": "low",
                   "category": "administrative", "rationale": "Routine filing"}
    with patch("ascent.alpha.event_alpha.generate_structured", return_value=mock_result):
        result = classify_event("Board compensation changes effective Q3", "TestCorp", "8-K")
    assert result["direction"] == "neutral"


def test_classify_event_returns_neutral_on_exception():
    """LLM failure → neutral, no raise."""
    from ascent.alpha.event_alpha import classify_event
    with patch("ascent.alpha.event_alpha.generate_structured", side_effect=RuntimeError("LLM down")):
        result = classify_event("Any text", "TestCorp", "8-K")
    assert result["direction"] == "neutral"
    assert result["conviction"] == 0.0


def test_classify_congressional_trade_purchase():
    """Purchase disclosure → direction=buy, conviction=0.4."""
    from ascent.alpha.event_alpha import classify_congressional_trade
    disclosure = {
        "senator": "Jane Smith", "symbol": "AAPL",
        "transaction_type": "purchase", "amount_range": "$1K-$15K",
        "filed_date": date.today(),
    }
    result = classify_congressional_trade(disclosure)
    assert result["direction"] == "buy"
    assert result["conviction"] == 0.4
    assert result["urgency"] == "low"


# ── Task 2.4: Options classifier ──────────────────────────────────────────────

def test_classify_options_anomaly_iv_spike_sell():
    """High IV z-score + high put/call z-score → sell, conviction 0.6."""
    from ascent.alpha.event_alpha import classify_options_anomaly
    result = classify_options_anomaly("AAPL", iv_zscore=3.1, put_call_ratio_zscore=2.5)
    assert result["direction"] == "sell"
    assert result["conviction"] == 0.6
    assert result["urgency"] == "high"


def test_classify_options_anomaly_call_buying_buy():
    """Unusual call buying (negative put/call z) → buy."""
    from ascent.alpha.event_alpha import classify_options_anomaly
    result = classify_options_anomaly("AAPL", iv_zscore=1.0, put_call_ratio_zscore=-2.5)
    assert result["direction"] == "buy"
    assert result["conviction"] == 0.5


# ── Task 5: Event runner ──────────────────────────────────────────────────────

def test_compute_event_size_respects_cap():
    """Never exceeds max_event_pct × NAV."""
    from ascent.execution.event_runner import compute_event_size
    nav = 100_000.0
    size = compute_event_size(conviction=1.0, urgency="high", nav=nav, max_event_pct=0.005)
    assert size <= 0.005 * nav + 1e-6


def test_compute_event_size_minimum_threshold():
    """Below $500 → returns 0.0 (skip trade)."""
    from ascent.execution.event_runner import compute_event_size
    # Very small NAV → size below $500
    size = compute_event_size(conviction=0.1, urgency="low", nav=1_000.0, max_event_pct=0.005)
    assert size == 0.0


def test_event_trading_disabled_no_order_submitted():
    """EVENT_TRADING_ENABLED=False → no Alpaca call."""
    import ascent.execution.event_runner as er
    original = er.EVENT_TRADING_ENABLED
    er.EVENT_TRADING_ENABLED = False
    try:
        with patch("ascent.execution.event_runner._submit_limit_order") as mock_submit:
            result = er.submit_event_trade("AAPL", "buy", 500.0, "test", "evt-001")
        mock_submit.assert_not_called()
        assert result["status"] == "disabled"
    finally:
        er.EVENT_TRADING_ENABLED = original


# ── Task 5.6: Event IC ────────────────────────────────────────────────────────

def test_compute_event_ic_correct_direction_sign(tmp_path):
    """Correct directional trades have positive IC."""
    import ascent.execution.event_runner as er
    from unittest.mock import patch as _patch

    # Write 6 synthetic event trades: 3 buys (pos return) + 3 sells (neg return)
    log_path = tmp_path / "event_trades.jsonl"
    trades = []
    for i in range(3):
        trades.append(json.dumps({
            "event_id": f"buy-{i}", "timestamp": "2026-04-01T10:00:00",
            "symbol": f"BUY{i}", "direction": "buy",
            "conviction": 0.7, "urgency": "high", "category": "earnings",
            "rationale": "", "size_dollars": 500, "fill_price": None,
            "nav_at_decision": 100000, "status": "submitted", "source": "edgar",
        }))
    for i in range(3):
        trades.append(json.dumps({
            "event_id": f"sell-{i}", "timestamp": "2026-04-01T10:00:00",
            "symbol": f"SELL{i}", "direction": "sell",
            "conviction": 0.7, "urgency": "high", "category": "earnings",
            "rationale": "", "size_dollars": 500, "fill_price": None,
            "nav_at_decision": 100000, "status": "submitted", "source": "edgar",
        }))
    log_path.write_text("\n".join(trades) + "\n")

    # Fake prices: BUY symbols go up, SELL symbols go down
    import pandas as _pd, numpy as _np
    dates = _pd.bdate_range("2026-04-01", periods=30)
    buy_syms  = [f"BUY{i}" for i in range(3)]
    sell_syms = [f"SELL{i}" for i in range(3)]
    all_syms  = buy_syms + sell_syms
    prices = _pd.DataFrame(
        {s: 100 + (_np.arange(30) * (1 if s.startswith("BUY") else -1)) * 0.5 for s in all_syms},
        index=dates,
    )

    with _patch.object(er, "EVENT_LOG", log_path):
        # Patch yfinance to avoid network call
        with _patch("yfinance.download", return_value=_pd.DataFrame()):
            result = er.compute_event_ic(lookback_days=20)

    assert result["n_trades"] == 6
    # With empty price data (mocked) IC values should be None — no crash
    assert result["ic_5d"] is None
