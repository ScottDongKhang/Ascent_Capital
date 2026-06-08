import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_ticker_memory(tmp_path):
    """Import ticker_memory with JSONL path redirected to tmp_path."""
    import importlib, sys
    # Temporarily patch the path constant after import
    import memory.ticker_memory as tm
    tm.TICKER_MEMORY_PATH = tmp_path / "ticker_memory.jsonl"
    return tm


def test_record_decision_appends_entry(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    tm.record_decision(
        symbol="CAT", date_str="2026-06-10", ai_w=0.10, quant_w=0.07,
        decision_type="amplify", rationale_snippet="Strong capex cycle momentum"
    )
    lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["symbol"] == "CAT"
    assert entry["ai_w"] == pytest.approx(0.10)
    assert entry["quant_w"] == pytest.approx(0.07)
    assert entry["scored"] is False


def test_record_decision_multiple_symbols(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    tm.record_decision("CAT", "2026-06-10", 0.10, 0.07, "amplify", "reason A")
    tm.record_decision("MRK", "2026-06-10", 0.06, 0.08, "reduce",  "reason B")
    lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_score_outcomes_skips_recent(tmp_path):
    """Entries less than 10 days old should not be scored."""
    tm = _make_ticker_memory(tmp_path)
    today_str = date.today().isoformat()
    tm.record_decision("CAT", today_str, 0.10, 0.07, "amplify", "recent")
    scored = tm.score_outcomes(date.today())
    assert scored == 0
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    assert entry["scored"] is False


def test_score_outcomes_scores_old_entry(tmp_path):
    """Entries 15 days old should be scored with mocked price data."""
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "old entry")

    with patch("memory.ticker_memory._fetch_return", side_effect=[0.05, 0.08]):
        scored = tm.score_outcomes(date.today())

    assert scored == 1
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    assert entry["scored"] is True
    assert entry["verdict"] in ("win", "miss", "fade", "early")
    # incremental_alpha = (ai_w - quant_w) * return = (0.10 - 0.07) * 0.05 = +0.0015
    assert entry["outcome_10d"] == pytest.approx((0.10 - 0.07) * 0.05, abs=1e-6)


def test_get_ticker_context_empty(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    ctx = tm.get_ticker_context("CAT")
    assert ctx == ""


def test_get_ticker_context_formats_history(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "capex thesis")
    # Manually mark as scored
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    entry.update(scored=True, outcome_10d=0.0015, outcome_21d=0.002, verdict="win")
    (tmp_path / "ticker_memory.jsonl").write_text(json.dumps(entry) + "\n")

    ctx = tm.get_ticker_context("CAT")
    assert "CAT" in ctx
    assert "amplify" in ctx
    assert "WIN" in ctx


def test_get_ticker_context_only_returns_that_symbol(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "reason")
    tm.record_decision("MRK", old_date, 0.06, 0.08, "reduce",  "reason")
    # score both
    for sym in ["CAT", "MRK"]:
        lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
        updated = []
        for line in lines:
            e = json.loads(line)
            if e["symbol"] == sym:
                e.update(scored=True, outcome_10d=0.001, outcome_21d=0.002, verdict="win")
            updated.append(json.dumps(e))
        (tmp_path / "ticker_memory.jsonl").write_text("\n".join(updated) + "\n")

    cat_ctx = tm.get_ticker_context("CAT")
    assert "CAT" in cat_ctx
    assert "MRK" not in cat_ctx


def test_get_cross_ticker_lessons_returns_recent(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    for sym in ["CAT", "MRK", "WMT"]:
        tm.record_decision(sym, old_date, 0.10, 0.07, "amplify", "reason")
        lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
        updated = []
        for line in lines:
            e = json.loads(line)
            if e["symbol"] == sym and not e["scored"]:
                e.update(scored=True, outcome_10d=0.001, outcome_21d=0.002, verdict="win")
            updated.append(json.dumps(e))
        (tmp_path / "ticker_memory.jsonl").write_text("\n".join(updated) + "\n")

    ctx = tm.get_cross_ticker_lessons(n=2)
    assert ctx != ""
    # Should include at most 2 entries
    assert ctx.count("amplify") <= 2
