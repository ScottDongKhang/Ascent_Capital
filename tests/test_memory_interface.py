# tests/test_memory_interface.py
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# ── Local fallback tests (no R2R key required) ──────────────────────────────

def _write_fake_verdict(tmpdir: Path, filename: str, data: dict):
    """Helper: write a fake verdict JSON to the temp debate log dir."""
    (tmpdir / filename).write_text(json.dumps(data))


def test_local_fallback_returns_empty_on_no_verdicts(tmp_path):
    from memory.r2r_interface import _local_search
    result = _local_search("calm_bull AAPL MSFT", debate_log_dir=tmp_path, n=3)
    assert result == []


def test_local_fallback_finds_regime_match(tmp_path):
    _write_fake_verdict(tmp_path, "verdict_2026-01-15.json", {
        "date": "2026-01-15",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {"AAPL": 0.2}},
        "verdict": {"recommendation": "proceed", "reasoning": "Momentum is strong in tech."},
    })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull AAPL", debate_log_dir=tmp_path, n=3)
    assert len(results) == 1
    assert results[0]["date"] == "2026-01-15"
    assert results[0]["recommendation"] == "proceed"


def test_local_fallback_ranks_by_overlap(tmp_path):
    _write_fake_verdict(tmp_path, "verdict_2026-01-10.json", {
        "date": "2026-01-10",
        "portfolio_state": {"us_regime": "stressed", "weights": {"TLT": 0.3}},
        "verdict": {"recommendation": "reduce_size", "reasoning": "Stress regime."},
    })
    _write_fake_verdict(tmp_path, "verdict_2026-02-01.json", {
        "date": "2026-02-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {"AAPL": 0.2, "MSFT": 0.15}},
        "verdict": {"recommendation": "proceed", "reasoning": "Bull confirmed."},
    })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull AAPL", debate_log_dir=tmp_path, n=3)
    assert results[0]["date"] == "2026-02-01"


def test_local_fallback_limits_results(tmp_path):
    for i in range(5):
        _write_fake_verdict(tmp_path, f"verdict_2026-0{i+1}-01.json", {
            "date": f"2026-0{i+1}-01",
            "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
            "verdict": {"recommendation": "proceed", "reasoning": "calm_bull test"},
        })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull", debate_log_dir=tmp_path, n=2)
    assert len(results) <= 2


def test_local_fallback_skips_malformed_files(tmp_path):
    (tmp_path / "verdict_bad.json").write_text("not valid json{{{")
    _write_fake_verdict(tmp_path, "verdict_2026-03-01.json", {
        "date": "2026-03-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
        "verdict": {"recommendation": "proceed", "reasoning": "ok"},
    })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull", debate_log_dir=tmp_path, n=5)
    assert len(results) == 1


# ── query_memory() interface tests ──────────────────────────────────────────

def test_query_memory_uses_local_fallback_when_no_api_key(tmp_path):
    """Without R2R_API_KEY, query_memory falls through to local search."""
    _write_fake_verdict(tmp_path, "verdict_2026-03-10.json", {
        "date": "2026-03-10",
        "portfolio_state": {"us_regime": "stressed", "weights": {"GLD": 0.3}},
        "verdict": {"recommendation": "reduce_size", "reasoning": "Stress regime."},
    })

    import memory.r2r_interface as r2r
    with patch.object(r2r, "R2R_API_KEY", ""):
        results = r2r.query_memory(query="stressed GLD", n=3, debate_log_dir=tmp_path)
    assert isinstance(results, list)


def test_query_memory_calls_r2r_api_when_key_present(tmp_path):
    """With R2R_API_KEY set, query_memory calls the R2R HTTP endpoint."""
    fake_r2r_response = {
        "results": [
            {
                "metadata": {"date": "2026-03-01", "recommendation": "proceed"},
                "text": "Past verdict: calm_bull, proceed. Reasoning: strong momentum.",
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_r2r_response

    import memory.r2r_interface as r2r
    with patch.object(r2r, "R2R_API_KEY", "test-key-123"), \
         patch("requests.post", return_value=mock_resp):
        results = r2r.query_memory(query="calm_bull AAPL", n=3, debate_log_dir=tmp_path)
    assert isinstance(results, list)


def test_query_memory_falls_back_on_r2r_failure(tmp_path):
    """If R2R API call fails, falls back to local search without raising."""
    _write_fake_verdict(tmp_path, "verdict_2026-02-01.json", {
        "date": "2026-02-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
        "verdict": {"recommendation": "proceed", "reasoning": "ok"},
    })

    import memory.r2r_interface as r2r
    with patch.object(r2r, "R2R_API_KEY", "test-key-123"), \
         patch("requests.post", side_effect=Exception("timeout")):
        results = r2r.query_memory(query="calm_bull", n=3, debate_log_dir=tmp_path)
    assert isinstance(results, list)


# ── format_memory_context() tests ───────────────────────────────────────────

def test_format_memory_context_empty():
    from memory.r2r_interface import format_memory_context
    text = format_memory_context([])
    assert "no relevant" in text.lower()


def test_format_memory_context_formats_results():
    from memory.r2r_interface import format_memory_context
    results = [
        {"date": "2026-03-01", "recommendation": "proceed",
         "reasoning": "Momentum was strong.", "regime": "calm_bull", "key_risks": []},
        {"date": "2026-02-15", "recommendation": "reduce_size",
         "reasoning": "Stressed market conditions.", "regime": "stressed", "key_risks": []},
    ]
    text = format_memory_context(results)
    assert "2026-03-01" in text
    assert "proceed" in text
    assert "Momentum was strong" in text


# ── debate_runner integration tests ─────────────────────────────────────────

def test_debate_runner_injects_memory_context(tmp_path):
    """debate_runner calls query_memory and injects result into portfolio_state."""
    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    fake_memory = [
        {"date": "2026-02-01", "recommendation": "proceed",
         "reasoning": "Momentum strong.", "regime": "calm_bull", "key_risks": []},
    ]

    import debate.debate_runner as dr
    with patch.object(dr, "query_memory", return_value=fake_memory) as mock_qm, \
         patch.object(dr, "ingest_verdict"), \
         patch.object(dr, "score_pending_verdicts", return_value=0), \
         patch.object(dr, "run_pending_debriefs", return_value=0), \
         patch.object(dr, "detect_blind_spots"), \
         patch.object(dr, "load_blind_spot_context", return_value=""), \
         patch.object(dr, "run_all_scenarios", return_value=[]), \
         patch.object(dr, "scan_catalysts", return_value={"upcoming_events": [], "catalyst_text": ""}), \
         patch.object(dr, "run_bull_agent", return_value="bull"), \
         patch.object(dr, "run_bear_agent", return_value="bear"), \
         patch.object(dr, "run_devils_advocate", return_value="devil"), \
         patch.object(dr, "run_regime_specialist", return_value="regime"), \
         patch.object(dr, "run_quant_sanity_check", return_value="quant"), \
         patch.object(dr, "run_bull_rebuttal", return_value="bull_r2"), \
         patch.object(dr, "run_bear_rebuttal", return_value="bear_r2"), \
         patch.object(dr, "run_devils_advocate_rebuttal", return_value="devil_r2"), \
         patch.object(dr, "run_regime_specialist_rebuttal", return_value="regime_r2"), \
         patch.object(dr, "run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        dr.run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    mock_qm.assert_called_once()
    assert "memory_context" in portfolio_state
    assert portfolio_state["memory_context"] == fake_memory


def test_debate_runner_ingests_verdict_after_writing():
    """debate_runner calls ingest_verdict after writing the verdict JSON."""
    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 1,
        "allocation": {},
        "weights": {"AAPL": 1.0},
    }

    import debate.debate_runner as dr
    with patch.object(dr, "query_memory", return_value=[]), \
         patch.object(dr, "ingest_verdict") as mock_ingest, \
         patch.object(dr, "score_pending_verdicts", return_value=0), \
         patch.object(dr, "run_pending_debriefs", return_value=0), \
         patch.object(dr, "detect_blind_spots"), \
         patch.object(dr, "load_blind_spot_context", return_value=""), \
         patch.object(dr, "run_all_scenarios", return_value=[]), \
         patch.object(dr, "scan_catalysts", return_value={"upcoming_events": [], "catalyst_text": ""}), \
         patch.object(dr, "run_bull_agent", return_value="bull"), \
         patch.object(dr, "run_bear_agent", return_value="bear"), \
         patch.object(dr, "run_devils_advocate", return_value="devil"), \
         patch.object(dr, "run_regime_specialist", return_value="regime"), \
         patch.object(dr, "run_quant_sanity_check", return_value="quant"), \
         patch.object(dr, "run_bull_rebuttal", return_value="bull_r2"), \
         patch.object(dr, "run_bear_rebuttal", return_value="bear_r2"), \
         patch.object(dr, "run_devils_advocate_rebuttal", return_value="devil_r2"), \
         patch.object(dr, "run_regime_specialist_rebuttal", return_value="regime_r2"), \
         patch.object(dr, "run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        dr.run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    mock_ingest.assert_called_once()


def test_build_context_includes_memory_context():
    """_build_context includes formatted memory context when present."""
    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 1,
        "allocation": {},
        "weights": {"AAPL": 1.0},
        "memory_context": [
            {"date": "2026-02-01", "recommendation": "proceed",
             "reasoning": "Bull momentum.", "regime": "calm_bull", "key_risks": []},
        ],
    }
    from debate.agents import _build_context
    ctx = _build_context(portfolio_state)
    assert "2026-02-01" in ctx
    assert "proceed" in ctx
    assert "Bull momentum" in ctx
