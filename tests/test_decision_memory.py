# tests/test_decision_memory.py
import json
import pytest
from pathlib import Path


def _write_memory_log(tmp_path, records):
    p = tmp_path / "logs" / "decision_memory.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_ingest_override_writes_record(tmp_path):
    from ascent.memory.decision_memory import ingest_override
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    ingest_override(
        rebalance_date="2026-05-19",
        symbol="AAPL",
        override_type="data_quality",
        regime="calm_bull",
        ai_action="REMOVED",
        ai_weight=0.0,
        quant_weight=0.08,
        momentum_252d=0.45,
        log_path=log_path,
    )
    assert log_path.exists()
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "AAPL"
    assert row["override_type"] == "data_quality"
    assert row["regime"] == "calm_bull"
    assert row["wedge_21d"] is None
    assert abs(row["weight_delta"] - (-0.08)) < 1e-9


def test_ingest_override_deduplicates(tmp_path):
    from ascent.memory.decision_memory import ingest_override
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    for _ in range(3):
        ingest_override(
            rebalance_date="2026-05-19",
            symbol="MSFT",
            override_type="valuation",
            regime="calm_bull",
            ai_action="REDUCED from 10% to 4%",
            ai_weight=0.04,
            quant_weight=0.10,
            log_path=log_path,
        )
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1


def test_update_outcomes_fills_wedge(tmp_path):
    from ascent.memory.decision_memory import update_outcomes
    log_path = _write_memory_log(tmp_path, [
        {
            "entry_id": "2026-05-01_AAPL",
            "rebalance_date": "2026-05-01",
            "symbol": "AAPL",
            "override_type": "regime_macro",
            "regime": "stressed",
            "ai_action": "REDUCED",
            "ai_weight": 0.05,
            "quant_weight": 0.10,
            "weight_delta": -0.05,
            "momentum_252d": 0.30,
            "wedge_21d": None,
        }
    ])
    update_outcomes("2026-05-01", 0.025, log_path=log_path)
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert abs(rows[0]["wedge_21d"] - 0.025) < 1e-9


def test_update_outcomes_does_not_overwrite_existing(tmp_path):
    from ascent.memory.decision_memory import update_outcomes
    log_path = _write_memory_log(tmp_path, [
        {
            "entry_id": "2026-04-01_MSFT",
            "rebalance_date": "2026-04-01",
            "symbol": "MSFT",
            "override_type": "valuation",
            "regime": "calm_bull",
            "ai_action": "REMOVED",
            "ai_weight": 0.0,
            "quant_weight": 0.09,
            "weight_delta": -0.09,
            "momentum_252d": None,
            "wedge_21d": 0.01,  # already filled
        }
    ])
    update_outcomes("2026-04-01", 0.99, log_path=log_path)
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    # wedge_21d already set — must not overwrite
    assert abs(rows[0]["wedge_21d"] - 0.01) < 1e-9


def test_query_filters_by_type_and_regime(tmp_path):
    from ascent.memory.decision_memory import query
    log_path = _write_memory_log(tmp_path, [
        {"entry_id": "2026-04-01_A", "rebalance_date": "2026-04-01", "symbol": "A",
         "override_type": "data_quality", "regime": "calm_bull", "ai_action": "REMOVED",
         "ai_weight": 0.0, "quant_weight": 0.05, "weight_delta": -0.05,
         "momentum_252d": None, "wedge_21d": 0.01},
        {"entry_id": "2026-04-01_B", "rebalance_date": "2026-04-01", "symbol": "B",
         "override_type": "valuation", "regime": "calm_bull", "ai_action": "REMOVED",
         "ai_weight": 0.0, "quant_weight": 0.06, "weight_delta": -0.06,
         "momentum_252d": None, "wedge_21d": -0.02},
        {"entry_id": "2026-04-01_C", "rebalance_date": "2026-04-01", "symbol": "C",
         "override_type": "data_quality", "regime": "stressed", "ai_action": "REMOVED",
         "ai_weight": 0.0, "quant_weight": 0.04, "weight_delta": -0.04,
         "momentum_252d": None, "wedge_21d": 0.005},
    ])
    results = query(override_type="data_quality", regime="calm_bull", log_path=log_path)
    assert len(results) == 1
    assert results[0].symbol == "A"


def test_get_statistics_win_rate(tmp_path):
    from ascent.memory.decision_memory import get_statistics
    log_path = _write_memory_log(tmp_path, [
        {"entry_id": f"2026-04-{i:02d}_X", "rebalance_date": f"2026-04-{i:02d}",
         "symbol": "X", "override_type": "regime_macro", "regime": "calm_bull",
         "ai_action": "REDUCED", "ai_weight": 0.05, "quant_weight": 0.10,
         "weight_delta": -0.05, "momentum_252d": None,
         "wedge_21d": 0.01 if i % 2 == 0 else -0.01}
        for i in range(1, 11)
    ])
    stats = get_statistics(override_type="regime_macro", regime="calm_bull", log_path=log_path)
    assert stats["n_cases"] == 10
    assert abs(stats["win_rate"] - 0.50) < 0.01
    assert stats["avg_wedge"] is not None


def test_get_statistics_no_data(tmp_path):
    from ascent.memory.decision_memory import get_statistics
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    stats = get_statistics(log_path=log_path)
    assert stats["n_cases"] == 0
    assert stats["win_rate"] is None


def test_format_query_result_no_data(tmp_path):
    from ascent.memory.decision_memory import format_query_result
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    result = format_query_result(log_path=log_path)
    assert "no matured outcomes" in result


def test_format_query_result_with_data(tmp_path):
    from ascent.memory.decision_memory import format_query_result
    log_path = _write_memory_log(tmp_path, [
        {"entry_id": f"2026-04-{i:02d}_Y", "rebalance_date": f"2026-04-{i:02d}",
         "symbol": "Y", "override_type": "news_event", "regime": "calm_bull",
         "ai_action": "ADDED", "ai_weight": 0.05, "quant_weight": 0.0,
         "weight_delta": 0.05, "momentum_252d": 0.20,
         "wedge_21d": 0.02 if i <= 7 else -0.01}
        for i in range(1, 11)
    ])
    result = format_query_result(override_type="news_event", regime="calm_bull", log_path=log_path)
    assert "matured outcome" in result
    assert "Win rate" in result
    assert "Recommendation" in result


def test_ingest_override_stores_altdata_fields(tmp_path):
    """With mocked caches, new altdata fields are written to the JSONL record."""
    import json
    import pandas as pd
    from unittest.mock import patch
    from ascent.memory.decision_memory import ingest_override

    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    detail_path = tmp_path / "data_cache" / "altdata_sec_detail.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps({"AAPL": {"tone": 0.65, "as_of": "2026-05-24"}}))

    trends_df = pd.DataFrame({"AAPL": [0.42]},
                              index=pd.DatetimeIndex(["2026-05-24"]))
    trends_df.index.name = "date"
    trends_path = tmp_path / "data_cache" / "altdata_trends.parquet"
    trends_df.to_parquet(trends_path)

    with patch("ascent.memory.decision_memory._REPO_ROOT", tmp_path):
        ingest_override(
            rebalance_date="2026-05-24",
            symbol="AAPL",
            override_type="valuation",
            regime="calm_bull",
            ai_action="REDUCED from 10% to 5%",
            ai_weight=0.05,
            quant_weight=0.10,
            log_path=log_path,
        )

    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["sec_tone"] == 0.65
    assert rows[0]["trends_direction"] == pytest.approx(0.42, abs=0.01)


def test_ingest_override_backward_compat_old_records(tmp_path):
    """Old records missing altdata fields deserialize without KeyError."""
    import json
    from ascent.memory.decision_memory import query, OverrideRecord

    old_record = {
        "entry_id": "2026-04-01_AAPL",
        "rebalance_date": "2026-04-01",
        "symbol": "AAPL",
        "override_type": "valuation",
        "regime": "calm_bull",
        "ai_action": "REDUCED",
        "ai_weight": 0.05,
        "quant_weight": 0.10,
        "weight_delta": -0.05,
        "momentum_252d": 0.30,
        "wedge_21d": 0.02,
    }
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps(old_record) + "\n")

    records = query(log_path=log_path)
    assert len(records) == 1
    r = records[0]
    assert r.sec_tone is None
    assert r.transcript_sentiment is None
    assert r.reddit_buzz is None
    assert r.trends_direction is None


def test_read_altdata_context_returns_all_none_when_no_caches(tmp_path):
    """No cache files → all four fields are None, no exception."""
    from unittest.mock import patch
    from ascent.memory.decision_memory import _read_altdata_context
    with patch("ascent.memory.decision_memory._REPO_ROOT", tmp_path):
        ctx = _read_altdata_context("AAPL")
    assert ctx == {
        "sec_tone": None,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": None,
    }
