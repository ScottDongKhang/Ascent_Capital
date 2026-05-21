import json
import pytest
from pathlib import Path
from ascent.monitoring.signal_health import compute_signal_health


def _write_ic_log(tmp_path, entries):
    p = tmp_path / "sleeve_ic_log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return str(p)


def _write_state(tmp_path, sleeve_ics):
    p = tmp_path / "last_rebalance_state.json"
    p.write_text(json.dumps({
        "date": "2026-05-10",
        "weights": {}, "alpha_ranks": {}, "alpha_scores": {},
        "sleeve_ics": sleeve_ics,
        "regime": "calm_bull", "regime_stability_10d": 0.9,
    }))
    return str(p)


def test_signal_health_detects_decay():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        entries = [
            {"date": f"2026-05-{11+i:02d}", "sleeves": {
                "trend": {"mean_ic": 0.014 - i * 0.002, "ic_t": 3.0, "n": 1500}
            }} for i in range(7)
        ]
        ic_path    = _write_ic_log(Path(tmp), entries)
        state_path = _write_state(Path(tmp), {"trend": 0.014})

        result = compute_signal_health("2026-05-20", ic_log_path=ic_path, state_path=state_path)

        assert "trend" in result
        assert result["trend"]["ic_at_rebalance"] == pytest.approx(0.014)
        assert result["trend"]["ic_5d_avg"] < 0.014
        assert result["trend"]["status"] in ("healthy", "weakening", "deteriorating")


def test_signal_health_returns_empty_without_log(tmp_path):
    result = compute_signal_health(
        "2026-05-20",
        ic_log_path=str(tmp_path / "missing.jsonl"),
        state_path=str(tmp_path / "missing.json"),
    )
    assert result == {}
