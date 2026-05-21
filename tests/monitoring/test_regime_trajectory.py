import json
import pytest
from pathlib import Path
from ascent.monitoring.regime_trajectory import compute_regime_trajectory


def _write_regime_signal(tmp_path, series, current="calm_bull"):
    data = {
        "regime": current, "label": current,
        "as_of": series[-1]["d"], "last_refit_date": series[-1]["d"],
        "series": series
    }
    p = tmp_path / "regime_signal.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_stable_regime(tmp_path):
    series = [
        {"d": f"2026-05-{i:02d}", "label": "calm_bull", "risk_mult": 1.0, "rs": 0.0}
        for i in range(10, 21)
    ]
    path = _write_regime_signal(tmp_path, series)
    result = compute_regime_trajectory("2026-05-20", signal_path=path)

    assert result["current_label"] == "calm_bull"
    assert result["stability_10d"] == pytest.approx(1.0)
    assert result["days_in_regime"] >= 10


def test_unstable_regime(tmp_path):
    labels = ["calm_bull", "stressed", "calm_bull", "stressed", "calm_bull",
              "stressed", "calm_bull", "calm_bull", "stressed", "calm_bull"]
    series = [
        {"d": f"2026-05-{i+10:02d}", "label": labels[i], "risk_mult": 1.0, "rs": 0.1 * i}
        for i in range(10)
    ]
    path = _write_regime_signal(tmp_path, series, current="calm_bull")
    result = compute_regime_trajectory("2026-05-20", signal_path=path)

    assert result["stability_10d"] < 0.8
    assert result["rs_trend"] in ("rising", "flat", "falling")


def test_returns_empty_without_file(tmp_path):
    result = compute_regime_trajectory("2026-05-20", signal_path=str(tmp_path / "missing.json"))
    assert result == {}
