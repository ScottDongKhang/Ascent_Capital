import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_performance_page import (
    compute_stats,
    _sparkline_paths,
    _redaction_label,
    _REDACTION_LABELS,
    _position_reasoning,
    _construction_section_html,
    _CONSTRUCTION_STAGES,
)


def test_compute_stats_spy_alpha_ignores_trailing_nan():
    records = [
        {"date": "2026-04-01", "equity": 100000.0, "day_return": 0.0},
        {"date": "2026-04-02", "equity": 101000.0, "day_return": 0.01},
        {"date": "2026-04-03", "equity": 108710.0, "day_return": 0.02},
    ]
    # last SPY bar is NaN (today's unpublished bar) — must fall back to last finite
    spy = {"2026-04-01": 100000.0, "2026-04-02": 105000.0, "2026-04-03": float("nan")}
    s = compute_stats(records, spy)
    assert s["spy_return"] is not None and math.isfinite(s["spy_return"])
    assert s["alpha"] is not None and math.isfinite(s["alpha"])
    assert s["spy_return"] == 5.0  # (105000/100000 - 1) * 100


def test_sparkline_paths_basic_and_missing(tmp_path, monkeypatch):
    import pandas as pd
    import scripts.generate_performance_page as g
    df = pd.DataFrame({
        "symbol": ["AAA"] * 4 + ["BBB"] * 1,
        "date":   ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-04"],
        "close":  [10.0, 11.0, 9.0, 12.0, 50.0],
    })
    p = tmp_path / "prices_live.parquet"
    df.to_parquet(p)
    monkeypatch.setattr(g, "PRICES_LIVE_PATH", str(p), raising=False)
    out = _sparkline_paths(["AAA", "BBB", "ZZZ"])
    assert "AAA" in out and out["AAA"]["d"].startswith("M")
    assert out["AAA"]["up"] is True            # 12 >= 10
    assert "BBB" not in out                     # only 1 point
    assert "ZZZ" not in out                     # absent symbol


def test_redaction_label_deterministic_and_in_set():
    assert _redaction_label("IFRA") == _redaction_label("IFRA")   # deterministic
    assert _redaction_label("IFRA") in _REDACTION_LABELS


def test_position_reasoning_conviction_and_flag():
    prethesis = {"high_conviction_names": [
        {"symbol": "IFRA", "reason": "AI data-center build-out beneficiary."}]}
    verdict = {"verdict": {"key_risks": [
        "BYD earnings imminent at 7.2% weight — unhedgeable binary."]}}
    ifra = _position_reasoning("IFRA", verdict, prethesis)
    assert "build-out" in ifra["why"]
    assert ifra["flagged"] is False
    byd = _position_reasoning("BYD", verdict, prethesis)
    assert byd["flagged"] is True and "BYD" in byd["committee"]
    zzz = _position_reasoning("ZZZ", verdict, prethesis)
    assert zzz["why"] and zzz["flagged"] is False
    assert "No adversarial flag" in zzz["committee"]


def test_construction_section_renders_and_is_sealed():
    html = _construction_section_html(
        "calm_bull",
        {"verdict": {"recommendation": "proceed", "confidence": 0.62}},
        {"level": 1, "title": "Analyst", "ai_weight": 0.05}, 17, 500)
    assert "How the book is built" in html
    assert "Sealed" in html and "calm_bull" in html
    for banned in ("0.70", "sleeve_weight", "DEFAULT_ALPHA_WEIGHTS", "trend=", "0.45"):
        assert banned not in html
    # never raises on empty inputs
    assert _construction_section_html("", {}, {}, 0, 0)
    assert len(_CONSTRUCTION_STAGES) == 7
