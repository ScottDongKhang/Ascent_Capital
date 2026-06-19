import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_performance_page import compute_stats


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
