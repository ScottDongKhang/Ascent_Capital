"""Tests for analyst revision alpha sleeve."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest


# ── build_analyst_panel ────────────────────────────────────────────────────────

def _make_analyst_df(symbols, dates, scores_by_sym):
    """Helper: build long-format analyst_df from {sym: [(date, score), ...]}."""
    rows = []
    for sym, events in scores_by_sym.items():
        for dt, score in events:
            rows.append({"symbol": sym, "signal_date": pd.Timestamp(dt), "score": score})
    return pd.DataFrame(rows)


def test_build_analyst_panel_basic():
    """Rolling sum of scores fills in correctly for each symbol."""
    from ascent.features.feature_defs import build_analyst_panel

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    syms  = ["A", "B"]

    # A gets +1 upgrade on day 2, B gets -1 downgrade on day 4
    analyst_df = _make_analyst_df(syms, dates, {
        "A": [(dates[1], 1)],
        "B": [(dates[3], -1)],
    })

    result = build_analyst_panel(analyst_df, dates, syms, window=63)
    assert "analyst_revision" in result

    panel = result["analyst_revision"]
    # A's score should be 0 before day 2, 1 on/after day 2
    assert panel.loc[dates[0], "A"] == 0.0
    assert panel.loc[dates[2], "A"] == 1.0
    # B's score should be 0 before day 4, -1 on/after day 4
    assert panel.loc[dates[2], "B"] == 0.0
    assert panel.loc[dates[4], "B"] == -1.0


def test_build_analyst_panel_rolling_decay():
    """Scores roll off after the window elapses."""
    from ascent.features.feature_defs import build_analyst_panel

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    syms = ["A"]

    # Single +1 event on day 0, window=3
    analyst_df = _make_analyst_df(syms, dates, {"A": [(dates[0], 1)]})
    result = build_analyst_panel(analyst_df, dates, syms, window=3)
    panel = result["analyst_revision"]

    # Score present within window
    assert panel.loc[dates[2], "A"] == 1.0
    # Score rolls off after window (day 3+: day 0 event outside [day3-3, day3])
    assert panel.loc[dates[3], "A"] == 0.0


def test_build_analyst_panel_accumulates():
    """Multiple upgrades on different days accumulate in the rolling window."""
    from ascent.features.feature_defs import build_analyst_panel

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    syms = ["A"]
    analyst_df = _make_analyst_df(syms, dates, {
        "A": [(dates[0], 1), (dates[1], 1), (dates[2], -1)],
    })
    result = build_analyst_panel(analyst_df, dates, syms, window=63)
    panel = result["analyst_revision"]
    # By day 3: +1+1-1 = +1
    assert panel.loc[dates[3], "A"] == 1.0


def test_build_analyst_panel_empty_returns_empty():
    """Empty input returns empty dict."""
    from ascent.features.feature_defs import build_analyst_panel
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    assert build_analyst_panel(None, dates, ["A"]) == {}
    assert build_analyst_panel(pd.DataFrame(), dates, ["A"]) == {}


def test_build_analyst_panel_missing_columns_returns_empty():
    from ascent.features.feature_defs import build_analyst_panel
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    bad_df = pd.DataFrame({"symbol": ["A"], "wrong_col": [1]})
    assert build_analyst_panel(bad_df, dates, ["A"]) == {}


def test_build_analyst_panel_normalizes_non_midnight_signal_date():
    """A signal_date with a nonzero time-of-day must still align with a midnight price index."""
    from ascent.features.feature_defs import build_analyst_panel

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    syms = ["A"]

    analyst_df = pd.DataFrame([
        {"symbol": "A", "signal_date": dates[2] + pd.Timedelta(hours=16, minutes=30), "score": 1},
    ])

    result = build_analyst_panel(analyst_df, dates, syms, window=63)
    panel = result["analyst_revision"]
    assert panel.loc[dates[2], "A"] == 1.0


# ── analyst_alpha ──────────────────────────────────────────────────────────────

def test_analyst_alpha_returns_zscore():
    """analyst_alpha cross-sectionally z-scores the revision panel."""
    from ascent.alpha.analyst import analyst_alpha

    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    syms  = ["A", "B", "C"]
    data  = pd.DataFrame(
        {"A": [2.0] * 20, "B": [0.5] * 20, "C": [-1.0] * 20},
        index=dates,
    )
    features = {"analyst_revision": data}
    result = analyst_alpha(features)

    assert not result.empty
    assert result.shape == (20, 3)
    # A > B > C after z-scoring
    assert result["A"].mean() > result["B"].mean() > result["C"].mean()


def test_analyst_alpha_missing_feature_returns_empty():
    from ascent.alpha.analyst import analyst_alpha
    assert analyst_alpha({}).empty
    assert analyst_alpha({"other": pd.DataFrame()}).empty


def test_analyst_alpha_all_zero_returns_empty():
    """No analyst coverage → all zeros → dropped before z-score."""
    from ascent.alpha.analyst import analyst_alpha
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    data  = pd.DataFrame({"A": [0.0] * 5, "B": [0.0] * 5}, index=dates)
    result = analyst_alpha({"analyst_revision": data})
    assert result.empty


# ── ingest helpers ─────────────────────────────────────────────────────────────

def test_action_score_mapping():
    """_action_score maps actions to ±1/0 correctly."""
    from ascent.data.ingest.analyst import _action_score

    assert _action_score("up") == 1
    assert _action_score("upgraded") == 1
    assert _action_score("down") == -1
    assert _action_score("downgraded") == -1
    assert _action_score("lower") == -1
    assert _action_score("reit") == 0
    assert _action_score("main") == 0
    assert _action_score("maintained") == 0
    assert _action_score("init", "Buy") == 1
    assert _action_score("init", "Sell") == -1
    assert _action_score("init", "Neutral") == 0


def test_grade_score_mapping():
    from ascent.data.ingest.analyst import _grade_score

    assert _grade_score("Strong Buy") == 1
    assert _grade_score("Outperform") == 1
    assert _grade_score("Overweight") == 1
    assert _grade_score("Underperform") == -1
    assert _grade_score("Underweight") == -1
    assert _grade_score("Hold") == 0
    assert _grade_score("Neutral") == 0
    assert _grade_score("") == 0


# ── stack integration ──────────────────────────────────────────────────────────

def test_default_alpha_weights_excludes_analyst():
    """Post proof-audit reduction: analyst didn't clear the significance bar and is
    excluded-unmeasured, not live-weighted. DEFAULT_ALPHA_WEIGHTS is meanrev/statarb only."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert "analyst" not in DEFAULT_ALPHA_WEIGHTS
    assert abs(sum(DEFAULT_ALPHA_WEIGHTS.values()) - 1.0) < 0.001


def test_self_improve_weights_match_stack():
    """self_improve.py DEFAULT_ALPHA_WEIGHTS must mirror stack.py exactly."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_w
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_w
    assert set(stack_w.keys()) == set(si_w.keys()), (
        f"stack sleeves {set(stack_w.keys())} != self_improve sleeves {set(si_w.keys())}"
    )
    for k in stack_w:
        assert abs(stack_w[k] - si_w[k]) < 0.001, f"sleeve '{k}' mismatch"


def test_analyst_floor_removed_from_self_improve():
    """analyst is no longer a live sleeve — its MIN_SLEEVE_WEIGHTS floor is pruned."""
    from ascent.research.self_improve import MIN_SLEEVE_WEIGHTS
    assert "analyst" not in MIN_SLEEVE_WEIGHTS


def test_analyst_floor_in_shadow_promoter():
    """_SLEEVE_FLOORS in shadow_promoter must protect analyst sleeve."""
    from ascent.research.shadow_promoter import _SLEEVE_FLOORS
    assert "analyst" in _SLEEVE_FLOORS
    assert _SLEEVE_FLOORS["analyst"] >= 0.01
