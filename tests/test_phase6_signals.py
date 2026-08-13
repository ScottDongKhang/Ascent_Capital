# tests/test_phase6_signals.py
"""Tests for Phase 6 alpha sleeves: options flow, insider transactions, short interest."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _bdate(n):
    return pd.bdate_range(end="2026-04-17", periods=n)


def _syms(n=5):
    return [f"S{i}" for i in range(n)]


# ── build_options_panel ───────────────────────────────────────────────────────

def test_options_panel_basic():
    from ascent.features.feature_defs import build_options_panel

    dates = _bdate(10)
    syms  = _syms(3)

    df = pd.DataFrame([
        {"symbol": "S0", "date": dates[3], "iv_skew": 0.05, "put_call_ratio": 0.55},
        {"symbol": "S1", "date": dates[3], "iv_skew": -0.02, "put_call_ratio": 0.45},
    ])
    result = build_options_panel(df, dates, syms)
    assert "iv_skew" in result
    assert "put_call_ratio" in result
    assert result["iv_skew"].shape[0] == len(dates)


def test_options_panel_forward_fills():
    from ascent.features.feature_defs import build_options_panel

    dates = _bdate(10)
    syms  = _syms(2)

    df = pd.DataFrame([
        {"symbol": "S0", "date": dates[2], "iv_skew": 0.10, "put_call_ratio": 0.60},
    ])
    result = build_options_panel(df, dates, syms)
    assert "iv_skew" in result
    # Day 2 → forward-filled to day 3, 4, 5, 6, 7 (limit=5)
    panel = result["iv_skew"]
    assert panel.loc[dates[3], "S0"] == pytest.approx(0.10)
    assert panel.loc[dates[7], "S0"] == pytest.approx(0.10)
    # Day 8+ beyond limit → NaN
    assert pd.isna(panel.loc[dates[8], "S0"])


def test_options_panel_empty_returns_empty():
    from ascent.features.feature_defs import build_options_panel

    dates = _bdate(5)
    assert build_options_panel(None, dates, _syms()) == {}
    assert build_options_panel(pd.DataFrame(), dates, _syms()) == {}


def test_options_panel_missing_columns():
    from ascent.features.feature_defs import build_options_panel

    dates = _bdate(5)
    bad = pd.DataFrame({"symbol": ["S0"], "wrong": [1]})
    assert build_options_panel(bad, dates, _syms()) == {}


# ── build_insider_panel ───────────────────────────────────────────────────────

def test_insider_panel_basic():
    from ascent.features.feature_defs import build_insider_panel

    dates = _bdate(20)
    syms  = _syms(3)

    df = pd.DataFrame([
        {"symbol": "S0", "signal_date": dates[5],  "score": 1},
        {"symbol": "S0", "signal_date": dates[8],  "score": 1},
        {"symbol": "S1", "signal_date": dates[10], "score": -1},
    ])
    result = build_insider_panel(df, dates, syms, window=63)
    assert "insider_net_score" in result

    panel = result["insider_net_score"]
    # S0 has 2 buys by day 9: rolling sum should be 2.0
    assert panel.loc[dates[9], "S0"] == pytest.approx(2.0)
    # S1 has 1 sell by day 11
    assert panel.loc[dates[11], "S1"] == pytest.approx(-1.0)


def test_insider_panel_rolling_decay():
    from ascent.features.feature_defs import build_insider_panel

    dates = _bdate(15)
    syms  = ["A"]

    df = pd.DataFrame([{"symbol": "A", "signal_date": dates[0], "score": 1}])
    result = build_insider_panel(df, dates, syms, window=5)
    panel = result["insider_net_score"]

    # Within window
    assert panel.loc[dates[4], "A"] == pytest.approx(1.0)
    # Beyond window: day 5 is the 6th day (0-indexed), outside window=5
    assert panel.loc[dates[5], "A"] == pytest.approx(0.0)


def test_insider_panel_empty_returns_empty():
    from ascent.features.feature_defs import build_insider_panel

    dates = _bdate(5)
    assert build_insider_panel(None, dates, _syms()) == {}
    assert build_insider_panel(pd.DataFrame(), dates, _syms()) == {}


def test_insider_panel_normalizes_non_midnight_signal_date():
    """A signal_date with a nonzero time-of-day must still align with a midnight price index."""
    from ascent.features.feature_defs import build_insider_panel

    dates = _bdate(20)
    syms = ["A"]

    df = pd.DataFrame([
        {"symbol": "A", "signal_date": dates[5] + pd.Timedelta(hours=9, minutes=15), "score": 1},
    ])
    result = build_insider_panel(df, dates, syms, window=63)
    panel = result["insider_net_score"]
    assert panel.loc[dates[5], "A"] == pytest.approx(1.0)


# ── build_short_panel ─────────────────────────────────────────────────────────

def test_short_panel_basic():
    from ascent.features.feature_defs import build_short_panel

    dates = _bdate(10)
    syms  = _syms(3)

    df = pd.DataFrame([
        {"symbol": "S0", "date": dates[2], "short_pct_float": 0.15, "short_ratio": 5.0},
        {"symbol": "S1", "date": dates[2], "short_pct_float": 0.05, "short_ratio": 1.5},
    ])
    result = build_short_panel(df, dates, syms)
    assert "short_pct_float" in result
    assert "short_ratio" in result
    assert result["short_pct_float"].loc[dates[2], "S0"] == pytest.approx(0.15)


def test_short_panel_forward_fills_15d():
    from ascent.features.feature_defs import build_short_panel

    dates = _bdate(25)
    syms  = ["A"]

    df = pd.DataFrame([{"symbol": "A", "date": dates[0], "short_pct_float": 0.20}])
    result = build_short_panel(df, dates, syms)
    panel = result["short_pct_float"]

    # limit=15: fills positions 1..15 (15 consecutive NaN values) after index 0
    assert panel.loc[dates[15], "A"] == pytest.approx(0.20)
    # Day 16+ is beyond limit=15
    assert pd.isna(panel.loc[dates[16], "A"])


def test_short_panel_empty_returns_empty():
    from ascent.features.feature_defs import build_short_panel

    dates = _bdate(5)
    assert build_short_panel(None, dates, _syms()) == {}
    assert build_short_panel(pd.DataFrame(), dates, _syms()) == {}


# ── options_flow_alpha ────────────────────────────────────────────────────────

def test_options_flow_alpha_z_scores():
    from ascent.alpha.options_flow import options_flow_alpha

    dates = _bdate(10)
    # B at 0.5 centers to 0 and gets dropped by the all-zero filter — use 0.55
    pcr = pd.DataFrame({"A": [0.70]*10, "B": [0.55]*10, "C": [0.30]*10}, index=dates)
    features = {"put_call_ratio": pcr}

    result = options_flow_alpha(features)
    assert not result.empty
    # A (bearish, high PC ratio) should have lower alpha than C (bullish, low PC ratio)
    assert result["A"].mean() < result["C"].mean()


def test_options_flow_alpha_missing_returns_empty():
    from ascent.alpha.options_flow import options_flow_alpha

    assert options_flow_alpha({}).empty
    assert options_flow_alpha({"other": pd.DataFrame()}).empty


def test_options_flow_alpha_iv_skew_inverted():
    """High IV skew (bearish) → negative alpha."""
    from ascent.alpha.options_flow import options_flow_alpha

    dates = _bdate(10)
    syms  = ["A", "B"]

    skew = pd.DataFrame({"A": [0.10]*10, "B": [-0.10]*10}, index=dates)
    features = {"iv_skew": skew}

    result = options_flow_alpha(features)
    # A has positive skew (bearish) → should have lower (more negative) alpha than B
    assert result["A"].mean() < result["B"].mean()


# ── insider_alpha ─────────────────────────────────────────────────────────────

def test_insider_alpha_z_scores():
    from ascent.alpha.insider import insider_alpha

    dates = _bdate(10)
    panel = pd.DataFrame({"A": [3.0]*10, "B": [0.0]*10, "C": [-2.0]*10}, index=dates)
    features = {"insider_net_score": panel}

    result = insider_alpha(features)
    assert not result.empty
    # A (net buys) > C (net sells)
    assert result["A"].mean() > result["C"].mean()


def test_insider_alpha_drops_all_zero():
    from ascent.alpha.insider import insider_alpha

    dates = _bdate(5)
    panel = pd.DataFrame({"A": [0.0]*5, "B": [0.0]*5}, index=dates)
    result = insider_alpha({"insider_net_score": panel})
    assert result.empty


def test_insider_alpha_missing_returns_empty():
    from ascent.alpha.insider import insider_alpha

    assert insider_alpha({}).empty


# ── short_interest_alpha ──────────────────────────────────────────────────────

def test_short_interest_alpha_z_scores():
    from ascent.alpha.short_interest import short_interest_alpha

    dates = _bdate(10)
    # A: high short float → high squeeze potential → positive alpha
    panel = pd.DataFrame({"A": [0.30]*10, "B": [0.10]*10, "C": [0.05]*10}, index=dates)
    features = {"short_pct_float": panel}

    result = short_interest_alpha(features)
    assert not result.empty
    assert result["A"].mean() > result["C"].mean()


def test_short_interest_alpha_missing_returns_empty():
    from ascent.alpha.short_interest import short_interest_alpha

    assert short_interest_alpha({}).empty


# ── ingest helpers ────────────────────────────────────────────────────────────

def test_transaction_score_mapping():
    from ascent.data.ingest.insider import _transaction_score

    assert _transaction_score("Purchase") == 1
    assert _transaction_score("open market purchase") == 1
    assert _transaction_score("Sale") == -1
    assert _transaction_score("open market sale") == -1
    assert _transaction_score("Option Exercise") == 0
    assert _transaction_score("") == 0


# ── stack integration ─────────────────────────────────────────────────────────

def test_default_weights_include_phase6():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS

    assert "options_flow"  in DEFAULT_ALPHA_WEIGHTS
    assert "insider"       in DEFAULT_ALPHA_WEIGHTS
    assert "short_interest" in DEFAULT_ALPHA_WEIGHTS
    assert DEFAULT_ALPHA_WEIGHTS["options_flow"]  >= 0.01
    assert DEFAULT_ALPHA_WEIGHTS["insider"]       >= 0.01
    assert DEFAULT_ALPHA_WEIGHTS["short_interest"] >= 0.01
    assert abs(sum(DEFAULT_ALPHA_WEIGHTS.values()) - 1.0) < 0.001


def test_self_improve_weights_match_stack():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS as stack_w
    from ascent.research.self_improve import DEFAULT_ALPHA_WEIGHTS as si_w

    assert set(stack_w.keys()) == set(si_w.keys()), (
        f"stack={set(stack_w.keys())} != self_improve={set(si_w.keys())}"
    )
    for k in stack_w:
        assert abs(stack_w[k] - si_w[k]) < 0.001, f"sleeve '{k}' mismatch"


def test_phase6_floors_in_self_improve():
    from ascent.research.self_improve import MIN_SLEEVE_WEIGHTS

    for sleeve in ("options_flow", "insider", "short_interest"):
        assert sleeve in MIN_SLEEVE_WEIGHTS
        assert MIN_SLEEVE_WEIGHTS[sleeve] >= 0.01


def test_phase6_floors_in_shadow_promoter():
    from ascent.research.shadow_promoter import _SLEEVE_FLOORS

    for sleeve in ("options_flow", "insider", "short_interest"):
        assert sleeve in _SLEEVE_FLOORS
        assert _SLEEVE_FLOORS[sleeve] >= 0.01
