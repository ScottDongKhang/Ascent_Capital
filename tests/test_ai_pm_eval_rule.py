# tests/test_ai_pm_eval_rule.py
"""
Tests for ascent/monitoring/ai_pm_eval_rule.py — the PRE-REGISTERED AI PM
evaluation rule.

The rule's whole purpose is to stop us acting on noise. Two invariants matter
most and are tested hardest:
  1. Below the minimum independent-decision count, the verdict is ALWAYS HOLD,
     no matter how good or bad the mean looks. (n=3 is a coin flip.)
  2. The unit of evidence is an INDEPENDENT DECISION, never a daily mark of the
     same book. 23 autocorrelated daily diffs are not 23 observations.
"""
from __future__ import annotations

import pytest

from ascent.monitoring.ai_pm_eval_rule import (
    MIN_DECISIONS,
    T_THRESHOLD,
    evaluate_rule,
)


def test_below_min_decisions_always_holds_even_if_strongly_positive():
    # A small handful of decisions, all strongly favouring the AI book.
    diffs = [0.05, 0.04, 0.06]  # n=3, mean huge and positive
    out = evaluate_rule(diffs)
    assert out["verdict"] == "HOLD"
    assert out["n"] == 3
    assert "insufficient" in out["reason"].lower()


def test_below_min_decisions_always_holds_even_if_strongly_negative():
    diffs = [-0.05, -0.04, -0.06]  # the current real situation: looks bad, n tiny
    out = evaluate_rule(diffs)
    assert out["verdict"] == "HOLD"
    assert "insufficient" in out["reason"].lower()


def test_promote_when_enough_decisions_and_significant_positive():
    # Enough decisions, consistently positive, low variance → significant.
    diffs = [0.01] * MIN_DECISIONS
    out = evaluate_rule(diffs)
    assert out["n"] == MIN_DECISIONS
    assert out["t_stat"] > T_THRESHOLD
    assert out["verdict"] == "PROMOTE"


def test_demote_when_enough_decisions_and_significant_negative():
    diffs = [-0.01] * MIN_DECISIONS
    out = evaluate_rule(diffs)
    assert out["t_stat"] < -T_THRESHOLD
    assert out["verdict"] == "DEMOTE"


def test_hold_when_enough_decisions_but_not_significant():
    # Enough decisions but noisy / mean near zero → cannot conclude.
    diffs = ([0.02, -0.02] * (MIN_DECISIONS // 2))
    out = evaluate_rule(diffs)
    assert out["n"] >= MIN_DECISIONS
    assert abs(out["t_stat"]) < T_THRESHOLD
    assert out["verdict"] == "HOLD"


def test_empty_input_holds():
    out = evaluate_rule([])
    assert out["verdict"] == "HOLD"
    assert out["n"] == 0


def test_verdict_payload_shape():
    out = evaluate_rule([0.01] * MIN_DECISIONS)
    for key in ("verdict", "n", "mean_diff", "t_stat", "reason", "min_decisions", "t_threshold"):
        assert key in out, f"missing key {key}"
