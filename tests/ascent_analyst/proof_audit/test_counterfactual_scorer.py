"""Path B: subsystems are scored by with-vs-without return delta on the counterfactual tracks."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.counterfactual_scorer import (
    SUBSYSTEM_TRACK_PAIRS,
    score_subsystem,
)


def test_all_named_subsystems_have_track_pairs():
    from ascent.analyst.proof_audit.components import COMPONENTS

    counterfactual_subsystems = {
        c.name for c in COMPONENTS if c.kind == "subsystem" and c.method == "counterfactual"
    }
    assert counterfactual_subsystems == set(SUBSYSTEM_TRACK_PAIRS)


def test_score_subsystem_detects_planted_positive_delta(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    # Delta oscillates around a positive mean (not constant) so ttest_1samp sees real
    # variance -- a perfectly constant delta series triggers a scipy divide-by-zero
    # RuntimeWarning on a degenerate std of 0.
    deltas = [0.003 if i % 2 == 0 else 0.001 for i in range(30)]
    without_track = pd.Series([0.001] * 30, index=dates)
    with_track = pd.Series([0.001 + d for d in deltas], index=dates)

    def fake_load(name):
        return with_track if name == "counterfactual.track_d" else without_track

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.counterfactual_scorer.registry.load", fake_load
    )
    result = score_subsystem("earned_authority")
    assert result.ic_mean > 0
    assert result.n == 30


def test_score_subsystem_unknown_name_raises():
    with pytest.raises(KeyError):
        score_subsystem("not_a_subsystem")
