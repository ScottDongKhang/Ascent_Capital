"""Verdict rule is three-way and never silently defaults."""
import json

from ascent.analyst.proof_audit.scorecard import ScorecardRow, verdict, write_scorecard
from ascent.analyst.proof_audit.stats import ICResult


def test_significant_positive_is_keep():
    result = ICResult(ic_mean=0.03, ic_t=3.5, p_value=0.001, sharpe=1.2, n=50)
    assert verdict(result) == "KEEP"


def test_significant_negative_is_cut():
    result = ICResult(ic_mean=-0.02, ic_t=-3.1, p_value=0.002, sharpe=-0.8, n=50)
    assert verdict(result) == "CUT"


def test_not_significant_is_cut():
    result = ICResult(ic_mean=0.001, ic_t=0.4, p_value=0.7, sharpe=0.05, n=50)
    assert verdict(result) == "CUT"


def test_below_min_sample_is_insufficient_data():
    result = ICResult(ic_mean=0.05, ic_t=2.0, p_value=0.01, sharpe=1.0, n=5)
    assert verdict(result, min_sample=30) == "INSUFFICIENT_DATA"


def test_write_scorecard_round_trips(tmp_path):
    rows = [
        ScorecardRow(
            component="trend", kind="alpha_sleeve", method="wf_ic",
            metric=0.03, p_value=0.001, sample_size=50, verdict="KEEP",
        ),
        ScorecardRow(
            component="ml", kind="alpha_sleeve", method="deferred",
            metric=None, p_value=None, sample_size=0, verdict="INSUFFICIENT_DATA",
        ),
    ]
    out = tmp_path / "scorecard.json"
    write_scorecard(rows, out)
    loaded = json.loads(out.read_text())
    assert len(loaded) == 2
    assert loaded[0]["component"] == "trend"
    assert loaded[0]["verdict"] == "KEEP"
    assert loaded[1]["metric"] is None
