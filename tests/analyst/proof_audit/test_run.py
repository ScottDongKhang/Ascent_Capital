"""run() must make every non-clean row self-describing, and must never publish two agent
rows that are secretly the same measurement.

The three specialist agents are currently all fed the same US-equity price matrix, so two of
them can silently produce byte-identical scores. A scorecard that reports that as two
independent verdicts is worse than one that admits it cannot tell them apart.
"""
import json

import pandas as pd
import pytest

from ascent.analyst.proof_audit import run as run_module
from ascent.analyst.proof_audit.stats import ICResult
from ascent.analyst.proof_audit.wf_scorer import DegenerateSignalError


def _by_name(rows):
    return {r.component: r for r in rows}


def _install_stub_scorers(monkeypatch, agent_results, sleeve_result=None, subsystem_result=None):
    """Replace the three scoring entrypoints with deterministic stubs."""
    sleeve_result = sleeve_result or ICResult(
        ic_mean=0.02, ic_t=3.0, p_value=0.004, sharpe=0.9, n=400
    )
    subsystem_result = subsystem_result or ICResult(
        ic_mean=-0.0007, ic_t=-0.9, p_value=0.35, sharpe=-0.3, n=47
    )

    def fake_score_sleeve(name, features, prices, dates=None):
        return sleeve_result

    def fake_score_agent(name, prices, dates=None):
        result = agent_results[name]
        if isinstance(result, Exception):
            raise result
        return result

    def fake_score_subsystem(name):
        return subsystem_result

    monkeypatch.setattr(run_module, "score_sleeve", fake_score_sleeve)
    monkeypatch.setattr(run_module, "score_agent", fake_score_agent)
    monkeypatch.setattr(run_module, "score_subsystem", fake_score_subsystem)


def _run(monkeypatch, tmp_path, agent_results):
    _install_stub_scorers(monkeypatch, agent_results)
    return run_module.run({}, pd.DataFrame(), out_path=tmp_path / "scorecard.json")


def test_identical_agent_scores_are_downgraded(monkeypatch, tmp_path):
    same = ICResult(ic_mean=-0.011, ic_t=-2.1, p_value=0.038, sharpe=-0.4, n=1634)
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": same,
            "international_agent": same,
            "alternatives_agent": ICResult(
                ic_mean=0.004, ic_t=2.2, p_value=0.03, sharpe=0.5, n=1600
            ),
        },
    )
    by_name = _by_name(rows)
    for name in ("macro_agent", "international_agent"):
        assert by_name[name].verdict == "INSUFFICIENT_DATA"
        assert "identical" in by_name[name].reason
        assert "universe" in by_name[name].reason
    # macro's reason must name the agent it collided with, and vice versa.
    assert "international_agent" in by_name["macro_agent"].reason
    assert "macro_agent" in by_name["international_agent"].reason
    # The genuinely distinct agent is untouched.
    assert by_name["alternatives_agent"].verdict == "KEEP"
    assert by_name["alternatives_agent"].reason is None


def test_distinct_agent_scores_are_not_downgraded(monkeypatch, tmp_path):
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    by_name = _by_name(rows)
    assert by_name["macro_agent"].verdict == "KEEP"
    assert by_name["international_agent"].verdict == "KEEP"
    assert by_name["alternatives_agent"].verdict == "CUT"
    for name in ("macro_agent", "international_agent", "alternatives_agent"):
        assert by_name[name].reason is None


def test_sleeve_rows_are_never_duplicate_checked(monkeypatch, tmp_path):
    """Every wf_ic sleeve gets the same stub result here; sleeves may legitimately
    correlate, so identical sleeve scores must NOT be downgraded."""
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    sleeve_rows = [r for r in rows if r.kind == "alpha_sleeve" and r.method == "wf_ic"]
    assert len(sleeve_rows) > 1
    assert all(r.verdict == "KEEP" for r in sleeve_rows)


def test_degenerate_signal_gets_a_density_reason(monkeypatch, tmp_path):
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": DegenerateSignalError(
                "signal matrix has insufficient non-NaN density (0 of 1636 candidate dates "
                "carry at least 10 non-NaN values; need 30)"
            ),
        },
    )
    row = _by_name(rows)["alternatives_agent"]
    assert row.verdict == "INSUFFICIENT_DATA"
    assert "density" in row.reason
    assert "wrong universe" in row.reason


def test_deferred_and_covered_rows_carry_their_documented_reason(monkeypatch, tmp_path):
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    by_name = _by_name(rows)
    for name in ("ml", "llm_fundamental", "narrative"):
        assert by_name[name].reason == run_module._DEFERRED_REASON["deferred"]
    assert by_name["us_equities_agent"].reason == run_module._DEFERRED_REASON["covered_by_sleeves"]


def test_shared_track_pair_subsystems_disclose_the_approximation(monkeypatch, tmp_path):
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    by_name = _by_name(rows)
    for name in ("regime_overlay", "hedge_overlay"):
        assert "earned_authority" in by_name[name].reason
        assert "approximat" in by_name[name].reason
    assert "canonical" in by_name["earned_authority"].reason
    assert "total-return" in by_name["debate_judge_intervention"].reason
    assert "split-only" in by_name["debate_judge_intervention"].reason


def test_written_json_carries_every_reason(monkeypatch, tmp_path):
    out = tmp_path / "scorecard.json"
    _install_stub_scorers(
        monkeypatch,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    run_module.run({}, pd.DataFrame(), out_path=out)
    loaded = json.loads(out.read_text())
    assert all("reason" in row for row in loaded)
    insufficient = [r for r in loaded if r["verdict"] == "INSUFFICIENT_DATA"]
    assert insufficient
    assert all(r["reason"] for r in insufficient)


def test_unknown_kind_is_not_silently_dropped(monkeypatch, tmp_path):
    """Every pinned component must produce exactly one row."""
    rows = _run(
        monkeypatch,
        tmp_path,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    from ascent.analyst.proof_audit.components import COMPONENTS

    assert [r.component for r in rows] == [c.name for c in COMPONENTS]


def test_missing_input_sleeve_failure_names_the_missing_inputs(monkeypatch, tmp_path):
    # "altdata" (not "fundamental") is the live example here: the real-data CLI now loads
    # fundamentals/earnings/analyst/options/insider/short frames (see run.py's __main__), so
    # "fundamental" scores for real and would fail this assertion. "altdata" has no parquet
    # cache wired up at all, so it is still a genuine missing-input gap.
    def failing_sleeve(name, features, prices, dates=None):
        if name == "altdata":
            raise KeyError("pe_ratio")
        return ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.004, sharpe=0.9, n=400)

    _install_stub_scorers(
        monkeypatch,
        {
            "macro_agent": ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634),
            "international_agent": ICResult(
                ic_mean=0.011, ic_t=2.4, p_value=0.017, sharpe=0.6, n=1630
            ),
            "alternatives_agent": ICResult(
                ic_mean=-0.03, ic_t=-4.0, p_value=0.0001, sharpe=-1.1, n=1600
            ),
        },
    )
    monkeypatch.setattr(run_module, "score_sleeve", failing_sleeve)
    rows = run_module.run({}, pd.DataFrame(), out_path=tmp_path / "scorecard.json")
    row = _by_name(rows)["altdata"]
    assert row.verdict == "INSUFFICIENT_DATA"
    assert "not loaded by this CLI" in row.reason


def test_dedupe_prices_by_calendar_day_collapses_intraday_timestamps():
    """The helper lives in run.py; scripts/run_proof_audit.py must not carry a second copy."""
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2022-12-21 00:00:00",
                    "2022-12-21 19:00:00",
                    "2022-12-21 20:00:00",
                    "2022-12-22 00:00:00",
                ]
            ),
            "symbol": ["AAA", "AAA", "BBB", "AAA"],
            "close": [10.0, 11.0, 5.0, 12.0],
        }
    )
    out = run_module._dedupe_prices_by_calendar_day(price_df)
    assert len(out) == 3
    aaa_21 = out[(out["symbol"] == "AAA") & (out["date"] == pd.Timestamp("2022-12-21"))]
    assert len(aaa_21) == 1
    assert aaa_21["close"].iloc[0] == pytest.approx(11.0)  # keep="last" by timestamp
