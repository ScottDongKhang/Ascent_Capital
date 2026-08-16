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


def test_sleeve_scoring_exception_gets_a_generic_failure_reason(monkeypatch, tmp_path):
    # There is no more special-cased "missing input" reason for any sleeve (including
    # "altdata"/"earnings_tone") -- see run.py's module docstring comment above
    # _DEGENERATE_SUFFIX. altdata's own self-loading logic returning empty on missing
    # sources is exercised for real by the density guard (see
    # test_degenerate_signal_gets_a_density_reason); this test just pins that a plain
    # (non-DegenerateSignalError) exception from ANY sleeve gets the generic
    # "scoring failed: ..." reason, naming the real exception, not an invented one.
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
    assert "scoring failed" in row.reason
    assert "pe_ratio" in row.reason


def test_run_without_agent_prices_falls_back_to_shared_prices(monkeypatch, tmp_path):
    """run() must remain callable exactly as before -- `agent_prices` is additive, not
    required. Every pre-existing call in this file omits it; this pins that fallback."""
    from ascent.analyst.proof_audit.components import COMPONENTS

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
    assert [r.component for r in rows] == [c.name for c in COMPONENTS]
    by_name = _by_name(rows)
    assert by_name["macro_agent"].verdict == "KEEP"
    assert by_name["international_agent"].verdict == "KEEP"
    assert by_name["alternatives_agent"].verdict == "CUT"


def test_agent_prices_routes_each_agent_to_its_own_matrix(monkeypatch, tmp_path):
    """When `agent_prices` supplies a per-agent matrix, `score_agent` must be called with
    THAT agent's own matrix, not the shared `prices` -- this is the actual fix: prior to it,
    every agent was scored against the same 938-symbol US-equity matrix regardless of what
    was passed in `agent_prices`."""
    shared_prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]})
    macro_prices = pd.DataFrame({"TLT": [10.0, 11.0, 12.0]})
    intl_prices = pd.DataFrame({"EEM": [20.0, 21.0, 22.0]})
    seen: dict[str, pd.DataFrame] = {}

    def fake_score_sleeve(name, features, prices, dates=None):
        return ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.004, sharpe=0.9, n=400)

    def fake_score_agent(name, prices, dates=None):
        seen[name] = prices
        return ICResult(ic_mean=0.02, ic_t=3.0, p_value=0.003, sharpe=0.8, n=1634)

    def fake_score_subsystem(name):
        return ICResult(ic_mean=-0.0007, ic_t=-0.9, p_value=0.35, sharpe=-0.3, n=47)

    monkeypatch.setattr(run_module, "score_sleeve", fake_score_sleeve)
    monkeypatch.setattr(run_module, "score_agent", fake_score_agent)
    monkeypatch.setattr(run_module, "score_subsystem", fake_score_subsystem)

    run_module.run(
        {},
        shared_prices,
        out_path=tmp_path / "scorecard.json",
        agent_prices={"macro_agent": macro_prices, "international_agent": intl_prices},
    )

    assert seen["macro_agent"] is macro_prices
    assert seen["international_agent"] is intl_prices
    # alternatives_agent has no entry in agent_prices -> falls back to the shared matrix.
    assert seen["alternatives_agent"] is shared_prices


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


def test_dedupe_wide_prices_by_calendar_day_collapses_intraday_timestamps():
    """Wide-format analogue: macro/international/alternatives caches are pivoted (symbol
    columns) with the date as the DataFrame INDEX, not a `date` column, so they need their
    own dedupe helper rather than reusing `_dedupe_prices_by_calendar_day`."""
    idx = pd.to_datetime(
        ["2022-12-21 00:00:00", "2022-12-21 19:00:00", "2022-12-22 00:00:00"]
    )
    price_df = pd.DataFrame({"TLT": [100.0, 101.0, 102.0]}, index=idx)
    out = run_module._dedupe_wide_prices_by_calendar_day(price_df)
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2022-12-21"), "TLT"] == pytest.approx(101.0)  # keep="last"
    assert out.loc[pd.Timestamp("2022-12-22"), "TLT"] == pytest.approx(102.0)


def test_load_agent_price_matrix_skips_a_cache_with_no_date_index_or_column(monkeypatch, tmp_path):
    """A cache written before ecaccf9's save_parquet fix (or otherwise still corrupted) can
    load back with neither a DatetimeIndex nor a `date` column -- a bare RangeIndex over raw
    values. `_load_agent_price_matrix` must detect that and return None (fall back to shared
    prices) instead of scoring against a matrix with no real dates."""
    dateless = pd.DataFrame({"TLT": [100.0, 101.0, 102.0]})  # RangeIndex, no dates, no date col
    monkeypatch.setattr(run_module, "load_parquet", lambda name: dateless)
    assert run_module._load_agent_price_matrix("macro_agent", "prices_macro") is None


def test_load_agent_price_matrix_dedupes_a_cache_with_a_real_date_index(monkeypatch):
    idx = pd.to_datetime(
        ["2022-12-21 00:00:00", "2022-12-21 19:00:00", "2022-12-22 00:00:00"]
    )
    dated = pd.DataFrame({"TLT": [100.0, 101.0, 102.0]}, index=idx)
    monkeypatch.setattr(run_module, "load_parquet", lambda name: dated)
    out = run_module._load_agent_price_matrix("macro_agent", "prices_macro")
    assert out is not None
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2022-12-21"), "TLT"] == pytest.approx(101.0)


def test_load_agent_price_matrix_restores_index_from_a_date_column(monkeypatch):
    """As of ecaccf9, save_parquet persists wide-format agent caches with the date as a
    `date` column (RangeIndex on load), not the DataFrame index. `_load_agent_price_matrix`
    must restore the DatetimeIndex from that column -- this is the fixed-upstream good case,
    as opposed to the still-corrupted case above (neither index nor column)."""
    dates = pd.to_datetime(
        ["2022-12-21 00:00:00", "2022-12-21 19:00:00", "2022-12-22 00:00:00"]
    )
    with_date_column = pd.DataFrame({"date": dates, "TLT": [100.0, 101.0, 102.0]})  # RangeIndex
    monkeypatch.setattr(run_module, "load_parquet", lambda name: with_date_column)
    out = run_module._load_agent_price_matrix("macro_agent", "prices_macro")
    assert out is not None
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2022-12-21"), "TLT"] == pytest.approx(101.0)
    assert out.loc[pd.Timestamp("2022-12-22"), "TLT"] == pytest.approx(102.0)


def test_agent_fallback_reason_is_disclosed_on_the_row(monkeypatch, tmp_path):
    """When a caller (the real-data CLI) knows WHY an agent fell back to the shared prices
    matrix -- e.g. its own cache has no usable date index -- that reason must land on the
    row, not just a stderr log line. Distinct scores here (no duplicate-check collision)."""
    # _run() doesn't thread agent_fallback_reasons; call run() directly instead.
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
    rows = run_module.run(
        {}, pd.DataFrame(), out_path=tmp_path / "scorecard.json",
        agent_fallback_reasons={"macro_agent": "own price cache prices_macro has no usable date index"},
    )
    by_name = _by_name(rows)
    assert "prices_macro" in by_name["macro_agent"].reason
    assert "no usable date index" in by_name["macro_agent"].reason
    # Agents with no fallback reason supplied are unaffected.
    assert by_name["international_agent"].reason is None
    assert by_name["alternatives_agent"].reason is None


def test_agent_fallback_reason_survives_duplicate_downgrade(monkeypatch, tmp_path):
    """If a fallback-scored agent ALSO collides with another agent's score, the row must
    keep both the fallback explanation and the duplicate-check explanation -- a reader must
    be able to tell the row is a fallback, not just that it looks suspicious."""
    same = ICResult(ic_mean=-0.011, ic_t=-2.1, p_value=0.038, sharpe=-0.4, n=1634)
    _install_stub_scorers(
        monkeypatch,
        {
            "macro_agent": same,
            "international_agent": same,
            "alternatives_agent": ICResult(
                ic_mean=0.004, ic_t=2.2, p_value=0.03, sharpe=0.5, n=1600
            ),
        },
    )
    rows = run_module.run(
        {}, pd.DataFrame(), out_path=tmp_path / "scorecard.json",
        agent_fallback_reasons={"macro_agent": "own price cache prices_macro has no usable date index"},
    )
    by_name = _by_name(rows)
    assert by_name["macro_agent"].verdict == "INSUFFICIENT_DATA"
    assert "no usable date index" in by_name["macro_agent"].reason
    assert "identical" in by_name["macro_agent"].reason
    # international_agent has no fallback reason of its own -- just the duplicate message.
    assert by_name["international_agent"].reason is not None
    assert "no usable date index" not in by_name["international_agent"].reason
    assert "identical" in by_name["international_agent"].reason
