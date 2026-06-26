"""
tests/test_ai_pm_counterfactual.py
Tests for ascent/monitoring/ai_pm_counterfactual.py
"""
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch


def test_snapshot_quant_star_is_idempotent():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_quant_star, QUANT_STAR_LOG
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_star.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_STAR_LOG", log_path):
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.5, "MSFT": 0.5})
            snapshot_quant_star(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.4})  # re-run same date
            lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1  # idempotent — second write skipped
    assert json.loads(lines[0])["weights"]["AAPL"] == 0.5  # first write preserved


def test_snapshot_ai_pm_normalizes_weights():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_ai.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 4), {"AAPL": 0.6, "MSFT": 0.6})  # sums to 1.2
            entry = json.loads(log_path.read_text().strip())
    total = sum(entry["weights"].values())
    assert abs(total - 1.0) < 0.001


def test_score_daily_appends_all_tracks():
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 5),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.012,
                spy_return=0.008,
                prices={
                    "AAPL": {"prev": 100.0, "curr": 101.5},
                    "MSFT": {"prev": 200.0, "curr": 201.0},
                },
            )
        entry = json.loads(log_path.read_text().strip())
    assert "track_astar_return" in entry
    assert "track_a_return" in entry
    assert "track_b_return" in entry
    assert "track_c_return" in entry
    assert "track_d_return" in entry
    assert abs(entry["track_b_return"] - 0.012) < 0.0001
    assert abs(entry["track_c_return"] - 0.008) < 0.0001


def test_score_daily_computes_track_d_correctly():
    """Track D = pure AI PM weighted returns. AAPL 60%, +1.5%; MSFT 40%, +0.5% → 1.1%"""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 5),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.012,
                spy_return=0.008,
                prices={
                    "AAPL": {"prev": 100.0, "curr": 101.5},  # +1.5%
                    "MSFT": {"prev": 200.0, "curr": 201.0},  # +0.5%
                },
            )
        entry = json.loads(log_path.read_text().strip())
    expected_d = 0.6 * 0.015 + 0.4 * 0.005  # 0.011
    assert abs(entry["track_d_return"] - expected_d) < 0.0001


def test_load_snapshots_returns_empty_on_missing_files():
    from ascent.monitoring.ai_pm_counterfactual import load_snapshots
    with tempfile.TemporaryDirectory() as tmp:
        star = Path(tmp) / "star.jsonl"
        quant = Path(tmp) / "quant.jsonl"
        ai = Path(tmp) / "ai.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_STAR_LOG", star):
            with patch("ascent.monitoring.ai_pm_counterfactual.QUANT_LOG", quant):
                with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", ai):
                    star_w, quant_w, ai_w = load_snapshots()
    assert star_w == {}
    assert quant_w == {}
    assert ai_w == {}


# ── Fix 3 regression tests: None inputs must not produce fabricated 0.0 values ──

def test_score_daily_returns_none_for_missing_ai_pm_weights():
    """score_daily with ai_pm_weights=None must set track_d_return to None, not 0.0."""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            record = score_daily(
                run_date=date(2026, 6, 10),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights=None,
                track_b_return=0.010,
                spy_return=0.005,
                prices={"AAPL": {"prev": 100.0, "curr": 101.0}, "MSFT": {"prev": 200.0, "curr": 201.0}},
            )
    assert record["track_d_return"] is None, (
        f"Expected None for missing ai_pm_weights, got {record['track_d_return']!r}"
    )
    # Other tracks with weights present must still compute normally
    assert record["track_astar_return"] is not None
    assert record["track_b_return"] == 0.010
    assert record["track_c_return"] == 0.005


def test_score_daily_returns_none_for_missing_quant_star_weights():
    """score_daily with quant_star_weights=None must set track_astar_return to None, not 0.0."""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            record = score_daily(
                run_date=date(2026, 6, 10),
                quant_star_weights=None,
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.010,
                spy_return=0.005,
                prices={"AAPL": {"prev": 100.0, "curr": 101.0}, "MSFT": {"prev": 200.0, "curr": 201.0}},
            )
    assert record["track_astar_return"] is None, (
        f"Expected None for missing quant_star_weights, got {record['track_astar_return']!r}"
    )
    # track_d should still compute normally
    assert record["track_d_return"] is not None


def test_score_daily_is_idempotent_per_date():
    """Re-running score_daily for the same date must REPLACE, not append.

    Regression for the June-10 rerun bug: the overnight rerun executed the
    pipeline ~9 times and each call appended another row, all of which
    get_cumulative_returns multiplied into the cumulative product."""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            for b in (-0.02, -0.01, 0.005):  # three reruns, final is 0.005
                score_daily(
                    run_date=date(2026, 6, 10),
                    quant_star_weights={"AAPL": 1.0},
                    quant_weights={"AAPL": 1.0},
                    ai_pm_weights={"AAPL": 1.0},
                    track_b_return=b,
                    spy_return=0.001,
                    prices={"AAPL": {"prev": 100.0, "curr": 100.0}},
                )
            lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1, f"Expected 1 row after 3 same-date reruns, got {len(lines)}"
    assert abs(json.loads(lines[0])["track_b_return"] - 0.005) < 1e-9  # last wins


def test_portfolio_return_skips_nan_prices():
    """yfinance period='5d' returns a trailing all-NaN row (today's unpublished bar);
    a single NaN price must not poison the whole track. If ALL prices are NaN the
    track is unpriced → None (skipped), not NaN/0.0 (frozen)."""
    from ascent.monitoring.ai_pm_counterfactual import _portfolio_return
    nan = float("nan")
    # one NaN, one good → return uses only the good leg
    r = _portfolio_return(
        {"AAPL": 0.5, "MSFT": 0.5},
        {"AAPL": {"prev": 100.0, "curr": 102.0}, "MSFT": {"prev": 200.0, "curr": nan}},
    )
    assert r is not None and abs(r - 0.5 * 0.02) < 1e-9
    # all NaN → None
    r2 = _portfolio_return(
        {"AAPL": 0.5, "MSFT": 0.5},
        {"AAPL": {"prev": 100.0, "curr": nan}, "MSFT": {"prev": 200.0, "curr": nan}},
    )
    assert r2 is None


def test_score_daily_none_when_weights_present_but_no_prices():
    """The freeze bug: snapshot weights exist but the price fetch failed, so prices={}.
    Track returns must be None (skipped), NOT a fabricated 0.0 that freezes the track."""
    from ascent.monitoring.ai_pm_counterfactual import score_daily
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            rec = score_daily(
                run_date=date(2026, 6, 11),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.01,
                spy_return=0.005,
                prices={},  # fetch failed
            )
    assert rec["track_astar_return"] is None
    assert rec["track_d_return"] is None
    assert rec["track_b_return"] == 0.01  # Track B from Alpaca still real


def test_cumulative_skips_none_and_compares_common_window():
    """get_cumulative_returns must (a) not crash on None days, (b) cumulate each
    track only over its own non-None days, and (c) difference B/D vs A* only over
    days where BOTH have data."""
    from ascent.monitoring import ai_pm_counterfactual as cf
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        recs = [
            # A* has data both days; D only day 2; B only day 1
            {"date": "2026-06-10", "track_astar_return": 0.02, "track_a_return": None,
             "track_b_return": 0.01, "track_c_return": 0.0, "track_d_return": None},
            {"date": "2026-06-11", "track_astar_return": 0.03, "track_a_return": None,
             "track_b_return": None, "track_c_return": 0.0, "track_d_return": 0.01},
        ]
        log_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            c = cf.get_cumulative_returns()
    # standalone cumret over each track's own non-None days
    assert abs(c["track_astar"] - round(((1.02 * 1.03) - 1) * 100, 3)) < 1e-6
    assert abs(c["track_d"] - 1.0) < 1e-6  # only day 2: +1%
    # common-window diffs must only use overlapping days
    # D vs A*: only 2026-06-11 overlaps → 0.01 - 0.03 = -2.0pp
    assert abs(c["ai_signal_d_vs_astar"] - (-2.0)) < 1e-6
    # B vs A*: only 2026-06-10 overlaps → 0.01 - 0.02 = -1.0pp
    assert abs(c["ai_value_add_b_vs_astar"] - (-1.0)) < 1e-6


def test_update_authority_none_inputs_skip_buffer_append():
    """update_authority(None, None) must not modify track_d_returns or track_astar_returns buffers."""
    from ascent.strategy.earned_authority import update_authority, _save_state, get_state
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "earned_authority.json"
        # Write a state with known non-empty buffers
        initial_state = {
            "level": 1,
            "title": "Analyst",
            "ai_weight": 0.05,
            "phase": 1,
            "level_start_date": "2026-06-04",
            "phase_start_date": "2026-06-04",
            "days_at_level": 3,
            "days_stuck": 3,
            "in_cooldown": False,
            "cooldown_until": None,
            "auto_revert_count": 0,
            "last_updated": "2026-06-09",  # yesterday, so today's call is allowed
            "track_d_returns": [0.001, 0.002, 0.003],
            "track_astar_returns": [0.002, 0.001, 0.002],
            "ai_returns_21d": [0.001, 0.002, 0.003],
            "quant_returns_21d": [0.002, 0.001, 0.002],
            "disable_sleeve_priors": False,
        }
        state_path.write_text(json.dumps(initial_state))
        with patch("ascent.strategy.earned_authority.STATE_PATH", state_path):
            with patch("ascent.strategy.earned_authority.SHADOW_RETURNS_PATH", Path(tmp) / "shadow.jsonl"):
                result = update_authority(
                    track_d_return=None,
                    track_astar_return=None,
                )
        # Buffers must be unchanged
        assert result["track_d_returns"] == [0.001, 0.002, 0.003], (
            f"Buffer was mutated with None input: {result['track_d_returns']}"
        )
        assert result["track_astar_returns"] == [0.002, 0.001, 0.002], (
            f"Buffer was mutated with None input: {result['track_astar_returns']}"
        )


def test_backfill_track_b_overwrites_provisional_and_fills_none():
    """Track B same-day value is unreliable (equity==last_equity → fake 0.0, or None);
    backfill must overwrite it with Alpaca's settled daily return for every date the
    published bar covers, and leave dates Alpaca doesn't have (e.g. today) untouched."""
    from ascent.monitoring.ai_pm_counterfactual import backfill_track_b
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        rows = [
            {"date": "2026-06-17", "track_astar_return": 0.01, "track_b_return": 0.0},    # provisional 0.0
            {"date": "2026-06-18", "track_astar_return": 0.02, "track_b_return": None},   # missing
            {"date": "2026-06-19", "track_astar_return": 0.03, "track_b_return": 0.0},    # today, not yet settled
        ]
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # Alpaca published settled bars through yesterday only.
        history = {"2026-06-17": 0.012, "2026-06-18": -0.004}
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            changed = backfill_track_b(history)
            out = [json.loads(l) for l in log_path.read_text().splitlines()]
        by_date = {r["date"]: r["track_b_return"] for r in out}
        assert changed == 2
        assert by_date["2026-06-17"] == 0.012   # provisional 0.0 overwritten with settled
        assert by_date["2026-06-18"] == -0.004  # None filled
        assert by_date["2026-06-19"] == 0.0     # today untouched (not in history)
        # Other fields preserved
        assert {r["date"]: r["track_astar_return"] for r in out}["2026-06-18"] == 0.02


def test_backfill_track_b_noops_on_empty_history():
    from ascent.monitoring.ai_pm_counterfactual import backfill_track_b
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cf_daily.jsonl"
        log_path.write_text(json.dumps({"date": "2026-06-17", "track_b_return": 0.0}) + "\n")
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            assert backfill_track_b({}) == 0


# ── backfill_astar_d (heal historical A★/A/D nulls) ──────────────────────────

def test_backfill_astar_d_fills_null_rows_from_snapshots():
    """A★/A/D rows that recorded None (pre-fix freeze / failed fetch) must be
    healed by recomputing from the as-of snapshot weights + historical closes —
    the missing analog of backfill_track_b. Idempotent: only fills None."""
    import ascent.monitoring.ai_pm_counterfactual as cf
    with tempfile.TemporaryDirectory() as tmp:
        daily = Path(tmp) / "daily.jsonl"
        star  = Path(tmp) / "star.jsonl"
        aipm  = Path(tmp) / "ai.jsonl"
        quant = Path(tmp) / "quant.jsonl"
        # Snapshot active from 06-15 onward
        star.write_text(json.dumps({"date": "2026-06-15", "weights": {"AAA": 0.5, "BBB": 0.5}}) + "\n")
        aipm.write_text(json.dumps({"date": "2026-06-15", "weights": {"AAA": 1.0}}) + "\n")
        quant.write_text(json.dumps({"date": "2026-06-15", "weights": {"AAA": 0.5, "BBB": 0.5}}) + "\n")
        # One null row to heal, one already-good row to leave alone
        daily.write_text(
            json.dumps({"date": "2026-06-16", "track_astar_return": None, "track_a_return": None,
                        "track_b_return": 0.01, "track_c_return": 0.0, "track_d_return": None}) + "\n" +
            json.dumps({"date": "2026-06-17", "track_astar_return": 0.009, "track_a_return": 0.009,
                        "track_b_return": 0.0, "track_c_return": 0.0, "track_d_return": 0.02}) + "\n"
        )
        # Historical closes: AAA +2% on 06-16, BBB -1% on 06-16
        closes = {
            "AAA": {"2026-06-15": 100.0, "2026-06-16": 102.0, "2026-06-17": 102.0},
            "BBB": {"2026-06-15": 50.0,  "2026-06-16": 49.5,  "2026-06-17": 49.5},
        }
        with patch.object(cf, "DAILY_LOG", daily), patch.object(cf, "QUANT_STAR_LOG", star), \
             patch.object(cf, "AI_PM_LOG", aipm), patch.object(cf, "QUANT_LOG", quant):
            changed = cf.backfill_astar_d(closes)
            out = {r["date"]: r for r in (json.loads(l) for l in daily.read_text().splitlines())}
        # 06-16 A★ = 0.5*0.02 + 0.5*(-0.01) = 0.005 ; D = 1.0*0.02 = 0.02
        assert changed >= 1
        assert abs(out["2026-06-16"]["track_astar_return"] - 0.005) < 1e-6
        assert abs(out["2026-06-16"]["track_d_return"] - 0.02) < 1e-6
        # already-good row untouched
        assert out["2026-06-17"]["track_astar_return"] == 0.009


def test_backfill_astar_d_idempotent_and_skips_unpriceable():
    import ascent.monitoring.ai_pm_counterfactual as cf
    with tempfile.TemporaryDirectory() as tmp:
        daily = Path(tmp) / "daily.jsonl"
        star  = Path(tmp) / "star.jsonl"
        star.write_text(json.dumps({"date": "2026-06-15", "weights": {"AAA": 1.0}}) + "\n")
        daily.write_text(json.dumps({"date": "2026-06-16", "track_astar_return": None,
                                     "track_b_return": 0.0, "track_c_return": 0.0}) + "\n")
        # No close data for AAA on 06-16 → cannot price → stays None, no crash
        with patch.object(cf, "DAILY_LOG", daily), patch.object(cf, "QUANT_STAR_LOG", star), \
             patch.object(cf, "AI_PM_LOG", Path(tmp)/"x.jsonl"), patch.object(cf, "QUANT_LOG", Path(tmp)/"y.jsonl"):
            assert cf.backfill_astar_d({"AAA": {"2026-06-15": 100.0}}) == 0  # no prior+curr pair
            assert cf.backfill_astar_d({}) == 0


# ── Task 1: force_sealed tagging ──────────────────────────────────────────────

def test_snapshot_ai_pm_writes_force_sealed_false_by_default():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm, AI_PM_LOG
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_snapshots.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 25), {"AAPL": 0.6, "MSFT": 0.4})
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("force_sealed") == False


def test_snapshot_ai_pm_writes_force_sealed_true():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm, AI_PM_LOG
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_snapshots.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 25), {"AAPL": 0.6, "MSFT": 0.4}, force_sealed=True)
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("force_sealed") == True


def test_aipm_result_has_force_sealed_field():
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={}, thesis={})
    assert r.force_sealed == False
    r2 = AIPMResult(portfolio={}, thesis={}, force_sealed=True)
    assert r2.force_sealed == True


def test_run_ai_pm_fallback_result_is_not_force_sealed():
    """Fallback result (portfolio={}) should not be marked force_sealed — it means no snapshot was taken."""
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={}, thesis={}, fallback=True)
    assert r.force_sealed == False


def test_aipm_result_force_sealed_defaults_false():
    """Normal result should default to not force_sealed."""
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={"AAPL": 0.5}, thesis={"market_view": "bullish"})
    assert r.force_sealed == False
