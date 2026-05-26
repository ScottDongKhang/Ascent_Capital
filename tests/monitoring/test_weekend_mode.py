"""
tests/monitoring/test_weekend_mode.py

Tests for weekend mode: gate logic, weekly debrief, scenario planner.
"""

import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Weekend gate ──────────────────────────────────────────────────────────────

class TestWeekendGate:
    def test_already_ran_returns_none_when_no_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH",
            tmp_path / "weekend_run_state.json"
        )
        from ascent.monitoring.weekend_runner import already_ran_this_weekend
        assert already_ran_this_weekend() is None

    def test_already_ran_returns_date_when_same_week(self, tmp_path, monkeypatch):
        state_path = tmp_path / "weekend_run_state.json"
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH", state_path
        )
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        state_path.write_text(json.dumps({
            "run_date": today.isoformat(),
            "iso_week": iso_week,
            "iso_year": iso_year,
            "completed_jobs": ["altdata_full_sweep"],
        }))
        from ascent.monitoring.weekend_runner import already_ran_this_weekend
        result = already_ran_this_weekend()
        assert result == today.isoformat()

    def test_already_ran_returns_none_for_different_week(self, tmp_path, monkeypatch):
        state_path = tmp_path / "weekend_run_state.json"
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH", state_path
        )
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        state_path.write_text(json.dumps({
            "run_date": "2026-01-01",
            "iso_week": 1,
            "iso_year": 2026,
            "completed_jobs": [],
        }))
        from ascent.monitoring.weekend_runner import already_ran_this_weekend
        assert already_ran_this_weekend() is None

    def test_mark_ran_writes_state(self, tmp_path, monkeypatch):
        state_path = tmp_path / "weekend_run_state.json"
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH", state_path
        )
        from ascent.monitoring.weekend_runner import _mark_ran
        _mark_ran(["job_a", "job_b"])
        state = json.loads(state_path.read_text())
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        assert state["iso_week"] == iso_week
        assert state["iso_year"] == iso_year
        assert "job_a" in state["completed_jobs"]
        assert "job_b" in state["completed_jobs"]

    def test_mark_ran_accumulates_across_runs(self, tmp_path, monkeypatch):
        state_path = tmp_path / "weekend_run_state.json"
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH", state_path
        )
        from ascent.monitoring.weekend_runner import _mark_ran
        _mark_ran(["job_a"])
        _mark_ran(["job_b"])
        state = json.loads(state_path.read_text())
        assert "job_a" in state["completed_jobs"]
        assert "job_b" in state["completed_jobs"]

    def test_job_ran_this_weekend_true_when_in_state(self, tmp_path, monkeypatch):
        state_path = tmp_path / "weekend_run_state.json"
        monkeypatch.setattr(
            "ascent.monitoring.weekend_runner.WEEKEND_STATE_PATH", state_path
        )
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        state_path.write_text(json.dumps({
            "run_date": today.isoformat(),
            "iso_week": iso_week,
            "iso_year": iso_year,
            "completed_jobs": ["ml_retrain"],
        }))
        from ascent.monitoring.weekend_runner import _job_ran_this_weekend
        assert _job_ran_this_weekend("ml_retrain") is True
        assert _job_ran_this_weekend("factor_discovery") is False


# ── Scenario planner ──────────────────────────────────────────────────────────

class TestScenarioPlanner:
    def test_compute_portfolio_impact_basic(self):
        from ascent.monitoring.scenario_planner import _compute_portfolio_impact
        weights = {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}
        shocks  = {"__market__": -0.03}
        nav     = 100_000
        result  = _compute_portfolio_impact(weights, shocks, nav)
        assert result["total_impact_pct"] == pytest.approx(-3.0, abs=0.01)
        assert result["total_impact_usd"] == pytest.approx(-3000, abs=1)

    def test_compute_portfolio_impact_symbol_specific(self):
        from ascent.monitoring.scenario_planner import _compute_portfolio_impact
        weights = {"EWY": 0.10, "SPY": 0.90}
        shocks  = {"EWY": -0.05}
        nav     = 100_000
        result  = _compute_portfolio_impact(weights, shocks, nav)
        assert result["total_impact_pct"] == pytest.approx(-0.5, abs=0.01)

    def test_build_scenarios_returns_5_plus_dynamic(self):
        from ascent.monitoring.scenario_planner import _build_scenarios
        weights = {"EWY": 0.115, "SNDK": 0.07, "SPY": 0.30, "TLT": 0.20, "BIL": 0.315}
        scenarios = _build_scenarios(weights, dynamic_risk="High rates risk")
        assert len(scenarios) == 6
        names = [s["name"] for s in scenarios]
        assert "risk_off_shock" in names
        assert "dynamic_debrief_risk" in names

    def test_build_scenarios_without_dynamic(self):
        from ascent.monitoring.scenario_planner import _build_scenarios
        weights = {"SPY": 0.50, "TLT": 0.50}
        scenarios = _build_scenarios(weights)
        assert len(scenarios) == 5

    def test_largest_position_scenario_uses_correct_symbol(self):
        from ascent.monitoring.scenario_planner import _build_scenarios
        weights = {"SNDK": 0.20, "SPY": 0.10, "GLD": 0.05}
        scenarios = _build_scenarios(weights)
        blowup = next(s for s in scenarios if s["name"] == "largest_position_blowup")
        assert "SNDK" in blowup["description"]
        assert blowup["shocks"].get("SNDK") == -0.15


# ── Weekly debrief ────────────────────────────────────────────────────────────

class TestWeeklyDebrief:
    def test_summarize_attribution_empty(self):
        from ascent.monitoring.weekly_debrief import _summarize_attribution
        assert _summarize_attribution([]) == {}

    def test_summarize_attribution_basic(self):
        from ascent.monitoring.weekly_debrief import _summarize_attribution
        rows = [
            {
                "date": "2026-05-20",
                "portfolio_return": 0.01,
                "spy_return": 0.005,
                "n_positions": 10,
                "top_contributors": [{"symbol": "VRT", "contribution": 0.002}],
                "top_drags": [{"symbol": "EWY", "contribution": -0.001}],
            },
            {
                "date": "2026-05-21",
                "portfolio_return": -0.005,
                "spy_return": 0.002,
                "n_positions": 10,
                "top_contributors": [],
                "top_drags": [{"symbol": "SNDK", "contribution": -0.003}],
            },
        ]
        result = _summarize_attribution(rows)
        assert result["trading_days"] == 2
        assert result["portfolio_return_pct"] == pytest.approx(0.49, abs=0.05)
        assert result["alpha_pct"] < 0  # underperformed SPY

    def test_summarize_verdicts_empty(self):
        from ascent.monitoring.weekly_debrief import _summarize_verdicts
        assert _summarize_verdicts([]) == {}

    def test_summarize_verdicts_counts(self):
        from ascent.monitoring.weekly_debrief import _summarize_verdicts
        verdicts = [
            {"verdict": "proceed"},
            {"verdict": "proceed"},
            {"verdict": "reduce_size"},
        ]
        result = _summarize_verdicts(verdicts)
        assert result["n_debates"] == 3
        assert result["proceed"] == 2
        assert result["reduce_size"] == 1
        assert result["halt_and_review"] == 0
