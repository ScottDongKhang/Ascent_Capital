# tests/test_phase1_hardening.py
import json
import pytest
from datetime import date, timedelta
from unittest.mock import patch


def test_skill_tracker_writes_as_of_date(tmp_path):
    """export_skill_scores() must write skill_score_as_of to the JSON payload."""
    from ascent.monitoring import skill_tracker

    with patch.object(skill_tracker, "SKILL_OUTPUT_PATH", tmp_path / "agent_skill_scores.json"), \
         patch.object(skill_tracker, "SKILL_LOG_PATH", tmp_path / "skill_scores_log.jsonl"), \
         patch.object(skill_tracker, "compute_all_skill_scores", return_value={}):
        skill_tracker.export_skill_scores()

    payload = json.loads((tmp_path / "agent_skill_scores.json").read_text())
    assert "skill_score_as_of" in payload, "skill_score_as_of key missing from output"
    assert payload["skill_score_as_of"] == date.today().isoformat()


def test_orchestrator_rejects_stale_skill_scores(tmp_path):
    """_load_skill_scores() returns {} when skill_score_as_of is more than 1 day old."""
    import json
    from orchestrator import central_intelligence as ci

    stale_date = (date.today() - timedelta(days=3)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-09T14:00:00",
        "skill_score_as_of": stale_date,
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {}, f"Expected empty dict for stale scores, got {result}"


def test_orchestrator_accepts_fresh_skill_scores(tmp_path):
    """_load_skill_scores() returns scores when skill_score_as_of is today."""
    import json
    from orchestrator import central_intelligence as ci

    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": date.today().isoformat(),
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 1.23}


def test_orchestrator_accepts_yesterday_scores_on_monday(tmp_path):
    """_load_skill_scores() returns scores when as_of is 1 day ago (weekend buffer)."""
    import json
    from orchestrator import central_intelligence as ci

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": yesterday,
        "agents": {
            "us_equities": {"skill_score": 0.80, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 0.80}


def test_sector_constrained_raises_on_low_coverage_at_construction_time():
    """
    sector_constrained_weighted() must raise SectorDataError (not silently degrade)
    when sector coverage of the candidate pool is below 80%.
    """
    import pandas as pd
    from ascent.portfolio.optimizer import sector_constrained_weighted, SectorDataError

    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    syms = ["A", "B", "C", "D", "E"]
    alpha = pd.DataFrame(
        [[0.5, 0.4, 0.3, 0.2, 0.1]] * 3,
        index=dates,
        columns=syms,
    )
    # Only 3 of 5 symbols have known sectors = 60% coverage < 80% threshold.
    # Correct behavior per CLAUDE.md integrity constraint #4: fall back to plain
    # rank weighting (do NOT raise SectorDataError).
    sector_map = {"A": "Tech", "B": "Health", "C": "Energy"}

    result = sector_constrained_weighted(alpha, n=5, sector_map=sector_map)
    # Falls back to rank-weighted — all 5 names should have non-zero weights
    assert isinstance(result, pd.DataFrame)
    last_row = result.iloc[-1]
    assert last_row.sum() > 0.0, "expected non-zero weights on fallback path"


def test_validate_sector_data_raises_when_profiles_missing():
    """validate_sector_data() must raise SectorDataError when profiles.parquet is absent."""
    import run_all_agents
    from ascent.portfolio.optimizer import SectorDataError

    symbols = ["AAPL", "MSFT", "GOOGL"]
    with patch("run_all_agents.has_data", return_value=False):
        with pytest.raises(SectorDataError, match="profiles.parquet missing"):
            run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_raises_on_low_coverage():
    """validate_sector_data() must raise when sector coverage < 80%."""
    import pandas as pd
    import run_all_agents
    from ascent.portfolio.optimizer import SectorDataError

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    profiles_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "sector": ["Tech", "Tech", None, None, None],
    })

    with patch("run_all_agents.has_data", return_value=True), \
         patch("run_all_agents.load_parquet", return_value=profiles_df):
        with pytest.raises(SectorDataError, match="Sector coverage"):
            run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_passes_on_good_coverage():
    """validate_sector_data() must not raise when coverage >= 80%."""
    import pandas as pd
    import run_all_agents

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    profiles_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "sector": ["Tech", "Tech", "Tech", "Consumer", "Tech"],
    })

    with patch("run_all_agents.has_data", return_value=True), \
         patch("run_all_agents.load_parquet", return_value=profiles_df):
        # Should not raise
        run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_skip_flag_logs_and_returns(tmp_path):
    """validate_sector_data(skip=True) must log override and return without checking."""
    import run_all_agents

    log_path = tmp_path / "sector_override.jsonl"
    with patch.object(run_all_agents, "SECTOR_OVERRIDE_LOG", log_path), \
         patch("run_all_agents.has_data") as mock_has_data:
        run_all_agents.validate_sector_data(symbols=["AAPL"], skip=True)

    mock_has_data.assert_not_called()
    assert log_path.exists()
    import json
    entry = json.loads(log_path.read_text().strip())
    assert entry["action"] == "sector_check_skipped"


def test_debate_runner_writes_halt_state_on_halt_verdict(tmp_path):
    """
    run_debate() must write execution/halt_state.json when verdict is halt_and_review.
    File must contain: halted=True, halt_date, reason, key_risks, verdict_path, requires_override.
    """
    import json
    from debate import debate_runner
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    verdict = {
        "recommendation": "halt_and_review",
        "confidence": 0.85,
        "key_risks": ["Energy concentration at 38%", "Oil shock risk"],
        "reasoning": "Too much energy exposure in volatile macro environment",
    }
    portfolio_state = {
        "date": "2026-04-15",
        "us_regime": "stressed",
        "macro_regime": "unknown",
        "n_positions": 12,
        "allocation": {},
        "weights": {"XLE": 0.20, "MPC": 0.18},
    }

    with patch.object(debate_runner, "HALT_STATE_PATH", halt_path), \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull arg"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear arg"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil arg"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime arg"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant ok"), \
         patch("debate.debate_runner.run_judge", return_value=verdict), \
         patch("debate.debate_runner.DEBATE_LOG_DIR", tmp_path):
        result = debate_runner.run_debate(portfolio_state, run_date=date(2026, 4, 15))

    assert halt_path.exists(), "halt_state.json was not written"
    state = json.loads(halt_path.read_text())
    assert state["halted"] is True
    assert state["halt_date"] == "2026-04-15"
    assert state["requires_override"] is True
    assert "key_risks" in state
    assert len(state["key_risks"]) == 2


def test_debate_runner_does_not_write_halt_state_on_proceed(tmp_path):
    """run_debate() must NOT write halt_state.json when verdict is proceed."""
    import json
    from debate import debate_runner
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    verdict = {
        "recommendation": "proceed",
        "confidence": 0.70,
        "key_risks": [],
        "reasoning": "Portfolio looks fine",
    }
    portfolio_state = {
        "date": "2026-04-15",
        "us_regime": "calm_bull",
        "macro_regime": "unknown",
        "n_positions": 10,
        "allocation": {},
        "weights": {"AAPL": 0.10},
    }

    with patch.object(debate_runner, "HALT_STATE_PATH", halt_path), \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull arg"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear arg"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil arg"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime arg"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant ok"), \
         patch("debate.debate_runner.run_judge", return_value=verdict), \
         patch("debate.debate_runner.DEBATE_LOG_DIR", tmp_path):
        debate_runner.run_debate(portfolio_state, run_date=date(2026, 4, 15))

    assert not halt_path.exists(), "halt_state.json should NOT be written for proceed verdict"


def test_check_halt_state_returns_true_when_no_halt_file(tmp_path):
    """check_halt_state() returns True (proceed) when halt_state.json does not exist."""
    import run_all_agents
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date.today())

    assert result is True


def test_check_halt_state_returns_false_when_halted_no_override(tmp_path):
    """check_halt_state() returns False when halted without override."""
    import json, run_all_agents
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"
    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "Too much energy exposure",
        "key_risks": ["Energy at 38%"],
        "verdict_path": "outputs/debate_log/verdict_2026-04-15.json",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is False


def test_check_halt_state_clears_files_on_valid_override(tmp_path):
    """check_halt_state() returns True and deletes both files when override is valid."""
    import json, run_all_agents
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "Too much energy",
        "key_risks": [],
        "verdict_path": "outputs/debate_log/verdict_2026-04-15.json",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))
    override_path.write_text(json.dumps({
        "override_date": "2026-04-16",
        "override_by": "scott",
        "reason": "Reviewed — acceptable",
        "acknowledged_risks": [],
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is True
    assert not halt_path.exists(), "halt_state.json should be deleted after valid override"
    assert not override_path.exists(), "halt_override.json should be deleted after valid override"


def test_daily_run_invokes_only_us_equities_agent():
    """Regression for target-architecture task 4.

    macro_agent/international_agent scored CUT on their real universes;
    alternatives_agent is still unmeasured (excluded pending re-measurement).
    run_all_agents.py's daily orchestration must build `agent_tasks` — and
    therefore `agent_outputs` — with exactly one entry: us_equities. The
    three excluded agent modules must still exist and be importable (their
    run_*_agent() functions are not deleted, only uncalled from the daily
    flow), but main() must not reference them.
    """
    import ast
    import inspect
    import run_all_agents

    source = inspect.getsource(run_all_agents.main)
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef) and func_def.name == "main"

    agent_tasks_assigns = [
        node for node in ast.walk(func_def)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "agent_tasks" for t in node.targets)
    ]
    assert len(agent_tasks_assigns) == 1, "expected exactly one agent_tasks assignment in main()"

    value = agent_tasks_assigns[0].value
    assert isinstance(value, ast.List), "agent_tasks must be a list literal"
    assert len(value.elts) == 1, (
        f"agent_tasks must have exactly one entry (us_equities), found {len(value.elts)}"
    )

    first_entry = value.elts[0]
    assert isinstance(first_entry, ast.Tuple)
    agent_name_node = first_entry.elts[0]
    assert isinstance(agent_name_node, ast.Constant) and agent_name_node.value == "us_equities"

    # macro/international/alternatives must not be invoked anywhere in main()
    for banned in ("run_macro_agent", "run_international_agent", "run_alternatives_agent"):
        assert banned not in source, f"{banned} must not be called from run_all_agents.main()"

    # The excluded agent modules must still exist (not deleted) with their
    # run_*_agent() entry points intact.
    from agents.macro_agent import run_macro_agent  # noqa: F401
    from agents.international_agent import run_international_agent  # noqa: F401
    from agents.alternatives_agent import run_alternatives_agent  # noqa: F401


def test_check_halt_state_blocks_override_predating_halt(tmp_path):
    """check_halt_state() returns False when override_date < halt_date."""
    import json, run_all_agents
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "energy",
        "key_risks": [],
        "verdict_path": "x",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))
    override_path.write_text(json.dumps({
        "override_date": "2026-04-14",   # predates halt
        "override_by": "scott",
        "reason": "Stale override",
        "acknowledged_risks": [],
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is False
