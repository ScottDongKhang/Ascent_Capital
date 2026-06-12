"""Tests for the Phase 2 force-seal pass in run_ai_pm().

Mirror of the Phase 1 (propose_prethesis) force-seal: when the main
tool_completion loop exhausts max_tool_calls without calling propose_portfolio,
run_ai_pm must make a direct Anthropic API call with
tool_choice={"type": "tool", "name": "propose_portfolio"} before falling back.
"""
import pytest
from unittest.mock import patch, MagicMock


def _valid_seal_input() -> dict:
    return {
        "weights": {"AAPL": 0.10, "MSFT": 0.10, "NVDA": 0.08},
        "thesis": {
            "market_view": "calm_bull continuation",
            "feedback_acknowledged": True,
            "worst_call_response": "Overweighted energy into falling crude.",
            "position_rationale": {},
            "quant_overrides": [],
        },
    }


def _tool_use_response(tool_input: dict):
    """Build a mock Anthropic messages.create response containing one tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "propose_portfolio"
    block.input = tool_input
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    return resp


def _patches(tmp_path, client):
    """Common patch set: hermetic run_ai_pm with a mocked Anthropic client."""
    return [
        patch("agents.ai_pm_agent._REPO_ROOT", tmp_path),
        patch("agents.ai_pm_agent._build_data_grounding", return_value=""),
        patch("agents.ai_pm_agent._build_temporal_context", return_value=""),
        patch("agents.ai_pm_agent._build_system_prompt", return_value="SYS"),
        patch("agents.ai_pm_agent._get_calibration_ic_safe", return_value=None),
        patch("agents.ai_pm_agent._get_current_regime", return_value="calm_bull"),
        patch("agents.red_team_agent.run_red_team", return_value=None),
        patch("ascent.llm.client._check_api_key"),
        patch("ascent.llm.client._record_usage"),
        patch("ascent.llm.client._get_client", return_value=client),
    ]


class TestPhase2ForceSeal:
    def test_force_seal_runs_when_main_pass_does_not_seal(self, tmp_path):
        """Main tool loop never seals -> force-seal direct call must produce the result."""
        (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)
        client = MagicMock()
        client.messages.create.return_value = _tool_use_response(_valid_seal_input())

        from agents import ai_pm_agent

        ctxs = _patches(tmp_path, client)
        with patch.object(ai_pm_agent, "tool_completion") as mock_tc:
            mock_tc.return_value = None  # main pass runs but never seals
            for c in ctxs:
                c.start()
            try:
                result = ai_pm_agent.run_ai_pm(quant_outputs=[], merged_weights={"AAPL": 0.10})
            finally:
                for c in ctxs:
                    c.stop()

        assert not result.fallback, "force-seal should rescue the run from fallback"
        assert result.portfolio == _valid_seal_input()["weights"]
        # The forced call must hard-require propose_portfolio
        _, kwargs = client.messages.create.call_args
        assert kwargs.get("tool_choice") == {"type": "tool", "name": "propose_portfolio"}
        tool_names = [t["name"] for t in kwargs.get("tools", [])]
        assert tool_names == ["propose_portfolio"]

    def test_force_seal_not_run_when_main_pass_seals(self, tmp_path):
        """If the main loop seals normally, no direct API call is made."""
        (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)
        client = MagicMock()

        from agents import ai_pm_agent

        def fake_tool_completion(**kwargs):
            kwargs["tool_executor"]("propose_portfolio", _valid_seal_input())

        ctxs = _patches(tmp_path, client)
        with patch.object(ai_pm_agent, "tool_completion", side_effect=fake_tool_completion):
            for c in ctxs:
                c.start()
            try:
                result = ai_pm_agent.run_ai_pm(quant_outputs=[], merged_weights={"AAPL": 0.10})
            finally:
                for c in ctxs:
                    c.stop()

        assert not result.fallback
        client.messages.create.assert_not_called()

    def test_force_seal_failure_still_falls_back(self, tmp_path):
        """If the forced call itself fails, run_ai_pm must return the quant fallback."""
        (tmp_path / "data_cache").mkdir(parents=True, exist_ok=True)
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("api down")

        from agents import ai_pm_agent

        ctxs = _patches(tmp_path, client)
        with patch.object(ai_pm_agent, "tool_completion", return_value=None):
            for c in ctxs:
                c.start()
            try:
                result = ai_pm_agent.run_ai_pm(quant_outputs=[], merged_weights={"AAPL": 0.10})
            finally:
                for c in ctxs:
                    c.stop()

        assert result.fallback
        assert result.portfolio == {}

    def test_force_seal_retries_after_gate_rejection(self, tmp_path):
        """A forced submission that trips the feedback gate must be retried once
        with the rejection message, not silently dropped."""
        dc = tmp_path / "data_cache"
        dc.mkdir(parents=True, exist_ok=True)
        (dc / "ai_pm_perf_feedback.json").write_text('{"worst_call": "energy overweight"}')

        bad = _valid_seal_input()
        bad["thesis"] = dict(bad["thesis"], feedback_acknowledged=False)
        good = _valid_seal_input()

        client = MagicMock()
        client.messages.create.side_effect = [
            _tool_use_response(bad),
            _tool_use_response(good),
        ]

        from agents import ai_pm_agent

        ctxs = _patches(tmp_path, client)
        with patch.object(ai_pm_agent, "tool_completion", return_value=None):
            for c in ctxs:
                c.start()
            try:
                result = ai_pm_agent.run_ai_pm(quant_outputs=[], merged_weights={"AAPL": 0.10})
            finally:
                for c in ctxs:
                    c.stop()

        assert client.messages.create.call_count == 2
        assert not result.fallback
        assert result.portfolio == good["weights"]
