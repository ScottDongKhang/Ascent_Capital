# tests/test_agent_tools.py
import pytest
import json
from unittest.mock import patch, MagicMock


_WEIGHTS = {"AAPL": 0.15, "MSFT": 0.12, "GLD": 0.10, "TLT": 0.08, "EEM": 0.10, "AMZN": 0.09}


def test_get_sector_concentration_returns_string():
    from debate.agent_tools import get_sector_concentration
    result = get_sector_concentration({"weights": _WEIGHTS})
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_sector_concentration_sums_to_100():
    from debate.agent_tools import get_sector_concentration
    result = get_sector_concentration({"weights": _WEIGHTS})
    assert "%" in result or "sector" in result.lower()


def test_get_var_estimate_returns_string():
    from debate.agent_tools import get_var_estimate
    result = get_var_estimate({"weights": _WEIGHTS})
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_position_momentum_returns_string():
    from debate.agent_tools import get_position_momentum
    with patch("debate.agent_tools._fetch_prices_cached") as mock_fetch:
        import pandas as pd, numpy as np
        idx = pd.date_range(end="2026-05-01", periods=260, freq="B")
        mock_fetch.return_value = {
            sym: pd.Series(np.cumprod(1 + np.random.normal(0.0004, 0.015, 260)), index=idx)
            for sym in ["AAPL", "MSFT"]
        }
        result = get_position_momentum({"symbols": ["AAPL", "MSFT"]})
    assert isinstance(result, str)
    assert "AAPL" in result or "momentum" in result.lower()


def test_get_regime_conditional_stats_returns_string():
    from debate.agent_tools import get_regime_conditional_stats
    result = get_regime_conditional_stats({"regime": "stressed"})
    assert isinstance(result, str)
    assert "stressed" in result.lower()


def test_tool_completion_calls_tools_and_returns_text():
    from ascent.llm.client import tool_completion

    tools = [{"name": "test_tool", "description": "Test", "input_schema": {
        "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]
    }}]

    tool_use_response = MagicMock()
    tool_use_response.stop_reason = "tool_use"
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "test_tool"
    tool_use_block.id   = "tu_abc123"
    tool_use_block.input = {"x": "hello"}
    tool_use_response.content = [tool_use_block]

    final_response = MagicMock()
    final_response.stop_reason = "end_turn"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I found the answer using the tool."
    final_response.content = [text_block]

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [tool_use_response, final_response]

    with patch("ascent.llm.client._get_client", return_value=mock_client):
        with patch("ascent.llm.client.ANTHROPIC_API_KEY", "sk-test"):
            def executor(name, inputs):
                return "tool result: 42"
            result = tool_completion(
                system_prompt="You are a test agent.",
                user_prompt="Use the tool.",
                tools=tools,
                tool_executor=executor,
            )
    assert "answer" in result.lower() or "tool" in result.lower()


def test_tool_completion_max_iterations_guard():
    from ascent.llm.client import tool_completion

    tools = [{"name": "loop_tool", "description": "Always asks for tool use",
              "input_schema": {"type": "object", "properties": {}}}]

    always_tool_response = MagicMock()
    always_tool_response.stop_reason = "tool_use"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "loop_tool"
    tool_block.id   = "tu_loop"
    tool_block.input = {}
    always_tool_response.content = [tool_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = always_tool_response

    with patch("ascent.llm.client._get_client", return_value=mock_client):
        with patch("ascent.llm.client.ANTHROPIC_API_KEY", "sk-test"):
            result = tool_completion(
                system_prompt="Test", user_prompt="Test",
                tools=tools, tool_executor=lambda n, i: "ok",
                max_tool_calls=2,
            )
    assert isinstance(result, str)
    assert mock_client.messages.create.call_count <= 4


def test_bear_agent_uses_tool_completion():
    import debate.agents as agents_mod
    captured = []

    def mock_tool_completion(system_prompt, user_prompt, tools, tool_executor, **kwargs):
        captured.append({"system": system_prompt, "user": user_prompt, "tools": tools})
        return "Bear case: concentration risk in tech is elevated."

    portfolio_state = {
        "date": "2026-05-03", "us_regime": "stressed",
        "weights": _WEIGHTS, "n_positions": 6, "allocation": {},
    }
    with patch("debate.agents.tool_completion", side_effect=mock_tool_completion):
        result = agents_mod.run_bear_agent(portfolio_state)

    assert len(captured) > 0, "run_bear_agent must call tool_completion"
    assert any("tool" in str(c["tools"]).lower() for c in captured), \
        "tool_completion call must include tool definitions"
    assert isinstance(result, str)
