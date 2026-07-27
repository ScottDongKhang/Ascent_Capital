# tests/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_client(text='{"key": "value"}'):
    mock_response = MagicMock()
    # Real content blocks carry a `type`; extract_text() filters on it.
    mock_response.content = [MagicMock(type="text", text=text)]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 20
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ── Claude 5 migration regressions ────────────────────────────────────────────

def test_extract_text_skips_thinking_block():
    """Thinking is on by default on Claude 5, so content[0] is often a thinking
    block. extract_text must return the text block, not index position 0."""
    from ascent.llm.client import extract_text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [
        MagicMock(type="thinking", thinking="internal reasoning"),
        MagicMock(type="text", text="the answer"),
    ]
    assert extract_text(resp) == "the answer"


def test_extract_text_returns_empty_on_refusal():
    from ascent.llm.client import extract_text
    resp = MagicMock()
    resp.stop_reason = "refusal"
    resp.content = [MagicMock(type="text", text="should not be read")]
    assert extract_text(resp) == ""


def test_claude5_does_not_send_temperature():
    """temperature/top_p/top_k are rejected with a 400 on Claude 5."""
    from ascent.llm.client import chat_completion, DEFAULT_MODEL
    mock_client = _make_mock_client()
    with patch("ascent.llm.client._client", mock_client):
        chat_completion([{"role": "user", "content": "hi"}],
                        model=DEFAULT_MODEL, temperature=0.7)
    kwargs = mock_client.messages.create.call_args[1]
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs and "top_k" not in kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_legacy_model_still_receives_temperature():
    """Haiku 4.5 is not on the Claude 5 surface and keeps the old parameters."""
    from ascent.llm.client import chat_completion, HAIKU_MODEL
    mock_client = _make_mock_client()
    with patch("ascent.llm.client._client", mock_client):
        chat_completion([{"role": "user", "content": "hi"}],
                        model=HAIKU_MODEL, temperature=0.2, max_tokens=100)
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["temperature"] == 0.2
    assert "thinking" not in kwargs
    assert kwargs["max_tokens"] == 100  # no floor applied to legacy models


def test_max_tokens_floor_applied_when_thinking():
    """max_tokens caps thinking + text together; a tight cap would return no text."""
    from ascent.llm.client import chat_completion, DEFAULT_MODEL, _MIN_TOKENS_WITH_THINKING
    mock_client = _make_mock_client()
    with patch("ascent.llm.client._client", mock_client):
        chat_completion([{"role": "user", "content": "hi"}],
                        model=DEFAULT_MODEL, max_tokens=500)
    assert mock_client.messages.create.call_args[1]["max_tokens"] == _MIN_TOKENS_WITH_THINKING


def test_extended_thinking_sends_adaptive_not_budget_tokens():
    """budget_tokens is rejected with a 400 on Claude 5."""
    from ascent.llm.client import extended_thinking_completion, DEFAULT_MODEL
    mock_client = _make_mock_client(text="answer")
    with patch("ascent.llm.client._client", mock_client):
        extended_thinking_completion([{"role": "user", "content": "hi"}],
                                     model=DEFAULT_MODEL, thinking_budget=3000)
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in kwargs.get("thinking", {})
    assert "temperature" not in kwargs
    assert kwargs["output_config"]["effort"] == "high"


def test_generate_structured_no_schema_omits_output_config():
    """With no json_schema, output_config carries only effort (no format key)."""
    from ascent.llm.client import generate_structured
    mock_client = _make_mock_client()
    with patch("ascent.llm.client._client", mock_client):
        generate_structured("sys", "user")
    kwargs = mock_client.messages.create.call_args[1]
    assert "format" not in kwargs.get("output_config", {})


def test_generate_structured_with_schema_sends_output_config():
    """When json_schema is provided, messages.create receives the correct output_config."""
    from ascent.llm.client import generate_structured
    mock_client = _make_mock_client()
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    }
    with patch("ascent.llm.client._client", mock_client):
        generate_structured("sys", "user", json_schema=schema)
    kwargs = mock_client.messages.create.call_args[1]
    assert "output_config" in kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] == schema


def test_chat_completion_output_config_passthrough():
    """chat_completion passes output_config through to messages.create."""
    from ascent.llm.client import chat_completion
    mock_client = _make_mock_client()
    oc = {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {}, "additionalProperties": False}}}
    with patch("ascent.llm.client._client", mock_client):
        chat_completion([{"role": "user", "content": "hi"}], output_config=oc)
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["output_config"]["format"] == oc["format"]
