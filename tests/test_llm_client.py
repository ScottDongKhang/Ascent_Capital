# tests/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_client(text='{"key": "value"}'):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=text)]
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 20
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_generate_structured_no_schema_omits_output_config():
    """When json_schema is None, output_config must not appear in the API call."""
    from ascent.llm.client import generate_structured
    mock_client = _make_mock_client()
    with patch("ascent.llm.client._client", mock_client):
        generate_structured("sys", "user")
    kwargs = mock_client.messages.create.call_args[1]
    assert "output_config" not in kwargs


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
    assert kwargs.get("output_config") == oc
