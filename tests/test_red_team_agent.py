# tests/test_red_team_agent.py
"""
Tests for agents/red_team_agent.py — adversarial red team critique.
"""
from unittest.mock import MagicMock, patch


_SAMPLE_PORTFOLIO = {
    "VICR": 0.10,
    "AMKR": 0.09,
    "FIX":  0.08,
    "WDC":  0.07,
    "VRT":  0.06,
}

_SAMPLE_THESIS = {
    "market_view": "Calm bull, credit spreads tight, earnings season positive.",
    "regime_assessment": "calm_bull, confidence 0.78",
    "key_risks": ["Fed surprise", "VICR earnings miss"],
    "what_could_be_wrong": "Momentum crowding unwinds sharply.",
    "quant_overrides": [{"symbol": "VAL", "ai_action": "exclude", "reason": "declining earnings"}],
}


def _make_mock_client(return_text: str = "This portfolio will blow up.") -> MagicMock:
    """Return a mock Anthropic client that returns return_text on messages.create."""
    mock_content_block = MagicMock()
    mock_content_block.text = return_text

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ── test 1: happy path ────────────────────────────────────────────────────────

def test_run_red_team_returns_string():
    """run_red_team returns a non-empty string when the client succeeds."""
    mock_client = _make_mock_client("VICR is grossly overvalued. Kill shot: correlation spike.")

    with patch("agents.red_team_agent.get_client", return_value=mock_client):
        from agents.red_team_agent import run_red_team
        result = run_red_team(_SAMPLE_PORTFOLIO, _SAMPLE_THESIS, regime="calm_bull")

    assert isinstance(result, str)
    assert len(result) > 0
    # Verify the client was actually called
    mock_client.messages.create.assert_called_once()


# ── test 2: exception safety ───────────────────────────────────────────────────

def test_run_red_team_handles_exception():
    """run_red_team returns empty string (never raises) when the client raises."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("API timeout")

    with patch("agents.red_team_agent.get_client", return_value=mock_client):
        from agents.red_team_agent import run_red_team
        result = run_red_team(_SAMPLE_PORTFOLIO, _SAMPLE_THESIS, regime="stressed")

    assert result == ""


# ── test 3: empty portfolio ────────────────────────────────────────────────────

def test_run_red_team_empty_portfolio():
    """run_red_team does not crash on an empty portfolio dict."""
    mock_client = _make_mock_client("Nothing to attack here.")

    with patch("agents.red_team_agent.get_client", return_value=mock_client):
        from agents.red_team_agent import run_red_team
        result = run_red_team({}, {}, regime="")

    assert isinstance(result, str)
    mock_client.messages.create.assert_called_once()


# ── test 4: regime appears in prompt ──────────────────────────────────────────

def test_run_red_team_includes_regime():
    """The regime string is included in the prompt sent to the client."""
    mock_client = _make_mock_client("Kill shot: macro tail risk unpriced.")

    with patch("agents.red_team_agent.get_client", return_value=mock_client):
        from agents.red_team_agent import run_red_team
        run_red_team(_SAMPLE_PORTFOLIO, _SAMPLE_THESIS, regime="crisis")

    # Inspect the messages argument passed to messages.create
    call_kwargs = mock_client.messages.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else None
    if messages is None:
        # Try positional kwarg fallback
        all_kwargs = call_kwargs.kwargs
        messages = all_kwargs.get("messages", [])

    # The regime should appear somewhere in the user content
    prompt_text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            prompt_text += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    prompt_text += block.get("text", "")

    assert "crisis" in prompt_text, (
        f"Expected 'crisis' in prompt, but prompt was:\n{prompt_text[:500]}"
    )
