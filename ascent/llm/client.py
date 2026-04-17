"""
ascent/llm/client.py
Centralized LLM client for Ascent Capital.

Uses Anthropic SDK directly with Claude Opus 4.6.

Configuration:
    ANTHROPIC_API_KEY in .env
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger(__name__)


def _load_env():
    """Load .env file into os.environ without requiring python-dotenv."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL     = "claude-opus-4-6"
HAIKU_MODEL       = "claude-haiku-4-5-20251001"

# Retry config
_MAX_RETRIES = 3

# Lazy singleton — initialized on first call, reused thereafter
_client: Optional[object] = None


def _get_client():
    """Return the shared Anthropic client, creating it on first call."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _check_api_key():
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
        )


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7,
) -> str:
    """
    Send a chat completion request to Anthropic.

    Args:
        messages:    List of {"role": "user"/"system"/"assistant", "content": "..."}
                     System messages are extracted and passed as the system param.
        model:       Anthropic model string
        max_tokens:  Max output tokens
        temperature: Sampling temperature

    Returns:
        The assistant's response text.
    """
    _check_api_key()

    client = _get_client()

    # Extract system message if present
    system_prompt = ""
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=filtered_messages,
    )
    if system_prompt:
        kwargs["system"] = system_prompt

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning(f"[LLM] Attempt {attempt + 1} failed ({e}), retrying in {wait}s")
            time.sleep(wait)


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.4,
) -> str:
    """
    Convenience wrapper for structured generation tasks.
    Lower temperature for more deterministic output.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return chat_completion(messages, model=model, max_tokens=max_tokens, temperature=temperature)


if __name__ == "__main__":
    try:
        result = chat_completion(
            [{"role": "user", "content": "Say 'Ascent Capital LLM client working' and nothing else."}],
            max_tokens=50,
        )
        print(f"[LLM] Test response: {result}")
    except Exception as e:
        print(f"[LLM] Test failed: {e}")
