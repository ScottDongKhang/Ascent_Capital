"""Minimal Anthropic wrapper.

Standalone by design -- this package imports nothing from `ascent`. The Claude 5
rules it honours are the same ones that repo learned the hard way:

- never index `content[0].text`; thinking is on by default so block 0 is usually
  a thinking block. Walk the blocks and take the text ones.
- never pass `temperature` / `top_p` / `top_k`; they 400.
- never pass `thinking={...}`; depth comes from `output_config.effort`.
- `max_tokens` caps thinking plus visible text together, so leave headroom.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
_MAX_TOKENS = 8000


def _load_env() -> None:
    """Read ANTHROPIC_API_KEY out of a repo-root .env if it is not exported."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def extract_text(resp) -> str:
    """Concatenate every text block, skipping thinking blocks."""
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()


def complete(system: str, user: str, effort: str = "medium") -> str:
    """One turn, text back. No tools, no streaming."""
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set and no .env provided one")

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    )
    text = extract_text(resp)
    if not text:
        raise RuntimeError("model returned no text block")
    return text
