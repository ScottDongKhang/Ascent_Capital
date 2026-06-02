"""ascent/llm/prompt_loader.py

Centralized prompt loader for the Ascent Capital LLM layer.

In the public repository, all proprietary system prompts, chain-of-thought
instructions, and reasoning models are stored in `private_prompts.yaml`,
which is excluded from version control via .gitignore.

Usage:
    from ascent.llm.prompt_loader import PromptLoader
    loader = PromptLoader()
    prompt = loader.get("ai_pm.synthesis")

The YAML file uses dot-notation keys, e.g.:
    ai_pm:
      synthesis: |
        You are the portfolio manager of...
      pre_thesis: |
        ...
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROMPTS_PATH = Path("private_prompts.yaml")

_PLACEHOLDER = (
    "[PROMPT REDACTED — populate private_prompts.yaml to enable this feature. "
    "See docs/private_prompts_schema.md for the expected structure.]"
)


class PromptLoader:
    """Loads and caches prompts from private_prompts.yaml.

    Falls back to a visible placeholder string when the file is absent
    so the pipeline fails loudly rather than silently sending empty prompts.
    """

    _instance: "PromptLoader | None" = None
    _cache: dict[str, Any] = {}

    def __new__(cls) -> "PromptLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import yaml  # optional dep — only needed at runtime
            if _PROMPTS_PATH.exists():
                with _PROMPTS_PATH.open("r") as fh:
                    self._cache = yaml.safe_load(fh) or {}
                log.info("[PromptLoader] Loaded %d top-level keys from %s",
                         len(self._cache), _PROMPTS_PATH)
            else:
                log.warning(
                    "[PromptLoader] %s not found — all prompts will return placeholder text. "
                    "Copy private_prompts.yaml.example to private_prompts.yaml and populate it.",
                    _PROMPTS_PATH,
                )
                self._cache = {}
        except Exception as exc:
            log.error("[PromptLoader] Failed to load %s: %s", _PROMPTS_PATH, exc)
            self._cache = {}
        self._loaded = True

    def get(self, key: str, default: str | None = None) -> str:
        """Return prompt for dot-notation key, e.g. 'ai_pm.synthesis'.

        Args:
            key:     Dot-separated path into the YAML tree.
            default: Fallback string. If None, returns the visible placeholder.
        """
        self._load()
        parts = key.split(".")
        node: Any = self._cache
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                fallback = default if default is not None else _PLACEHOLDER
                log.debug("[PromptLoader] Key %r not found — using fallback", key)
                return fallback
            node = node[part]
        if not isinstance(node, str):
            return default if default is not None else _PLACEHOLDER
        return node

    def get_with_format(self, key: str, **kwargs: Any) -> str:
        """Return prompt for key and call .format(**kwargs) on it."""
        template = self.get(key)
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError) as exc:
            log.warning("[PromptLoader] Format failed for key %r: %s", key, exc)
            return template


def get_prompt(key: str, default: str | None = None) -> str:
    """Module-level convenience wrapper around the singleton PromptLoader."""
    return PromptLoader().get(key, default=default)


def get_prompt_formatted(key: str, **kwargs: Any) -> str:
    """Module-level convenience wrapper for formatted prompts."""
    return PromptLoader().get_with_format(key, **kwargs)
