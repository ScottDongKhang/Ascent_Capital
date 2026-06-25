"""
ascent/llm/client.py
Centralized LLM client for Ascent Capital.

Uses Anthropic SDK directly with Claude Opus 4.6.

Configuration:
    ANTHROPIC_API_KEY in .env
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

# ── Cost accumulator (per-process, reset on each run) ─────────────────────────
_PRICING = {
    # (input $/MTok, output $/MTok)
    "claude-opus-4-6":           (15.00, 75.00),
    "claude-opus-4-7":           (15.00, 75.00),
    "claude-sonnet-4-6":         ( 3.00, 15.00),
    "claude-haiku-4-5-20251001": ( 0.80,  4.00),
}
_DEFAULT_PRICING = (3.00, 15.00)  # fallback for unknown models

_usage: Dict[str, Dict] = {}  # keyed by model


def _record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    if model not in _usage:
        _usage[model] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    _usage[model]["input_tokens"]  += input_tokens
    _usage[model]["output_tokens"] += output_tokens
    _usage[model]["calls"]         += 1


def get_usage_summary() -> Dict:
    """Return token counts and estimated cost across all models this process."""
    rows = []
    total_cost = 0.0
    for model, counts in _usage.items():
        in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (counts["input_tokens"] / 1_000_000 * in_price
              + counts["output_tokens"] / 1_000_000 * out_price)
        total_cost += cost
        rows.append({
            "model":         model,
            "input_tokens":  counts["input_tokens"],
            "output_tokens": counts["output_tokens"],
            "calls":         counts["calls"],
            "cost_usd":      round(cost, 4),
        })
    return {"models": rows, "total_cost_usd": round(total_cost, 4)}


def log_costs(date_str: str, log_dir: str = "logs") -> None:
    """Append today's token usage + estimated cost to logs/cost_log.jsonl."""
    summary = get_usage_summary()
    if not summary["models"]:
        return
    entry = {"date": date_str, **summary}
    path = Path(log_dir) / "cost_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    tc = summary["total_cost_usd"]
    print(f"[LLM] Cost logged — ${tc:.4f} total | {path}")
    for row in summary["models"]:
        print(f"  {row['model']}: {row['calls']} calls, "
              f"{row['input_tokens']:,} in / {row['output_tokens']:,} out → ${row['cost_usd']:.4f}")


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
SONNET_MODEL      = "claude-sonnet-4-6"
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


# Public alias — preferred import for external callers
get_client = _get_client


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
    use_cache: bool = False,
    output_config: dict | None = None,
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
        if use_cache:
            kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            kwargs["system"] = system_prompt
    if output_config is not None:
        kwargs["output_config"] = output_config

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)
            _record_usage(model, response.usage.input_tokens, response.usage.output_tokens)
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
    use_cache: bool = False,
    json_schema: dict | None = None,
) -> str:
    """
    Convenience wrapper for structured generation tasks.
    Lower temperature for more deterministic output.
    Pass json_schema to enforce response shape via structured outputs.
    """
    output_config = (
        {"format": {"type": "json_schema", "schema": json_schema}}
        if json_schema is not None else None
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return chat_completion(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
        output_config=output_config,
    )


def extended_thinking_completion(
    messages,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 5000,
    thinking_budget: int = 3000,
) -> str:
    """Extended thinking mode. temperature=1 required by Anthropic API. Falls back on rejection."""
    _check_api_key()
    client = _get_client()
    system_prompt = ""
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered.append(m)
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=1,
        thinking={"type": "enabled", "budget_tokens": thinking_budget},
        messages=filtered,
    )
    if system_prompt:
        kwargs["system"] = system_prompt
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.messages.create(**kwargs)
            _record_usage(model, resp.usage.input_tokens, resp.usage.output_tokens)
            text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(text_parts)
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("thinking", "budget", "unsupported", "temperature")):
                log.warning(f"[LLM] Extended thinking unavailable ({e}), falling back")
                return chat_completion(messages, model=model, max_tokens=max_tokens, temperature=0.3)
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


_FINAL_TURN_NUDGE = (
    "You have nearly used your entire tool-call budget. This is your FINAL turn — "
    "do NOT request more tools. Give your final answer now, and if you have a "
    "submission tool (e.g. propose_portfolio / propose_prethesis) you MUST call it "
    "this turn. If you see no reason to deviate from the baseline, submit it unchanged."
)


def tool_completion(
    system_prompt: str,
    user_prompt: str,
    tools: list,
    tool_executor,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    max_tool_calls: int = 3,
    use_cache: bool = False,
) -> str:
    """
    Execute an LLM call with Anthropic tool use, running the tool loop until
    stop_reason == 'end_turn' or max_tool_calls is reached.

    Args:
        system_prompt:  System instructions for the agent.
        user_prompt:    Initial user message.
        tools:          List of Anthropic tool schema dicts (name, description, input_schema).
        tool_executor:  Callable(tool_name: str, tool_inputs: dict) -> str result.
        model:          Anthropic model string.
        max_tokens:     Max output tokens per call.
        max_tool_calls: Maximum number of tool call iterations before forcing a final response.

    Returns:
        The agent's final text response after all tool calls.
    """
    _check_api_key()
    client   = _get_client()
    messages = [{"role": "user", "content": user_prompt}]

    for _iteration in range(max_tool_calls + 1):
        system_block = (
            [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            if use_cache else system_prompt
        )
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
            system=system_block,
        )
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.messages.create(**kwargs)
                _record_usage(model, resp.usage.input_tokens, resp.usage.output_tokens)
                break
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                log.warning("[LLM/Tools] Attempt %d failed (%s), retry in %ds", attempt + 1, e, wait)
                time.sleep(wait)

        if resp.stop_reason == "end_turn":
            text_parts = [
                block.text for block in resp.content
                if getattr(block, "type", "") == "text"
            ]
            return "\n".join(text_parts)

        if resp.stop_reason == "tool_use":
            tool_use_blocks = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_use_blocks:
                break

            assistant_content = []
            for block in resp.content:
                block_type = getattr(block, "type", "")
                if block_type == "tool_use":
                    assistant_content.append({
                        "type":  "tool_use",
                        "id":    block.id,
                        "name":  block.name,
                        "input": block.input,
                    })
                elif block_type == "text":
                    assistant_content.append({"type": "text", "text": block.text})

            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in tool_use_blocks:
                result = tool_executor(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     str(result),
                })
            # One round before the budget is exhausted, tell the model this is its
            # final turn so it finalizes / calls its submission tool instead of
            # over-grounding into a "[max iterations reached]" non-answer (the AI PM
            # force-seal path). Appended to the tool-result user turn so roles stay
            # valid; only fires on the penultimate iteration.
            if _iteration == max_tool_calls - 1:
                tool_results.append({"type": "text", "text": _FINAL_TURN_NUDGE})
            messages.append({"role": "user", "content": tool_results})

        else:
            text_parts = [
                block.text for block in resp.content
                if getattr(block, "type", "") == "text"
            ]
            return "\n".join(text_parts) if text_parts else f"[Stopped: {resp.stop_reason}]"

    try:
        text_parts = [
            block.text for block in resp.content
            if getattr(block, "type", "") == "text"
        ]
        return "\n".join(text_parts) if text_parts else "[Tool loop: max iterations reached]"
    except Exception:
        return "[Tool loop: max iterations reached]"


if __name__ == "__main__":
    try:
        result = chat_completion(
            [{"role": "user", "content": "Say 'Ascent Capital LLM client working' and nothing else."}],
            max_tokens=50,
        )
        print(f"[LLM] Test response: {result}")
    except Exception as e:
        print(f"[LLM] Test failed: {e}")
