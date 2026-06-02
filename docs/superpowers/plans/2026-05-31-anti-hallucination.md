# Anti-Hallucination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent LLMs in Ascent Capital from fabricating financial data by adding structured output enforcement, evidence citation requirements, and grounding constraints across all three LLM-facing alpha/debate surfaces.

**Architecture:** Four independent changes — (1) add `output_config` passthrough to the central LLM client, (2) enforce JSON schema + evidence quoting in `llm_fundamental`, (3) enforce JSON schema + grounding in `narrative_alpha`, (4) add evidence-citation language to debate agent prompts. Tasks 2–4 depend on Task 1 completing first; after that they are parallel.

**Tech Stack:** Python 3.12, Anthropic SDK (`anthropic`), pytest, unittest.mock

---

## Files

| File | Change |
|---|---|
| `ascent/llm/client.py` | Add `output_config: dict \| None = None` param to `chat_completion` and `generate_structured` |
| `ascent/alpha/llm_fundamental.py` | Add amnesia system prompt, `quoted_evidence` field to schema, use `json_schema` param |
| `ascent/alpha/narrative_alpha.py` | Add grounding instruction, use `json_schema` param |
| `debate/agents.py` | Add evidence-citation instruction to bull, bear, devil's advocate system prompts |
| `tests/test_llm_fundamental_alpha.py` | New tests for amnesia prompt and `quoted_evidence` |
| `tests/test_narrative_alpha.py` | New test for grounding instruction |
| `tests/test_llm_client.py` | New test for `output_config` passthrough |

---

## Task 1: Add `output_config` passthrough to `client.py`

**Files:**
- Modify: `ascent/llm/client.py`
- Test: `tests/test_llm_client.py`

### Background

`chat_completion` and `generate_structured` currently hard-code the kwargs passed to `messages.create()`. Structured outputs require an `output_config={"format": {"type": "json_schema", "schema": {...}}}` kwarg. We need to thread it through without breaking existing callers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_client.py` if it does not exist:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_llm_client.py -v 2>&1 | tail -20
```

Expected: 3 FAILs (AttributeError or TypeError on `json_schema` param not existing)

- [ ] **Step 3: Implement the changes in `client.py`**

In `chat_completion` (line 122), add `output_config: dict | None = None` to the signature and pass it through:

```python
def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    use_cache: bool = False,
    output_config: dict | None = None,   # NEW
) -> str:
```

Inside `chat_completion`, after building `kwargs`, add (before the retry loop):

```python
    if output_config is not None:
        kwargs["output_config"] = output_config
```

The full `kwargs` block (lines 155-168) becomes:

```python
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
```

In `generate_structured` (line 182), add `json_schema: dict | None = None` to signature and pass through:

```python
def generate_structured(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    use_cache: bool = False,
    json_schema: dict | None = None,   # NEW
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_llm_client.py -v 2>&1 | tail -10
```

Expected: 3 PASSes

- [ ] **Step 5: Run full suite to check no regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -15
```

Expected: same pass count as before (627 + 3 new = 630), 0 failures

- [ ] **Step 6: Commit**

```bash
git add ascent/llm/client.py tests/test_llm_client.py
git commit -m "feat: add output_config/json_schema passthrough to llm client"
```

---

## Task 2: Harden `llm_fundamental.py` — structured outputs + amnesia + evidence

**Files:**
- Modify: `ascent/alpha/llm_fundamental.py`
- Modify: `tests/test_llm_fundamental_alpha.py`

### Background

The current system prompt says "You do not know the company name, ticker, or exact dates" but doesn't prevent the model from blending training-data knowledge of specific companies. The response schema is enforced only by prompt instruction, not by the API. Adding `quoted_evidence` to the JSON schema forces the model to pull an actual quote from the provided metrics table rather than fabricating a narrative.

The cache currently stores `{"direction": ..., "confidence": ...}`. The new schema adds `key_trend`, `uncertainty`, and `quoted_evidence` fields. Existing cache entries (missing `quoted_evidence`) remain valid — the scoring path only reads `direction` and `confidence`.

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_llm_fundamental_alpha.py`:

```python
def test_system_prompt_contains_amnesia_instruction():
    """The system prompt must explicitly forbid using training-data knowledge."""
    from ascent.alpha.llm_fundamental import _SYSTEM_PROMPT
    assert "training" in _SYSTEM_PROMPT.lower() or "amnesia" in _SYSTEM_PROMPT.lower() or \
           "do not use" in _SYSTEM_PROMPT.lower(), \
           "System prompt must instruct model not to use training-data company knowledge"


def test_user_template_contains_quoted_evidence_field():
    """The user template must ask for a quoted_evidence field."""
    from ascent.alpha.llm_fundamental import _USER_TEMPLATE
    assert "quoted_evidence" in _USER_TEMPLATE, \
           "User template must include quoted_evidence in JSON schema"


def test_call_llm_uses_json_schema(tmp_path):
    """_call_llm must pass a json_schema to generate_structured."""
    from ascent.alpha.llm_fundamental import _call_llm
    import ascent.alpha.llm_fundamental as mod
    calls = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        calls.append(kwargs)
        return '{"direction": "UP", "confidence": 0.8, "key_trend": "improving", "uncertainty": "rates", "quoted_evidence": "Q0 gross_profitability=0.350"}'

    with patch.object(mod, "generate_structured", mock_generate):
        result = _call_llm("AAPL", "Quarter | ...\n---\nQ0 | 0.350 | 0.01 | 0.05")

    assert result is not None
    assert any("json_schema" in c for c in calls), \
           "_call_llm must pass json_schema= to generate_structured"


def test_quoted_evidence_stored_in_cache(tmp_path):
    """quoted_evidence from LLM response must be stored in the cache entry."""
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    import ascent.alpha.llm_fundamental as mod
    import json

    fund = _make_fundamentals(symbols=["AAPL"])

    def mock_call(symbol, table):
        return {
            "direction": "UP",
            "confidence": 0.8,
            "key_trend": "improving",
            "uncertainty": "macro",
            "quoted_evidence": "Q0 gross_profitability=0.400",
        }

    cache_path = tmp_path / "c.json"
    with patch.object(mod, "CACHE_PATH", cache_path):
        with patch.object(mod, "_call_llm", side_effect=mock_call):
            llm_fundamental_alpha(fund)

    cache = json.loads(cache_path.read_text())
    entries = list(cache.values())
    assert len(entries) == 1
    assert "quoted_evidence" in entries[0], \
           "Cache entry must store quoted_evidence for auditability"
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/python -m pytest tests/test_llm_fundamental_alpha.py::test_system_prompt_contains_amnesia_instruction tests/test_llm_fundamental_alpha.py::test_user_template_contains_quoted_evidence_field tests/test_llm_fundamental_alpha.py::test_call_llm_uses_json_schema tests/test_llm_fundamental_alpha.py::test_quoted_evidence_stored_in_cache -v 2>&1 | tail -15
```

Expected: 4 FAILs

- [ ] **Step 3: Update `_SYSTEM_PROMPT` and `_USER_TEMPLATE`**

Replace lines 26–45 in `ascent/alpha/llm_fundamental.py`:

```python
_SYSTEM_PROMPT = (
    "You are a financial analyst evaluating anonymized company financials. "
    "You do not know the company name, ticker, sector, or exact dates. "
    "Do not use any knowledge about specific companies from your training data — "
    "treat yourself as having amnesia about all individual companies. "
    "Base your analysis ONLY on the numerical data provided in each prompt. "
    "Respond only with valid JSON matching the specified schema. No other text."
)

_LLM_FUNDAMENTAL_SCHEMA = {
    "type": "object",
    "properties": {
        "direction":       {"type": "string", "enum": ["UP", "DOWN", "NEUTRAL"]},
        "confidence":      {"type": "number"},
        "key_trend":       {"type": "string"},
        "uncertainty":     {"type": "string"},
        "quoted_evidence": {
            "type": "string",
            "description": (
                "A direct quote of one or two specific numbers from the provided "
                "metrics table that most support your forecast direction. "
                "Example: 'Q0 gross_profitability=0.412, Q-1=0.389 (+0.023)'. "
                "If no supporting number exists, write 'no clear numerical support'."
            ),
        },
    },
    "required": ["direction", "confidence", "key_trend", "uncertainty", "quoted_evidence"],
    "additionalProperties": False,
}

_USER_TEMPLATE = """Analyze these quarterly financial metrics for an anonymous company.

Financial Data (Q-3 = three quarters ago, Q0 = most recent quarter):
{metrics_table}

Step 1: Identify 3 key trends in revenue growth, gross margin, and asset base (cite specific numbers from the table above).
Step 2: Compute: (a) gross margin change Q-3→Q0, (b) accruals ratio trend, (c) asset growth rate Q-3→Q0.
Step 3: Interpret each economically — improving, stable, or deteriorating, and why.
Step 4: Identify any inflection points in the last 2 quarters.
Step 5: Forecast next-quarter earnings direction. State confidence (0.0–1.0) and primary reason.
Step 6: State the single most important uncertainty in your forecast.

Respond ONLY with a JSON object matching the provided schema. The quoted_evidence field must contain actual numbers copied from the table above."""
```

- [ ] **Step 4: Update `_call_llm` to use `json_schema` and store `quoted_evidence`**

Replace the `_call_llm` function (lines 73–103):

```python
def _call_llm(symbol: str, metrics_table: str) -> Optional[dict]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        user_prompt = _USER_TEMPLATE.format(metrics_table=metrics_table)
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=512,
            temperature=0.2,
            use_cache=True,
            json_schema=_LLM_FUNDAMENTAL_SCHEMA,
        )
        # Structured outputs guarantee valid JSON; parse defensively anyway
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("[LLM Fundamental] No JSON found in response for %s", symbol)
            return None
        parsed = json.loads(raw[start:end])
        direction  = parsed.get("direction", "").upper()
        confidence = float(parsed.get("confidence", 0.0))
        if direction not in ("UP", "DOWN", "NEUTRAL"):
            log.warning("[LLM Fundamental] Invalid direction '%s' for %s", direction, symbol)
            return None
        if not (0.0 <= confidence <= 1.0):
            log.warning("[LLM Fundamental] Confidence out of range %.3f for %s", confidence, symbol)
            return None
        return {
            "direction":       direction,
            "confidence":      confidence,
            "key_trend":       parsed.get("key_trend", ""),
            "uncertainty":     parsed.get("uncertainty", ""),
            "quoted_evidence": parsed.get("quoted_evidence", ""),
        }
    except Exception as exc:
        log.warning("[LLM Fundamental] Call failed for %s: %s", symbol, exc)
        return None
```

- [ ] **Step 5: Run the 4 new tests**

```bash
.venv/bin/python -m pytest tests/test_llm_fundamental_alpha.py::test_system_prompt_contains_amnesia_instruction tests/test_llm_fundamental_alpha.py::test_user_template_contains_quoted_evidence_field tests/test_llm_fundamental_alpha.py::test_call_llm_uses_json_schema tests/test_llm_fundamental_alpha.py::test_quoted_evidence_stored_in_cache -v 2>&1 | tail -15
```

Expected: 4 PASSes

- [ ] **Step 6: Run full `llm_fundamental` test file**

```bash
.venv/bin/python -m pytest tests/test_llm_fundamental_alpha.py -v 2>&1 | tail -20
```

Expected: all existing tests pass plus 4 new ones

- [ ] **Step 7: Run full suite for regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: 0 failures

- [ ] **Step 8: Commit**

```bash
git add ascent/alpha/llm_fundamental.py tests/test_llm_fundamental_alpha.py
git commit -m "feat: add structured outputs + amnesia prompt + quoted_evidence to llm_fundamental"
```

---

## Task 3: Harden `narrative_alpha.py` — structured outputs + grounding instruction

**Files:**
- Modify: `ascent/alpha/narrative_alpha.py`
- Modify: `tests/test_narrative_alpha.py`

### Background

`_compute_shift` sends a summary of two cached analyses (direction, confidence, key_trend) to Haiku and asks it to score the narrative shift. The risk is the model importing outside knowledge about the company implied by the `key_trend` text. Adding a grounding instruction + structured schema closes that gap.

- [ ] **Step 1: Write failing tests**

Append these to `tests/test_narrative_alpha.py`:

```python
def test_system_prompt_contains_grounding_instruction():
    """System prompt must instruct model to reason only from provided summaries."""
    from ascent.alpha.narrative_alpha import _SYSTEM_PROMPT
    lowered = _SYSTEM_PROMPT.lower()
    assert "only" in lowered or "provided" in lowered, \
        "System prompt must restrict model to the provided data"
    assert "training" in lowered or "outside" in lowered or "do not" in lowered, \
        "System prompt must forbid using external knowledge"


def test_compute_shift_uses_json_schema():
    """_compute_shift must pass json_schema to generate_structured."""
    import ascent.alpha.narrative_alpha as mod
    from unittest.mock import patch

    calls = []

    def mock_generate(system_prompt, user_prompt, **kwargs):
        calls.append(kwargs)
        return '{"shift": 0.5, "reason": "direction improved"}'

    current = {"direction": "UP",   "confidence": 0.8, "key_trend": "improving margins"}
    prior   = {"direction": "DOWN", "confidence": 0.6, "key_trend": "declining margins"}

    with patch.object(mod, "generate_structured", mock_generate):
        with patch.object(mod, "_load_narrative_cache", return_value={}):
            with patch.object(mod, "_save_narrative_cache", return_value=None):
                mod._compute_shift("AAPL", current, prior)

    assert any("json_schema" in c for c in calls), \
        "_compute_shift must pass json_schema= to generate_structured"
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/python -m pytest tests/test_narrative_alpha.py::test_system_prompt_contains_grounding_instruction tests/test_narrative_alpha.py::test_compute_shift_uses_json_schema -v 2>&1 | tail -10
```

Expected: 2 FAILs

- [ ] **Step 3: Update `_SYSTEM_PROMPT` and add schema**

Replace lines 31–34 in `ascent/alpha/narrative_alpha.py`:

```python
_SYSTEM_PROMPT = (
    "You are a financial analyst comparing two quarterly investment narrative summaries. "
    "Reason ONLY from the two summaries provided — do not use any outside knowledge "
    "about the company, sector, or market conditions from your training data. "
    "Your entire analysis must be grounded in the direction, confidence, and trend text given. "
    "Respond only with valid JSON. No other text."
)

_NARRATIVE_SHIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "shift":  {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["shift", "reason"],
    "additionalProperties": False,
}
```

- [ ] **Step 4: Update `_compute_shift` to use `json_schema`**

In `_compute_shift` (the `generate_structured` call, ~line 114), add `json_schema=_NARRATIVE_SHIFT_SCHEMA`:

```python
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=_HAIKU_MODEL,
            max_tokens=128,
            temperature=0.1,
            use_cache=False,
            json_schema=_NARRATIVE_SHIFT_SCHEMA,
        )
```

- [ ] **Step 5: Run the 2 new tests**

```bash
.venv/bin/python -m pytest tests/test_narrative_alpha.py::test_system_prompt_contains_grounding_instruction tests/test_narrative_alpha.py::test_compute_shift_uses_json_schema -v 2>&1 | tail -10
```

Expected: 2 PASSes

- [ ] **Step 6: Run full narrative_alpha test file**

```bash
.venv/bin/python -m pytest tests/test_narrative_alpha.py -v 2>&1 | tail -15
```

Expected: all existing tests pass plus 2 new ones

- [ ] **Step 7: Run full suite for regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: 0 failures

- [ ] **Step 8: Commit**

```bash
git add ascent/alpha/narrative_alpha.py tests/test_narrative_alpha.py
git commit -m "feat: add structured outputs + grounding instruction to narrative_alpha"
```

---

## Task 4: Evidence-citation requirement in debate agent prompts

**Files:**
- Modify: `debate/agents.py` (lines 434–441, 465–474, 552–565)
- Test: `tests/test_debate_agents.py` (add 3 tests)

### Background

Bull, Bear, and Devil's Advocate can fabricate statistics. The prompts already instruct agents to use provided context, but they don't explicitly prohibit citing outside-knowledge numbers. Adding a citation rule forces agents to attribute every number they state to the context block they were given. The Quant Sanity agent (pure Python, no LLM) already validates math — this change targets the *assertion* surface, not the validation surface.

The rule to add to each agent's system prompt:

```
EVIDENCE RULE: Every number you cite (return %, position weight, percentile, ratio) must be present in the Portfolio context above. Write [FROM CONTEXT] after any number you quote from it. If you cannot find a number in the context, state the observation qualitatively rather than guessing a value.
```

- [ ] **Step 1: Write failing tests**

Append to `tests/test_debate_agents.py` (create if it does not exist):

```python
# tests/test_debate_agents.py
import pytest


_EVIDENCE_RULE_FRAGMENT = "EVIDENCE RULE"


def _get_bull_system_prompt():
    """Extract bull system prompt by inspecting run_bull_agent source."""
    import inspect
    import debate.agents as mod
    src = inspect.getsource(mod.run_bull_agent)
    # The system_prompt string is in the source
    return src


def _get_bear_system_prompt():
    import inspect
    import debate.agents as mod
    return inspect.getsource(mod.run_bear_agent)


def _get_da_system_prompt():
    import inspect
    import debate.agents as mod
    return inspect.getsource(mod.run_devils_advocate)


def test_bull_system_prompt_contains_evidence_rule():
    src = _get_bull_system_prompt()
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Bull agent system prompt must contain EVIDENCE RULE citation instruction"


def test_bear_system_prompt_contains_evidence_rule():
    src = _get_bear_system_prompt()
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Bear agent system prompt must contain EVIDENCE RULE citation instruction"


def test_devils_advocate_system_prompt_contains_evidence_rule():
    src = _get_da_system_prompt()
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Devil's Advocate system prompt must contain EVIDENCE RULE citation instruction"
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/python -m pytest tests/test_debate_agents.py -v 2>&1 | tail -10
```

Expected: 3 FAILs

- [ ] **Step 3: Add EVIDENCE RULE to bull agent (lines 434–441)**

The `_EVIDENCE_RULE` text to prepend to each agent's system prompt. Add it as a module-level constant just before `run_bull_agent`:

```python
# Append to the end of each debate agent system prompt — do not move above personality text
_EVIDENCE_RULE = (
    " EVIDENCE RULE: Every number you cite (return %, position weight, scenario "
    "percentile, ratio) must appear in the Portfolio context you were given. Write "
    "[FROM CONTEXT] immediately after any number you quote from it. If you cannot "
    "find a number in the context, state the observation qualitatively rather than "
    "inventing a value."
)
```

Place this constant at approximately line 420 (before `run_bull_agent`).

Update the bull system_prompt string to append `_EVIDENCE_RULE`:

```python
    return generate_structured(
        system_prompt=(
            "You are the Bull Analyst at Ascent Capital. Your job is to make "
            "the strongest case FOR executing the proposed trades as-is. Reference specific "
            "positions, the current regime, and momentum signals. Be specific and data-driven. "
            "You have been given historical accuracy data for each debater — use it to "
            "understand where the bear and devil tend to over-warn. Keep your argument under 200 words."
            f"{track_record}"
            f"{_EVIDENCE_RULE}"
        ),
        user_prompt=user_prompt,
        model=DEBATE_MODEL,
        temperature=0.6,
        use_cache=True,
    )
```

- [ ] **Step 4: Add EVIDENCE RULE to bear agent (lines 465–474 and fallback 486–493)**

Primary bear path (tool_completion system_prompt, ~line 465):

```python
            system_prompt=(
                "You are the Bear Analyst at Ascent Capital. Your job is to argue "
                "for REDUCING risk or WAITING. Identify the weakest positions, concentration risks, "
                "regime fragility, or macro headwinds. Use the provided tools to compute "
                "sector concentration, VaR, and momentum BEFORE making claims -- do not guess "
                "at numbers you can look up. Be specific. "
                "You have been given historical accuracy data -- use it to calibrate how often "
                "your past warnings were correct in this regime. Keep under 200 words."
                f"{track_record}"
                f"{_EVIDENCE_RULE}"
            ),
```

Fallback path (~line 486):

```python
            system_prompt=(
                "You are the Bear Analyst at Ascent Capital. Argue for reducing risk. "
                "Be specific. Keep under 200 words."
                f"{_EVIDENCE_RULE}"
            ),
```

- [ ] **Step 5: Add EVIDENCE RULE to devil's advocate (line 552)**

```python
    _da_system_prompt = (
        "You are the Devil's Advocate at Ascent Capital. Your job is to "
        "find the SINGLE most dangerous assumption in the current portfolio construction. "
        "What could go catastrophically wrong that the quant signals would NOT catch? "
        "You have been given Monte Carlo scenario analysis showing worst-case portfolio "
        "impacts. Use these numbers to make a specific, quantified argument. "
        "You also have historical accuracy data — use it to understand when your "
        "past warnings were prescient vs. over-cautious. "
        "Use the available tools to look up sector concentration, VaR, and momentum data "
        "to make quantitative arguments. "
        "Think about: earnings surprises, geopolitical events, liquidity gaps, "
        f"correlation breakdowns. Be specific. Keep under 150 words."
        f"{track_record}"
        f"{_EVIDENCE_RULE}"
    )
```

- [ ] **Step 6: Run the 3 new tests**

```bash
.venv/bin/python -m pytest tests/test_debate_agents.py -v 2>&1 | tail -10
```

Expected: 3 PASSes

- [ ] **Step 7: Run full suite for regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: 0 failures

- [ ] **Step 8: Commit**

```bash
git add debate/agents.py tests/test_debate_agents.py
git commit -m "feat: add evidence citation rule to bull/bear/devil advocate prompts"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Structured outputs for `llm_fundamental` | Task 2 |
| Structured outputs for `narrative_alpha` | Task 3 |
| Amnesia system prompt in `llm_fundamental` | Task 2 |
| Grounding instruction in `narrative_alpha` | Task 3 |
| `quoted_evidence` field to force citation | Task 2 |
| Evidence citation rule in debate agents | Task 4 |
| Central `output_config` support in client | Task 1 |
| Cache stores `quoted_evidence` for audit | Task 2 |

### Type consistency

- `_call_llm` now returns `dict` with keys `direction`, `confidence`, `key_trend`, `uncertainty`, `quoted_evidence`. The scoring path in `llm_fundamental_alpha` only reads `direction` and `confidence` — no change needed downstream.
- `generate_structured` signature gains `json_schema: dict | None = None`. All existing callers pass no `json_schema`, so they're unaffected.
- `chat_completion` signature gains `output_config: dict | None = None`. All existing callers (including `tool_completion`) pass no `output_config`, so they're unaffected.

### Placeholder scan

No TBD/TODO/placeholder steps. All code blocks are complete and runnable.
