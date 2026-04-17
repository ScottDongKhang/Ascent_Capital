# Multi-Turn Debate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-shot sequential debate (each agent argues independently) with a two-round debate where each agent sees all Round 1 arguments before making their Round 2 rebuttal, improving verdict quality without requiring any new data sources.

**Architecture:** Each LLM debate agent gains a `_rebuttal` variant function that accepts the full Round 1 argument dict as additional context. `debate_runner.py` runs Round 1 (existing), then Round 2 (new), then passes both rounds to `run_judge()`. The judge's existing signature is extended with an optional `round2_args` dict. The quant sanity check (pure Python, no opinion) does not participate in Round 2. All changes are additive — Round 1 behavior is unchanged.

**Tech Stack:** Python 3.12, `ascent/llm/client.py` (`generate_structured`, `chat_completion`)

---

## File Structure

- **Modify:** `debate/agents.py` — add `run_bull_rebuttal()`, `run_bear_rebuttal()`, `run_devils_advocate_rebuttal()`, `run_regime_specialist_rebuttal()`
- **Modify:** `debate/judge.py` — accept optional `round2_args: dict` and include Round 2 in synthesis context
- **Modify:** `debate/debate_runner.py` — add Round 2 block after Round 1, pass `round2_args` to judge
- **Create:** `tests/test_multi_turn_debate.py` — unit tests (mocked LLM, no network)

---

### Task 1: Add rebuttal functions to `debate/agents.py`

Each rebuttal function follows the same pattern: takes `portfolio_state` + `round1_args` dict, and returns a shorter (≤100 words) response focused on the strongest counter-argument.

**Files:**
- Modify: `debate/agents.py`
- Test: `tests/test_multi_turn_debate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_turn_debate.py
from unittest.mock import patch
from datetime import date
import pytest


PORTFOLIO_STATE = {
    "date": "2026-04-12",
    "us_regime": "calm_bull",
    "macro_regime": "neutral",
    "n_positions": 5,
    "allocation": {"us_equities": 0.6},
    "weights": {"AAPL": 0.2, "MSFT": 0.2, "CAT": 0.2, "WMT": 0.2, "MRK": 0.2},
}

ROUND1_ARGS = {
    "bull": "Strong momentum in tech and industrials justifies full deployment.",
    "bear": "Valuations are stretched; a 10% drawdown is likely in Q2.",
    "devils_advocate": "The biggest risk is a surprise Fed hike that breaks momentum entirely.",
    "regime_specialist": "Calm bull regime supports full exposure.",
}


def test_bull_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Bull rebuttal text") as mock_gen:
        from debate.agents import run_bull_rebuttal
        result = run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Bull rebuttal text"


def test_bear_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Bear rebuttal text") as mock_gen:
        from debate.agents import run_bear_rebuttal
        result = run_bear_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Bear rebuttal text"


def test_devils_advocate_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Devil rebuttal text") as mock_gen:
        from debate.agents import run_devils_advocate_rebuttal
        result = run_devils_advocate_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Devil rebuttal text"


def test_regime_specialist_rebuttal_calls_llm():
    with patch("debate.agents.generate_structured", return_value="Regime rebuttal text") as mock_gen:
        from debate.agents import run_regime_specialist_rebuttal
        result = run_regime_specialist_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    mock_gen.assert_called_once()
    assert result == "Regime rebuttal text"


def test_rebuttal_prompt_includes_all_round1_arguments():
    """The user prompt passed to the LLM must contain all Round 1 arguments."""
    captured_prompts = {}

    def mock_gen(system_prompt, user_prompt, **kwargs):
        captured_prompts["user"] = user_prompt
        return "rebuttal"

    with patch("debate.agents.generate_structured", side_effect=mock_gen):
        from debate.agents import run_bull_rebuttal
        run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)

    assert "Strong momentum" in captured_prompts["user"]   # bull's own round1
    assert "Valuations are stretched" in captured_prompts["user"]   # bear's round1
    assert "surprise Fed hike" in captured_prompts["user"]   # devil's round1


def test_rebuttal_failure_returns_fallback():
    """If LLM fails, rebuttal returns a non-crashing fallback string."""
    with patch("debate.agents.generate_structured", side_effect=Exception("API down")):
        from debate.agents import run_bull_rebuttal
        result = run_bull_rebuttal(PORTFOLIO_STATE, ROUND1_ARGS)
    assert "failed" in result.lower() or "error" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'run_bull_rebuttal' from 'debate.agents'`

- [ ] **Step 3: Add rebuttal functions to `debate/agents.py`**

Append the following to the end of `debate/agents.py` (after `run_quant_sanity_check`):

```python
# ── Round 2: Rebuttal functions ────────────────────────────────────────────────

def _format_round1_for_rebuttal(round1_args: dict) -> str:
    """Format all Round 1 arguments into a compact block for Round 2 prompts."""
    parts = []
    labels = {
        "bull": "BULL (Round 1)",
        "bear": "BEAR (Round 1)",
        "devils_advocate": "DEVIL'S ADVOCATE (Round 1)",
        "regime_specialist": "REGIME SPECIALIST (Round 1)",
    }
    for key, label in labels.items():
        arg = round1_args.get(key, "")
        if arg and "failed" not in arg.lower()[:20]:
            parts.append(f"{label}:\n{arg}")
    return "\n\n".join(parts)


def run_bull_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """
    Round 2: Bull agent responds to all Round 1 arguments.
    Focuses on the strongest counter to the bear and devil's positions.
    """
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer, respond to the bear and devil's strongest points. "
        "What did they get wrong or overweight?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Bull Analyst at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Rebut the bear's and devil's "
                "strongest concerns concisely. Do not repeat your Round 1 argument — "
                "engage directly with their specific claims. Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
        )
    except Exception as e:
        return f"Bull rebuttal failed: {e}"


def run_bear_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """
    Round 2: Bear agent responds to all Round 1 arguments.
    Focuses on what the bull and regime specialist missed.
    """
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer, respond to the bull and regime specialist's key points. "
        "What critical risk did they dismiss or underweight?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Bear Analyst at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Rebut the bull's and regime "
                "specialist's strongest points. Identify the risk they are still not "
                "taking seriously. Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
        )
    except Exception as e:
        return f"Bear rebuttal failed: {e}"


def run_devils_advocate_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """
    Round 2: Devil's advocate responds to all Round 1 arguments.
    Focuses on the assumption that every other debater is still making.
    """
    context = _build_context(portfolio_state)
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Portfolio context:\n{context}\n\n"
        f"Round 1 arguments from all debaters:\n{round1_block}\n\n"
        "In 75 words or fewer: what dangerous assumption is EVERY Round 1 debater "
        "still making — including the bear?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Devil's Advocate at Ascent Capital in Round 2 of debate. "
                "You have read all Round 1 arguments. Find the shared blind spot — "
                "the assumption that even the bear is making. Make it concrete and "
                "quantified if possible. Under 75 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.7,
        )
    except Exception as e:
        return f"Devil's advocate rebuttal failed: {e}"


def run_regime_specialist_rebuttal(portfolio_state: dict, round1_args: dict) -> str:
    """
    Round 2: Regime specialist responds to the debate.
    Focuses only on regime implications — not individual positions.
    Uses Haiku (same as Round 1).
    """
    regime = str(portfolio_state.get("us_regime", "unknown")).lower()
    round1_block = _format_round1_for_rebuttal(round1_args)
    user_prompt = (
        f"Current regime: {regime.upper()}\n\n"
        f"Round 1 arguments:\n{round1_block}\n\n"
        "In 60 words or fewer: is the regime posture question settled by Round 1, "
        "or is there a regime-specific risk still unaddressed?"
    )
    try:
        return generate_structured(
            system_prompt=(
                "You are the Regime Specialist at Ascent Capital in Round 2 of debate. "
                "You argue only from regime signal — not from individual stock opinions. "
                "Has Round 1 adequately addressed the regime-specific risk? "
                "If not, name what is still missing. Under 60 words."
            ),
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            temperature=0.4,
        )
    except Exception as e:
        return f"Regime specialist rebuttal failed: {e}"
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add debate/agents.py tests/test_multi_turn_debate.py
git commit -m "feat: add Round 2 rebuttal functions for all debate agents"
```

---

### Task 2: Extend `run_judge()` to accept Round 2 arguments

**Files:**
- Modify: `debate/judge.py:17–96`
- Test: `tests/test_multi_turn_debate.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_multi_turn_debate.py`:

```python
def test_judge_accepts_round2_args():
    """Judge synthesizes Round 2 when provided."""
    from unittest.mock import patch
    import json

    round2_args = {
        "bull_rebuttal": "The bear is wrong — momentum is intact.",
        "bear_rebuttal": "The bull ignores valuation risk.",
        "devils_advocate_rebuttal": "Both sides miss the liquidity risk.",
        "regime_specialist_rebuttal": "Regime posture is fine.",
    }

    fake_verdict_json = json.dumps({
        "confidence": 0.75,
        "recommendation": "proceed",
        "key_risks": ["valuation"],
        "reasoning": "Bull case wins.",
    })

    with patch("debate.judge.generate_structured", return_value=fake_verdict_json):
        from debate.judge import run_judge
        verdict = run_judge(
            "bull round 1", "bear round 1", "devil round 1",
            PORTFOLIO_STATE,
            regime_arg="regime round 1",
            quant_check="QUANT SANITY CHECK:\n  ✓ Clean",
            round2_args=round2_args,
        )

    assert verdict["recommendation"] == "proceed"


def test_judge_round2_prompt_includes_rebuttals():
    """The judge's user prompt must include Round 2 rebuttal content."""
    from unittest.mock import patch
    import json

    captured = {}

    def mock_gen(system_prompt, user_prompt, **kwargs):
        captured["user"] = user_prompt
        return json.dumps({
            "confidence": 0.5,
            "recommendation": "reduce_size",
            "key_risks": [],
            "reasoning": "test",
        })

    round2_args = {
        "bull_rebuttal": "UniqueStringBullR2",
        "bear_rebuttal": "UniqueStringBearR2",
        "devils_advocate_rebuttal": "UniqueStringDevilR2",
        "regime_specialist_rebuttal": "UniqueStringRegimeR2",
    }

    with patch("debate.judge.generate_structured", side_effect=mock_gen):
        from debate.judge import run_judge
        run_judge(
            "bull r1", "bear r1", "devil r1",
            PORTFOLIO_STATE,
            round2_args=round2_args,
        )

    assert "UniqueStringBullR2" in captured["user"]
    assert "UniqueStringBearR2" in captured["user"]
    assert "UniqueStringDevilR2" in captured["user"]


def test_judge_works_without_round2_args():
    """run_judge still works when round2_args is not passed (backward compatible)."""
    from unittest.mock import patch
    import json

    fake_verdict_json = json.dumps({
        "confidence": 0.6,
        "recommendation": "proceed",
        "key_risks": [],
        "reasoning": "ok",
    })

    with patch("debate.judge.generate_structured", return_value=fake_verdict_json):
        from debate.judge import run_judge
        verdict = run_judge("bull", "bear", "devil", PORTFOLIO_STATE)

    assert verdict["recommendation"] == "proceed"
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py::test_judge_accepts_round2_args tests/test_multi_turn_debate.py::test_judge_round2_prompt_includes_rebuttals -v
```

Expected: FAIL — `run_judge()` doesn't accept `round2_args`

- [ ] **Step 3: Extend `run_judge()` in `debate/judge.py`**

Change the function signature from:

```python
def run_judge(
    bull_argument: str,
    bear_argument: str,
    devils_argument: str,
    portfolio_state: dict,
    regime_arg: str = "",
    quant_check: str = "",
) -> dict:
```

to:

```python
def run_judge(
    bull_argument: str,
    bear_argument: str,
    devils_argument: str,
    portfolio_state: dict,
    regime_arg: str = "",
    quant_check: str = "",
    round2_args: dict = None,
) -> dict:
```

Then, in `run_judge()`, after the existing `quant_block` line and before the `context = (...)` block, add:

```python
    round2_args = round2_args or {}
```

In the `context = (...)` f-string, add a Round 2 section after the quant_check block:

```python
        + (
            "\n\nROUND 2 — REBUTTALS (agents responding to each other):\n"
            + (f"BULL REBUTTAL:\n{round2_args['bull_rebuttal']}\n\n" if round2_args.get("bull_rebuttal") else "")
            + (f"BEAR REBUTTAL:\n{round2_args['bear_rebuttal']}\n\n" if round2_args.get("bear_rebuttal") else "")
            + (f"DEVIL'S ADVOCATE REBUTTAL:\n{round2_args['devils_advocate_rebuttal']}\n\n" if round2_args.get("devils_advocate_rebuttal") else "")
            + (f"REGIME SPECIALIST REBUTTAL:\n{round2_args['regime_specialist_rebuttal']}\n\n" if round2_args.get("regime_specialist_rebuttal") else "")
            if round2_args else ""
        )
```

Also update the judge's system prompt to mention Round 2. Change the sentence:

```
"Synthesize ALL arguments — bull, bear, devil's advocate, regime specialist, "
"and quant sanity check — into a single verdict.\n\n"
```

to:

```
"Synthesize ALL arguments — Round 1 (bull, bear, devil's advocate, regime specialist, "
"quant sanity check) AND Round 2 rebuttals where agents engaged with each other — "
"into a single verdict. Round 2 arguments are more focused; weight them accordingly.\n\n"
```

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add debate/judge.py tests/test_multi_turn_debate.py
git commit -m "feat: extend run_judge() to accept and synthesize Round 2 rebuttal arguments"
```

---

### Task 3: Wire Round 2 into `debate_runner.py`

**Files:**
- Modify: `debate/debate_runner.py:154–201` — add Round 2 block after Round 1, pass `round2_args` to judge

- [ ] **Step 1: Write the failing test**

Add to `tests/test_multi_turn_debate.py`:

```python
def test_debate_runner_executes_two_rounds(monkeypatch):
    """debate_runner calls both Round 1 and Round 2 agent functions."""
    from unittest.mock import patch, call
    import json

    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    verdict_data = {
        "confidence": 0.7,
        "recommendation": "proceed",
        "key_risks": [],
        "reasoning": "ok",
    }

    patches = {
        "debate.debate_runner.score_pending_verdicts": 0,
        "debate.debate_runner.run_pending_debriefs": 0,
        "debate.debate_runner.detect_blind_spots": None,
        "debate.debate_runner.load_blind_spot_context": "",
        "debate.debate_runner.run_all_scenarios": [],
        "debate.debate_runner.run_bull_agent": "bull_r1",
        "debate.debate_runner.run_bear_agent": "bear_r1",
        "debate.debate_runner.run_devils_advocate": "devil_r1",
        "debate.debate_runner.run_regime_specialist": "regime_r1",
        "debate.debate_runner.run_quant_sanity_check": "quant_r1",
        "debate.debate_runner.run_bull_rebuttal": "bull_r2",
        "debate.debate_runner.run_bear_rebuttal": "bear_r2",
        "debate.debate_runner.run_devils_advocate_rebuttal": "devil_r2",
        "debate.debate_runner.run_regime_specialist_rebuttal": "regime_r2",
        "debate.debate_runner.run_judge": verdict_data,
    }

    with patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull_r1"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear_r1"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil_r1"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime_r1"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant_r1"), \
         patch("debate.debate_runner.run_bull_rebuttal", return_value="bull_r2") as mock_bull_r2, \
         patch("debate.debate_runner.run_bear_rebuttal", return_value="bear_r2") as mock_bear_r2, \
         patch("debate.debate_runner.run_devils_advocate_rebuttal", return_value="devil_r2") as mock_devil_r2, \
         patch("debate.debate_runner.run_regime_specialist_rebuttal", return_value="regime_r2") as mock_regime_r2, \
         patch("debate.debate_runner.run_judge", return_value=verdict_data) as mock_judge:
        from debate.debate_runner import run_debate
        run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    # All Round 2 functions must have been called
    mock_bull_r2.assert_called_once()
    mock_bear_r2.assert_called_once()
    mock_devil_r2.assert_called_once()
    mock_regime_r2.assert_called_once()

    # Judge must have received round2_args
    judge_call_kwargs = mock_judge.call_args
    assert judge_call_kwargs is not None
    # round2_args is a keyword argument
    _, kwargs = judge_call_kwargs
    round2 = kwargs.get("round2_args", {})
    assert round2.get("bull_rebuttal") == "bull_r2"
    assert round2.get("bear_rebuttal") == "bear_r2"
```

- [ ] **Step 2: Run failing test**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py::test_debate_runner_executes_two_rounds -v
```

Expected: FAIL — `run_bull_rebuttal` not imported in `debate_runner.py`

- [ ] **Step 3: Add Round 2 block to `debate_runner.py`**

Add the new imports at the top of `debate/debate_runner.py` (line ~23):

```python
from debate.agents import (
    run_bull_agent, run_bear_agent, run_devils_advocate,
    run_regime_specialist, run_quant_sanity_check,
    run_bull_rebuttal, run_bear_rebuttal,
    run_devils_advocate_rebuttal, run_regime_specialist_rebuttal,
)
```

(Replace the existing `from debate.agents import ...` line.)

Then insert this Round 2 block in `debate_runner.py` **after** the quant sanity check block and **before** the `# Judge synthesizes` comment (approximately line 200):

```python
    # ── Round 2: Agents rebut each other ──────────────────────────────────────
    round1_args = {
        "bull": bull,
        "bear": bear,
        "devils_advocate": devil,
        "regime_specialist": regime_arg,
    }

    print("[Debate] Round 2 — agents rebut each other...")

    try:
        bull_r2 = run_bull_rebuttal(portfolio_state, round1_args)
        print(f"[Debate] Bull R2: {bull_r2[:80]}...")
    except Exception as e:
        bull_r2 = f"Bull rebuttal failed: {e}"

    try:
        bear_r2 = run_bear_rebuttal(portfolio_state, round1_args)
        print(f"[Debate] Bear R2: {bear_r2[:80]}...")
    except Exception as e:
        bear_r2 = f"Bear rebuttal failed: {e}"

    try:
        devil_r2 = run_devils_advocate_rebuttal(portfolio_state, round1_args)
        print(f"[Debate] Devil R2: {devil_r2[:80]}...")
    except Exception as e:
        devil_r2 = f"Devil's advocate rebuttal failed: {e}"

    try:
        regime_r2 = run_regime_specialist_rebuttal(portfolio_state, round1_args)
        print(f"[Debate] Regime R2: {regime_r2[:80]}...")
    except Exception as e:
        regime_r2 = f"Regime specialist rebuttal failed: {e}"

    round2_args = {
        "bull_rebuttal": bull_r2,
        "bear_rebuttal": bear_r2,
        "devils_advocate_rebuttal": devil_r2,
        "regime_specialist_rebuttal": regime_r2,
    }
```

Then update the `run_judge(...)` call to pass `round2_args`:

```python
    verdict = run_judge(
        bull, bear, devil, portfolio_state,
        regime_arg=regime_arg,
        quant_check=quant_check,
        round2_args=round2_args,
    )
```

Also extend the `record["arguments"]` dict to include Round 2:

```python
        "arguments": {
            "bull":                       bull,
            "bear":                       bear,
            "devils_advocate":            devil,
            "regime_specialist":          regime_arg,
            "quant_sanity":               quant_check,
            "bull_rebuttal":              bull_r2,
            "bear_rebuttal":              bear_r2,
            "devils_advocate_rebuttal":   devil_r2,
            "regime_specialist_rebuttal": regime_r2,
        },
```

- [ ] **Step 4: Run all tests**

```bash
.venv/bin/pytest tests/test_multi_turn_debate.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add debate/debate_runner.py tests/test_multi_turn_debate.py
git commit -m "feat: wire two-round debate into debate_runner — agents now rebut each other before judge synthesizes"
```

---

## Self-Review

**Spec coverage:**
- Round 1: unchanged (existing behavior) ✓
- Round 2 rebuttal functions for all 4 LLM agents: Task 1 ✓
- Quant sanity check does not participate in Round 2: correct — no `run_quant_sanity_rebuttal` added ✓
- Judge receives and synthesizes Round 2: Task 2 ✓
- Backward compatibility (`run_judge()` works without `round2_args`): Task 2 test `test_judge_works_without_round2_args` ✓
- Round 2 saved in verdict record: Task 3 (`record["arguments"]` extension) ✓
- Non-fatal failure handling: each rebuttal call wrapped in try/except ✓

**Placeholder scan:** None found.

**Type consistency:**
- `round2_args` is a `dict` with keys `bull_rebuttal`, `bear_rebuttal`, `devils_advocate_rebuttal`, `regime_specialist_rebuttal` — used consistently in Task 1 (runner builds it), Task 2 (judge reads it), Task 3 (runner passes it).
- `run_bull_rebuttal(portfolio_state: dict, round1_args: dict) -> str` — consistent in agents.py and all import/call sites.
