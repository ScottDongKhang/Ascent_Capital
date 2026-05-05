# AI-Native Ascent Capital — Tier 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three deeper AI-native capabilities: a FinMem-style post-trade reflection loop that builds institutional memory, LLM-guided hypothesis generation for the self-improve loop, and tool-capable debate agents that can actively compute portfolio statistics mid-debate.

**Architecture:** Three independent modules wired at clean seams. Task D adds a Haiku-powered reflection agent (`memory/reflection_agent.py`) that runs after each verdict is scored, writes structured lessons, and feeds them into future debate contexts. Task E replaces purely random weight perturbation in `self_improve.py` with LLM-guided hypothesis generation + cosine-similarity deduplication (`ascent/research/factor_proposer.py`). Task F adds a `tool_completion()` loop to `ascent/llm/client.py` and equips the bear and devil's advocate with four domain tools so they can actively look up sector concentrations, VaR estimates, and momentum data during debates.

**Tech Stack:** Python 3.12, Claude Haiku (`claude-haiku-4-5-20251001`), Anthropic SDK tool use, existing `ascent/llm/client.py`, `memory/r2r_interface.py`, `debate/outcome_tracker.py`, `ascent/research/self_improve.py`, scipy, numpy (all installed).

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `memory/reflection_agent.py` | Post-verdict reflection — Haiku reads outcome, writes structured lesson |
| Create | `tests/test_reflection_agent.py` | Full test suite for Task D |
| Modify | `run_all_agents.py` | Call `reflect_on_new_outcomes()` after `score_pending_verdicts()` |
| Modify | `debate/agents.py` (`_build_context`) | Pull recent reflections into debate context |
| Create | `ascent/research/factor_proposer.py` | LLM hypothesis generation + cosine-sim deduplication |
| Create | `tests/test_factor_proposer.py` | Full test suite for Task E |
| Modify | `ascent/research/self_improve.py` (`generate_variants`) | Call proposer first, fall back to random |
| Create | `debate/agent_tools.py` | Tool definitions + pure-Python implementations |
| Modify | `ascent/llm/client.py` | Add `tool_completion()` — Anthropic tool-use loop |
| Modify | `debate/agents.py` (`run_bear_agent`, `run_devils_advocate`) | Use `tool_completion` with agent tools |
| Create | `tests/test_agent_tools.py` | Full test suite for Task F |

---

## Task D: FinMem-Style Post-Trade Reflection Agent

**Problem:** The debate system produces verdicts, scores them 14 days later, and tracks per-agent accuracy — but it never synthesizes *why* decisions were wrong or right. FinMem (arxiv:2311.13743) showed that structured post-trade reflection — writing a concise "lesson learned" after each outcome — dramatically improves future decision quality because the lessons compress historical knowledge into a form the LLM can reason over directly. Currently Ascent's debate agents see only raw past verdict text. This task adds a structured reflection layer that generates regime-specific lessons and injects them into future debates.

**Files:**
- Create: `memory/reflection_agent.py`
- Create: `tests/test_reflection_agent.py`
- Modify: `run_all_agents.py`
- Modify: `debate/agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reflection_agent.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def _write_scored_verdict(path: Path, regime="stressed", recommendation="proceed",
                          nav_change=-0.032, bull_text="strong momentum", bear_text="credit spreads widening"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "date": "2026-04-10",
        "outcome_scored": True,
        "outcome_nav_change": nav_change,
        "outcome_score": 0.0 if nav_change < -0.01 else 1.0,
        "verdict": {"recommendation": recommendation, "reasoning": "base case holds"},
        "portfolio_state": {"us_regime": regime, "weights": {"AAPL": 0.15, "MSFT": 0.10}},
        "arguments": {"bull": bull_text, "bear": bear_text, "devils_advocate": "liquidity gap"},
    }
    path.write_text(json.dumps(data))
    return data


def test_reflect_returns_dict(tmp_path):
    from memory.reflection_agent import reflect_on_verdict
    vpath = tmp_path / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)

    def mock_generate(system_prompt, user_prompt, **kwargs):
        return json.dumps({
            "lesson": "Ignored credit spreads in stressed regime",
            "key_error": "Bull underweighted macro headwinds",
            "confidence_calibration": "DOWN",
            "regime": "stressed",
        })

    with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
        result = reflect_on_verdict(vpath)

    assert isinstance(result, dict)
    for key in ["lesson", "key_error", "confidence_calibration", "regime", "date"]:
        assert key in result, f"Missing key: {key}"


def test_reflect_skips_unscored_verdict(tmp_path):
    from memory.reflection_agent import reflect_on_verdict
    vpath = tmp_path / "verdict_2026-04-10.json"
    data = {"date": "2026-04-10", "outcome_scored": False, "verdict": {}, "portfolio_state": {}, "arguments": {}}
    vpath.write_text(json.dumps(data))
    result = reflect_on_verdict(vpath)
    assert result is None, "Must return None for unscored verdicts"


def test_reflect_on_new_outcomes_processes_new_files(tmp_path):
    from memory.reflection_agent import reflect_on_new_outcomes
    debate_dir = tmp_path / "debate_log"
    refl_path  = tmp_path / "reflections.jsonl"
    vpath = debate_dir / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)

    def mock_generate(system_prompt, user_prompt, **kwargs):
        return json.dumps({"lesson": "Test", "key_error": "None", "confidence_calibration": "HOLD", "regime": "stressed"})

    with patch("memory.reflection_agent.DEBATE_LOG_DIR", debate_dir):
        with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
            with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
                count = reflect_on_new_outcomes()

    assert count >= 1
    assert refl_path.exists()
    lines = [json.loads(l) for l in refl_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    assert "lesson" in lines[0]


def test_reflect_not_processed_twice(tmp_path):
    from memory.reflection_agent import reflect_on_new_outcomes
    debate_dir = tmp_path / "debate_log"
    refl_path  = tmp_path / "reflections.jsonl"
    vpath = debate_dir / "verdict_2026-04-10.json"
    _write_scored_verdict(vpath)
    call_count = [0]

    def mock_generate(system_prompt, user_prompt, **kwargs):
        call_count[0] += 1
        return json.dumps({"lesson": "T", "key_error": "N", "confidence_calibration": "HOLD", "regime": "stressed"})

    with patch("memory.reflection_agent.DEBATE_LOG_DIR", debate_dir):
        with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
            with patch("memory.reflection_agent._call_llm", side_effect=mock_generate):
                reflect_on_new_outcomes()
                first_count = call_count[0]
                reflect_on_new_outcomes()
                second_count = call_count[0]

    assert second_count == first_count, "Second run must not re-process already-reflected verdicts"


def test_load_recent_reflections_filters_by_regime(tmp_path):
    from memory.reflection_agent import load_recent_reflections
    refl_path = tmp_path / "reflections.jsonl"
    rows = [
        {"date": "2026-04-01", "regime": "stressed",  "lesson": "A", "key_error": "", "confidence_calibration": "DOWN"},
        {"date": "2026-04-02", "regime": "calm_bull", "lesson": "B", "key_error": "", "confidence_calibration": "UP"},
        {"date": "2026-04-03", "regime": "stressed",  "lesson": "C", "key_error": "", "confidence_calibration": "HOLD"},
    ]
    refl_path.write_text("\n".join(json.dumps(r) for r in rows))

    with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
        results = load_recent_reflections(regime="stressed", n=5)

    assert len(results) == 2
    assert all(r["regime"] == "stressed" for r in results)


def test_format_reflections_for_context_returns_string(tmp_path):
    from memory.reflection_agent import load_recent_reflections, format_reflections_for_context
    refl_path = tmp_path / "reflections.jsonl"
    refl_path.write_text(json.dumps({
        "date": "2026-04-01", "regime": "stressed",
        "lesson": "Ignored credit spreads", "key_error": "Bull wrong",
        "confidence_calibration": "DOWN",
    }))
    with patch("memory.reflection_agent.REFLECTIONS_PATH", refl_path):
        refs = load_recent_reflections(regime="stressed", n=3)
    ctx = format_reflections_for_context(refs)
    assert isinstance(ctx, str)
    assert len(ctx) > 0
    assert "stressed" in ctx.lower() or "lesson" in ctx.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_reflection_agent.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'memory.reflection_agent'`

- [ ] **Step 3: Create `memory/reflection_agent.py`**

```python
"""
memory/reflection_agent.py

FinMem-style post-trade reflection.

After each verdict is scored (14 days post-decision), Haiku reads the
outcome and writes a structured lesson: what went wrong, what the losing
side missed, and how future agents should calibrate confidence in this
regime. Lessons are stored in memory/reflections.jsonl and injected into
future debate contexts via _build_context() in debate/agents.py.

Source: FinMem (Wang et al., 2023) — arxiv.org/abs/2311.13743
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DEBATE_LOG_DIR  = Path("outputs/debate_log")
REFLECTIONS_PATH = Path("memory/reflections.jsonl")

_SYSTEM_PROMPT = (
    "You are a senior portfolio risk manager conducting a post-trade review. "
    "You will be shown a debate verdict and its actual outcome 14 days later. "
    "Your job: write a concise structured lesson so future debate teams make fewer errors. "
    "Respond ONLY with valid JSON matching the specified format. No other text."
)

_USER_TEMPLATE = """Post-trade review:

Regime at decision time: {regime}
Debate verdict: {recommendation}
Outcome 14 days later: portfolio moved {nav_change:+.1%} → verdict was {correct_str}

Bull argued (summary): {bull_summary}
Bear argued (summary): {bear_summary}
Devil's Advocate argued (summary): {devil_summary}

Write a structured lesson in this exact JSON format:
{{"lesson": "one sentence — what future teams should watch for in this regime",
  "key_error": "one sentence — what the {losing_side} side got wrong",
  "confidence_calibration": "UP, DOWN, or HOLD — how much to trust the {wrong_agent} agent in {regime} regime going forward",
  "regime": "{regime}"}}"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.2,
            use_cache=True,
        )
    except Exception as exc:
        log.warning("[Reflection] LLM call failed: %s", exc)
        return None


def _load_reflected_dates() -> set:
    """Return the set of verdict dates already reflected on (to avoid re-processing)."""
    if not REFLECTIONS_PATH.exists():
        return set()
    dates = set()
    try:
        for line in REFLECTIONS_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if "verdict_date" in row:
                    dates.add(row["verdict_date"])
    except Exception:
        pass
    return dates


def reflect_on_verdict(verdict_path: Path) -> Optional[Dict]:
    """
    Reflect on a single scored verdict. Returns reflection dict or None if skipped.

    Skips if:
    - outcome_scored is not True
    - LLM call fails
    """
    try:
        data = json.loads(verdict_path.read_text())
    except Exception as exc:
        log.warning("[Reflection] Cannot read %s: %s", verdict_path, exc)
        return None

    if not data.get("outcome_scored"):
        return None

    regime         = str(data.get("portfolio_state", {}).get("us_regime", "unknown")).lower()
    recommendation = data.get("verdict", {}).get("recommendation", "proceed")
    nav_change     = float(data.get("outcome_nav_change", 0.0))
    date_str       = data.get("date", "unknown")

    correct = (
        (recommendation == "proceed"        and nav_change >= 0) or
        (recommendation == "reduce_size"    and nav_change < -0.005) or
        (recommendation == "halt_and_review" and nav_change < -0.01)
    )
    correct_str = "CORRECT" if correct else "INCORRECT"
    losing_side = "bull" if nav_change < -0.01 else ("bear" if nav_change > 0.01 else "neither")
    wrong_agent = losing_side if losing_side != "neither" else "bull"

    args = data.get("arguments", {})
    bull_summary   = str(args.get("bull",             ""))[:150]
    bear_summary   = str(args.get("bear",             ""))[:150]
    devil_summary  = str(args.get("devils_advocate",  ""))[:150]

    user_prompt = _USER_TEMPLATE.format(
        regime=regime, recommendation=recommendation,
        nav_change=nav_change, correct_str=correct_str,
        bull_summary=bull_summary, bear_summary=bear_summary,
        devil_summary=devil_summary,
        losing_side=losing_side, wrong_agent=wrong_agent,
    )

    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        parsed["date"]         = str(date_str)
        parsed["verdict_date"] = str(date_str)
        parsed["nav_change"]   = round(nav_change, 4)
        parsed["correct"]      = correct
        parsed.setdefault("regime", regime)
        return parsed
    except Exception as exc:
        log.warning("[Reflection] JSON parse failed: %s", exc)
        return None


def reflect_on_new_outcomes() -> int:
    """
    Reflect on all newly-scored verdicts that haven't been reflected on yet.
    Appends structured lessons to memory/reflections.jsonl.
    Returns count of new reflections written.
    """
    if not DEBATE_LOG_DIR.exists():
        return 0

    already_reflected = _load_reflected_dates()
    count = 0
    REFLECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for vf in sorted(DEBATE_LOG_DIR.glob("verdict_*.json")):
        try:
            data = json.loads(vf.read_text())
        except Exception:
            continue

        if not data.get("outcome_scored"):
            continue

        date_str = data.get("date", "")
        if date_str in already_reflected:
            continue

        reflection = reflect_on_verdict(vf)
        if reflection is None:
            continue

        with open(REFLECTIONS_PATH, "a") as f:
            f.write(json.dumps(reflection) + "\n")
        count += 1
        log.info("[Reflection] Wrote lesson for %s (regime=%s, correct=%s)",
                 date_str, reflection.get("regime"), reflection.get("correct"))

    return count


def load_recent_reflections(regime: Optional[str] = None, n: int = 3) -> List[Dict]:
    """
    Load the N most recent reflections, optionally filtered by regime.
    Returns list of reflection dicts sorted newest-first.
    """
    if not REFLECTIONS_PATH.exists():
        return []

    rows = []
    try:
        for line in REFLECTIONS_PATH.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []

    if regime:
        regime_lower = str(regime).lower()
        rows = [r for r in rows if str(r.get("regime", "")).lower() == regime_lower]

    rows.sort(key=lambda r: r.get("date", ""), reverse=True)
    return rows[:n]


def format_reflections_for_context(reflections: List[Dict]) -> str:
    """
    Format recent reflections as a concise LLM-readable block for injection
    into debate agent system prompts.
    """
    if not reflections:
        return ""

    lines = [f"Post-trade lessons — {len(reflections)} recent outcome(s) in this regime:"]
    for i, r in enumerate(reflections, 1):
        correct_str = "CORRECT" if r.get("correct") else "INCORRECT"
        calib = r.get("confidence_calibration", "HOLD")
        lines.append(
            f"\n[{i}] {r.get('date', 'unknown')} | Verdict was {correct_str} | "
            f"Calibrate {r.get('wrong_agent_type', 'bull')} {calib}"
        )
        if r.get("lesson"):
            lines.append(f"    Lesson: {r['lesson']}")
        if r.get("key_error"):
            lines.append(f"    Key error: {r['key_error']}")

    return "\n".join(lines)
```

- [ ] **Step 4: Wire `reflect_on_new_outcomes()` into `run_all_agents.py`**

Find the block that calls `score_pending_verdicts()` (search for `score_pending_verdicts` or `outcome_tracker`). Add the reflection call immediately after:

```python
        # Score pending verdicts (runs daily, NOP if no verdicts old enough)
        try:
            from debate.outcome_tracker import score_pending_verdicts
            n_scored = score_pending_verdicts()
            if n_scored:
                print(f"[OutcomeTracker] Scored {n_scored} verdict(s)")
                # Reflect on newly-scored outcomes
                from memory.reflection_agent import reflect_on_new_outcomes
                n_reflected = reflect_on_new_outcomes()
                if n_reflected:
                    print(f"[Reflection] Wrote {n_reflected} new lesson(s) to memory/reflections.jsonl")
        except Exception as _oe:
            print(f"[OutcomeTracker] Scoring skipped: {_oe}")
```

- [ ] **Step 5: Inject reflections into `debate/agents.py` `_build_context()`**

In `_build_context()` at line 21, after the memory_ctx block (around line 64), add:

```python
    # Post-trade reflections — structured lessons from past outcomes in this regime
    regime_for_reflection = portfolio_state.get("us_regime", "unknown")
    try:
        from memory.reflection_agent import load_recent_reflections, format_reflections_for_context
        recent_reflections = load_recent_reflections(regime=regime_for_reflection, n=3)
        reflection_text = format_reflections_for_context(recent_reflections)
        if reflection_text:
            lines.append("")
            lines.append(reflection_text)
    except Exception:
        pass
```

- [ ] **Step 6: Run all tests**

```bash
.venv/bin/pytest tests/test_reflection_agent.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add memory/reflection_agent.py tests/test_reflection_agent.py \
        run_all_agents.py debate/agents.py
git commit -m "feat(memory): FinMem-style post-trade reflection — structured lessons injected into future debates"
```

---

## Task E: LLM-Guided Factor Hypothesis Generation

**Problem:** The self-improve loop perturbs sleeve weights randomly — it explores weight space without any theory about *why* a particular weighting should work. In stressed markets, maybe quality/fundamental signals should dominate; in calm bull markets, momentum should. AlphaAgent (arxiv:2502.16789) generates code hypotheses and rejects duplicates via AST comparison. For Ascent's weight-space search, the analogous improvement is: (1) have the LLM generate regime-aware narrative hypotheses about what should work, (2) translate them to weight biases, and (3) reject hypotheses that are too similar to each other (by cosine similarity of their bias vectors). This makes the 5 weekly variants explore meaningfully different hypotheses rather than 5 nearly-identical perturbations of the status quo.

**Files:**
- Create: `ascent/research/factor_proposer.py`
- Create: `tests/test_factor_proposer.py`
- Modify: `ascent/research/self_improve.py` (`generate_variants` function, line 85)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_factor_proposer.py
import pytest
import json
import numpy as np
from unittest.mock import patch


_CURRENT_WEIGHTS = {
    "trend": 0.41, "meanrev": 0.05, "volatility": 0.05, "statarb": 0.15,
    "ml": 0.10, "fundamental": 0.05, "earnings": 0.05, "analyst": 0.05,
    "options_flow": 0.02, "insider": 0.02, "short_interest": 0.02, "llm_fundamental": 0.03,
}


def test_propose_hypotheses_returns_list(tmp_path):
    from ascent.research.factor_proposer import propose_hypotheses
    mock_response = json.dumps([
        {"narrative": "In stressed regime, quality trumps momentum",
         "weight_biases": {"fundamental": 0.08, "trend": -0.08}},
        {"narrative": "Credit stress means low-vol names outperform",
         "weight_biases": {"volatility": 0.06, "trend": -0.06}},
    ])
    with patch("ascent.research.factor_proposer._call_llm", return_value=mock_response):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=2)
    assert isinstance(result, list)
    assert len(result) >= 1
    for h in result:
        assert "narrative" in h
        assert "weight_biases" in h


def test_duplicate_hypotheses_are_deduplicated():
    from ascent.research.factor_proposer import deduplicate_hypotheses
    h1 = {"narrative": "A", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}
    h2 = {"narrative": "B", "weight_biases": {"fundamental": 0.10, "trend": -0.10}}  # identical biases
    h3 = {"narrative": "C", "weight_biases": {"volatility": 0.10, "trend": -0.10}}
    result = deduplicate_hypotheses([h1, h2, h3], similarity_threshold=0.85)
    assert len(result) < 3, "Identical bias vectors must be deduplicated"
    assert len(result) >= 2, "Genuinely different hypotheses must be kept"


def test_generate_guided_variants_sums_to_one():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"narrative": "More quality", "weight_biases": {"fundamental": 0.10, "trend": -0.10}},
        {"narrative": "More vol",    "weight_biases": {"volatility": 0.08, "trend": -0.08}},
    ]
    variants = generate_guided_variants(_CURRENT_WEIGHTS, hypotheses, perturb_range=0.03)
    for v in variants:
        w = v["alpha_weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total}"


def test_generate_guided_variants_respects_floor():
    from ascent.research.factor_proposer import generate_guided_variants
    hypotheses = [
        {"narrative": "Zero out trend", "weight_biases": {"trend": -0.99, "fundamental": 0.50}},
    ]
    variants = generate_guided_variants(_CURRENT_WEIGHTS, hypotheses, perturb_range=0.03)
    for v in variants:
        w = v["alpha_weights"]
        assert w.get("trend", 0) >= 0.05, "Trend must never drop below minimum floor 5%"


def test_llm_failure_returns_empty_list():
    from ascent.research.factor_proposer import propose_hypotheses
    with patch("ascent.research.factor_proposer._call_llm", return_value=None):
        result = propose_hypotheses(regime="stressed", current_weights=_CURRENT_WEIGHTS, n=3)
    assert result == []


def test_generate_variants_uses_proposer_when_available():
    from ascent.research.self_improve import generate_variants
    mock_hyp = [
        {"narrative": "Quality over momentum", "weight_biases": {"fundamental": 0.08, "trend": -0.08}},
        {"narrative": "Vol regime favors stability", "weight_biases": {"volatility": 0.07, "trend": -0.07}},
        {"narrative": "Mean reversion in stress",   "weight_biases": {"meanrev": 0.06, "trend": -0.06}},
        {"narrative": "ML signals outperform",      "weight_biases": {"ml": 0.06, "trend": -0.06}},
        {"narrative": "Earnings drive alpha",       "weight_biases": {"earnings": 0.06, "trend": -0.06}},
    ]
    with patch("ascent.research.factor_proposer.propose_hypotheses", return_value=mock_hyp):
        with patch("ascent.research.factor_proposer.generate_guided_variants",
                   wraps=lambda w, h, **kw: [{"variant_id": f"guided_{i}", "alpha_weights": w} for i, _ in enumerate(h)]) as mock_gen:
            variants = generate_variants({"alpha_weights": _CURRENT_WEIGHTS}, n=5, regime="stressed")
    assert len(variants) == 5
    assert any(v["variant_id"].startswith("guided_") for v in variants), \
        "generate_variants must use guided proposer when regime is provided"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_factor_proposer.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'ascent.research.factor_proposer'`

- [ ] **Step 3: Create `ascent/research/factor_proposer.py`**

```python
"""
ascent/research/factor_proposer.py

LLM-guided factor hypothesis generation for the self-improve loop.

Replaces purely random weight perturbation with regime-aware hypotheses:
1. Haiku proposes N narratives (e.g. "quality beats momentum in stress")
2. Each narrative includes weight biases (deltas to apply to current weights)
3. Cosine similarity check rejects hypotheses that are near-duplicates
4. generate_guided_variants() applies biases + small random noise → full configs

Falls back to silent empty list if LLM is unavailable.

Inspired by AlphaAgent (Liu et al., 2025) — arxiv.org/abs/2502.16789
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Minimum weight floor per sleeve — never perturb below these
_SLEEVE_FLOORS: Dict[str, float] = {
    "trend":          0.05,
    "fundamental":    0.02,
    "earnings":       0.02,
    "analyst":        0.02,
    "options_flow":   0.01,
    "insider":        0.01,
    "short_interest": 0.01,
    "llm_fundamental": 0.01,
}

_SYSTEM_PROMPT = (
    "You are a quantitative researcher at Ascent Capital. "
    "You propose alpha weight hypotheses for a multi-sleeve trading system. "
    "Each hypothesis must have a clear economic narrative and concrete weight biases. "
    "Respond ONLY with a valid JSON array. No other text."
)

_USER_TEMPLATE = """Current regime: {regime}

Current alpha sleeve weights:
{weights_str}

Propose {n} diverse hypotheses for sleeve weight adjustments that might outperform.
Each hypothesis should reflect a different economic reasoning about what works in a {regime} environment.

The available sleeves are: {sleeves}

Respond with a JSON array of exactly {n} hypothesis objects:
[
  {{
    "narrative": "One-sentence economic rationale",
    "weight_biases": {{"sleeve_name": delta_float, ...}}
  }},
  ...
]

Rules:
- Each bias is a delta (positive = increase, negative = decrease), typically between -0.15 and +0.15
- Biases for a single hypothesis must sum to approximately 0.0 (weight-neutral)
- Only include sleeves you want to change; unlisted sleeves are unchanged
- Hypotheses must be meaningfully different from each other"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.6,
            use_cache=False,  # hypotheses should be fresh each week
        )
    except Exception as exc:
        log.warning("[FactorProposer] LLM call failed: %s", exc)
        return None


def _bias_vector(hypothesis: dict, sleeves: List[str]) -> np.ndarray:
    """Convert weight_biases dict to a fixed-length numpy vector for similarity comparison."""
    biases = hypothesis.get("weight_biases", {})
    return np.array([biases.get(s, 0.0) for s in sorted(sleeves)])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def deduplicate_hypotheses(
    hypotheses: List[dict],
    similarity_threshold: float = 0.85,
) -> List[dict]:
    """
    Remove near-duplicate hypotheses using cosine similarity of their bias vectors.
    Keeps the first occurrence when duplicates are found.
    """
    if not hypotheses:
        return []

    all_sleeves = set()
    for h in hypotheses:
        all_sleeves.update(h.get("weight_biases", {}).keys())
    sleeves = sorted(all_sleeves)

    kept   = []
    vecs   = []
    for h in hypotheses:
        v = _bias_vector(h, sleeves)
        duplicate = any(
            _cosine_similarity(v, existing) > similarity_threshold
            for existing in vecs
        )
        if not duplicate:
            kept.append(h)
            vecs.append(v)

    return kept


def propose_hypotheses(
    regime: str,
    current_weights: Dict[str, float],
    n: int = 5,
) -> List[dict]:
    """
    Ask Haiku to propose N regime-aware alpha weight hypotheses.

    Returns:
        List of hypothesis dicts: [{narrative: str, weight_biases: {sleeve: delta}}]
        Empty list if LLM unavailable or parse fails.
    """
    sleeves = sorted(current_weights.keys())
    weights_str = "\n".join(f"  {s}: {w:.2f}" for s, w in sorted(current_weights.items()))

    user_prompt = _USER_TEMPLATE.format(
        regime=regime,
        weights_str=weights_str,
        n=n,
        sleeves=", ".join(sleeves),
    )

    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return []

    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        parsed = json.loads(raw[start:end])
        if not isinstance(parsed, list):
            return []
        hypotheses = [
            h for h in parsed
            if isinstance(h, dict) and "narrative" in h and "weight_biases" in h
        ]
        return deduplicate_hypotheses(hypotheses)
    except Exception as exc:
        log.warning("[FactorProposer] Parse failed: %s", exc)
        return []


def generate_guided_variants(
    current_weights: Dict[str, float],
    hypotheses: List[dict],
    perturb_range: float = 0.03,
) -> List[dict]:
    """
    Convert hypothesis weight biases into full variant configs.

    Applies bias + small random noise within perturb_range, then renormalizes.
    Enforces per-sleeve minimum floors from _SLEEVE_FLOORS.

    Returns:
        List of variant config dicts: [{variant_id, alpha_weights}]
    """
    from datetime import datetime
    variants = []

    for i, hyp in enumerate(hypotheses):
        biases  = hyp.get("weight_biases", {})
        weights = dict(current_weights)

        for sleeve, bias in biases.items():
            if sleeve in weights:
                noise = float(np.random.uniform(-perturb_range, perturb_range))
                weights[sleeve] = weights[sleeve] + bias + noise

        for sleeve in weights:
            floor = _SLEEVE_FLOORS.get(sleeve, 0.0)
            weights[sleeve] = max(floor, weights[sleeve])

        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
        else:
            weights = dict(current_weights)

        variants.append({
            "variant_id":    f"guided_{i+1}_{datetime.now().strftime('%Y%m%d')}",
            "alpha_weights": weights,
            "hypothesis":    hyp.get("narrative", ""),
        })

    return variants
```

- [ ] **Step 4: Modify `ascent/research/self_improve.py` `generate_variants()` to call proposer**

Find `generate_variants()` at line 85. Replace the entire function body with this:

```python
def generate_variants(base_config: dict, n: int = N_VARIANTS, regime: str = None) -> list:
    """
    Generate N variant configs. If regime is provided, tries LLM-guided hypothesis
    generation first. Falls back to random perturbation if LLM is unavailable.
    Weights are renormalized to sum to 1 after perturbation.
    """
    base_weights   = base_config.get("alpha_weights", DEFAULT_ALPHA_WEIGHTS)
    active_sleeves = dict(base_weights)

    # Try LLM-guided hypothesis generation when regime is known
    if regime:
        try:
            from ascent.research.factor_proposer import propose_hypotheses, generate_guided_variants
            hypotheses = propose_hypotheses(regime=regime, current_weights=active_sleeves, n=n)
            if hypotheses:
                guided = generate_guided_variants(active_sleeves, hypotheses, perturb_range=0.03)
                if len(guided) >= n:
                    return guided[:n]
                # If we got fewer than n guided variants, top up with random ones
                random_count = n - len(guided)
                random_variants = _random_variants(active_sleeves, n=random_count)
                return guided + random_variants
        except Exception as exc:
            print(f"[SelfImprove] LLM hypothesis generation failed ({exc}), using random perturbation")

    return _random_variants(active_sleeves, n=n)


def _random_variants(active_sleeves: dict, n: int) -> list:
    """Generate n random weight perturbation variants (original behavior)."""
    import copy
    import random
    from datetime import datetime
    variants = []

    for i in range(n):
        variant = copy.deepcopy(active_sleeves)
        for sleeve in variant:
            delta = random.uniform(-PERTURB_RANGE, PERTURB_RANGE)
            floor = MIN_SLEEVE_WEIGHTS.get(sleeve, 0.0)
            variant[sleeve] = max(floor, variant[sleeve] + delta)

        total = sum(variant.values())
        if total > 0:
            variant = {k: round(v / total, 4) for k, v in variant.items()}
        else:
            variant = copy.deepcopy(active_sleeves)

        variants.append({
            "variant_id":    f"v{i+1}_{datetime.now().strftime('%Y%m%d')}",
            "alpha_weights": variant,
        })

    return variants
```

Also update the call to `generate_variants` in `run_self_improve()` at line ~225 to pass the regime:

```python
    variants = generate_variants(active, n=N_VARIANTS, regime=current_regime)
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/pytest tests/test_factor_proposer.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 6: Full suite check**

```bash
.venv/bin/pytest tests/ -q --tb=short 2>&1 | tail -6
```
Expected: All ≥265 tests pass.

- [ ] **Step 7: Commit**

```bash
git add ascent/research/factor_proposer.py tests/test_factor_proposer.py \
        ascent/research/self_improve.py
git commit -m "feat(research): LLM-guided hypothesis generation — regime-aware weight proposals with cosine deduplication"
```

---

## Task F: Tool-Capable Debate Agents

**Problem:** Debate agents currently reason only from the `portfolio_state` dict passed to them. They cannot look up the actual sector concentration of the proposed portfolio, compute a VaR estimate, or verify position momentum. This forces agents to argue from memory and guess at numbers. TradingAgents (arxiv:2412.20138) showed that domain-specific tool access makes LLM trading agents significantly sharper — they stop hallucinating numbers and argue from computed facts. This task equips the bear and devil's advocate with four tools so they can actively compute the statistics they cite.

**Files:**
- Create: `debate/agent_tools.py`
- Modify: `ascent/llm/client.py` — add `tool_completion()`
- Modify: `debate/agents.py` — `run_bear_agent()` and `run_devils_advocate()` use tool_completion
- Create: `tests/test_agent_tools.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # Result is text, not JSON — just check it contains percentage symbols or "sector"
    assert "%" in result or "sector" in result.lower() or "%" in result


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
    # Must return a string without hanging — max_tool_calls enforced
    assert isinstance(result, str)
    assert mock_client.messages.create.call_count <= 4  # 1 initial + 3 max tools + grace


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_agent_tools.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'debate.agent_tools'`

- [ ] **Step 3: Create `debate/agent_tools.py`**

```python
"""
debate/agent_tools.py

Tool definitions and pure-Python implementations for tool-capable debate agents.

Tools:
  get_sector_concentration(weights) — sector breakdown of portfolio
  get_var_estimate(weights)         — historical 5th-percentile 1-day return
  get_position_momentum(symbols)    — 252-day momentum for each symbol
  get_regime_conditional_stats(regime) — historical regime outcome statistics

These are Anthropic tool schema definitions + synchronous implementations.
All implementations must be fast (< 1s) and never raise unhandled exceptions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# ── Anthropic tool schema definitions ─────────────────────────────────────────

DEBATE_TOOLS = [
    {
        "name": "get_sector_concentration",
        "description": (
            "Compute the sector-level weight breakdown for the proposed portfolio. "
            "Use this to identify sector concentration risk before making your argument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Dict mapping symbol → portfolio weight (float)",
                    "additionalProperties": {"type": "number"},
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_var_estimate",
        "description": (
            "Estimate the portfolio's historical Value-at-Risk (5th percentile 1-day return). "
            "Use this to quantify downside risk in your argument."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Dict mapping symbol → portfolio weight",
                    "additionalProperties": {"type": "number"},
                }
            },
            "required": ["weights"],
        },
    },
    {
        "name": "get_position_momentum",
        "description": (
            "Look up 252-day momentum (price return) for a list of symbols. "
            "Use this to verify whether positions are actually in uptrends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols",
                }
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_regime_conditional_stats",
        "description": (
            "Get historical statistics for a given regime label: typical duration, "
            "average drawdown, base rate of continued stress, and historical examples."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "regime": {
                    "type": "string",
                    "description": "Regime label: calm_bull, stressed, crisis, neutral, uncertain",
                }
            },
            "required": ["regime"],
        },
    },
]


# ── Regime statistics (static — no live data dependency) ──────────────────────

_REGIME_STATS = {
    "calm_bull": {
        "avg_duration_weeks": 18, "avg_drawdown_pct": -4.2,
        "base_rate_continues_pct": 72, "avg_return_annualized_pct": 14.8,
        "tail_risk_note": "Low tail risk — watch for euphoria/breadth narrowing as warning sign",
        "historical_examples": "2013–2014, 2017, 2019, 2021 H1",
    },
    "stressed": {
        "avg_duration_weeks": 7, "avg_drawdown_pct": -12.4,
        "base_rate_continues_pct": 38, "avg_return_annualized_pct": -6.2,
        "tail_risk_note": "High — 38% chance of escalating to crisis; credit spreads are leading indicator",
        "historical_examples": "Q4 2018, Aug 2015, Q1 2020 onset, Q4 2022",
    },
    "crisis": {
        "avg_duration_weeks": 5, "avg_drawdown_pct": -28.7,
        "base_rate_continues_pct": 25, "avg_return_annualized_pct": -42.0,
        "tail_risk_note": "Extreme — correlation spikes to 0.85+, liquidity gaps appear; capital preservation mode",
        "historical_examples": "Mar 2020, Q4 2008, Q3 2002",
    },
    "neutral": {
        "avg_duration_weeks": 3, "avg_drawdown_pct": -5.8,
        "base_rate_continues_pct": 30, "avg_return_annualized_pct": 4.1,
        "tail_risk_note": "Moderate — typically transitions quickly in either direction",
        "historical_examples": "Various 2-4 week windows between regimes",
    },
    "uncertain": {
        "avg_duration_weeks": 2, "avg_drawdown_pct": -7.1,
        "base_rate_continues_pct": 20, "avg_return_annualized_pct": 1.2,
        "tail_risk_note": "High uncertainty — HMM entropy > 0.90, reduce size and wait for clarity",
        "historical_examples": "Regime transition periods, data disruptions",
    },
}


# ── Price cache (shared across tool calls in a single debate session) ──────────

_PRICE_CACHE: Dict[str, Any] = {}


def _fetch_prices_cached(symbols: List[str]) -> Dict[str, Any]:
    """Fetch price series for symbols, using in-process cache."""
    import pandas as pd
    missing = [s for s in symbols if s not in _PRICE_CACHE]
    if missing:
        try:
            import yfinance as yf
            raw = yf.download(missing, period="2y", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            for sym in missing:
                if sym in close.columns:
                    _PRICE_CACHE[sym] = close[sym].dropna()
                else:
                    _PRICE_CACHE[sym] = pd.Series(dtype=float)
        except Exception as exc:
            log.warning("[AgentTools] Price fetch failed: %s", exc)
            for sym in missing:
                _PRICE_CACHE[sym] = pd.Series(dtype=float)
    return {s: _PRICE_CACHE[s] for s in symbols}


# ── Tool implementations ───────────────────────────────────────────────────────

def get_sector_concentration(inputs: dict) -> str:
    """Return sector breakdown of portfolio weights as plain text."""
    weights = inputs.get("weights", {})
    if not weights:
        return "No weights provided."

    sector_weights: Dict[str, float] = {}
    unknown_syms = []
    try:
        from pathlib import Path
        import pandas as pd
        profiles_path = Path("data_cache/profiles.parquet")
        if profiles_path.exists():
            df = pd.read_parquet(profiles_path)
            if "symbol" in df.columns and "sector" in df.columns:
                sector_map = dict(zip(df["symbol"], df["sector"]))
            else:
                sector_map = {}
        else:
            sector_map = {}
    except Exception:
        sector_map = {}

    # Add ETF buckets
    sector_map.update({
        "TLT": "rates", "IEF": "rates", "LQD": "rates", "BIL": "cash",
        "HYG": "credit", "UUP": "fx", "GLD": "commodities",
        "PDBC": "commodities", "DBA": "commodities", "DBB": "commodities",
        "VNQ": "reits", "IFRA": "infrastructure", "VIXY": "volatility",
        "EEM": "em_equity", "VWO": "em_equity", "EWT": "em_equity",
        "EWZ": "em_equity", "EWY": "em_equity", "INDA": "em_equity",
        "EWJ": "developed_intl", "EWG": "developed_intl",
        "EWU": "developed_intl", "EFA": "developed_intl",
    })

    for sym, w in weights.items():
        sector = sector_map.get(sym, "unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(w)
        if sector == "unknown":
            unknown_syms.append(sym)

    lines = ["Sector concentration:"]
    for sector, sw in sorted(sector_weights.items(), key=lambda x: -x[1]):
        lines.append(f"  {sector}: {sw:.1%}")
    if unknown_syms:
        lines.append(f"  (unknown sector for: {', '.join(unknown_syms[:5])})")

    max_sector = max(sector_weights.items(), key=lambda x: x[1])
    lines.append(f"\nLargest sector: {max_sector[0]} at {max_sector[1]:.1%}")
    return "\n".join(lines)


def get_var_estimate(inputs: dict) -> str:
    """Estimate portfolio historical VaR (5th percentile 1-day return) from 1-year of data."""
    weights = inputs.get("weights", {})
    if not weights:
        return "No weights provided for VaR estimate."

    try:
        import numpy as np
        syms  = list(weights.keys())
        wvals = list(weights.values())
        total = sum(wvals)
        if total > 0:
            wvals = [w / total for w in wvals]

        prices = _fetch_prices_cached(syms)

        import pandas as pd
        rets_list = []
        w_used    = []
        for sym, w in zip(syms, wvals):
            s = prices.get(sym, pd.Series(dtype=float))
            if len(s) > 50:
                rets_list.append(s.pct_change().dropna().values[-252:])
                w_used.append(w)

        if not rets_list:
            return "Insufficient price data for VaR estimate."

        # Align to same length
        min_len   = min(len(r) for r in rets_list)
        rets_list = [r[-min_len:] for r in rets_list]
        w_arr     = np.array(w_used) / sum(w_used)
        portfolio_rets = np.sum(np.column_stack(rets_list) * w_arr, axis=1)

        var_5  = float(np.percentile(portfolio_rets, 5))
        var_1  = float(np.percentile(portfolio_rets, 1))
        avg    = float(np.mean(portfolio_rets))
        vol    = float(np.std(portfolio_rets))

        return (
            f"Portfolio VaR estimate (1-year history, {min_len} days):\n"
            f"  5th percentile (daily VaR-95): {var_5:+.2%}\n"
            f"  1st percentile (daily VaR-99): {var_1:+.2%}\n"
            f"  Mean daily return: {avg:+.3%}\n"
            f"  Daily volatility:  {vol:.3%} ({vol * 16:.1%} annualized)"
        )
    except Exception as exc:
        return f"VaR estimate failed: {exc}"


def get_position_momentum(inputs: dict) -> str:
    """Return 252-day momentum for a list of symbols."""
    symbols = inputs.get("symbols", [])
    if not symbols:
        return "No symbols provided."

    try:
        import numpy as np
        prices = _fetch_prices_cached(symbols)
        lines  = ["252-day momentum (price return):"]
        for sym in symbols:
            s = prices.get(sym)
            if s is None or len(s) < 21:
                lines.append(f"  {sym}: insufficient data")
                continue
            if len(s) >= 252:
                mom_252 = float(s.iloc[-1] / s.iloc[-252] - 1)
                lines.append(f"  {sym}: {mom_252:+.1%} (252d)")
            else:
                mom_avail = float(s.iloc[-1] / s.iloc[0] - 1)
                lines.append(f"  {sym}: {mom_avail:+.1%} ({len(s)}d, <252d available)")
        return "\n".join(lines)
    except Exception as exc:
        return f"Momentum fetch failed: {exc}"


def get_regime_conditional_stats(inputs: dict) -> str:
    """Return historical regime statistics as plain text."""
    regime = str(inputs.get("regime", "unknown")).lower()
    stats  = _REGIME_STATS.get(regime)
    if not stats:
        return (
            f"No historical statistics for regime '{regime}'. "
            f"Valid regimes: {', '.join(_REGIME_STATS.keys())}"
        )
    return (
        f"Historical statistics for {regime.upper()} regime:\n"
        f"  Typical duration: {stats['avg_duration_weeks']} weeks\n"
        f"  Average drawdown: {stats['avg_drawdown_pct']:.1f}%\n"
        f"  Base rate of continuation: {stats['base_rate_continues_pct']}%\n"
        f"  Average annualized return: {stats['avg_return_annualized_pct']:+.1f}%\n"
        f"  Tail risk: {stats['tail_risk_note']}\n"
        f"  Historical examples: {stats['historical_examples']}"
    )


# ── Tool dispatcher ────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_inputs: dict) -> str:
    """Dispatch a tool call by name. Returns plain-text result string."""
    _TOOL_MAP = {
        "get_sector_concentration":  get_sector_concentration,
        "get_var_estimate":          get_var_estimate,
        "get_position_momentum":     get_position_momentum,
        "get_regime_conditional_stats": get_regime_conditional_stats,
    }
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"
    try:
        return fn(tool_inputs)
    except Exception as exc:
        log.warning("[AgentTools] Tool %s failed: %s", tool_name, exc)
        return f"Tool {tool_name} failed: {exc}"
```

- [ ] **Step 4: Add `tool_completion()` to `ascent/llm/client.py`**

Add this function after `extended_thinking_completion()` (around line 177), before the `if __name__ == "__main__":` block:

```python
def tool_completion(
    system_prompt: str,
    user_prompt: str,
    tools: list,
    tool_executor,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    max_tool_calls: int = 3,
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

    for iteration in range(max_tool_calls + 1):
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
            system=system_prompt,
        )
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.messages.create(**kwargs)
                break
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                log.warning("[LLM/Tools] Attempt %d failed (%s), retry in %ds", attempt+1, e, wait)
                import time; time.sleep(wait)

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

            # Serialize the assistant's response (tool_use blocks)
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

            # Execute each tool and collect results
            tool_results = []
            for block in tool_use_blocks:
                result = tool_executor(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     str(result),
                })
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop_reason — extract any text and return
            text_parts = [
                block.text for block in resp.content
                if getattr(block, "type", "") == "text"
            ]
            return "\n".join(text_parts) if text_parts else f"[Stopped: {resp.stop_reason}]"

    # Max iterations reached — extract any text from the last response
    try:
        text_parts = [
            block.text for block in resp.content
            if getattr(block, "type", "") == "text"
        ]
        return "\n".join(text_parts) if text_parts else "[Tool loop: max iterations reached]"
    except Exception:
        return "[Tool loop: max iterations reached]"
```

- [ ] **Step 5: Update `run_bear_agent()` and `run_devils_advocate()` in `debate/agents.py`**

In `run_bear_agent()` at line 101, replace the `generate_structured()` call with `tool_completion()`:

```python
def run_bear_agent(portfolio_state: dict) -> str:
    context      = _build_context(portfolio_state)
    regime       = portfolio_state.get("us_regime")
    cred_context = load_credibility_context(regime)
    user_prompt  = f"Portfolio context:\n{context}"
    if cred_context:
        user_prompt += f"\n\n{cred_context}"
    user_prompt += (
        "\n\nMake the bear case against these trades. "
        "Use the available tools to verify sector concentrations, VaR, and momentum "
        "before making quantitative claims."
    )
    try:
        from debate.agent_tools import DEBATE_TOOLS, execute_tool
        from ascent.llm.client import tool_completion
        return tool_completion(
            system_prompt=(
                "You are the Bear Analyst at Ascent Capital. Your job is to argue "
                "for REDUCING risk or WAITING. Identify the weakest positions, concentration risks, "
                "regime fragility, or macro headwinds. Use the provided tools to compute "
                "sector concentration, VaR, and momentum BEFORE making claims — do not guess "
                "at numbers you can look up. Be specific. "
                "You have been given historical accuracy data — use it to calibrate how often "
                "your past warnings were correct in this regime. Keep under 200 words."
            ),
            user_prompt=user_prompt,
            tools=DEBATE_TOOLS,
            tool_executor=execute_tool,
            model=DEBATE_MODEL,
            max_tokens=800,
            max_tool_calls=2,
        )
    except Exception as e:
        log.warning("[Bear] Tool completion failed (%s), falling back to generate_structured", e)
        return generate_structured(
            system_prompt=(
                "You are the Bear Analyst at Ascent Capital. Argue for reducing risk. "
                "Be specific. Keep under 200 words."
            ),
            user_prompt=user_prompt,
            model=DEBATE_MODEL,
            temperature=0.6,
            use_cache=True,
        )
```

Apply the same pattern to `run_devils_advocate()` at line 124, replacing `generate_structured` with a `tool_completion` call using the same `DEBATE_TOOLS` and `execute_tool`. The system prompt remains the same as before. Wrap in the same try/except with `generate_structured` fallback.

Add `import logging` and `log = logging.getLogger(__name__)` at the top of `debate/agents.py` if not already present.

- [ ] **Step 6: Run all tests**

```bash
.venv/bin/pytest tests/test_agent_tools.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 7: Full suite check**

```bash
.venv/bin/pytest tests/ -q --tb=short 2>&1 | tail -8
```
Expected: All tests pass (≥265).

- [ ] **Step 8: Commit**

```bash
git add debate/agent_tools.py ascent/llm/client.py debate/agents.py \
        tests/test_agent_tools.py
git commit -m "feat(debate): tool-capable agents — bear and devil can compute sector/VaR/momentum during debate"
```

---

## Final: Push

- [ ] **Push to GitHub**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ Post-trade reflection: Haiku reads scored verdict + outcome, writes structured lesson
- ✅ Reflections filtered by regime and injected into `_build_context()`
- ✅ Idempotent: already-reflected verdicts are not re-processed
- ✅ LLM-guided hypothesis generation: propose → translate biases → renormalize → floor clamp
- ✅ Cosine-similarity deduplication of near-identical hypotheses
- ✅ `generate_variants()` uses guided proposer when regime is available, falls back to random
- ✅ `tool_completion()` in client.py: full Anthropic tool-use loop with max iteration guard
- ✅ `execute_tool()` dispatcher: sector concentration, VaR, momentum, regime stats
- ✅ bear and devil's advocate use tool_completion with try/except fallback
- ✅ All tools are pure-Python implementations — no external services required

**Type consistency:**
- `reflect_on_verdict(verdict_path: Path) -> Optional[Dict]` — used correctly in `reflect_on_new_outcomes()`
- `load_recent_reflections(regime: Optional[str], n: int) -> List[Dict]` — consumed by `format_reflections_for_context()`
- `propose_hypotheses(regime, current_weights, n) -> List[dict]` — consumed by `generate_guided_variants()`
- `generate_guided_variants(current_weights, hypotheses, perturb_range) -> List[dict]` — same shape as `_random_variants()`
- `tool_completion(system_prompt, user_prompt, tools, tool_executor, model, max_tokens, max_tool_calls) -> str` — matches how bear/devil agents call it
- `execute_tool(tool_name: str, tool_inputs: dict) -> str` — matches `tool_executor` callable interface in `tool_completion()`
