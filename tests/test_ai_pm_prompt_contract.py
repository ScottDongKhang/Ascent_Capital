"""The AI PM must be told what it is scored on and how much authority it has.

Audit, 2026-07-27:

- The prompt said "OBJECTIVE: Sharpe ratio, not raw return" and asked for a
  3-month view, while the layer is scored on `wedge_21d` — a 21-day RAW return
  difference vs the pure-quant baseline. A model correctly following
  "when in doubt, choose the lower-volatility expression" loses that metric by
  construction in an equity bull market. It was graded on a test it was told not
  to study for.

- It was never told its authority level. Measured dilution of its logged
  proposals was 11-45% of each intended delta (median ~14%), so a 10pp conviction
  landed as ~1pp and a full exit became a trim — while the prompt gave it no way
  to know that. The debate judge IS told its cap and visibly reasons about it.

- `get_alpha_wedge`'s own description says "Call in Phase 1", but it was absent
  from PRE_THESIS_TOOLS, so in the live two-phase path Phase 1 could not call it.

- Phase 2 (Opus, the highest-judgment call in the fund) ran at max_tokens=4000 —
  below the wrapper's own 4096 thinking floor — for thinking plus ~20 weights
  plus a 12-key thesis memo, and nothing in the repo ever used xhigh effort.
"""

import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src() -> str:
    with open(os.path.join(_ROOT, "agents/ai_pm_agent.py")) as f:
        return f.read()


class TestObjectiveMatchesTheMetric:
    def test_the_scored_metric_is_named(self):
        src = _src()
        assert "wedge_21d" in src, "the prompt must name the metric it is scored on"

    def test_the_contradictory_sharpe_objective_is_gone(self):
        assert "OBJECTIVE: Sharpe ratio, not raw return" not in _src()

    def test_baseline_is_not_presented_as_a_free_default(self):
        """Eleven stand-down instructions and no counterweight made 'do nothing'
        the highest-reward action the prompt offered."""
        src = _src()
        assert "not a safe default" in src or "non-decision" in src


class TestAuthorityIsDisclosed:
    def test_prompt_builder_reads_the_authority_state(self):
        src = _src()
        i = src.index("def _build_temporal_context")
        body = src[i:i + 4000]
        assert "earned_authority" in body, (
            "the AI PM must be told its own tracking-error budget"
        )

    def test_disclosure_appears_in_the_returned_context(self):
        src = _src()
        i = src.index("def _build_temporal_context")
        body = src[i:i + 6000]
        assert "authority_str" in body and "YOUR AUTHORITY" in body

    def test_context_builds_without_raising(self):
        """It reads state off disk; a missing file must not break the prompt."""
        from agents.ai_pm_agent import _build_temporal_context
        out = _build_temporal_context()
        assert isinstance(out, str) and out


class TestAlphaWedgeReachableInPhase1:
    def test_tool_is_registered_for_the_prethesis_phase(self):
        from agents.ai_pm_agent import PRE_THESIS_TOOLS
        names = {t["name"] for t in PRE_THESIS_TOOLS}
        assert "get_alpha_wedge" in names, (
            "its description tells the model to call it in Phase 1"
        )

    def test_every_prethesis_tool_has_an_executor(self):
        """A tool offered but unroutable burns a call from a hard budget."""
        from agents.ai_pm_agent import PRE_THESIS_TOOLS
        src = _src()
        for t in PRE_THESIS_TOOLS:
            name = t["name"]
            if name == "propose_prethesis":
                continue
            assert f'"{name}":' in src, f"{name} has no executor mapping"


class TestPhase2Budget:
    def _phase2_tool_completion_block(self) -> str:
        """The Phase 2 tool_completion(...) call. Anchored on tools=AI_PM_TOOLS
        and read FORWARD, since max_tokens/effort follow it."""
        src = _src()
        i = src.index("tools=AI_PM_TOOLS,")
        return src[i:i + 1200]

    def test_max_tokens_clears_the_thinking_floor(self):
        block = self._phase2_tool_completion_block()
        m = re.search(r"max_tokens=(\d+)", block)
        assert m, "could not find max_tokens on the Phase 2 call"
        assert int(m.group(1)) > 4096, (
            "max_tokens caps thinking AND visible text together; 4000 was below "
            "the wrapper's own 4096 floor"
        )

    def test_effort_is_raised_for_the_highest_judgment_call(self):
        assert 'effort="xhigh"' in self._phase2_tool_completion_block()

    def test_no_direct_create_call_sits_below_the_thinking_floor(self):
        """Direct messages.create() callers get no floor applied, so they must
        leave their own headroom."""
        src = _src()
        for m in re.finditer(r"_cli\.messages\.create\((.{0,400}?)\)", src, re.S):
            budget = re.search(r"max_tokens=(\d+)", m.group(1))
            if budget:
                assert int(budget.group(1)) >= 4096, (
                    f"direct create with max_tokens={budget.group(1)} — thinking "
                    f"can consume the whole budget and return no text"
                )


class TestForceSealKeepsItsInstructions:
    def test_phase1_force_seal_includes_the_prethesis_prompt(self):
        src = _src()
        i = src.index("Pre-thesis: main pass exhausted")
        block = src[i:i + 1500]
        assert "_PRE_THESIS_PROMPT" in block, (
            "the force-seal produced a thesis from a model never told what a "
            "pre-thesis is — and this is the path that produced 2026-06-24"
        )
