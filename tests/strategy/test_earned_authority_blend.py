"""
tests/strategy/test_earned_authority_blend.py

Tests for the active-weight-budget blend() function itself (Finding 2), and
(TestNoLiveBlendCallSite below) for its removal from the live write path.

blend() behaviour contract -- unchanged, still exercised directly below:
  - ai_weight is a one-way active-weight budget (e.g. 0.05 = 5pp max one-way deviation)
  - Deltas are scaled so gross one-way deviation never exceeds budget
  - Dust positions (<0.5%) are dropped and weights renormalized
  - Empty AI portfolio returns quant unchanged
  - Zero budget returns quant unchanged

Task 6, 2026-08-14: `earned_authority` scored CUT (p=0.35, track_d vs
track_astar) in the proof audit -- a distinct write path from the judge's
position-change (Task 5's `debate_judge_intervention`, CUT p=0.75), easy to
conflate by name but measuring a different mechanism: the AI PM's own
portfolio blended into `merged_weights` via `authority_blend()`/`blend()`.
The call site `merged_weights = authority_blend(ai_pm_result.portfolio,
merged_weights)` in run_all_agents.py is removed. `blend()` itself is left in
place (untouched, still tested above for its own pure behaviour) in case the
write path is reinstated later; it is simply uncalled from
run_all_agents.py now. `update_authority()` is KEPT -- unlike Task 5's judge
write path, the earned-authority ladder it maintains (level, ai_weight,
in_cooldown, days_at_level) is read independently of the blend by: the
Opus-trigger logic in run_all_agents.main() (first-day-post-promotion
check), agents/ai_pm_agent.py's authority-disclosure prompt text, the
dashboard's earned-authority panel (scripts/generate_performance_page.py),
and ai_pm_perf_feedback.py's feedback reporting. Continuing to call
update_authority() keeps that ladder live for measurement even though it no
longer gates a real write.
"""
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch


LEVEL_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]
LEVEL_TITLES  = ["Shadow", "Analyst", "Associate", "Manager", "Director", "CEO"]


def _write_state(tmp_dir: str, level: int, ai_weight: float | None = None) -> Path:
    state = {
        "level": level,
        "title": LEVEL_TITLES[level],
        "ai_weight": ai_weight if ai_weight is not None else LEVEL_WEIGHTS[level],
        "level_start_date": "2026-06-04",
        "days_at_level": 5,
        "days_stuck": 5,
        "in_cooldown": False,
        "cooldown_until": None,
        "auto_revert_count": 0,
        "last_updated": "2026-06-03",
        "track_d_returns": [],
        "track_astar_returns": [],
        "disable_sleeve_priors": False,
        "phase": level,
        "ai_returns_21d": [],
        "quant_returns_21d": [],
    }
    p = Path(tmp_dir) / "earned_authority.json"
    p.write_text(json.dumps(state))
    return p


def _call_blend(ai_portfolio, quant_portfolio, level=1, ai_weight=None):
    """Helper: call blend() with a temp state file at the given level."""
    import ascent.strategy.earned_authority as ea

    with tempfile.TemporaryDirectory() as tmp:
        sp = _write_state(tmp, level=level, ai_weight=ai_weight)
        shadow = Path(tmp) / "shadow.jsonl"
        with patch.object(ea, "STATE_PATH", sp):
            with patch.object(ea, "SHADOW_RETURNS_PATH", shadow):
                return ea.blend(ai_portfolio, quant_portfolio)


# ── Test 1: ±2pp override survives at Level 1 ────────────────────────────────

def test_override_survives_at_level1_budget():
    """A ±2pp proposed override within a 5pp budget is applied at near-full scale."""
    # 15 equal-weight names (~6.667% each)
    syms = [f"S{i:02d}" for i in range(15)]
    quant = {s: 1 / 15 for s in syms}

    # AI proposes S00 up 2pp, S01 down 2pp — everything else unchanged
    ai = dict(quant)
    ai["S00"] = quant["S00"] + 0.02
    ai["S01"] = quant["S01"] - 0.02

    result = _call_blend(ai, quant, level=1)  # budget = 5pp

    # One-way gross = (|+0.02| + |-0.02|) / 2 = 0.02, budget = 0.05 → scale = 1.0
    # S00 should be near quant["S00"] + 0.02, S01 near quant["S01"] - 0.02
    expected_s00 = (quant["S00"] + 0.02) / sum(ai.values())  # after renorm
    assert "S00" in result, "S00 must survive after a ±2pp override"
    # Change must be near full-scale, not diluted to 0.10pp (the old bug)
    # Allow 0.2pp tolerance for renorm effects
    assert abs(result["S00"] - expected_s00) < 0.002, (
        f"S00 weight {result['S00']:.4f} should be near {expected_s00:.4f} "
        f"(not diluted to ~{quant['S00']:.4f})"
    )


# ── Test 2: Aggregate one-way deviation ≤ ai_weight ─────────────────────────

def test_aggregate_deviation_capped_at_budget():
    """When AI proposes large changes, gross one-way deviation is capped at budget."""
    quant = {f"S{i}": 0.10 for i in range(10)}  # 10 names × 10% each
    # AI wants to concentrate into 5 names at 20% each (huge overrides)
    ai = {f"S{i}": 0.20 for i in range(5)}
    # S5–S9 absent from AI portfolio (AI proposes 0% weight)

    budget = 0.05
    result = _call_blend(ai, quant, level=1, ai_weight=budget)

    # Compute one-way deviation of result vs quant
    all_syms = set(result) | set(quant)
    one_way = sum(abs(result.get(s, 0.0) - quant.get(s, 0.0)) for s in all_syms) / 2.0
    assert one_way <= budget + 1e-6, (
        f"One-way deviation {one_way:.4f} exceeds budget {budget:.4f}"
    )


# ── Test 3: Empty AI portfolio returns pure quant ────────────────────────────

def test_empty_ai_returns_quant():
    """blend({}, quant) must return quant unchanged."""
    quant = {"AAPL": 0.5, "MSFT": 0.5}
    result = _call_blend({}, quant, level=1)
    assert result == quant, f"Expected {quant}, got {result}"


# ── Test 4: Zero budget returns pure quant ───────────────────────────────────

def test_zero_budget_returns_quant():
    """At ai_weight=0.0 (Level 0 / Shadow), blend returns quant unchanged."""
    quant = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
    ai    = {"NVDA": 0.5, "TSLA": 0.5}  # completely different portfolio
    result = _call_blend(ai, quant, level=0)  # Level 0 → ai_weight = 0.0
    assert result == quant, f"Expected pure quant at budget=0, got {result}"


# ── Test 5: Weights sum to 1.0 after blend ───────────────────────────────────

def test_weights_sum_to_one():
    """blend() output must always sum to 1.0 ± 1e-6."""
    quant = {"AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10, "AMZN": 0.10,
             "META": 0.10, "NVDA": 0.10, "TSLA": 0.10, "JPM": 0.10,
             "WMT": 0.10, "PG": 0.10}
    # AI proposes a shifted portfolio
    ai = {"AAPL": 0.15, "MSFT": 0.15, "GOOG": 0.15, "AMZN": 0.15,
          "META": 0.05, "NVDA": 0.05, "TSLA": 0.05, "JPM": 0.05,
          "WMT": 0.05, "PG": 0.10}

    for level in [1, 2, 3]:
        result = _call_blend(ai, quant, level=level)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Level {level}: weights sum to {total:.8f}, expected 1.0"
        )


# ── Integration: the live blend-into-merged_weights write path is gone ───────

class TestNoLiveBlendCallSite:
    """The whole point: run_all_agents.py must no longer blend the AI PM's
    portfolio into merged_weights, while blend() itself (tested above) and
    update_authority() stay in place and callable."""

    def _src(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(root, "run_all_agents.py")) as f:
            return f.read()

    def test_authority_blend_is_not_imported(self):
        src = self._src()
        assert "authority_blend" not in src, (
            "the `blend as authority_blend` import/alias must be gone -- "
            "the live blend call site is removed"
        )

    def test_merged_weights_is_never_assigned_from_blend(self):
        src = self._src()
        assert "merged_weights = authority_blend(" not in src, (
            "merged_weights must never be reassigned from the AI PM blend -- "
            "earned_authority scored CUT (p=0.35) in the proof audit"
        )

    def test_update_authority_call_site_still_present(self):
        """update_authority() is KEPT: the ladder it maintains is read
        independently of the blend (Opus trigger, AI PM authority-disclosure
        prompt, dashboard, ai_pm_perf_feedback.py) for continued measurement."""
        src = self._src()
        assert "update_authority(" in src, (
            "update_authority() call site should remain -- it feeds reads "
            "elsewhere that are independent of the removed blend write"
        )
