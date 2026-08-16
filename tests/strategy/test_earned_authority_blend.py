"""
tests/strategy/test_earned_authority_blend.py

(TestNoLiveBlendCallSite below) covers removal of the active-weight-budget
blend() write path from run_all_agents.py.

Task 6, 2026-08-14: `earned_authority` scored CUT (p=0.35, track_d vs
track_astar) in the proof audit -- a distinct write path from the judge's
position-change (Task 5's `debate_judge_intervention`, CUT p=0.75), easy to
conflate by name but measuring a different mechanism: the AI PM's own
portfolio blended into `merged_weights` via `authority_blend()`/`blend()`.
The call site `merged_weights = authority_blend(ai_pm_result.portfolio,
merged_weights)` in run_all_agents.py was removed 2026-08-14, and
`ascent.strategy.earned_authority.blend()` itself (the pure function it
called) was deleted 2026-08-15 after a repo-wide confirm of zero remaining
callers. `update_authority()` is KEPT -- unlike Task 5's judge write path,
the earned-authority ladder it maintains (level, ai_weight, in_cooldown,
days_at_level) is read independently of the blend by: the Opus-trigger logic
in run_all_agents.main() (first-day-post-promotion check),
agents/ai_pm_agent.py's authority-disclosure prompt text, the dashboard's
earned-authority panel (scripts/generate_performance_page.py), and
ai_pm_perf_feedback.py's feedback reporting. Continuing to call
update_authority() keeps that ladder live for measurement even though it no
longer gates a real write.
"""
# ── Integration: the live blend-into-merged_weights write path is gone ───────

class TestNoLiveBlendCallSite:
    """The whole point: run_all_agents.py must no longer blend the AI PM's
    portfolio into merged_weights. blend() itself has been deleted from
    ascent/strategy/earned_authority.py (zero callers); update_authority()
    stays in place and callable."""

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
