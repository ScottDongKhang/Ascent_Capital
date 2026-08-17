# Lean fix: alpha-stack gating + dead-code deletion — report

Date: 2026-08-15

## Task 1: Gate all sleeve computation in `ascent/alpha/stack.py::build_alpha_stack()`

Wrapped the existing try/except (or if/elif) blocks for `trend`, `volatility` (the inline
`vol_of_vol_21d`/`vol_21d` block), `fundamental`, `earnings`, `analyst`, `options_flow`,
`insider`, `short_interest`, `earnings_tone`, `narrative` in
`if alpha_weights.get("<name>", 0) > 0:` — same pattern already used by `llm_fundamental`
(line 277) and `altdata` (line 345). Bodies of the try/except blocks were not touched.
`meanrev` and `statarb` stay unconditional (both are in `DEFAULT_ALPHA_WEIGHTS`). `ml` was
left exactly as-is: it's gated by `"targets" in features` (data availability), not by
`alpha_weights`, and `ml` is not itself a key in `DEFAULT_ALPHA_WEIGHTS` — a genuinely
different gating mechanism, per the task's instruction to leave it alone if in doubt.

**Bug found and fixed as instructed**: `narrative`'s comment claimed "0% weight until
narrative cache has ≥30 days history" but the code only checked `if _symbols:` — no weight
check at all. It is now actually gated on `alpha_weights.get("narrative", 0) > 0`, matching
the comment's stated intent.

### Verification before/after

1. **Static reasoning**: `total_w = sum(alpha_weights.get(k, 0.0) for k in alphas)` and
   `w = alpha_weights.get(name, 0.0) / total_w` in the blend loop already treat any sleeve
   absent from `alpha_weights` (or present at 0.0) as contributing zero weight regardless of
   whether it was computed — so skipping computation for a 0-weight sleeve cannot change the
   composite.
2. **Empirical A/B test**: loaded `git show HEAD:ascent/alpha/stack.py` (pre-edit) and the
   post-edit file as two separate Python modules via `importlib.util.spec_from_file_location`,
   built a synthetic `features` dict (close, mom_5d/10d/21d/252d, zscore_20d, rsi_14,
   bb_pct_20d, vol_21d — no `targets`, so `ml` is naturally excluded on both sides), and ran
   `build_alpha_stack()` on both:
   - Under `DEFAULT_ALPHA_WEIGHTS` (`{"meanrev": 0.5, "statarb": 0.5}`): old code loaded
     `['trend', 'meanrev', 'volatility', 'statarb', 'narrative']` (4 sleeves computed for
     nothing — confirms the waste and confirms the narrative gating bug, since narrative ran
     despite 0 weight); new code loaded exactly `['meanrev', 'statarb']`.
   - **Composite output: max abs diff = 0.0** (bit-identical) between old and new.
   - Repeated with `trend` weight set to 0.1 (to confirm the gate lets a genuinely-weighted
     sleeve through correctly): old computed the same 5 sleeves, new computed
     `['trend', 'meanrev', 'statarb']`. **Composite output: max abs diff = 0.0** again.
3. **Existing test suite** for everything under `ascent/alpha/`:
   ```
   .venv/bin/python -m pytest tests/test_altdata_pipeline.py tests/test_ic_gate_redistribution.py \
     tests/test_narrative_alpha.py tests/test_fundamental_alpha.py tests/test_self_evolving_alpha.py \
     tests/test_llm_fundamental_alpha.py tests/test_phase6_signals.py tests/test_analyst_alpha.py \
     tests/alpha/test_stack_weights.py tests/alpha/test_meta_learner.py tests/test_alpha_stack_weights.py \
     tests/test_earnings_alpha.py -q
   ```
   → **159 passed**, 2 unrelated deprecation warnings.

Commit: `7e8b55f`

## Task 2: Delete orphaned `ascent/portfolio/hedge_overlay.py` — BLOCKED, not done

Ran `grep -rln --include="*.py" "hedge_overlay|apply_hedge_overlay" .` (excluding
`.venv`/`.git`/`.claude/worktrees`) and found a real, live caller:

```
scripts/evaluate_hedge.py:26: from ascent.portfolio.hedge_overlay import compute_hedge_weight
scripts/evaluate_hedge.py:112:    hedge_weights[dt] = compute_hedge_weight(label, confidence)
```

`scripts/evaluate_hedge.py` is a standalone historical-hedge-evaluation script
(`.venv/bin/python scripts/evaluate_hedge.py`) that imports and calls
`compute_hedge_weight` from `hedge_overlay.py` for real, not as a string/comment reference.
The other hits (`ascent/analyst/proof_audit/run.py`, `components.py`,
`counterfactual_scorer.py`) are all string labels/comments (`"hedge_overlay"` as a component
name), not imports or calls — those are fine and irrelevant to the blocker.

Per the task's explicit instruction ("if you find even one real caller, STOP and report it as
a blocker instead of deleting"), **`hedge_overlay.py` and `tests/test_hedge_overlay.py` were
left untouched**. `tests/test_hedge_overlay.py` still passes (12 tests) unmodified.

If this is still meant to be deleted, `scripts/evaluate_hedge.py`'s dependency needs to be
resolved first (e.g. confirm the script itself is dead and can go too, or inline the one
function it needs).

## Task 3: Delete two dead functions

### `ascent/strategy/earned_authority.py::blend()`

Verified zero production callers with `grep -rn "\.blend(|authority_blend" --include=*.py .`
before deleting. The only hits were:
- Test files calling `ea.blend(...)` directly to test the function's own math (removed, see
  below).
- Comment-only references to `authority_blend()` in `ascent/monitoring/ai_pm_counterfactual.py`
  and `ascent/strategy/ai_pm_guardrails.py` (prose, not calls — untouched).
- `run_all_agents.py` no longer contains an `authority_blend` import or call (that call site
  was already removed 2026-08-14 per CLAUDE.md constraint #5).

Deleted the `blend()` function body (lines 298–332). `update_authority()` and `get_state()`
were not touched — confirmed still called from `run_all_agents.py`, `agents/ai_pm_agent.py`,
`scripts/generate_performance_page.py`, and `ai_pm_perf_feedback.py`.

### `run_all_agents.py::apply_judge_position_change`

Verified zero callers with `grep -rn "apply_judge_position_change" --include=*.py .` before
deleting — all hits besides the `def` itself were prose comments (already noting the call
sites were removed 2026-08-14) or source-inspection test assertions. Deleted the function body
(the `def apply_judge_position_change(...)` block, ~81 lines, up to but not including
`def already_ran_for_session`). `_apply_position_change_to_weights` (the pure helper it wrapped)
and `_record_advisory_judge_proposal` (which now carries the sole live
`record_intervention()`/`add_judge_falsifier()` call sites) were untouched.

### Test changes

- `tests/test_ai_pm_agent.py`: removed 4 test functions that called `ea.blend()` directly
  (`test_shadow_phase_blend_returns_pure_quant`, `test_blend_union_of_positions`,
  `test_blend_min_weight_filter`, `test_blend_renormalizes_to_1`). `_make_state()` helper kept
  — still used by the authority-ladder tests in the same file.
- `tests/test_ai_pm_authority.py`: removed 3 test functions that called `ea.blend()` directly
  (`test_blend_at_level1_uses_5pct`, `test_blend_shadow_returns_pure_quant`,
  `test_blend_renormalizes`).
- `tests/strategy/test_earned_authority_blend.py`: this file was dedicated to `blend()`'s pure
  math (Tests 1–5) plus a `TestNoLiveBlendCallSite` class asserting the write path is gone.
  Removed Tests 1–5 and the now-unused `_call_blend`/`_write_state` helpers they depended on —
  kept `TestNoLiveBlendCallSite` (still meaningful: asserts `run_all_agents.py` never imports
  `authority_blend` or assigns `merged_weights` from it, and that `update_authority()` is still
  called) and updated its docstring/module docstring to say `blend()` was deleted rather than
  "left in place."
- `tests/test_judge_change_applied_everywhere.py`: this file mostly tests
  `_apply_position_change_to_weights` (kept, untouched — 10 tests). Replaced the
  `TestNeitherPathAppliesTheJudgeChange` class's three tests (which asserted the deleted
  function's def was present exactly once and uncalled) with a single
  `test_apply_helper_is_gone_entirely` that asserts `"apply_judge_position_change("` no longer
  appears anywhere in `run_all_agents.py`.
- `tests/test_advisory_write_paths.py`: this file covers several unrelated advisory-write-path
  fixes (falsifier trim, early-zeros gate, etc. — untouched). Fixed the one test that indexed
  `src.index("def apply_judge_position_change")` (which would now raise `ValueError`) —
  `test_record_intervention_reaches_a_live_call_site` now asserts
  `"def apply_judge_position_change" not in src` and that `record_intervention(` is reachable
  inside `_record_advisory_judge_proposal` (the surviving advisory recorder).

### Test run

```
.venv/bin/python -m pytest tests/test_ai_pm_agent.py tests/test_ai_pm_cheap_bugs.py \
  tests/test_ai_pm_eval_rule.py tests/test_judge_change_applied_everywhere.py \
  tests/test_advisory_write_paths.py tests/test_ai_pm_counterfactual.py \
  tests/test_ai_pm_perf_feedback.py tests/test_ai_pm_authority.py \
  tests/test_ai_pm_prompt_contract.py tests/test_ai_pm_prethesis_causal.py \
  tests/agents/ tests/debate/test_judge_symmetric.py tests/strategy/ tests/test_hedge_overlay.py -q
```
→ **431 passed**, 2 unrelated Pandas4Warning deprecation warnings (pre-existing, from
`test_altdata_pipeline.py` / `reddit_sentiment.py`, unrelated to this work).

Commit: `91188f6`

## Composite alpha output confirmation

Confirmed unchanged under default weights: the A/B comparison in Task 1 (old module vs new
module, same synthetic features, `alpha_weights={"meanrev": 0.5, "statarb": 0.5}`) produced
**max abs diff = 0.0** on the full composite DataFrame. This is the strongest evidence
available short of running the live pipeline (not done — no live-trading actions were taken).

## Files touched

- `ascent/alpha/stack.py` (Task 1)
- `ascent/strategy/earned_authority.py` (Task 3)
- `run_all_agents.py` (Task 3)
- `tests/strategy/test_earned_authority_blend.py` (Task 3)
- `tests/test_advisory_write_paths.py` (Task 3)
- `tests/test_ai_pm_agent.py` (Task 3)
- `tests/test_ai_pm_authority.py` (Task 3)
- `tests/test_judge_change_applied_everywhere.py` (Task 3)

Not touched (as required): `ascent/alpha/trend.py`, `fundamental.py`, `earnings.py`,
`analyst.py`, `options_flow.py`, `insider.py`, `short_interest.py`, `earnings_tone.py`,
`narrative_alpha.py`; `_EVENING_ROLLOVER_HOUR`/`_calendar_day_key`; wide-format price caches;
anything under `debate/`; `ascent/strategy/` besides `blend()`; `ascent/portfolio/hedge_overlay.py`
(blocked, see Task 2); no live order submission code touched.
