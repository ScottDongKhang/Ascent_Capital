# Target Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewire the live (currently paused) trading pipeline to match what sub-projects 1/1b/1c
proved: a 2-sleeve alpha stack, no regime/hedge overlay, one capital-allocating agent, and no
unproven AI-PM/debate write paths — while keeping every excluded component's code intact and the
debate/AI-PM analysis layer running for continued measurement.

**Architecture:** Eight surgical removals/reductions across the live pipeline, each independently
testable and revertible. No new subsystems, no rewrites of surviving logic — `orchestrator/
central_intelligence.py`'s merge function already degrades gracefully to fewer agents, so most
tasks are deletions plus the tests that pin the new (smaller) surface.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pytest.

## Global Constraints

- Always use `.venv/bin/python`. Never bare `python`.
- `import logging`; never `from loguru import logger`.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- Live paper trading stays paused (`com.ascentcapital.eod`/`.heartbeat`) — this plan does not
  reload it; that's sub-project 4's decision, after validation (sub-project 3).
- **Never delete code for an excluded component** — `macro_agent`/`international_agent`/
  `alternatives_agent`, all 9 `INSUFFICIENT_DATA` sleeves' code, `debate/adversarial_authority.py`,
  `ascent/strategy/earned_authority.py`, `debate_runner`/`debate/agents.py`/`debate/judge.py`,
  `agents/ai_pm_agent.py` all stay in the repo. Only call sites that *invoke* them for live
  capital/write purposes are removed, per the design spec's §3/§4.
- `DEFAULT_ALPHA_WEIGHTS` in `ascent/alpha/stack.py` and `ascent/research/self_improve.py` must
  keep matching key sets (CLAUDE.md constraint #6) — every task touching either file must verify
  both stay in sync.
- Every task's removal must be verified against the CURRENT code (read the file first) — this
  plan's line numbers come from a research pass and may have shifted; do not paste a diff
  against a line number without confirming the surrounding code first.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ascent/alpha/stack.py` (modify `DEFAULT_ALPHA_WEIGHTS`, delete `DEFAULT_ALPHA_WEIGHTS_BY_REGIME`) | 2-sleeve stack | 1 |
| `ascent/research/self_improve.py` (modify `DEFAULT_ALPHA_WEIGHTS`, `MIN_SLEEVE_WEIGHTS`) | matching key set | 1 |
| `ascent/alpha/stack.py` (modify `build_alpha_stack`) | remove regime-conditional weights | 2 |
| `run_all_agents.py` (modify) | remove hedge overlay call | 3 |
| `run_all_agents.py`, `orchestrator/central_intelligence.py` (modify) | exclude 3 agents from live capital | 4 |
| `run_all_agents.py` (modify) | remove judge position-change write path | 5 |
| `run_all_agents.py` (modify) | remove AI PM earned-authority blend write path | 6 |
| `CLAUDE.md` (modify) | update constraints #5, #6 | 7 |
| (verification only) | full suite + dry-run check | 8 |

---

## Task 1: Alpha stack reduces to 2 sleeves

**Why:** Only `meanrev`/`statarb` cleared the audit's significance bar. The other 13 keys in
`DEFAULT_ALPHA_WEIGHTS` (and `self_improve.py`'s parallel copy) represent measured-CUT or
excluded-unmeasured sleeves.

**Files:**
- Modify: `ascent/alpha/stack.py` — `DEFAULT_ALPHA_WEIGHTS` (~line 16-32), delete
  `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` (~line 36-63) entirely (Task 2 removes its only reader)
- Modify: `ascent/research/self_improve.py` — `DEFAULT_ALPHA_WEIGHTS` (~line 37-53),
  `MIN_SLEEVE_WEIGHTS` (~line 67-74)
- Test: whatever existing test(s) already pin these dicts' key sets —
  `tests/test_alpha_stack_weights.py`, `tests/alpha/test_stack_weights.py`,
  `tests/test_self_evolving_alpha.py` (read each first; update assertions to the new 2-key set,
  don't just delete failing assertions)

**Interfaces:**
- Consumes: nothing new
- Produces: `DEFAULT_ALPHA_WEIGHTS = {"meanrev": 0.50, "statarb": 0.50}` in both files, matching
  key sets (CLAUDE.md constraint #6's guard must still pass)

- [ ] **Step 1: Read the current exact dicts in both files**

```bash
sed -n '1,70p' ascent/alpha/stack.py
sed -n '1,80p' ascent/research/self_improve.py
```

- [ ] **Step 2: Read the existing tests that pin these dicts**

```bash
cat tests/test_alpha_stack_weights.py 2>/dev/null
cat tests/alpha/test_stack_weights.py 2>/dev/null
cat tests/test_self_evolving_alpha.py 2>/dev/null
```

Confirm which assertions check the key set / weight sum / specific sleeve presence — these will
need updating, not deleting (a test asserting `sum(weights.values()) == 1.0` should still pass
and should stay; a test asserting `"trend" in weights` needs to change to assert it's *absent*
now, or be removed if it was testing trend-specific logic no longer applicable).

- [ ] **Step 3: Apply the reduction**

In `ascent/alpha/stack.py`:
```python
DEFAULT_ALPHA_WEIGHTS = {
    "meanrev": 0.50,
    "statarb": 0.50,
}
```
Delete `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` entirely — Task 2 removes the code that reads it, so
leaving it here first is fine but it must be gone by the end of Task 2 too; doing it now in one
place is simpler than tracking a stub.

In `ascent/research/self_improve.py`, mirror the exact same `DEFAULT_ALPHA_WEIGHTS`, and prune
`MIN_SLEEVE_WEIGHTS` to only keys that still exist in the reduced dict (if `meanrev`/`statarb`
had floors defined, keep those; delete floors for `trend`/`earnings`/`analyst`/`options_flow`/
`insider`/`short_interest`).

- [ ] **Step 4: Update the tests to match**

Adjust assertions per Step 2's findings. Add one new test if none exists:
`test_default_alpha_weights_key_sets_match()` asserting
`set(stack.DEFAULT_ALPHA_WEIGHTS) == set(self_improve.DEFAULT_ALPHA_WEIGHTS) == {"meanrev", "statarb"}`
— check first whether the existing test suite already has this exact assertion (CLAUDE.md says
"the guard enforces" this, so a guard/check likely exists somewhere — find it, e.g.
`scripts/verify_docs.py`, rather than assuming you need to write it from scratch).

- [ ] **Step 5: Run tests, syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/alpha/stack.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('ascent/research/self_improve.py').read())"
.venv/bin/python -m pytest tests/test_alpha_stack_weights.py tests/alpha/ tests/test_self_evolving_alpha.py -v
.venv/bin/python scripts/verify_docs.py --quiet   # if this checks the key-set guard, confirm it still passes
```

- [ ] **Step 6: Commit**

```bash
git add ascent/alpha/stack.py ascent/research/self_improve.py tests/
git commit -m "feat(alpha): reduce DEFAULT_ALPHA_WEIGHTS to meanrev+statarb

Only these 2 of 15 sleeves showed a statistically significant positive
walk-forward IC in the proof audit (sub-projects 1/1b/1c). The other 13
are either measured-CUT or excluded-unmeasured, per the target
architecture design (docs/superpowers/specs/2026-08-14-target-architecture-design.md).
Sleeve code itself is untouched -- only the live weight table shrinks.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Remove regime-conditional weight adjustment

**Why:** `regime_overlay` scored CUT (p=0.35, no proven value). `regime_adjust_sleeve_weights`
is the mechanism that varies sleeve weights by regime — remove its call site so
`build_alpha_stack()` always uses the flat `DEFAULT_ALPHA_WEIGHTS`.

**Files:**
- Modify: `ascent/alpha/stack.py` — the regime-label resolution and `regime_adjust_sleeve_weights`
  call inside `build_alpha_stack()` (research found this around line 212-228), and the
  regime-keyed logic inside `_load_active_alpha_weights()` (around line 86-108)
- Test: whatever exists for `build_alpha_stack` — find and read first

**Interfaces:**
- Consumes: nothing new
- Produces: `build_alpha_stack()` keeps its existing signature (still accepts a `regime_signal`
  parameter, since callers pass one) but no longer branches on it for weight adjustment — the
  parameter becomes accepted-but-unused for this purpose. Confirm no other logic inside
  `build_alpha_stack()` still needs `regime_signal` before deciding whether to drop the parameter
  entirely or just stop using it for weights (check the function body — CLAUDE.md mentions regime
  signals are used elsewhere too, e.g. `apply_hedge_overlay` takes a regime separately in Task 3,
  and hedge overlay's regime comes from `agent_outputs`, not from inside `build_alpha_stack`, so
  this may be independent — verify, don't assume).

- [ ] **Step 1: Read `build_alpha_stack()` and `_load_active_alpha_weights()` in full, current state**

```bash
sed -n '1,260p' ascent/alpha/stack.py
```

Confirm the exact current call site of `regime_adjust_sleeve_weights` and every other use of
`regime_signal` inside this file — do not remove a use that's unrelated to weight adjustment.

- [ ] **Step 2: Find and read existing tests**

```bash
grep -rln "build_alpha_stack\|regime_adjust_sleeve_weights" tests/
```

- [ ] **Step 3: Remove the regime-conditional weight branch**

Delete the `if regime_signal is not None:` block that calls `regime_adjust_sleeve_weights`
(and the regime-label resolution feeding `_load_active_alpha_weights(regime=...)`, since
`DEFAULT_ALPHA_WEIGHTS_BY_REGIME` no longer exists after Task 1) — leave any *other*
`regime_signal` use in the function untouched if Step 1 found one.

- [ ] **Step 4: Update/add tests**

Confirm `build_alpha_stack()` produces the same composite regardless of what `regime_signal` is
passed (a test constructing two calls, one with a regime signal and one without, asserting
identical output, is a strong regression guard here).

- [ ] **Step 5: Run tests, syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/alpha/stack.py').read())"
.venv/bin/python -m pytest <test files found in Step 2> -v
```

- [ ] **Step 6: Commit**

```bash
git add ascent/alpha/stack.py tests/
git commit -m "feat(alpha): remove regime-conditional sleeve weight adjustment

regime_overlay scored CUT (p=0.35) in the proof audit -- no proven value.
build_alpha_stack() now always uses the flat DEFAULT_ALPHA_WEIGHTS.
apply_regime_to_portfolio (a different function, backtest-path only) is
untouched.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Remove hedge overlay

**Why:** `hedge_overlay` scored CUT (p=0.35, same measurement pair as `regime_overlay` — both
approximated onto the same counterfactual track pending a dedicated one, but both share the same
"no proven value" verdict).

**Files:**
- Modify: `run_all_agents.py` — the `apply_hedge_overlay` call site (research found it at
  ~Step 5b, right after `run_orchestrator()`, ~line 1202-1229 including the log-append)
- Test: `tests/test_hedge_overlay.py` (likely tests `apply_hedge_overlay` itself — keep those,
  since the function isn't deleted, only its call site in the live pipeline)

**Interfaces:**
- Consumes: nothing new
- Produces: no interface change to `ascent/portfolio/hedge_overlay.py` itself (not deleted, per
  global constraints) — only `run_all_agents.py` stops calling it.

- [ ] **Step 1: Read the current exact call site**

```bash
sed -n '1180,1240p' run_all_agents.py
```

Confirm the exact block boundaries (where `_hedge_regime` resolution starts, where the
log-append after `apply_hedge_overlay` ends) before deleting.

- [ ] **Step 2: Delete the block**

Remove the `_hedge_regime` resolution, the `apply_hedge_overlay(merged_weights, _hedge_regime)`
call, and its log-append to `logs/hedge_log.jsonl`. Confirm `merged_weights` (or whatever
variable the overlay was mutating) is used correctly by the code immediately after this block
with the overlay simply absent — i.e., the pipeline continues with the pre-overlay weights.

- [ ] **Step 3: Confirm `ascent/portfolio/hedge_overlay.py` and its tests are untouched**

```bash
.venv/bin/python -m pytest tests/test_hedge_overlay.py -v
```
Expected: still passes — this task doesn't touch that file.

- [ ] **Step 4: Syntax check, and confirm `run_all_agents.py` still imports cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -c "import run_all_agents"
```

- [ ] **Step 5: Commit**

```bash
git add run_all_agents.py
git commit -m "feat(portfolio): remove hedge overlay from the live pipeline

hedge_overlay scored CUT in the proof audit -- no proven value.
ascent/portfolio/hedge_overlay.py itself is untouched (not deleted);
only the run_all_agents.py call site is removed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: Exclude macro/international/alternatives agents from live capital

**Why:** `macro_agent`/`international_agent` both scored CUT on their real universes.
`alternatives_agent` is still `INSUFFICIENT_DATA` (unexplained density issue, unrelated to the
now-fixed cache bug) — excluded per the "revisit unmeasured components individually" decision,
not proven negative. `orchestrator/central_intelligence.py`'s merge already degrades gracefully
to fewer agents (confirmed by research: `_compute_allocation()` only builds entries for agents
present in `agent_outputs`, `merge_agent_outputs()` defaults an absent agent to 0.0) — this task
is primarily about NOT INVOKING these three agents in the daily run, not restructuring the merge.

**Files:**
- Modify: `run_all_agents.py` — wherever it invokes the 4 specialist agents (investigate exact
  call sites first, this plan does not assume a specific mechanism — subprocess, direct import,
  or something else)
- Modify: `orchestrator/central_intelligence.py` — simplify `BASE_ALLOCATION`/
  `STRESSED_ALLOCATION`/`CRISIS_ALLOCATION` and `_apply_crisis_veto()` (research found these
  become inert with 1 agent present; simplify rather than leave dead branches, but do NOT
  restructure `_compute_allocation()`/`merge_agent_outputs()` themselves — they already handle
  this correctly)
- Test: `tests/test_phase1_hardening.py`, `tests/test_plan_b.py` (read first, find what they
  assert about the 4-agent case, update to the 1-agent case)

**Interfaces:**
- Consumes: nothing new
- Produces: `run_all_agents.py`'s daily orchestration invokes only `us_equities_agent`.
  `orchestrator/central_intelligence.py`'s public merge functions keep their existing signatures.

- [ ] **Step 1: Investigate the exact agent-invocation mechanism**

```bash
grep -n "macro_agent\|international_agent\|alternatives_agent\|run_macro_agent\|run_international_agent\|run_alternatives_agent" run_all_agents.py
```

Read enough surrounding context to understand exactly how each agent is invoked (function call,
subprocess, thread — CLAUDE.md mentions `start_event_agent_thread` as one pattern used elsewhere
in this file, worth checking if agents use a similar pattern) and what happens to its output
afterward (passed into `agent_outputs` list feeding the orchestrator, presumably).

- [ ] **Step 2: Read `orchestrator/central_intelligence.py`'s allocation dicts and crisis veto**

```bash
sed -n '190,230p' orchestrator/central_intelligence.py
sed -n '395,530p' orchestrator/central_intelligence.py
sed -n '585,645p' orchestrator/central_intelligence.py
```

- [ ] **Step 3: Read the existing tests**

```bash
cat tests/test_phase1_hardening.py 2>/dev/null | head -100
cat tests/test_plan_b.py 2>/dev/null | head -100
```

Identify which tests assert on multi-agent behavior (4 agents present) vs. single-agent
behavior — you'll need to update the former, the latter should already pass unchanged.

- [ ] **Step 4: Remove the three agents' invocation in `run_all_agents.py`**

Stop calling `run_macro_agent`/`run_international_agent`/`run_alternatives_agent` (or whatever
the real call pattern is) in the daily orchestration flow. `agent_outputs` should end up
containing only the `us_equities_agent`'s output. Do not delete `agents/macro_agent.py`,
`agents/international_agent.py`, `agents/alternatives_agent.py`, or their `run_*_agent()`
functions — only the call sites in `run_all_agents.py`.

- [ ] **Step 5: Simplify `orchestrator/central_intelligence.py`'s now-inert branches**

Simplify `BASE_ALLOCATION`/`STRESSED_ALLOCATION`/`CRISIS_ALLOCATION` and `_apply_crisis_veto()`
to reflect a single-agent reality — read the current code carefully first (Step 2's output) and
make the minimal change that removes dead multi-agent branching without altering
`_compute_allocation()`/`merge_agent_outputs()`'s core degradation logic, which research confirmed
already works correctly for this case. If in doubt about how far to simplify, err toward less
change here — the core merge logic doesn't need restructuring, only the dead per-agent
allocation constants/veto logic that assumed 4 agents would ever be present.

- [ ] **Step 6: Update tests**

Update `tests/test_phase1_hardening.py`/`tests/test_plan_b.py`'s multi-agent assertions to match
the new single-agent reality. Add a regression test confirming `run_all_agents.py`'s daily flow
produces `agent_outputs` with exactly one entry (`us_equities`).

- [ ] **Step 7: Run tests, syntax check, import check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('orchestrator/central_intelligence.py').read())"
.venv/bin/python -c "import run_all_agents"
.venv/bin/python -m pytest tests/test_phase1_hardening.py tests/test_plan_b.py -v
```

- [ ] **Step 8: Commit**

```bash
git add run_all_agents.py orchestrator/central_intelligence.py tests/
git commit -m "feat(orchestrator): only us_equities_agent allocates live capital

macro_agent/international_agent scored CUT on their real universes;
alternatives_agent is still unmeasured (unexplained density issue,
unrelated to the now-fixed save_parquet bug) -- excluded pending future
re-measurement, not proven negative. Agent code is untouched, not
deleted; only the daily invocation and orchestrator's now-inert
multi-agent allocation branches are removed. central_intelligence.py's
core merge/allocation functions already degraded gracefully to fewer
agents and needed no restructuring.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: Remove the judge's live position-change write path

**Why:** `debate_judge_intervention` scored CUT (p=0.75, n=47 — underpowered but still no proven
value; same evidence bar applied to every other component in this rebuild). Two call sites,
confirmed by research.

**Files:**
- Modify: `run_all_agents.py` — delete both `apply_judge_position_change` call sites
  (scheduled-rebalance path, research found ~line 1913-1916; discovery/mini-rebalance path,
  ~line 2979-2982). Do NOT delete `apply_judge_position_change` itself, `run_debate()`, or any
  of `debate/`.
- Test: `tests/test_judge_change_applied_everywhere.py` — read first; this test's name implies
  it currently asserts the write path IS applied everywhere. Its assertions need to flip to
  confirm the write path is now a no-op / not called, not simply be deleted (a passing test that
  no longer tests the removed behavior is worse than no test).

**Interfaces:**
- Consumes: nothing new
- Produces: `run_debate()` still runs, still produces a verdict with `position_changes`, but
  nothing in `run_all_agents.py` applies it to `merged_weights` anymore.

- [ ] **Step 1: Read both call sites in full, current state**

```bash
sed -n '1895,1930p' run_all_agents.py
sed -n '2965,2995p' run_all_agents.py
```

Confirm exact boundaries — the call itself plus any surrounding logging/bookkeeping that should
either stay (e.g. logging that the verdict *would have* changed a position, for continued
visibility) or go with it. Prefer keeping observability (log what the verdict proposed) while
removing the actual weight mutation — check whether existing code already separates "log the
verdict" from "apply the verdict" before deciding whether any logging needs to be preserved
explicitly.

- [ ] **Step 2: Read the existing test**

```bash
cat tests/test_judge_change_applied_everywhere.py
```

- [ ] **Step 3: Remove both call sites**

Delete the `apply_judge_position_change(...)` calls at both locations. Leave `run_debate()` and
verdict-logging untouched.

- [ ] **Step 4: Rewrite the test to assert the new behavior**

Rename/rewrite `tests/test_judge_change_applied_everywhere.py` to assert the *opposite* of its
current name — the judge's position change is no longer applied on either path, even when a
verdict proposes one. Keep the test's fixture/setup pattern, flip the assertions.

- [ ] **Step 5: Run tests, syntax check, import check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -c "import run_all_agents"
.venv/bin/python -m pytest tests/test_judge_change_applied_everywhere.py -v
```

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py tests/test_judge_change_applied_everywhere.py
git commit -m "feat(debate): remove the judge's live position-change write path

debate_judge_intervention scored CUT (p=0.75) in the proof audit. Both
call sites (scheduled-rebalance and discovery paths) are removed; debate
itself keeps running and logging verdicts for continued measurement.
apply_judge_position_change, debate/, and adversarial_authority.py are
untouched -- only the two call sites that invoked the write are gone.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: Remove the AI PM's earned-authority blend write path

**Why:** `earned_authority` scored CUT (p=0.35, `track_d` vs `track_astar`) — this measures the
AI PM's own portfolio blended into `merged_weights` via `ascent/strategy/earned_authority.py`'s
`authority_blend`/`blend()`, a *distinct* mechanism from Task 5's judge write path. Same evidence
bar, same treatment: remove the write, keep the analysis layer running.

**Files:**
- Modify: `run_all_agents.py` — delete the AI PM blend-into-`merged_weights` call site (research
  found `authority_blend(ai_pm_result.portfolio, merged_weights)` around line 1524) and the
  `update_authority(...)` call (~line 2299) that feeds `data_cache/earned_authority.json`'s
  ladder. Do NOT delete `ascent/strategy/earned_authority.py`, `agents/ai_pm_agent.py`, or the AI
  PM's Phase 1/Phase 2 execution itself.
- Test: `tests/strategy/test_earned_authority_blend.py`, `tests/test_derive_overrides.py` — read
  first; likely need the same "flip assertions to confirm the blend no longer applies" treatment
  as Task 5.

**Interfaces:**
- Consumes: nothing new
- Produces: `run_ai_pm(...)` still runs, still produces `ai_pm_result.portfolio` and reasoning,
  but nothing blends it into `merged_weights` anymore. `agents/ai_pm_agent.py:1110` and
  `ascent/strategy/ai_pm_perf_feedback.py`'s reads of `earned_authority.json` keep working (the
  file just stops being updated by the live loop once `update_authority` is no longer called from
  it — confirm whether that breaks anything reading it, or whether it should keep being called
  for measurement purposes even though its output no longer gates a live write. **Investigate
  this before deciding** — if `update_authority` genuinely has no other effect than gating the now
  removed blend, removing its call site is correct; if it also does something read elsewhere
  that this task shouldn't disturb, keep the call and only remove the blend application itself).

- [ ] **Step 1: Read the AI PM blend call site and `update_authority` call site, current state**

```bash
sed -n '1505,1540p' run_all_agents.py
sed -n '2285,2315p' run_all_agents.py
```

- [ ] **Step 2: Read `ascent/strategy/earned_authority.py`'s `authority_blend`/`blend()`/`update_authority`**

```bash
cat ascent/strategy/earned_authority.py
```

Confirm what `update_authority` actually does — does it ONLY feed the gate that decides how much
of the AI PM's portfolio gets blended in, or does it have side effects read elsewhere (e.g.
`ai_pm_perf_feedback.py`)? This determines whether Step 4 removes both calls or just the blend
application.

- [ ] **Step 3: Read the existing tests**

```bash
cat tests/strategy/test_earned_authority_blend.py
cat tests/test_derive_overrides.py
```

- [ ] **Step 4: Remove the write path (scope per Step 2's finding)**

Remove the blend-into-`merged_weights` call. Remove `update_authority`'s call site too, UNLESS
Step 2 found it has a genuinely independent, still-needed side effect — if so, keep that call and
document why in the commit message.

- [ ] **Step 5: Rewrite tests to assert the new behavior**

Same pattern as Task 5: flip assertions to confirm the AI PM's portfolio no longer gets blended
into `merged_weights`, keeping existing fixture/setup structure.

- [ ] **Step 6: Run tests, syntax check, import check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -c "import run_all_agents"
.venv/bin/python -m pytest tests/strategy/test_earned_authority_blend.py tests/test_derive_overrides.py -v
```

- [ ] **Step 7: Commit**

```bash
git add run_all_agents.py tests/
git commit -m "feat(ai-pm): remove the AI PM's earned-authority blend write path

earned_authority scored CUT (p=0.35, track_d vs track_astar) in the proof
audit -- a distinct write path from the judge's position-change (Task 5),
easy to conflate by name but measuring a different mechanism. AI PM Phase
1/2 execution and ascent/strategy/earned_authority.py are untouched; only
the blend-into-merged_weights call site is removed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: Update CLAUDE.md constraints

**Why:** Constraint #5 describes the judge write-path exception this plan just removed —
leaving it as-is would make CLAUDE.md assert something false about the live code, the exact
drift failure mode CLAUDE.md's own preamble warns against. Constraint #6 stays true but should
reflect the new 2-key set, not describe it as if it were still 15 keys.

**Files:**
- Modify: `CLAUDE.md` — constraints #5 and #6 in the "Integrity constraints" section

- [ ] **Step 1: Read the current exact text of both constraints**

```bash
grep -n "^5\.\|^6\." CLAUDE.md | head -5
sed -n '108,130p' CLAUDE.md
```

- [ ] **Step 2: Rewrite constraint #5**

Replace the judge-exception carve-out with: debate is advisory-only, full stop — no live write
exception. State that this was a deliberate change (sub-project 2, 2026-08-14) after
`debate_judge_intervention` scored CUT in the proof audit, and that debate keeps running and
logging for continued measurement. Keep the constraint numbered #5 for continuity with any
other doc cross-references — check `docs/REPO_MAP.md` and any other file that might reference
"constraint #5" by number before renumbering anything.

- [ ] **Step 3: Rewrite constraint #6**

State the new 2-key `DEFAULT_ALPHA_WEIGHTS` set and that the guard still enforces the two files'
key sets match — just on a smaller set now.

- [ ] **Step 4: Grep for other now-stale references**

```bash
grep -n "apply_judge_position_change\|DEFAULT_ALPHA_WEIGHTS\|earned_authority\|hedge_overlay\|regime_adjust_sleeve_weights" CLAUDE.md docs/REPO_MAP.md
```

Update any other passage in these two files that describes the removed mechanisms as if they
were still live (e.g. the "Non-obvious gotchas" section may reference `LONG_SHORT_ENABLED`'s
precondition in a way that assumes the judge write path exists — read carefully, don't
over-edit unrelated gotchas).

- [ ] **Step 5: Run the doc guard**

```bash
.venv/bin/python scripts/verify_docs.py --quiet
```

Expected: 0 failures (or the same pre-existing failures noted in sub-project 1c's final review,
not new ones).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/REPO_MAP.md
git commit -m "docs(claude-md): update integrity constraints for the target architecture

Constraint 5's judge write-path exception is removed -- debate is now
advisory-only, no exception, after debate_judge_intervention scored CUT.
Constraint 6 reflects the 2-key DEFAULT_ALPHA_WEIGHTS set.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ -q -W error 2>&1 | tail -60
```

Compare failures against sub-project 1c's already-documented pre-existing/environmental failure
list (API keys, network, deprecation warnings, unrelated to this plan's files) — any NEW failure
in a file this plan touched is a real regression, investigate before proceeding.

- [ ] **Step 2: Confirm `run_all_agents.py` and `orchestrator/central_intelligence.py` still import cleanly**

```bash
.venv/bin/python -c "import run_all_agents"
.venv/bin/python -c "import orchestrator.central_intelligence"
```

- [ ] **Step 3: If a dry-run mode exists for `run_all_agents.py`, exercise it**

```bash
grep -n "dry.run\|dry_run\|--dry" run_all_agents.py | head -10
```

If a dry-run flag exists, run it and confirm the pipeline completes without error end-to-end
with the new (smaller) agent/write-path surface — report the outcome even if you can't run it
(e.g. if it requires live credentials this session doesn't have), noting exactly why.

- [ ] **Step 4: Report final state**

Summarize: which of the 8 tasks' changes are confirmed working together (not just individually
tested), any residual concern, and whether this plan's done-criteria are met.

---

## Done criteria

```bash
.venv/bin/python -m pytest tests/ -q -W error
.venv/bin/python -c "import run_all_agents"
.venv/bin/python scripts/verify_docs.py --quiet
```

- No new test failures beyond the already-documented pre-existing/environmental set.
- `DEFAULT_ALPHA_WEIGHTS` is `{"meanrev": 0.50, "statarb": 0.50}` in both files, matching.
- `run_all_agents.py` no longer calls `apply_hedge_overlay`, `apply_judge_position_change`, or
  the AI PM earned-authority blend; no longer invokes `macro_agent`/`international_agent`/
  `alternatives_agent`.
- None of the excluded components' source files were deleted.
- `CLAUDE.md` accurately describes the new state.

## Explicitly out of scope

- Deleting any excluded component's code.
- `ascent/main.py`'s backtest-path `apply_regime_to_portfolio`.
- Any kill-switch flag changes.
- Resuming `com.ascentcapital.eod`/`.heartbeat` — that's sub-project 4, after sub-project 3
  (subagent-driven rebuild of anything this plan's investigation reveals needs deeper work) and
  validation.
- Re-splitting `meanrev`/`statarb`'s weight ratio.
