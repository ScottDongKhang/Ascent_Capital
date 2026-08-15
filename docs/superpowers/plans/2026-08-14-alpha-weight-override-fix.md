# Alpha Weight Runtime Override Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the bug validation found — stale pre-rebuild files silently override
`DEFAULT_ALPHA_WEIGHTS` at runtime in both the backtest and the live pipeline — then re-validate
with a real walk-forward run.

**Architecture:** A data cleanup (delete 2 stale files) plus a small, precise code fix in
`ascent/alpha/stack.py::_get_gated_weights()`'s redistribution logic, then re-run the same
validation that found the bug.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pytest.

## Global Constraints

- Always use `.venv/bin/python`.
- This touches shared, live-trading-adjacent code (`ascent/alpha/stack.py`) — TDD, full test
  suite, no shortcuts.
- Do not change `IC_GATE_THRESHOLD`, the gating decision logic itself, or whether gating applies
  with only 2 sleeves — only fix the redistribution target.
- Confirmed by design-spec research: `_load_active_alpha_weights()` already falls through to
  `DEFAULT_ALPHA_WEIGHTS.copy()` when `data_cache/active_alpha_config.json` doesn't exist, and
  `_get_gated_weights()` already returns `alpha_weights` unchanged when
  `logs/sleeve_ic_log.jsonl` doesn't exist — deleting both files needs no code change for that
  half of the fix. Verify this is still true by reading the current code before deleting
  anything (don't assume the plan's earlier research is still accurate).

---

## Task 1: Delete the two stale override files

**Why:** `data_cache/active_alpha_config.json` (dated 2026-05-02, pre-rebuild 15-sleeve
snapshot) and `logs/sleeve_ic_log.jsonl` (references sleeves like `insider` that no longer
exist) both silently override the current `DEFAULT_ALPHA_WEIGHTS` at runtime.

**Files:** none (data operation only)

- [ ] **Step 1: Confirm the fallback behavior in the CURRENT code**

```bash
sed -n '1,130p' ascent/alpha/stack.py
```

Confirm `_load_active_alpha_weights()` genuinely falls through to `DEFAULT_ALPHA_WEIGHTS.copy()`
when `data_cache/active_alpha_config.json` doesn't exist, and `_get_gated_weights()` genuinely
returns `alpha_weights` unchanged when `logs/sleeve_ic_log.jsonl` doesn't exist. If either
assumption is wrong, STOP and report — do not delete the file until the fallback is confirmed.

- [ ] **Step 2: Back up both files, then delete them**

```bash
mkdir -p data_cache/.pre_delete_backup_2026-08-14 logs/.pre_delete_backup_2026-08-14
cp data_cache/active_alpha_config.json data_cache/.pre_delete_backup_2026-08-14/ 2>/dev/null || echo "already absent"
cp logs/sleeve_ic_log.jsonl logs/.pre_delete_backup_2026-08-14/ 2>/dev/null || echo "already absent"
rm -f data_cache/active_alpha_config.json logs/sleeve_ic_log.jsonl
```

- [ ] **Step 3: Confirm the fallback actually fires**

```bash
.venv/bin/python -c "
from ascent.alpha.stack import _load_active_alpha_weights, _get_gated_weights, DEFAULT_ALPHA_WEIGHTS
w = _load_active_alpha_weights()
print('loaded weights:', w)
assert set(w) == set(DEFAULT_ALPHA_WEIGHTS), f'expected {set(DEFAULT_ALPHA_WEIGHTS)}, got {set(w)}'
g = _get_gated_weights(w)
print('gated weights:', g)
assert g == w, 'gating should be a no-op with no IC log'
print('OK')
"
```

- [ ] **Step 4: Commit** (data_cache/ and logs/ are gitignored per this repo's convention —
confirm this before attempting to commit; if either file was actually tracked, that's worth
flagging since it contradicts the project's stated gitignore rules)

```bash
git status --short data_cache/active_alpha_config.json logs/sleeve_ic_log.jsonl
```

If untracked/gitignored (expected), there's nothing to commit for this step — note the deletion
in your report instead. If tracked, commit the deletion with a clear message explaining why.

---

## Task 2: Fix `_get_gated_weights()`'s hardcoded "redistribute to trend"

**Why:** The current code (`ascent/alpha/stack.py`) only redistributes freed weight when
`"trend" in result and result.get("trend", 0) > 0` — with the 2-sleeve stack, `trend` isn't a
key in `DEFAULT_ALPHA_WEIGHTS` at all, so freed weight from a gated sleeve is silently dropped
(the portfolio ends up under-invested, not reallocated), rather than the trend-dominant failure
mode seen when the stale config file was still present. Both are bugs; fix the general case.

**Files:**
- Modify: `ascent/alpha/stack.py::_get_gated_weights()` (currently ~line 66-129 — confirm exact
  lines by reading the file first, Task 1 doesn't touch this function so line numbers should
  still match, but verify)
- Test: find existing tests for `_get_gated_weights`/IC gating (`grep -rln
  "_get_gated_weights\|IC gate\|ic_gate" tests/`) — read them first

**Interfaces:**
- Consumes: nothing new
- Produces: `_get_gated_weights(alpha_weights, ic_log_path=..., window=...)` keeps its existing
  signature and return type (`dict`). Only the redistribution logic inside changes.

- [ ] **Step 1: Read the current exact function and any existing tests**

```bash
sed -n '66,130p' ascent/alpha/stack.py
grep -rln "_get_gated_weights\|IC gate\|ic_gate" tests/
```

- [ ] **Step 2: Write failing tests for the new behavior**

Cover at minimum:
1. A 2-sleeve dict (`{"meanrev": 0.5, "statarb": 0.5}`) with `meanrev` gated (via a synthetic
   IC log showing negative rolling IC for `meanrev`) — freed weight (0.5) should go entirely to
   `statarb` (→ `{"meanrev": 0.0, "statarb": 1.0}`), not be dropped, and NOT create a `trend`
   key.
2. A 3+ sleeve case with one gated — freed weight redistributes PROPORTIONALLY among the
   remaining non-zero sleeves (preserve their relative ratios), matching whatever the most
   sensible generalization of "give it to the survivors" is — write the test to pin the exact
   expected math once you've decided the algorithm (simple proportional redistribution by
   existing weight share is the natural choice; use it unless you find a reason not to).
3. **Every** sleeve in `alpha_weights` gated simultaneously — must return the ORIGINAL
   `alpha_weights` unchanged (fail-safe: don't produce an all-zero dict), not an empty/degenerate
   result.
4. The old `trend`-specific behavior — a legacy 15-sleeve-shaped dict where `trend` is present
   and other sleeves are gated — should still redistribute sensibly (proportionally, same
   algorithm as case 2 now applies uniformly; this doesn't need to specifically favor `trend`
   anymore, since the whole point of the fix is removing that hardcoding — write the test to
   confirm proportional behavior, not to preserve the old trend-favoring quirk).

Follow this codebase's established pattern for synthetic IC-log fixtures — check how existing
IC-gating tests (if any) construct a `sleeve_ic_log.jsonl`-shaped fixture, reuse that pattern.

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest <the test file> -v
```

- [ ] **Step 4: Implement the fix**

Replace the hardcoded `if freed > 0 and "trend" in result and result.get("trend", 0) > 0:
result["trend"] = round(result["trend"] + freed, 4)` block with proportional redistribution
among the surviving (non-gated, non-zero) sleeves in `result`. Handle the all-gated case (no
survivors) by returning the original `alpha_weights` unchanged instead of applying any zeroing.

- [ ] **Step 5: Run tests, syntax check, run the broader alpha test suite**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/alpha/stack.py').read())"
.venv/bin/python -m pytest <the test file> -v
.venv/bin/python -m pytest tests/alpha/ tests/test_alpha_stack_weights.py tests/test_self_evolving_alpha.py -v
```

- [ ] **Step 6: Commit**

```bash
git add ascent/alpha/stack.py <test file>
git commit -m "fix(alpha): redistribute IC-gated weight proportionally, not to hardcoded trend

_get_gated_weights() only redistributed freed weight when \"trend\" was
present in the weight dict -- with the 2-sleeve meanrev/statarb stack,
trend isn't a key at all, so freed weight from a gated sleeve was
silently dropped rather than reallocated, leaving the portfolio
under-invested. Now redistributes proportionally among surviving
sleeves; if every sleeve is gated simultaneously, returns the original
weights unchanged rather than producing an all-zero dict.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Re-run validation and report real results

**Why:** Confirm the fix actually resolves the observed failure mode before any cutover
decision is reconsidered.

- [ ] **Step 1: Confirm the stale files are still absent**

```bash
ls data_cache/active_alpha_config.json logs/sleeve_ic_log.jsonl 2>&1
```

Expected: both "No such file or directory".

- [ ] **Step 2: Re-run the real walk-forward validation**

```bash
.venv/bin/python -c "
from ascent.research.walk_forward_runner import walk_forward_pipeline
walk_forward_pipeline()
" > /tmp/wf_run_output_fixed.log 2>&1
```

This may take a while (the prior run completed in well under 30 minutes based on this
session's own timing — expect similar). Let it run to completion.

- [ ] **Step 3: Verify the fix actually took effect in this run**

```bash
grep "\[alpha_stack\] loaded=" /tmp/wf_run_output_fixed.log | sort -u
grep "\[Stack\] IC gate:" /tmp/wf_run_output_fixed.log | sort -u
```

Expected: `[alpha_stack] loaded=` shows only `['meanrev', 'statarb']` (or a subset if one is
occasionally gated) — NOT `trend`/`insider`/`fundamental`/etc. If IC gate messages appear, they
should show DIFFERENT values across different folds (genuine rolling computation), not one
static value repeated 300+ times. **If either check fails, the fix is incomplete — investigate
before reporting results, do not report broken numbers as if they were valid** (this is exactly
the mistake the prior validation attempt correctly avoided).

- [ ] **Step 4: Report the real performance numbers**

Extract Sharpe, Sortino, CAGR, max drawdown, hit rate, turnover from the run's printed
"PERFORMANCE REPORT" section. Compare against the prior (broken) run's numbers to confirm this
is meaningfully different (a real fix should produce a plausible hit rate, not ~4.6%).

- [ ] **Step 5: Copy artifacts and give an explicit recommendation**

Copy the log and any ledger CSVs into `outputs/wf_results/` with a dated name distinct from the
prior broken run's artifacts (e.g. `wf_run_target_architecture_2026-08-14_fixed.log`). State
plainly: does this now clear a reasonable bar for resuming live trading? Cite
`CURRENT_VERIFIED_NUMBERS.md`'s pre-rebuild figures for directional comparison, with the same
caveat as before (different configuration, not apples-to-apples).

- [ ] **Step 6: Commit the validation artifacts**

```bash
git add outputs/wf_results/
git commit -m "docs(wf): re-validation after alpha-weight override fix

<real numbers, cited>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Done criteria

- `data_cache/active_alpha_config.json` and `logs/sleeve_ic_log.jsonl` are absent.
- `_get_gated_weights()`'s tests pass, including the all-gated fail-safe case.
- A fresh validation run's `[alpha_stack] loaded=` lines show only the intended live sleeves.
- A clear, evidence-based recommendation exists for whether to resume live trading.

## Explicitly out of scope

- The cutover action itself (reloading the launchd scheduler) — a separate step after this
  plan's Task 3 produces trustworthy numbers, not automatic within this plan.
- Rebuilding `active_alpha_config.json`/`sleeve_ic_log.jsonl` with fresh correct content.
- Whether IC-gating should apply at all with only 2 live sleeves.
