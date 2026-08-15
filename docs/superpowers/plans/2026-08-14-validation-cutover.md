# Validation & Cutover Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce real walk-forward evidence for the rebuilt 2-sleeve alpha stack, then decide
whether to resume live trading based on that evidence.

**Architecture:** This is a validation-and-decision plan, not a code-change plan — most tasks
are "run something real and report the real numbers," not TDD. Task 3 may involve a real,
irreversible-feeling action (reloading the live scheduler) gated on Task 2's results.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`.

## Global Constraints

- Always use `.venv/bin/python`.
- Use `ascent/research/walk_forward_runner.py::walk_forward_pipeline()`, NOT
  `scripts/run_ascent_wf.py`'s default path — research confirmed the latter's
  `AscentPortfolioStrategy` silently overrides the rebuilt 2-sleeve weights with its own
  grid-searched trend blend, which would validate the wrong system. `walk_forward_runner.py`
  correctly falls through to the current `DEFAULT_ALPHA_WEIGHTS` since it calls
  `build_alpha_stack()` with no weight override.
- `walk_forward_pipeline()` does NOT natively emit the `wf_report_*.json` schema (CAGR/Sharpe/
  Sortino/WFE) — that belongs to the other (unused) framework. It prints daily-return summary
  stats and writes `ascent_daily_ledger.csv`/`ascent_holdings_ledger.csv` to the working
  directory. Compute standard metrics (Sharpe, Sortino, CAGR, max drawdown) from the resulting
  equity curve — check whether `ascent/research/wf_framework/metrics.py::PerformanceAnalyzer`
  (or a similar existing helper) can be reused for this rather than writing new metric code from
  scratch; investigate before deciding.
- Do not fix the "3 dead folds" bug (`AscentPortfolioStrategy`'s memoization collision,
  `ascent_strategy.py:67-68,159`) — first confirm whether `walk_forward_runner.py` even shares
  it (it's a different framework/class), and only disclose if it does.
- The cutover decision (Task 3) is NOT automatic — only resume live trading if Task 2's results
  clear a reasonable bar (positive Sharpe, drawdown not obviously worse than
  `CURRENT_VERIFIED_NUMBERS.md`'s pre-rebuild figures, no crash in the run). If the run fails or
  the numbers are bad, report that and do NOT reload the scheduler.

---

## Task 1: Confirm the real invocation and pre-flight checks

**Why:** `walk_forward_pipeline()`'s exact parameter defaults (`train_days`, `purge_days`,
`top_n`, `max_weight`) come from `get_config()` — confirm what they resolve to before a ~30min
run, and confirm the function can actually run standalone (data availability, no missing
imports) with a fast smoke check before committing to the full run.

- [ ] **Step 1: Read `walk_forward_pipeline()`'s full signature and defaults**

```bash
sed -n '1,120p' ascent/research/walk_forward_runner.py
.venv/bin/python -c "from ascent.config.settings import get_config; c = get_config(); print(c.walk_forward.train_days, c.walk_forward.purge_days, c.backtest.top_n, c.backtest.max_weight)"
```

- [ ] **Step 2: Check for a smoke-test / fast-mode option**

Look for a `--smoke` equivalent or a way to run a short slice first (e.g. limit fold count) to
confirm the pipeline runs end-to-end before committing to the full ~30min run. If none exists,
skip this step and proceed directly to Task 2, but note the risk in your report.

- [ ] **Step 3: Confirm `ascent/research/wf_framework/metrics.py::PerformanceAnalyzer`'s real
interface** (or whatever the right metrics helper turns out to be)

```bash
sed -n '1,120p' ascent/research/wf_framework/metrics.py
```

Confirm what inputs it needs (an equity curve? a returns series?) and whether it can be
constructed from `walk_forward_pipeline()`'s output (`ascent_daily_ledger.csv` or the in-memory
`fold_results`/`combined_weights`) without modification.

- [ ] **Step 4: Report findings** — real config values, smoke-test feasibility, metrics helper
interface. No commit needed (investigation only).

---

## Task 2: Run the real walk-forward validation

**Why:** This is the actual evidence this whole sub-project exists to produce.

- [ ] **Step 1: Run the full walk-forward pipeline**

```bash
.venv/bin/python -c "
from ascent.research.walk_forward_runner import walk_forward_pipeline
walk_forward_pipeline()
"
```

Expected runtime: allow up to ~30 minutes based on the other framework's documented runtime;
this framework's actual runtime is unconfirmed — if it runs meaningfully longer, let it
complete rather than killing it, and report the real elapsed time.

- [ ] **Step 2: Compute standard metrics from the output**

Using whatever Task 1 found is the right approach — construct Sharpe, Sortino, CAGR, max
drawdown (and WFE if computable — check whether this framework has an OOS-vs-IS split to derive
it from, since WFE specifically requires that) from `ascent_daily_ledger.csv` or the in-memory
results. Report every number with a citation to how it was computed (which file/column), not
reconstructed from memory or assumption.

- [ ] **Step 3: Disclose the dead-folds question**

Confirm whether `walk_forward_runner.py` shares the memoization-collision bug found in
`AscentPortfolioStrategy` (a different class — check if `walk_forward_runner.py` even uses that
class, or has its own fold-execution path free of this specific bug). Report which is true, with
evidence (grep for the collision's actual mechanism — class-level memoization keyed without
`pit_boundary` — inside whatever strategy/execution code `walk_forward_runner.py` actually
calls).

- [ ] **Step 4: Compare against the pre-rebuild baseline**

Read `CURRENT_VERIFIED_NUMBERS.md` for the pre-rebuild system's own walk-forward Sharpe/drawdown
figures (if present and current) and report a side-by-side comparison — the new 2-sleeve
system's numbers vs. the old system's, with the caveat that the old numbers came from a
different (15-sleeve, overlay-included) configuration, so this is directional context, not an
apples-to-apples backtest comparison.

- [ ] **Step 5: Report the full real result** — every metric, its computation source, the
dead-folds disclosure, the baseline comparison, and an explicit recommendation: does this clear
a reasonable bar for resuming live trading (positive Sharpe, drawdown not obviously worse, clean
run) or not? State the recommendation plainly — this feeds Task 3's decision directly.

No commit needed for this task (a validation run, not a code change) — unless the run produces
an artifact worth preserving in the repo (e.g. copy `ascent_daily_ledger.csv` into
`outputs/wf_results/` with a dated name and commit it, matching this project's existing
convention of keeping dated backtest artifacts).

---

## Task 3: Cutover decision

**Why:** The final step of the entire strip-down/rebuild effort.

- [ ] **Step 1: Read Task 2's full report and recommendation.**

- [ ] **Step 2: If Task 2's recommendation is negative** (bad numbers, a crash, or any
reasonable doubt): report this clearly, do NOT reload the scheduler, and stop here. This is a
valid, expected outcome — do not force a positive recommendation to complete the plan.

- [ ] **Step 3: If Task 2's recommendation is positive:**

```bash
launchctl list | grep ascentcapital   # confirm still unloaded before touching anything
launchctl load ~/Library/LaunchAgents/com.ascentcapital.eod.plist
launchctl load ~/Library/LaunchAgents/com.ascentcapital.heartbeat.plist
launchctl list | grep ascentcapital   # confirm both now loaded
```

- [ ] **Step 4: Update `CLAUDE.md`'s "Current state" section and `CHECKPOINTS.md`** to reflect
that live trading has resumed, with the date and a pointer to this validation's results.

- [ ] **Step 5: Commit the doc updates.**

```bash
git add CLAUDE.md CHECKPOINTS.md outputs/wf_results/
git commit -m "chore: resume live trading after target-architecture validation

<real Sharpe/drawdown numbers from Task 2, cited>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Done criteria

- Task 2 produces real, artifact-backed walk-forward metrics for the rebuilt 2-sleeve system.
- Task 3 makes an evidence-based cutover decision — either resumes trading with clear
  justification, or explicitly declines to with clear reasoning. Both are valid completions of
  this plan; a negative result is not a failure of the plan.

## Explicitly out of scope

- Building/adapting orchestrator-level (`--live-system`) backtesting.
- Fixing the 3-dead-folds bug.
- Re-running the proof audit.
- Paper-shadow testing (that's the natural monitoring phase after cutover, not a precondition).
