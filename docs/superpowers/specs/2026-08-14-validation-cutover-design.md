# Validation & Cutover — Design Spec

**Sub-project 4** (final) of the strip-down/rebuild. Sub-project 3 ("subagent-driven rebuild")
was effectively already executed as part of sub-project 2 — the target architecture was built
task-by-task via `subagent-driven-development`, not designed-then-separately-rebuilt. This spec
covers only what remains: validating the rebuilt pipeline with real walk-forward evidence, then
deciding whether to resume live trading.

## A real validation trap, found by research before writing this spec

Two separate walk-forward frameworks exist in this repo, and only one is safe to use:

- **`scripts/run_ascent_wf.py`'s default path** (`AscentPortfolioStrategy`) does NOT read
  `ascent/alpha/stack.py::DEFAULT_ALPHA_WEIGHTS` as-is — `_make_alpha_weights()`
  (`ascent/research/wf_framework/ascent_strategy.py:629-666`) unconditionally reintroduces
  `trend` at a grid-searched weight and renormalizes `meanrev` down to fill the remainder. A
  naive run through this path would silently backtest a 3-sleeve trend/statarb/meanrev blend —
  **not** the rebuilt 2-sleeve stack — and would produce numbers that look like validation but
  aren't.
- **`ascent/research/walk_forward_runner.py::walk_forward_pipeline()`** calls
  `build_alpha_stack(hist_features, agent_id="us_equities")` with no weight override, which
  correctly falls through to the real, current `DEFAULT_ALPHA_WEIGHTS` (`{"meanrev": 0.50,
  "statarb": 0.50}`). **This is the only framework that validates what was actually built.**

**Decision: use `walk_forward_runner.py`, not `run_ascent_wf.py`'s default path.**

## Scope: alpha/portfolio-layer validation, not a new orchestrator-level backtest

Neither framework exercises the orchestrator's agent-blend by default (that requires
`run_ascent_wf.py --live-system`, which uses a separately-optimized `FullOrchestrationStrategy`
class that would need real adaptation work to reflect single-agent allocation — out of scope
here, would be its own engineering effort). This is an acceptable scope boundary, not a gap:
`macro_agent`/`international_agent`/`alternatives_agent` were excluded because they scored CUT
or stayed unmeasured, and `regime_overlay`/`hedge_overlay` were removed because they scored
CUT — none of their removal requires new backtest evidence to justify, since the audit already
supplied the evidence for removing them. What genuinely needs walk-forward validation is
whether the *surviving* 2-sleeve alpha stack, sized through the existing (unchanged) portfolio
construction machinery, produces a sound risk-adjusted return profile on its own — that is
exactly what `walk_forward_runner.py` tests.

## What the validation run must report

Real numbers from a real run, not reconstructed: CAGR, volatility, Sharpe, Sortino, max
drawdown, win rate, WFE, `n_folds`, `n_oos_days`, alpha, beta — matching the existing
`wf_report_*.json` schema so the new result is directly comparable to prior published numbers
(e.g. `outputs/wf_results/wf_report_stratvol_2026-07-27.json`, the most recent).

**Known, disclosed caveat carried into this run, not fixed here:** 3 of 21 folds (17, 18, 20)
return `OOS Sharpe = 0.0` from a stale-cache collision in `AscentPortfolioStrategy`'s
memoization (`ascent_strategy.py:67-68,159` colliding with `optimizer.py:105`) — this is a
`run_ascent_wf.py`-specific bug; confirm whether `walk_forward_runner.py` shares it (research
did not confirm either way) before citing fold counts. If it does, report both the raw and the
dead-fold-adjusted Sharpe/WFE, per this project's own standing rule that a known-buggy number
must be disclosed as such, not silently used.

## Cutover decision

After a real validation run, decide whether to resume live trading
(`launchctl load com.ascentcapital.eod.plist` + `.heartbeat.plist`) based on whether the result
clears a reasonable bar: positive Sharpe, drawdown not obviously worse than the pre-rebuild
system's own historical numbers (`CURRENT_VERIFIED_NUMBERS.md`), and no crash/error in the run
itself. This is not a rubber-stamp step — if the numbers are bad or the run fails, do not
resume live trading; report the failure and stop for reconsideration instead.

## Explicitly out of scope

- Building or adapting `FullOrchestrationStrategy`/`run_ascent_wf.py --live-system` to reflect
  the single-agent, overlay-free architecture — a separate engineering effort.
- Fixing the 3-dead-folds bug — disclose it, don't fix it (matches this project's established
  "a known-buggy number is often a real, disclosable bug, not something to silently patch
  mid-validation" discipline).
- Re-running the proof audit itself (`scripts/run_proof_audit.py`) — already current as of
  `checkpoint-4-cache-repair-final`.
- Paper-shadow testing against live market data in real time — that requires actually running
  the pipeline forward, which only makes sense after this backtest passes and trading resumes;
  it's the natural monitoring phase after cutover, not a precondition for it.
