# Dead risk-mechanism strip-down — report

Date: 2026-08-16

## Item 1+2: Regime `risk_multiplier` / `sleeve_adjustments` machinery

**Verified caller count before deletion:** zero live (non-test, non-self-referencing) callers
for all three functions:

- `apply_regime_to_portfolio` (`ascent/regime/integration.py`): referenced at
  `ascent/main.py:420` as an **import only** — never called anywhere in `main.py`. `git log -S`
  on `main.py` shows the string's occurrence count only ever incremented once (the import line
  itself) — it was never called and then removed; it was simply never called. `ascent/regime/
  engine.py` referenced it only in its module *docstring* ("Usage pattern" example) and via an
  unused import. `ascent/regime/__init__.py` re-exported it but nothing consumed the re-export.
  `PROJECT_STATUS.md:823` independently confirms the same finding in prose.
  - **Caveat found and judged not a blocker:** `patch.sh` (a heredoc shell script sitting in
    the repo root) contains an actual call to `apply_regime_to_portfolio` inside a block it
    would write into `ascent/main.py` if executed. It is not part of the running pipeline —
    it's a dormant/historical patch script, not imported or invoked by anything, and the code
    it would install does not match current `main.py`. Flagging it here per the task's
    "surprise caller" instruction, but treating it as non-live since nothing executes it.
  - Also found: two design docs (`docs/superpowers/plans/2026-08-14-target-architecture.md`,
    `docs/superpowers/specs/2026-08-14-target-architecture-design.md`) describe
    `apply_regime_to_portfolio` as "backtest-path only, out of scope" — this appears to
    describe an aspirational/historical state (matching `patch.sh`'s content) that does not
    match current `main.py`, which never calls it on any path, backtest or live. Current code
    wins; the docs are stale on this specific point.
- `regime_scale_weights` (`ascent/regime/integration.py`): zero external callers; only
  self-referenced inside `apply_regime_to_portfolio` (which is itself dead).
- `regime_adjust_sleeve_weights` (`ascent/regime/integration.py`): zero callers anywhere,
  including no callers in `ascent/alpha/stack.py::build_alpha_stack` — confirmed by the
  in-code comment there stating `regime_signal` is "accepted for call-site compatibility but
  no longer used" (this removal predates this session, per `stack.py`'s existing comment
  referencing the CUT proof-audit verdict for `regime_overlay`).

**What was done:**
- Deleted `regime_scale_weights`, `regime_adjust_sleeve_weights`, `apply_regime_to_portfolio`,
  and the now-orphaned `_BASE_SLEEVE_WEIGHTS` constant from `ascent/regime/integration.py`.
  Left `regime_max_weight`, `regime_sector_cap`, `regime_signal_threshold`,
  `regime_rebalance_band`, `regime_covariance_halflife`, `build_regime_series`,
  `get_signal_for_date` untouched — `regime_max_weight` in particular is the separate, live
  cap-tightening mechanism named in the task as explicitly off-limits.
- Rewrote the module docstring (stale "API contract" section referenced function names that
  never existed) to note what was removed and why.
- Removed the dead `apply_regime_to_portfolio` import from `ascent/regime/engine.py` (also
  removed from its "Usage pattern" docstring example) and from `ascent/regime/__init__.py`
  (which had two redundant import lines, the second fully subsuming the first).
- Updated the stale comment in `ascent/alpha/stack.py` that pointed at
  `apply_regime_to_portfolio` as "the separate, in-scope regime overlay on the backtest path."
- Removed the dead `apply_regime_to_portfolio` import at `ascent/main.py:420` (`from
  ascent.regime import RegimeEngine, apply_regime_to_portfolio` → `RegimeEngine` only).

**Judgment call — what was NOT removed:** `ascent/main.py`'s construction of `risk_multiplier`
/ `sleeve_adjustments` values (in the JSON-load fallback branch, ~line 388-398, and via
`regime_engine.get_signal()` in the full-fit branch) was **left in place**. Reasons:
1. These are required fields on the `RegimeSignal` dataclass — `RegimeSignal(...)` cannot be
   constructed without them, and `RegimeSignal` (specifically its `.label`) genuinely is
   consumed downstream (`regime_label=regime_signal.label.value` at `main.py:749`, and
   `regime_signal=regime_signal` passed into `build_alpha_stack`). Stripping the fields would
   require restructuring the dataclass, which is out of scope and riskier than leaving two
   inert attributes populated alongside a used one.
2. `regime_signal.risk_multiplier` **is** read once more, at the `print` around
   `ascent/main.py:512` (`f"risk_mult={regime_signal.risk_multiplier:.2f}"`). I left this
   print in place: it is a genuine diagnostic showing what the regime engine actually computed
   (useful for operators watching `[Regime]` log lines), not leftover dead-function plumbing —
   the value it prints is real, just not fed into portfolio weights. Removing it would reduce
   observability for no code-health benefit. `sleeve_adjustments` has zero remaining readers
   anywhere (including the print), but it's a single dataclass field with no separate
   construction cost, populated as part of the same `RegimeSignal(...)` call — not worth a
   partial dataclass surgery for.

## Item 3: Factor constraints builder

**Verified caller count before deletion:** `build_factor_constraints` and
`get_regime_factor_bounds` (`ascent/risk/factor_constraints.py`) had zero non-test callers.
The sole production reference was `ascent/main.py:752` (`factor_constraints=None,  # Plan 2
wires in...`), which never calls the builder — it hardcodes `None`. `ascent/portfolio/
mvo_optimizer.py` and `ascent/portfolio/optimizer.py` reference the builder only in docstring
comments describing the expected dict format for the (still-live, still-accepted)
`factor_constraints` parameter. `check_factor_bounds` in `ascent/risk/factor_exposure.py` is a
**different, confirmed-live** function (own module, own tests) — not touched.

**What was done:**
- Deleted `ascent/risk/factor_constraints.py` entirely (both functions, both private bound
  dicts — nothing else in the file had non-test callers).
- Left `ascent/main.py:752`'s `factor_constraints=None` call argument as-is, per instructions —
  the optimizer's parameter itself stays live and accepting; only the unused builder is gone.
- Updated the docstring comments in `ascent/portfolio/mvo_optimizer.py` (the
  `factor_constraints:` param doc) and `ascent/portfolio/optimizer.py` (a comment about the
  `"symbols"` key) to document the expected dict shape directly instead of pointing at a
  deleted function, and to note the builder's removal.
- Deleted the two dedicated tests in `tests/test_factor_risk_model.py`:
  `test_factor_constraints_builder_returns_list` and
  `test_regime_factor_bounds_tighter_in_crisis`. Left `test_check_factor_bounds_no_violation`
  and `test_check_factor_bounds_violation` untouched (they cover the separate, live
  `check_factor_bounds` in `factor_exposure.py`). `tests/test_mvo_optimizer.py`'s
  `test_mvo_factor_constraints_respected` was also left untouched — it hand-builds a
  `factor_constraints` list to test the live optimizer parameter, without importing the
  deleted builder.

## Item 4: AI PM guardrails table

**Verified caller count before deletion:** `apply_guardrails`
(`ascent/strategy/ai_pm_guardrails.py`) had zero non-test callers. `agents/ai_pm_agent.py`'s
only import from this module is `check_conviction_inflation` (a different function). The only
non-test references to `apply_guardrails` were 4 dedicated tests in
`tests/test_ai_pm_authority.py`. Its two private helpers, `_rolling_corr` and
`_apply_tracking_error_cap`, were called only from inside `apply_guardrails` itself — no other
callers. The module-level `_LEVEL_CONFIG`, `_CORR_BLOCK_THRESHOLD`, `_VAR_PROXY` constants
were consumed only by `apply_guardrails`/its helpers; `_ALPHA_QUALITY_PERCENTILE` was already
dead on arrival (defined, never referenced even inside `apply_guardrails`).

**What was done:**
- Deleted `apply_guardrails`, `_rolling_corr`, `_apply_tracking_error_cap`, `_LEVEL_CONFIG`,
  `_CORR_BLOCK_THRESHOLD`, `_ALPHA_QUALITY_PERCENTILE`, `_VAR_PROXY` from
  `ascent/strategy/ai_pm_guardrails.py`. Left `check_conviction_inflation` and
  `is_valuation_short` fully untouched — confirmed live, unrelated.
- Dropped now-unused imports (`math`, `os`, and `List`/`Optional`/`Tuple` from `typing`); kept
  `Dict` (used by both remaining functions).
- Rewrote the module docstring to describe current contents instead of the deleted function.
- Deleted the 4 dedicated tests in `tests/test_ai_pm_authority.py`:
  `test_level1_blocks_reduce`, `test_level1_blocks_amplify_bottom_50pct_alpha`,
  `test_level1_max_weight_change_capped_at_2pp`, `test_max_overrides_enforced`. Left
  `test_check_conviction_inflation` and everything else in that file untouched.

## Verification

```
.venv/bin/python -m py_compile ascent/regime/integration.py ascent/regime/engine.py \
    ascent/regime/__init__.py ascent/main.py ascent/strategy/ai_pm_guardrails.py \
    ascent/alpha/stack.py ascent/portfolio/mvo_optimizer.py ascent/portfolio/optimizer.py \
    tests/test_factor_risk_model.py tests/test_ai_pm_authority.py
# → clean, no output

.venv/bin/python -c "import ascent.main; import ascent.regime; ..."
# → ALL IMPORTS OK (also printed remaining public names in integration.py and
#   ai_pm_guardrails.py to visually confirm the right things survived)

.venv/bin/python -m pytest tests/test_factor_risk_model.py tests/test_ai_pm_authority.py \
    tests/regime/ tests/test_mvo_optimizer.py -q
# → 31 passed (factor_risk_model + ai_pm_authority + regime) + 15 passed, 1 skipped
#   (test_mvo_optimizer.py, pre-existing skip unrelated to this change)
```

No test failures. No other repo location (excluding `.claude/worktrees/*`, which are separate
worktree copies out of scope) references any of the deleted symbols.

## Blockers

None. All 4 items completed as scoped.
