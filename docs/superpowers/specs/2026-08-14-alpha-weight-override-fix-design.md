# Alpha Weight Runtime Override — Design Spec

**Sub-project 4b** (unplanned, found by sub-project 4's validation run). Fixes the bug documented
in `outputs/wf_results/vc-task-2-validation-report.md` and memory
`alpha-weights-runtime-override-not-fixed.md`: two stale, pre-rebuild files silently override
`ascent/alpha/stack.py::DEFAULT_ALPHA_WEIGHTS` at runtime, in both the backtest and the live
pipeline, defeating sub-project 2's 2-sleeve reduction.

## Fix 1: delete the two stale files

`data_cache/active_alpha_config.json` (dated 2026-05-02, `promoted_from: "v3_20260402"`, a
15-sleeve shadow-promotion snapshot) and `logs/sleeve_ic_log.jsonl` (references sleeves —
`insider` — that no longer exist in the live weight set) both predate the rebuild by 3+ months.
Both functions that read them already have a clean fallback when the file is absent:
`_load_active_alpha_weights()` falls through to `DEFAULT_ALPHA_WEIGHTS.copy()` when
`active_alpha_config.json` doesn't exist; `_get_gated_weights()` returns `alpha_weights`
unchanged when `sleeve_ic_log.jsonl` doesn't exist. Deleting both files is sufficient — no code
change needed for this half of the fix. `SELF_MODIFY_ENABLED = False` means
`active_alpha_config.json` won't be regenerated automatically; `sleeve_ic_log.jsonl` will
presumably start accumulating fresh entries for whatever sleeves are actually live going
forward (confirm this in the implementation task before deleting, don't assume).

## Fix 2: `_get_gated_weights()`'s hardcoded "redistribute to trend" is wrong now

`ascent/alpha/stack.py::_get_gated_weights()`'s docstring says "Freed weight redistributed to
trend" — hardcoded, regardless of whether `trend` is part of the current `alpha_weights` dict at
all. With the 2-sleeve stack, if `meanrev`'s rolling IC ever legitimately turns negative in the
future (a real, expected scenario the gate exists to protect against — not a bug), the freed
weight should NOT flow into `trend`, which build_alpha_stack would then compute and include even
though it's absent from `DEFAULT_ALPHA_WEIGHTS` — reproducing exactly the bug this fix is for.

**Fix:** redistribute freed weight proportionally among the OTHER sleeves already present in the
`alpha_weights` dict being gated (i.e., generically correct regardless of which sleeves are
currently live), not to a hardcoded name. If gating would zero every live sleeve simultaneously
(e.g. both `meanrev` and `statarb` gated in the same window), leave `alpha_weights` unchanged
rather than producing an all-zero dict — a fully-gated stack should fail safe to "don't touch
the weights the gate can't allocate," not to an empty portfolio.

## Verification

After both fixes, re-run the same walk-forward validation
(`ascent.research.walk_forward_runner.walk_forward_pipeline()`) and confirm via the
`[Stack] IC gate:` / `[alpha_stack] loaded=` log lines that the run genuinely uses only
`meanrev`/`statarb` throughout (or a legitimate, rolling — not static — gating event affecting
one of them, redistributing to the other, not to `trend`). Only then treat the resulting
Sharpe/drawdown/hit-rate numbers as real evidence for a cutover decision.

## Explicitly out of scope

- Whether IC-gating should apply at all with only 2 sleeves (a real policy question — zeroing
  either sleeve concentrates 100% into the other, which may be intended risk behavior, not a
  bug) — keep the gate's existence, only fix its redistribution target.
- Rebuilding `data_cache/active_alpha_config.json`/`logs/sleeve_ic_log.jsonl` with fresh,
  correct content — deletion is sufficient; they'll regenerate naturally (or stay absent) as
  the live system runs.
- The cutover decision itself — that's the next step after re-validation, not this fix.
