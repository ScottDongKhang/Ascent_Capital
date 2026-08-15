# Validation & Cutover — Task 2 Report: DO NOT TRUST THESE NUMBERS

**Status: validation run completed technically, but surfaced a real, unresolved bug that
invalidates its results and casts doubt on whether sub-project 2's rebuild is even effective at
runtime. Cutover decision: DO NOT resume live trading pending investigation.**

## What ran

```
.venv/bin/python -c "from ascent.research.walk_forward_runner import walk_forward_pipeline; walk_forward_pipeline()"
```

Full log: `outputs/wf_results/wf_run_target_architecture_2026-08-14_BROKEN.log` (2627 lines).
330 folds, all "succeeded" (no fold-level errors), OOS period 2020-01-01 to 2026-07-21.

## The numbers (do not cite these as real evidence)

```
Total Return:     -10.40%    CAGR:        -0.84%    Sharpe:  -0.433
Sortino:          -0.184     Max Drawdown: -13.13%   Hit Rate: 4.6%
Profit Factor:     0.79      Avg Turnover: 3.14%/day  Avg Positions: 11.9
```

A 4.6% hit rate over 3291 trading days is not a plausible result for any real strategy — it's
the signature of something structurally broken, not of a bad-but-real 2-sleeve stack.

## Root cause, confirmed

Every single fold's log shows:
```
[Stack] IC gate: zeroing meanrev (rolling mean_ic=-0.0140 < -0.005)
[Stack] IC gate: zeroing insider (rolling mean_ic=-0.0532 < -0.005)
```
**Identical, static values on all 330 folds** — not a rolling per-fold calculation. Two stale
artifacts, both predating sub-project 2's rebuild, are silently overriding the code-level
`DEFAULT_ALPHA_WEIGHTS = {"meanrev": 0.50, "statarb": 0.50}`:

1. **`ascent/alpha/stack.py::_get_gated_weights()`** reads `logs/sleeve_ic_log.jsonl` (a stale,
   pre-rebuild log — still has `insider`, a sleeve that no longer exists in
   `DEFAULT_ALPHA_WEIGHTS` at all) and zeros any sleeve with rolling mean IC below
   `IC_GATE_THRESHOLD = -0.005`. It zeroed `meanrev` — one of the only 2 surviving sleeves —
   using a stale, frozen IC reading. **Its docstring says freed weight is "redistributed to
   trend"** — hardcoded, regardless of whether `trend` is even part of the current weight set.
2. **`data_cache/active_alpha_config.json`** (`updated_at: "2026-05-02"`, `promoted_from:
   "v3_20260402"`) holds a completely different, pre-rebuild weight set —
   `{trend: 0.7428, meanrev: 0.0877, statarb: 0.0661, ml: 0.1034, volatility: 0.0}` — and
   `_load_active_alpha_weights()` reads this file in preference to `DEFAULT_ALPHA_WEIGHTS` when
   it exists.

Combined, these explain the log line `[alpha_stack] loaded=['trend', 'meanrev', 'volatility',
'statarb', 'fundamental', 'earnings', 'analyst', 'options_flow', 'insider', 'earnings_tone',
'narrative']` — 11 sleeves, not the intended 2, with `trend` dominant and `meanrev` zeroed.

## Why this matters beyond this one backtest run

`_load_active_alpha_weights()` is not backtest-only code — it's what `build_alpha_stack()`
calls in the **live pipeline** too (`ascent/alpha/stack.py`, same function, same file,
unconditionally). Sub-project 2 changed `DEFAULT_ALPHA_WEIGHTS` at the code level and verified
that change via unit tests, but **no task checked whether a stale `active_alpha_config.json` or
`logs/sleeve_ic_log.jsonl` would override it at runtime.** Both files predate the rebuild by
over 3 months. This means: as things stand right now, if live trading were resumed, it is NOT
confirmed that the pipeline would actually run the intended 2-sleeve `meanrev`/`statarb` stack
— it might silently run the stale 2026-05-02 trend-heavy configuration instead, the exact
opposite of what sub-project 2 was built to do.

## Recommendation

**Do not resume live trading.** Do not treat sub-project 2 (target architecture) as fully
"done" at the runtime level, even though its code-level changes and task reviews were correct
and thorough — this is a genuine gap those reviews had no way to catch, since it only surfaces
when the pipeline is actually run end-to-end against real cached state, which none of sub-project
2's task-scoped tests did (they tested `build_alpha_stack()`'s logic directly with mocked/
synthetic weights, not the real `data_cache`/`logs` files on disk).

**Before any re-validation:**
1. Reset or delete `data_cache/active_alpha_config.json` (it describes a promotion from a
   15-sleeve world that no longer exists) and `logs/sleeve_ic_log.jsonl` (references sleeves,
   like `insider`, that are no longer live) — or make `_load_active_alpha_weights()`/
   `_get_gated_weights()` explicitly aware that a config/log predating the rebuild should not
   override the current `DEFAULT_ALPHA_WEIGHTS`.
2. Decide what `_get_gated_weights()`'s "redistribute to trend" behavior should do now that
   `trend` isn't a live sleeve — redistribute proportionally to whatever sleeves ARE live
   (`meanrev`/`statarb`), or don't gate at all with only 2 sleeves (gating exists to protect a
   diversified stack from one bad sleeve; with only 2, zeroing either one concentrates 100% into
   the other, which may not be the intended behavior at all).
3. Re-run this validation only after both are addressed, and confirm via the `[Stack] IC gate:`
   / `[alpha_stack] loaded=` log lines that the run is genuinely using only `meanrev`+`statarb`
   before trusting any performance number from it.
