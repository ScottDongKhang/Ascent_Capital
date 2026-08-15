# Task 3 Re-validation Report: fix confirmed correct, but does NOT resolve bad results

**Status: the alpha-weight override fix (sub-project 4b) is real, independently verified, and
necessary — but the walk-forward validation still produces implausible results after it. There
is a SEPARATE, still-unidentified bug. Live trading remains NOT resumed.**

## What was re-run

Same command as the original (broken) validation, after Task 1 (deleted `data_cache/
active_alpha_config.json` + `logs/sleeve_ic_log.jsonl`) and Task 2 (fixed `_get_gated_weights()`'s
redistribution) were merged:

```
.venv/bin/python -c "from ascent.research.walk_forward_runner import walk_forward_pipeline; walk_forward_pipeline()"
```

Full log: `outputs/wf_results/wf_run_target_architecture_2026-08-14_fixed_still_broken.log`.

## Result: still bad, nearly identical to before the fix

```
                    BEFORE FIX        AFTER FIX
Total Return:       -10.40%           -8.65%
CAGR:               -0.84%            -0.69%
Sharpe Ratio:       -0.433            -0.305
Sortino Ratio:      -0.184            -0.129
Max Drawdown:       -13.13%           -11.45%
Hit Rate:            4.6%              4.7%
Profit Factor:        0.79              0.84
```

Marginally less bad, but still implausible (a 4.7% hit rate over 3291 days is not a real
strategy's signature). **The alpha-weight fix did not meaningfully change the outcome.**

## Why the original diagnosis was real but incomplete

I initially treated `[alpha_stack] loaded=['trend', 'meanrev', ..., 'narrative']` (11 sleeve
names) as proof the composite was contaminated with dead sleeves. **This was a misreading of
the log line.** Confirmed directly against `ascent/alpha/stack.py`:

```python
alphas["trend"] = trend       # unconditional -- every sleeve function that computes a
alphas["meanrev"] = mr        # non-empty result gets added here, regardless of alpha_weights
...
print(f"[alpha_stack] loaded={loaded}  skipped={skipped}")   # <- just lists alphas.keys()
...
total_w = sum(alpha_weights.get(k, 0.0) for k in alphas)
for name, alpha_df in alphas.items():
    w = alpha_weights.get(name, 0.0) / total_w    # <- any sleeve NOT in alpha_weights gets w=0
```

`alphas` is populated for every sleeve unconditionally; the `loaded=` line just reports which
sleeve functions ran successfully. The actual BLEND weight (`w`) for any sleeve absent from
`alpha_weights` (i.e. everything except `meanrev`/`statarb` now) is mathematically zero. **The
`loaded=` list was never a reliable signal of composite contamination** — my plan's own Task 3
verification step (checking this log line) was based on a wrong assumption, and I'm flagging
that explicitly rather than letting it stand as a validated check for next time.

The alpha-weight fix itself (Tasks 1-2) is still real and worth having: it fixed a genuine stale
config-file override and a genuine redistribution bug, both independently verified by a
whole-branch review that hand-derived the math against 6 examples. It just wasn't the cause of
THIS run's bad numbers.

## What's still unexplained

The one anomaly present identically in both runs (before and after the fix), unrelated to alpha
weighting: **every one of 327 `[WF] targets injected` log lines shows `valid rows: 0`** — the
per-fold training-target matrix (`FeatureBuilder(train_prices, train_macro).compute_targets(...)`)
appears to be entirely NaN across every single fold, every single run. This is a different code
path from `build_alpha_stack` (it feeds the `ml` sleeve, which isn't in `DEFAULT_ALPHA_WEIGHTS`
either way) — so it's not obviously the cause of the bad meanrev/statarb-only numbers, but it's
the one consistent, unexplained red flag across both runs and warrants its own investigation
before trusting this harness for anything.

**I have not identified the actual root cause of the bad Sharpe/hit-rate numbers.** It could be:
- A genuine property of `meanrev`+`statarb` alone on this exact 12-position, 330-fold setup
  (possible, but a 4.7% hit rate is extreme even for a bad strategy — this still smells like a
  bug, not a real result).
- A bug in `BacktestEngine`'s return/hit-rate calculation specific to a concentrated, low-sleeve-
  count portfolio.
- Something related to the unexplained zero-valid-target-rows anomaly above.
- A bug in how `walk_forward_runner.py` combines/forward-fills the 330 rebalance dates into 3291
  daily rows.

## Recommendation

**Still do not resume live trading.** The alpha-weight fix (Tasks 1-2) should be kept — it's a
real, correctly-verified fix for a real bug, independent of whether it explains this run's
numbers. But the walk-forward harness itself needs fresh, focused investigation into the
BacktestEngine/hit-rate mechanics (or the zero-valid-target-rows anomaly) before any further
validation attempt. This is a distinct, deeper problem from the one just fixed, not a small
follow-up — treat it as its own investigation, not another quick patch.
