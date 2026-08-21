# Beta Decomposition & Capacity/Crowding Analysis

Answers `21_alpha_edge_audit.md`'s second recommendation: how much of the
reported Sharpe 0.41 is genuine cross-sectional reversal edge versus
residual beta exposure (beta is reported at 0.947 to SPY).

## 1. Data available

`outputs/wf_results/wf_report_clean_2026-08-15.json` (the canonical
walk-forward artifact) is **summary-stats-only** — 19 scalar/string
fields, no per-fold or per-day return series anywhere in the artifact, its
3.2MB source log, or any sibling file for this specific run. A genuinely
matching per-day ledger does not exist on disk; the repo-root
`ascent_daily_ledger.csv` is stale (spans to 2026-05-01, not 2026-07-15,
with a mismatched +257.9% total return) and cannot be used as a
stand-in. Regenerating a real one requires a full 165-fold walk-forward
re-run (10-sleeve alpha stack + per-fold HMM regime fit) — a substantial,
uncosted compute job not attempted this session.

## 2. Computed beta-adjusted Sharpe (algebraic, not a re-run)

Because the project's beta is a standard OLS coefficient
(`ascent/research/evaluation.py:126`: `beta = r.cov(b) / b.var()`), the
reported beta plus independently-computed volatilities fully determine the
beta-hedged variance with no correlation assumption needed:

```
Var(hedged) = σ_s² − β²·σ_m²
```

**Inputs**: σ_s (strategy vol, reported) = 0.2458; σ_m (SPY vol) —
**computed directly** from `prices_live`, 2020-01-02→2026-07-15, n=1640 =
0.20333 (cross-check: implied SPY CAGR from this same series = 13.83%,
matching the artifact's own `spy_cagr_same_window: 0.1382` almost exactly
— validates the price series and window alignment); β (reported) = 0.947;
CAGR_s = 0.102; CAGR_m = 0.1382.

**Implied correlation**: ρ = β·σ_m/σ_s = 0.783 (strong positive — expected
for a reversal sleeve still carrying net long equity exposure).

**Residual (beta-hedged) vol**: √(0.2458² − 0.947²×0.20333²) = **0.1528**
(15.28%).

**Residual (beta-hedged) CAGR** (linear approximation, CAGR_hedged ≈
CAGR_s − β·CAGR_m): 0.102 − 0.947×0.1382 = **−0.0289** (−2.89%).

**Beta-hedged Sharpe: −0.0289 / 0.1528 ≈ −0.19.**

**Sanity check**: the artifact separately reports `excess_sharpe: −0.222`
(the naive β=1 version, strategy minus raw SPY). Independently recomputing
that same β=1 quantity from the inputs above gives −0.236 — within ~6%
relative of the artifact's own figure, bounding the approximation error in
this method.

**Central estimate: beta-hedged Sharpe ≈ −0.19, defensible range roughly
−0.15 to −0.25.** This is computed from reported summary statistics plus
an independently-verified SPY series, cross-validated against the
artifact's own excess-return field — not a correlation-assumption guess —
but it is a static-beta approximation, not a direct computation on the
actual daily hedged return series (which doesn't exist for this run).

## 3. Reconciling the two Sharpe/beta pairs in CURRENT_VERIFIED_NUMBERS.md

Already resolved in that file's own text (lines 54-71), not a live
discrepancy: Sharpe 0.41/beta 0.95 is the current canonical run
(`wf_report_clean_2026-08-15.json`); Sharpe 0.41/beta 0.73 is an explicitly
**superseded** 2026-06-22 run with a confirmed bug (force-injects a CUT
sleeve, bypasses the IC gate). Worth noting: both runs land on nearly
identical raw Sharpe (0.41 both times) while beta differs substantially —
consistent with Sharpe alone being a poor discriminator between genuine
edge and beta exposure across these two runs, reinforcing why the
beta-adjustment above matters.

## 4. Verdict on "beta wearing a governance costume"

**Supported, with moderate-to-high confidence.** The beta-hedged Sharpe
isn't merely lower than 0.41 — it's **negative**. Once the 0.947 SPY
loading is algebraically removed, the residual return stream over this
window has a negative risk-adjusted return, not a diminished-but-positive
one. This is directly corroborated by the artifact's own `alpha: −0.0362`
and `excess_sharpe: −0.222` fields — both already show the strategy
losing to its benchmark before any beta-adjustment. Beta-hedging doesn't
change the sign of the story; it confirms it algebraically.

**Confidence caveats**: this is a summary-statistics-derived computation
with an estimated ~5-10% relative error from the CAGR-linearization step —
not enough to flip the sign given the ~13pp margin between CAGR_s and
β·CAGR_m, but a rolling-beta variant (beta could itself drift toward 1 in
stressed regimes, compounding the problem) would be more informative than
this single static-beta computation. **What would resolve remaining
uncertainty**: patching `walk_forward_runner.py` to persist
`result.daily_ledger` to `outputs/wf_results/` (it currently doesn't),
then re-running once — after which a direct, non-linearized hedged Sharpe
and a rolling 63-day beta become computable exactly.

## 5. Capacity and crowding

Two cost/capacity modules already exist in the codebase —
`ascent/execution/cost_model.py` (live order-sizing, Almgren-Chriss-based)
and `ascent/execution/capacity_model.py` (a separate strategy-capacity
estimator) — but the latter is **dead code**: zero callers outside its own
file, no `capacity_log.jsonl` anywhere. This is the first time it's been
run.

**Capacity**: using measured universe ADV (901-symbol universe, median
45-day ADV $224M, worst-case single name $17.5M) against the model's own
parameters and its own default (unmeasured) IC placeholders (meanrev=0.02,
statarb=0.03): breakeven capacity is roughly **$50M-$400M**, with the two
independently-built modules' participation thresholds cross-validating
each other almost to the decimal (~10% ADV). **Not currently binding** —
actual paper AUM (~$104,640) is 3-4 orders of magnitude below this
ceiling; a max-weight position today is ~0.06% of even the thinnest
name's ADV, an order of magnitude below the model's own 5% warning
threshold.

**Crowding**: a separate, more fundamental question that capacity math
can't answer — it requires visibility into other market participants'
positioning that a solo-operator paper account structurally doesn't have.
Public commentary (Resonanz Capital on the Feb-Mar 2025 "quant unwind";
HedgeCo on early-2026 momentum/reversal squeezes) documents stat-arb and
reversal factors as actively crowded and prone to correlated
deleveraging, consistent with but not a substitute for the McLean &
Pontiff decay channel already established in `10`. **The honest
statement: crowding is a real, industry-documented risk for this specific
factor family, but its magnitude on Ascent's implementation can't be
estimated from data available in this repo.**

## 6. Bottom line

Capacity is not the binding constraint at Ascent's current scale — the
binding question is whether there's real edge here at all, and the
beta-hedged Sharpe computed in §2 says, with real evidence rather than
suspicion, probably not on this window. A capacity study without a
measured, crowding-adjusted IC would have been precise governance around
an unknown quantity; §2-4 here, combined with `23`'s per-sleeve IC
findings, are the actual answer `21` called for.
