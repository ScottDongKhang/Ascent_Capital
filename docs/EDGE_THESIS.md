# Ascent — Edge Thesis (one page, honest)

**Date:** 2026-06-23 · **Author:** Scott (with Claude) · **Status:** draft for falsification

> The job of this page is to answer one question in plain English: *why should
> this make money that a passive SPY-plus-cash portfolio doesn't?* If the answer
> isn't convincing, that is itself the finding — the next work is research to find
> an edge, not engineering to wrap one.

## 1. What the strategy actually is

Long-only, ~15–22 names, rebalanced ~every 10 business days. Alpha is a blend:
cross-sectional momentum / trend **70%**, sector stat-arb **15%**, short-term mean
reversion **5%**, XGBoost ML **10%** (fundamental sleeve **disabled** — IC-t −4.75,
an anti-signal). On top: a regime model (HMM, K=3), a SPY-200MA gross-exposure cut,
and a vol-target overlay. ~22% sits in defensive non-equity sleeves by design.

## 2. The honest evidence (verified, walk-forward OOS 2021-01 → 2026-01)

| Metric | Value |
|---|---|
| Sharpe | **0.41** |
| CAGR | **+10.3%** |
| Excess CAGR vs SPY | **+1.0pp** (10.42% vs 9.41%, same window) |
| Regression alpha vs SPY | **+2.24%/yr** at **beta 0.73** |
| Max drawdown | **−32.9%** |
| Win rate | **50.2%** |
| **Walk-forward efficiency** | **−0.65** (the IS optimizer adds NO OOS value — overfit) |

One backtest, **not a live track record**. The AI PM and debate layers were zeroed
in this run — this is the **pure quant** number.

## 3. Candidate edge hypotheses — and what would falsify each

**H1 — Cross-sectional momentum earns a real, persistent premium.**
- *For:* momentum is one of the most replicated factors in the literature; it is
  70% of the book and the regression alpha is positive at sub-1 beta.
- *Against / honest:* it is also one of the most crowded and arbitraged. WFE −0.65
  says our *parameter tuning* on top of it adds nothing out of sample.
- *Falsifier:* if a plain, untuned momentum portfolio over the same window matches
  or beats Ascent net of costs, our "edge" is just factor beta we could buy cheaper.

  **TESTED 2026-06-23 (same clean cache, same 21-fold/1134-day window):**

  | arm | Sharpe | CAGR | alpha | beta | WFE |
  |---|---|---|---|---|---|
  | FULL (all sleeves + IS optimizer) | 0.486 | 12.5% | +4.6% | 0.71 | **−0.50** |
  | TREND-only (fixed momentum params) | **0.427** | 10.7% | **+2.75%** | 0.72 | **+0.99** |

  **Result: H1 is largely confirmed — the edge is overwhelmingly momentum.**
  Trend-only with *untuned* params reproduces ~88% of the full Sharpe and nearly
  the whole verified alpha (+2.75% vs the canonical +2.24%), with WFE **+0.99**
  (it generalizes). The other 9 sleeves + the optimizer add only **+0.06 Sharpe**
  AND introduce all the overfitting (WFE +0.99 → −0.50). The complexity is mostly
  cost; the momentum sleeve is the spine.
  *Caveat: the FULL arm here scored 0.486, not the canonical 0.41 — the ML sleeve
  was skipped in this harness, so the +0.06 increment is uncertain; the trend-only
  number is clean. Artifact: `outputs/wf_results/edge_test1_momentum.json`.*

**H2 — The regime + defensive overlay is a genuine risk-management edge (better
drawdowns per unit return), not just lower beta.**
- *For:* beta 0.73 with +2.24%/yr regression alpha suggests *some* risk-adjusted add.
- *Against (my prior suspicion):* the overlay may be costing return in bull regimes
  without buying real crash protection.
- *Falsifier:* if removing the 200MA + vol overlay *improves* Sharpe, it's a cost.

  **TESTED 2026-06-23 (same harness as #1, overlays overridden to no-ops):**

  | arm | Sharpe | CAGR | alpha | beta | max DD |
  |---|---|---|---|---|---|
  | overlay ON (full) | 0.486 | 12.5% | +4.6% | 0.71 | −36.4% |
  | overlay OFF | **0.311** | 8.2% | **−0.96%** | **0.94** | **−43.0%** |

  **Result: H2 CONFIRMED, my prior suspicion was WRONG.** Removing the overlay
  cuts Sharpe by 0.175, turns alpha **negative**, pushes beta to 0.94, and deepens
  the drawdown. The overlay is not a drag — it is the mechanism that converts raw
  momentum beta into positive risk-adjusted alpha at sub-1 beta. Keep it.
  Artifact: `outputs/wf_results/edge_test2_overlay.json`.

**H3 — A solo operator's structural edge is small/uncrowded universes and
holding-period regimes large funds can't exploit.**
- *For:* this is the only place a retail-scale book can plausibly find durable edge.
- *Against:* the current universe (`ascent/data/universe.py`) is large-cap heavy —
  exactly where there is no capacity advantage.
- *Falsifier:* if 12-1 momentum IC is no higher in the low-liquidity subset than in
  mega-caps, there is no structural wedge here.

  **TESTED 2026-06-23 (59 monthly cross-sections, 2021-2026, Spearman IC by
  trailing-dollar-volume tercile):**

  | liquidity tercile | mean momentum IC | t vs 0 |
  |---|---|---|
  | low (least liquid) | 0.0072 | 0.30 |
  | mid | 0.0016 | 0.06 |
  | high (mega-cap) | 0.0095 | 0.32 |

  **Result: H3 FALSIFIED (within this universe).** Per-name momentum IC is
  indistinguishable from zero in every bucket, and low-liquidity names do NOT
  predict better than mega-caps (low − high = −0.002). No capacity/crowding wedge
  here. *Scope caveat: the bottom tercile of a 936-name large/mid-cap universe is
  still liquid; this does not rule out a wedge in genuinely micro-cap names Ascent
  does not currently trade — that is the next place to look, and it requires a
  different universe, not a config tweak.* Script:
  `scripts/edge_tests/edge_test3_liquidity_ic.py`.

  Note: a ~0.007 monthly IC is *weak* — the strategy works not because any single
  name's momentum signal is strong, but because the construction (concentration +
  the test-#2 risk overlay) aggregates many weak signals. That is fragile and worth
  remembering.

## 4. Verdict (as of today)

**There is no undeniable, revolutionary edge here, and claiming one would be
dishonest.** What the evidence supports: a *modest, thin* risk-adjusted tilt
(~+1pp/yr excess at 0.73 beta) that is **largely momentum factor exposure**
(test #1: trend-only reproduces ~88% of the Sharpe and nearly all the alpha, with
*better* out-of-sample robustness than the full stack), sitting on top of an
**overfit** parameter layer that the multi-sleeve machinery itself introduces.
That is a respectable learning result and a plausible *seed*, not a fund-grade edge.

The most actionable consequence of test #1: **the complexity is mostly cost.** If
the momentum sleeve alone (fixed params) gives WFE +0.99 and Sharpe 0.43, the
honest engineering move is to *simplify toward the robust core*, not add sleeves.

**The real, evidence-backed edge thesis (from tests #1 + #2):**
> Ascent = **cross-sectional momentum** (the raw-return engine — test #1) **+ a
> disciplined 200MA/vol-target risk overlay** (the alpha engine — test #2). The
> momentum sleeve generates the return; the overlay converts it from ~0.94-beta
> equity exposure with *negative* alpha into 0.71-beta exposure with positive
> alpha. Everything else — the 9 other alpha sleeves, the IS optimizer, the AI/
> debate layers — is, on current evidence, marginal-to-negative and the source of
> the overfitting (WFE +0.99 → −0.50). This is a real, modest, defensible *seed*.
> It is not yet undeniable edge, and it is two well-known components, not magic.

The defensible path is not to tune until the backtest glows (that is the WFE −0.65
trap). It is to **run tests #1–#3 above** and let them kill the hypotheses that
deserve killing. Whatever survives honest falsification is the real spine of the
fund. Whatever doesn't tells you where the actual research is.

## 5. The AI PM, specifically

The AI PM is **not yet judgeable**: 3 resolved independent decisions (+4.4%, +3.3%,
+0.9%), all absolute-positive, trailing a noisy quant counterfactual on effectively
n≈1. The promotion engine had been *starved of data* (return buffers empty,
`decisions_evaluated 0`); that measurement plumbing is now repaired. The verdict is
governed by the pre-registered rule in [`AI_PM_EVAL_RULE.md`](AI_PM_EVAL_RULE.md):
**HOLD until ≥20 independent decisions** — do not promote, do not disable.
