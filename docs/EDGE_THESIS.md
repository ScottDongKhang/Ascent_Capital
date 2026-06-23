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
- *Falsifier:* if a plain, untuned momentum decile portfolio over the same window
  matches or beats Ascent net of costs, our "edge" is just factor beta we could buy
  cheaper. **(Not yet run — this is test #1.)**

**H2 — The regime + defensive overlay is a genuine risk-management edge (better
drawdowns per unit return), not just lower beta.**
- *For:* beta 0.73 with +2.24%/yr regression alpha suggests *some* risk-adjusted add.
- *Against:* max DD −32.9% is **not** good for a strategy carrying 22% defensives;
  the overlay may be costing return in bull regimes without buying real crash
  protection.
- *Falsifier:* if removing the 200MA + vol overlay *improves* Sharpe over the OOS
  window, the overlay is a cost, not an edge. **(Not yet run — test #2.)**

**H3 — A solo operator's structural edge is small/uncrowded universes and
holding-period regimes large funds can't exploit.**
- *For:* this is the only place a retail-scale book can plausibly find durable edge.
- *Against:* the current universe (`ascent/data/universe.py`) is large-cap heavy —
  exactly where there is no capacity advantage.
- *Falsifier:* if per-sleeve IC is no higher in the small-cap / low-coverage subset
  than in mega-caps, there is no structural wedge here. **(Not yet run — test #3.)**

## 4. Verdict (as of today)

**There is no undeniable, revolutionary edge here, and claiming one would be
dishonest.** What the evidence supports: a *modest, thin* risk-adjusted tilt
(~+1pp/yr excess at 0.73 beta) that is **partly or wholly explained by known
factors**, sitting on top of an **overfit** parameter layer. That is a respectable
learning result and a plausible *seed*, not a fund-grade edge.

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
