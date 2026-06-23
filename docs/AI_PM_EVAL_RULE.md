# AI PM Evaluation Rule (PRE-REGISTERED)

**Status:** committed 2026-06-23, *before* the sample is large enough to judge.
**Implements:** `ascent/monitoring/ai_pm_eval_rule.py` · tested in `tests/test_ai_pm_eval_rule.py`

## Why this document exists

The whole point of pre-registration is to decide the rule **before** we have the
data, so the verdict cannot be rationalised after the fact. As of 2026-06-23 the
AI PM has **3 resolved independent decisions** (+4.4%, +3.3%, +0.9% realized 21d).
The running "D − A★ = −6.52pp" headline is one rebalance book marked daily for 23
autocorrelated days — effectively **n ≈ 1**. That is not evidence of anything. The
temptation to "make the AI PM profitable" on that sample is the same error as
WFE −0.65: acting on noise. This rule removes the temptation by making the call
mechanical.

## The rule

Unit of evidence: **one independent decision = one rebalance** with a resolved
21-day outcome. Never a daily mark of the same book.

Let `d_i = (Track D return − Track A★ return)` for decision `i`, where Track A★ is
the quant-only counterfactual. Over `n` independent decisions:

```
mean = mean(d_i)
t    = mean / (stdev(d_i) / sqrt(n))

PROMOTE   if  n >= MIN_DECISIONS  and  t >  +T_THRESHOLD
DEMOTE    if  n >= MIN_DECISIONS  and  t <  −T_THRESHOLD
HOLD      otherwise   (includes: not enough decisions yet)
```

Pre-registered constants (do **not** tune to fit an outcome):

| Constant | Value | Why |
|---|---|---|
| `MIN_DECISIONS` | 20 | Below ~20 independent rebalances there is no power to separate skill from luck for a 19-name book. At a ~10-day cadence that is ~9–12 months of paper trading. |
| `T_THRESHOLD` | 2.0 | Standard two-sided ~95% bar. We are not promoting an AI to allocate more capital on a hunch. |

## What it does NOT do

- It does not use daily marks as observations (autocorrelation inflates `t`).
- It does not issue PROMOTE/DEMOTE below `MIN_DECISIONS`, no matter the mean.
- It does not get "tuned" if we don't like the answer. Changing a threshold after
  seeing data voids the pre-registration; if the rule is genuinely wrong, write a
  new dated pre-registration explaining why, and supersede this one.

## Current verdict (auto-computed)

Run: `python -c "from ascent.monitoring.ai_pm_eval_rule import *; print(evaluate_rule(load_decision_diffs()))"`

As of 2026-06-23: **HOLD** — 3 independent decisions < 20 required. Mean is
positive (+2.89pp/decision) but is not yet evidence. Keep accumulating decisions;
do not promote, do not disable.

> Caveat to wire up next: `load_decision_diffs()` currently returns the AI book's
> own realized 21d return per decision, not the per-decision (D − A★) difference —
> the A★ counterfactual is persisted daily, not per-decision. Until a per-decision
> A★ is stored, feed `evaluate_rule` the paired diffs from a decision-level
> counterfactual. The rule itself (the gate + t-stat) is correct and tested.
