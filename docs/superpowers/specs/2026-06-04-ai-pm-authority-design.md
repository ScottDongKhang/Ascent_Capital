# AI PM Authority System — Design Spec
**Date:** 2026-06-04  
**Status:** Approved for implementation

---

## Problem

The AI PM has been in Phase 0 (0% capital authority) since April 1. The earliest it could advance under the old system was ~August 2026. Meanwhile the two-phase AI PM redesign (2026-05-28) has no live performance record at all. The system needs a way to start earning authority *now*, with real capital at stake but with guardrails that match its proven track record.

---

## Design Goals

1. AI PM gets capital allocation **starting today** (5%)
2. Career progression from Analyst → CEO as performance is proven
3. Gets **better every day** through a Python-computed feedback loop (no extra LLM cost)
4. Every decision is **logged** — what it proposed, why, what was blocked
5. **Three-track counterfactual** proves whether AI PM is adding value vs pure quant and vs SPY
6. **API cost under $25/year** at Level 3+, under $16/year at Level 1–2

---

## Career Ladder

| Level | Title | AI Weight | Starts |
|---|---|---|---|
| 0 | Shadow | 0% | Historical |
| 1 | Analyst | 5% | **NOW (manual advance)** |
| 2 | Associate | 15% | After 21d Sortino edge + hit rate |
| 3 | Manager | 30% | After 21d Sortino edge + hit rate |
| 4 | Director | 50% | After 21d Sortino edge + hit rate |
| 5 | CEO | 75% | After 42d Sortino edge + hit rate |

### Promotion Criteria (per level transition)

All four gates must clear simultaneously:

| Transition | Window | Sortino Edge vs Quant | Hit Rate | Profit Factor | Min Evaluated Decisions |
|---|---|---|---|---|---|
| 1 → 2 | 21 trading days | > 0.20 | ≥ 52% | > 1.2 | ≥ 5 |
| 2 → 3 | 21 trading days | > 0.30 | ≥ 55% | > 1.3 | ≥ 8 |
| 3 → 4 | 21 trading days | > 0.40 | ≥ 58% | > 1.4 | ≥ 10 |
| 4 → 5 | 42 trading days | > 0.50 | ≥ 60% | > 1.5 | ≥ 15 |

**Sortino ratio** rewards smooth upward equity curves — penalises downside volatility only.

**Profit factor** = gross winning alpha / gross losing alpha. Must exceed threshold to ensure the AI PM isn't winning frequently but losing big. A profit factor below 1.0 means it loses money in expectation regardless of hit rate.

**Minimum evaluated decisions** prevents statistical luck — the AI PM cannot be promoted until enough override decisions have been scored against actual outcomes. An "evaluated decision" is one where a 10d outcome has been computed.

**Incremental alpha measurement**: override wins and losses are measured as `(ai_weight − quant_weight) × return` — the *delta* contribution only. If the quant had STRL at 7% and the AI PM bumped it to 9%, the AI PM owns only the 2pp of extra weight, not the full 9%. This prevents the AI PM from claiming credit for the quant's signal.

**Fade penalty**: if more than 30% of the AI PM's evaluated decisions are classified as "fades" (positive outcome at 10d, negative at 21d), promotion is blocked regardless of Sortino or profit factor. Consistent fading indicates the AI PM is riding short-term momentum it mistakes for alpha.

**Regime diversity gate** (Level 1→2 and above): the AI PM must show it does not *lose badly* in any observed regime (no more than −0.5% cumulative alpha in any single regime). For Level 2→3 and above, it must show *positive* cumulative alpha in at least one regime other than `calm_bull`, with a minimum of **5 consecutive trading days** observed in that regime. One lucky day does not count.

### Demotion Criteria

- **Soft demotion**: AI PM max drawdown exceeds quant's by 3pp over the rolling window → drop 1 level. Requires ≥ 10 days of data before this check fires (avoids false demotion on tiny samples).
- **Hard demotion**: Single day AI PM return is 5pp worse than quant → immediate 1-level drop
- **Catastrophic**: Single day AI PM return is 10pp worse than quant → revert to Shadow (Level 0)
- **Cool-down after demotion**: 5 trading day lock-out before promotion evaluation resumes. Prevents thrashing and forces the AI PM to absorb feedback before trying again.

Demotion resets the evaluation buffer. The AI PM must re-earn each level.

---

## Guardrails by Level

Controls what the AI PM is *allowed to do* when blending with quant. Violations are **logged but blocked** — the AI PM can propose anything; the guardrail layer enforces limits before blending.

| Level | Max Weight Change vs Quant | New Symbols | Override Types | Max Overrides/Rebalance | Max Tracking Error Added |
|---|---|---|---|---|---|
| 1 | ±2pp | 0 | AMPLIFY only | 2 | 0.3% daily |
| 2 | ±4pp | 1 | AMPLIFY + HOLD | 3 | 0.5% daily |
| 3 | ±6pp | 2 | AMPLIFY + HOLD + REDUCE | 4 | 0.8% daily |
| 4 | ±8pp | 3 | All (conviction gate applies) | 5 | 1.2% daily |
| 5 | ±10pp | 5 | All | Unlimited | Uncapped |

**Level 1 cannot REDUCE any position.** It can only concentrate more weight on high-conviction names. The quant handles all defensive calls at this level. This ensures the AI PM proves it can find good longs before it earns the right to make risk calls.

**Level 1 amplification quality constraint**: the AI PM may only amplify names that the quant ranks in the **top 50% of alpha scores** on that rebalance day. Amplifying a name the quant considers mediocre is not allowed at this level — the AI PM must agree with the quant's direction before concentrating.

**Override correlation check**: simultaneous overrides in names with rolling 63-day correlation > 0.65 are blocked. The AI PM cannot burn two override slots on correlated names — they count as one concentrated bet, not two independent calls. This prevents disguised sector concentration.

**Tracking error cap**: after the authority blend, compute the expected daily tracking error of the blended portfolio vs the pure quant portfolio using a diagonal covariance proxy. If the blend would exceed the level's cap, proportionally scale back all AI PM weight changes until it fits. This algorithmically enforces equity curve smoothness — the AI PM cannot make the ride bumpier.

**Post-blend portfolio constraint validation**: after all AI PM adjustments and tracking error scaling, the blended portfolio is run through the existing portfolio validator. If any constraint is violated (max_weight > 10%, sector cap exceeded, total weight ≠ 1.0 ± 0.001), the AI PM's changes are rolled back to pure quant for that rebalance and the violation is logged. Portfolio integrity is non-negotiable.

---

## Daily Run Architecture

The AI PM runs once per day as part of `run_all_agents.py`.

### Non-Rebalance Days (226 days/year)
- **Phase 1 only**: Sonnet reads perf feedback file + held positions + today's signals
- Outputs a lightweight **daily conviction update** (not a full portfolio proposal): which names it's more/less bullish on today and why
- Written to `logs/ai_pm_daily_views.jsonl`
- Does **not** update `merged_weights.json` — portfolio stays frozen until rebalance
- Goes to counterfactual log as "what the AI PM would have done today"
- Cost: ~$0.05/day

### Rebalance Days (26 days/year)
- **Phase 1**: Sonnet pre-thesis (reads data, forms views)
- **Phase 2**: Sonnet (Level 1–2) or Opus (Level 3+) synthesis with perf feedback injected
- Produces full portfolio proposal → guardrail check → authority blend → `merged_weights.json`
- Decision logged to `logs/ai_pm_decision_log.jsonl`
- Cost: ~$0.12/rebalance (Level 1–2), ~$0.40/rebalance (Level 3+)

### Annual Cost
| Level | Annual |
|---|---|
| 1–2 | ~$16/yr |
| 3–4 | ~$23/yr |
| 5 | ~$23/yr |

---

## Daily Learning Brief (Zero LLM Cost)

Every day after `_log_holdings()`, a Python process computes `data_cache/ai_pm_perf_feedback.json`. No LLM calls. Pure arithmetic on existing logs.

**Contents:**
```json
{
  "as_of": "2026-06-04",
  "level": 1,
  "days_at_level": 3,
  "sortino_21d_ai": 0.82,
  "sortino_21d_quant": 0.74,
  "sortino_edge": 0.08,
  "hit_rate_21d": 0.52,
  "override_win_rate": 0.44,
  "amplify_avg_alpha_10d": +0.0031,
  "reduce_avg_alpha_10d": -0.0089,
  "new_position_avg_alpha_10d": +0.0012,
  "last_5_decisions": [
    {
      "date": "2026-05-27",
      "symbol": "STRL",
      "type": "amplify",
      "ai_w": 0.09,
      "quant_w": 0.07,
      "outcome_10d": +0.031,
      "verdict": "win"
    }
  ],
  "best_call_10d": {"symbol": "STRL", "type": "amplify", "alpha": +0.031},
  "worst_call_10d": {"symbol": "HUM", "type": "hold", "alpha": -0.028},
  "days_to_next_promotion": 18,
  "promotion_gap_sortino": 0.12
}
```

**Outcome windows**: each override decision is scored at **5d, 10d, and 21d**. The feedback file reports all three. The promotion gate uses the 10d window as primary, but the AI PM sees the full picture — a call that looks good at 5d but reverses by 21d is flagged as a "fade".

```json
"last_5_decisions": [
  {
    "date": "2026-05-27",
    "symbol": "STRL",
    "type": "amplify",
    "ai_w": 0.09,
    "quant_w": 0.07,
    "outcome_5d": +0.018,
    "outcome_10d": +0.031,
    "outcome_21d": +0.044,
    "verdict": "win",
    "fade": false
  }
],
"n_decisions_evaluated": 3,
"n_decisions_pending": 2
```

`n_decisions_evaluated` is gated before promotion — must meet minimum threshold per level.

On rebalance days, this file is **injected into the Phase 2 synthesis prompt** before the model reasons. The prompt includes a **mandatory feedback reference requirement**:

> "Before proposing any portfolio changes, you MUST explicitly state: (1) what your best recent call was and why it worked, (2) what your worst recent call was and what you would do differently today, (3) whether your current override type win rates justify the action you're about to take. If your `reduce_avg_alpha` is negative, you are banned from REDUCE overrides this rebalance regardless of conviction."

This is not optional guidance — it is a structural prompt constraint that forces the AI PM to confront its own track record before acting. This is how it gets better over time at zero extra cost.

---

## Decision Log

`logs/ai_pm_decision_log.jsonl` — one entry per rebalance day.

```json
{
  "date": "2026-06-10",
  "level": 1,
  "title": "Analyst",
  "ai_weight": 0.05,
  "phase2_model": "claude-sonnet-4-6",
  "perf_feedback_injected": true,
  "quant_proposed": {"STRL": 0.071, "WDC": 0.071},
  "ai_pm_proposed": {"STRL": 0.090, "WDC": 0.071},
  "guardrail_violations_blocked": [
    {"symbol": "NVDA", "proposed_change": +0.05, "limit": 0.02, "action": "capped"}
  ],
  "overrides_applied": [
    {"symbol": "STRL", "type": "amplify", "quant_w": 0.071, "ai_w": 0.090, "reason": "..."}
  ],
  "final_blended": {"STRL": 0.072, "WDC": 0.071},
  "thesis_summary": "Concentrating on infrastructure cycle names..."
}
```

---

## Three-Track Counterfactual Engine

### Tracks

| Track | Source | What It Measures |
|---|---|---|
| A — Quant Only | Snapshot before `authority_blend()` on rebalance day; held constant until next rebalance | What would have happened with no AI PM |
| B — Actual | Alpaca `last_equity` (real account) | What actually happened |
| C — SPY | `prices_live.parquet` | Market benchmark |
| D — Pure AI PM | `ai_pm_proposed` weights from decision log at 100% weight; held constant until next rebalance | AI PM signal quality, independent of dilution |

**Track D is critical** — at Level 1 (5% AI weight), Track B and Track A are 95% identical. The difference is too small to evaluate in noise. Track D shows what the AI PM would do with full authority, allowing you to assess whether it has genuine skill before the blending dilutes the signal. Track D never affects actual execution — it is purely diagnostic.

### Data Flow

**Rebalance days:**
- Snapshot pure quant `merged_weights` BEFORE `authority_blend()` → `logs/counterfactual_quant_snapshots.jsonl`
- Snapshot is **idempotent**: if an entry for today already exists (e.g. pipeline re-run), skip — never overwrite. This prevents a second run from corrupting Track A.
- Also snapshot `ai_pm_proposed` weights from the decision log for Track D.

**Every day (`_log_holdings()`):**
- Load last rebalance's quant snapshot weights
- Compute Track A daily return: quant weights × price changes
- Track B: from Alpaca `(equity - last_equity) / last_equity`
- Track C: SPY price change from `prices_live.parquet`
- Append all three to `logs/counterfactual_daily.jsonl`

### Key Metric

```
ai_value_add = Track B cumulative − Track A cumulative
```

Positive and growing = AI PM earning its authority.  
Negative = demotion triggers fire.

### Cumulative Report (printed daily)

```
[Counterfactual] Since AI PM went live (2026-06-04 → 2026-06-10, 5 days):
  Track A (Quant Only):    +1.24%
  Track B (Actual):        +1.31%
  Track C (SPY):           +0.82%
  Track D (Pure AI PM):    +1.68%
  AI value add (B−A):      +0.07pp vs quant | diluted by 95% quant weight
  AI signal quality (D−A): +0.44pp — what full authority would have added
```

---

## GitHub Pages Dashboard Addition

Add a "AI PM Performance" section to `docs/index.html` (auto-updated on every run):

- **Four-line equity curve**: Track A (blue/quant), Track B (green/actual), Track C (grey/SPY), Track D (orange/pure AI PM)
- **Current level badge**: "Analyst — Day 3 of 21 | Sortino edge: +0.08 (need +0.20)"
- **Override scorecard**: win rate, incremental alpha per override, best call, worst call, fade rate
- **Signal quality panel**: Track D vs Track A — "If AI PM ran the whole fund today, it would have added +0.44pp"
- **Gate progress**: all 7 promotion gates shown as pass/fail checklist

---

## Files

### Modified
| File | Change |
|---|---|
| `ascent/strategy/earned_authority.py` | 5-level ladder, Sortino+hit rate criteria, daily evaluation, level guardrails, model selection per level |
| `run_all_agents.py` | Daily AI PM run (Phase 1 on non-rebalance, full on rebalance), quant snapshot before blend, inject perf feedback, decision log write, counterfactual daily scoring |
| `scripts/generate_performance_page.py` | Add Track A/B/C chart and AI PM level badge |

### New
| File | Purpose |
|---|---|
| `ascent/monitoring/ai_pm_counterfactual.py` | Track A/B/C daily scoring, cumulative report, quant snapshot |
| `ascent/strategy/ai_pm_perf_feedback.py` | Python-computed daily learning brief, zero LLM cost |
| `logs/ai_pm_decision_log.jsonl` | Per-rebalance full decision record |
| `logs/ai_pm_daily_views.jsonl` | Per-day lightweight conviction update (non-rebalance) |
| `logs/counterfactual_daily.jsonl` | Daily Track A/B/C returns |
| `logs/counterfactual_quant_snapshots.jsonl` | Quant weights at each rebalance, frozen for Track A |
| `data_cache/ai_pm_perf_feedback.json` | Daily learning brief (current state) |

---

## Bootstrap

On implementation day:
1. Edit `data_cache/earned_authority.json` → `phase: 1, ai_weight: 0.05`
2. Today becomes Day 1 of the Analyst evaluation window
3. 21 trading days later (~Jul 3), system auto-checks for promotion to Associate

---

## Integrity Constraints

1. **Counterfactual Track A must use frozen quant weights** — never retroactively updated after the rebalance snapshot; write is idempotent (no overwrite on re-run)
2. **Guardrail violations logged before blocking** — AI PM's true proposals are preserved even when overridden
3. **Perf feedback outcome windows are forward-only** — a decision made on date T is scored using prices on T+5, T+10, T+21, never look-ahead
4. **Incremental alpha only** — override performance measured as `(ai_weight − quant_weight) × return`, not `ai_weight × return`
5. **Model selection enforced in code** — Level 1–2 physically cannot use Opus for Phase 2
6. **Authority blend still respects existing `conviction_gate`** — guardrail layer is additive, not a replacement
7. **All promotion gates must clear simultaneously** — Sortino, profit factor, hit rate, minimum decisions, fade penalty, regime gate all pass or nothing advances
8. **Level 1 amplification quality** — amplifying a name ranked in the bottom 50% of quant alpha scores is blocked and logged
9. **Override correlation** — simultaneous overrides in names with 63-day rolling correlation > 0.65 are blocked
10. **Tracking error cap is a hard block** — if the blend would exceed the level's daily tracking error cap, AI PM weight changes are proportionally scaled back; cap cannot be overridden
11. **Post-blend portfolio validation** — blended weights must pass the existing portfolio constraint validator; any violation rolls back AI PM changes to pure quant for that rebalance
12. **Cool-down is a hard lock** — no promotion evaluation during the 5-day cool-down after demotion
13. **Regime diversity requires ≥ 5 days** — a single lucky day in a non-calm_bull regime does not satisfy the gate
14. **Fade penalty blocks promotion** — >30% fading decisions (win 10d, lose 21d) blocks promotion regardless of other metrics
15. **Feedback file freshness gate** — if `ai_pm_perf_feedback.json` is older than 2 calendar days, Phase 2 does not receive it and a warning is logged; stale data is worse than no data
16. **Track D is diagnostic only** — pure AI PM weights never execute; they exist solely to measure signal quality independent of authority level dilution
17. **Feedback citation is required** — if Phase 2 response does not reference the feedback report, falls back to pure quant weights for that rebalance
