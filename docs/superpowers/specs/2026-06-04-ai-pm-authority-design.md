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

## Anti-Hallucination Guardrails

The AI PM is specifically prone to four hallucination types that current system protections don't cover: fabricated numbers, stale data cited as fresh, invented causal chains, and conviction inflation. The debate agents already have `[FROM CONTEXT]` tagging (added 2026-05-31) — the AI PM needs equivalent protection.

### 1. Source Tagging (Phase 1 output)

Every factual claim in Phase 1's output must carry a source tag. Tags are enforced via the Phase 1 JSON schema:

```json
"conviction_reasons": [
  {
    "symbol": "STRL",
    "claim": "Revenue +8.3% YoY, backlog at record $2.1B",
    "source": "earnings-2026-04-22",
    "data_date": "2026-04-22",
    "days_ago": 43
  }
]
```

Claims without a `source` field are **stripped by the parser before Phase 2 sees them**. Phase 2 cannot amplify Phase 1's unsourced assertions.

### 2. Recency Gate

Any data cited must pass a freshness check:
- **Price / flow data**: must be within 5 calendar days
- **Earnings / filings**: must be within 45 calendar days (matching the fundamental sleeve filing lag)
- **Analyst consensus**: within 30 calendar days

If a claim's `data_date` is older than the threshold, it is stripped and logged as `stale_claim`. The AI PM's Phase 2 prompt explicitly states: *"Today is {date}. Any data older than 45 days is stale. Do not cite it as current evidence."*

### 3. Numeric Cross-Reference Check (Python, post-Phase 2)

After Phase 2 completes but **before** guardrail processing, Python cross-checks numeric claims in the thesis against ground truth in the data cache:

```python
# Example: AI PM claims "STRL revenue +23%" — check against fundamentals cache
claimed = extract_numeric_claims(thesis_summary)  # regex parse
actual   = load_from_cache("fundamentals", symbol="STRL", field="revenue_growth")
if abs(claimed - actual) / abs(actual) > 0.15:   # >15% error
    log_hallucination_incident(symbol, claimed, actual)
    reduce_conviction(symbol)  # downgrade high → medium
```

Hallucination incidents are stored in the feedback file. Three incidents for the same symbol within 21 days → that symbol is **barred from AI PM overrides for the next rebalance**. This is logged and visible on the dashboard.

### 4. Conviction Inflation Check

If Phase 2 marks more than **40% of proposed names as "high conviction"**, all convictions above the threshold are automatically downgraded to "medium." The prompt states this rule explicitly so the AI PM learns to self-regulate. A model that thinks everything is high-conviction has no model of risk.

### 5. Temporal Context Injection

Every Phase 1 and Phase 2 prompt begins with a locked header that the model cannot contradict:

```
SYSTEM CONTEXT (authoritative — do not contradict):
Today: {date}
Last trading day: {prev_trading_day}
Current regime: {regime_label} (as of {regime_date})
Data freshness cutoff: {date - 45d} (do not cite anything older as current)
Your last rebalance: {last_rebalance_date}
Your worst recent call: {worst_call_symbol} ({worst_call_alpha:+.1%} over 10d)
```

### 6. Phase 1 → Phase 2 Context Strip

Only the following fields from Phase 1 pass into Phase 2:
- `high_conviction_names` (symbols only, no prose)
- `conviction_reasons` (sourced, recency-validated claims only)
- `regime_assessment` (structured dict, not prose)
- `causal_mechanisms` (structured)

Freeform prose from Phase 1 is **not** passed to Phase 2. Phase 2 must re-derive its reasoning from the structured claims and the quantitative context it receives directly. This prevents hallucinated chains: Phase 1 invents a narrative → Phase 2 amplifies it.

### Summary

| Guardrail | Where | What It Catches |
|---|---|---|
| Source tagging | Phase 1 schema | Unsourced factual claims |
| Recency gate | Phase 1 parser | Stale data cited as fresh |
| Numeric cross-reference | Post-Phase 2 Python | Fabricated or misremembered numbers |
| Conviction inflation | Post-Phase 2 Python | Everything marked "high conviction" |
| Temporal context injection | Both phases | Date confusion, stale anchoring |
| Phase 1 → 2 context strip | Handoff layer | Prose narrative hallucination chains |

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

**Graceful degradation on API failure:**
- Phase 2 exception/timeout → use Phase 1 result as a single-pass proposal (no synthesis, but still an informed view)
- Phase 1 also fails → fall back to pure quant weights, log `ai_pm_fallback: "api_failure"`
- Never block the rebalance execution due to AI PM failures — quant always runs independently

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
  "in_cooldown": false,
  "cooldown_days_remaining": 0,
  "days_stuck_at_level": 3,
  "stuck_alert": false,
  "sortino_21d_ai": 0.82,
  "sortino_21d_quant": 0.74,
  "sortino_edge": 0.08,
  "sortino_n_days": 14,
  "hit_rate_21d": 0.52,
  "override_win_rate": 0.44,
  "amplify_avg_alpha_10d": +0.0031,
  "amplify_n": 3,
  "amplify_confidence": "low",
  "reduce_avg_alpha_10d": -0.0089,
  "reduce_n": 1,
  "reduce_ban_active": false,
  "new_position_avg_alpha_10d": +0.0012,
  "new_position_n": 0,
  "daily_view_accuracy_2d": 0.58,
  "daily_view_accuracy_5d": 0.52,
  "daily_view_n": 12,
  "last_5_decisions": [...],
  "best_call_10d": {"symbol": "STRL", "type": "amplify", "alpha": +0.031, "n_basis": 3},
  "worst_call_10d": {"symbol": "HUM", "type": "hold", "alpha": -0.028, "n_basis": 3},
  "days_to_next_promotion": 18,
  "promotion_gap_sortino": 0.12,
  "promotion_gates": {
    "sortino_edge": {"pass": false, "value": 0.08, "threshold": 0.20},
    "hit_rate": {"pass": true, "value": 0.52, "threshold": 0.52},
    "profit_factor": {"pass": false, "value": 1.1, "threshold": 1.2},
    "min_decisions": {"pass": false, "value": 3, "threshold": 5},
    "fade_rate": {"pass": true, "value": 0.20, "threshold": 0.30},
    "regime_gate": {"pass": true, "value": "no bad regime yet"},
    "cooldown": {"pass": true, "value": "not in cooldown"}
  }
}
```

Every metric includes its **sample size** (`_n` field) and a **confidence label** (`low` < 5, `medium` 5–15, `high` > 15). The AI PM cannot claim a metric is meaningful when n < 5 — the prompt explicitly instructs it to treat low-confidence metrics as "insufficient data, do not act on."

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

On rebalance days, this file is **injected into the Phase 2 synthesis prompt** before the model reasons.

**Enforceable feedback citation**: Phase 2 response schema requires two structured fields:
```json
{
  "feedback_acknowledged": true,
  "worst_call_response": "HUM was held at 7.1% and dropped -27.6% on earnings. I should have reduced given crowding=WATCH and no catalyst. Going forward I will require crowding=CLEAN before amplifying healthcare names near earnings.",
  "reduce_ban_respected": true,
  ...portfolio proposal...
}
```
If `feedback_acknowledged` is absent or `false`, the response is **rejected and falls back to pure quant**. This is enforced by the response parser, not trusted from the LLM. The feedback citation is a schema constraint, not a prose instruction.

**REDUCE ban sample gate**: the REDUCE ban fires only when `reduce_n >= 5`. With fewer than 5 REDUCE decisions evaluated, the AI PM is permitted to REDUCE but receives a warning in its prompt: "REDUCE track record: only N decisions evaluated — treat as unproven."

**Stuck promotion alert**: `days_stuck_at_level` is tracked in the feedback file. When it exceeds **63 trading days** (3× the evaluation window), `stuck_alert: true` is set and the daily pipeline prints a warning: `[AIPMAuthority] WARNING: AI PM has been at Level 1 for 63+ days without promoting — review promotion gates.`

**Daily view scoring**: non-rebalance Phase 1 conviction updates are scored at 2d and 5d horizons. `daily_view_accuracy_2d` and `daily_view_accuracy_5d` in the feedback file show whether the AI PM's daily bullish/bearish calls are directionally correct. This is a leading indicator of skill — consistently accurate daily views (>55% over n≥20) support promotion; consistently inaccurate views (<45%) are a flag. This costs zero extra LLM calls — the scoring is Python arithmetic on existing price data.

This is how it gets better over time at zero extra cost.

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
- Also snapshot `ai_pm_proposed` weights from the decision log for Track D — **normalized to sum to 1.0** before storing. Raw AI PM proposals may not be weight-normalized pre-guardrail.

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
17. **Feedback citation is a schema constraint** — `feedback_acknowledged: true` and `worst_call_response` are required fields in Phase 2 response schema; absence is caught by the response parser and triggers quant fallback
18. **REDUCE ban requires n≥5** — ban only fires on sufficient sample; fewer than 5 REDUCE decisions produces a warning, not a ban
19. **Stuck promotion alert at 63 days** — `stuck_alert` flag set in feedback file; pipeline prints warning; no automatic action taken
20. **Daily view scoring** — Phase 1 daily conviction updates scored at 2d/5d in Python; accuracy metrics in feedback file as leading indicator; below 45% over n≥20 is a flag
21. **Track D weight normalization** — `ai_pm_proposed` weights normalized to sum=1.0 before Track D snapshot is written
22. **Orphaned decisions scored as 0** — if a symbol is unavailable at scoring date (halt, delist), outcome is recorded as 0.0 return; decision counts toward `n_decisions_evaluated` so the minimum-decisions gate remains satisfiable
23. **API failure never blocks rebalance** — Phase 2 fail → Phase 1 result; Phase 1 fail → pure quant; failure mode logged with `ai_pm_fallback` field
24. **Source tagging is a hard schema constraint** — Phase 1 claims without `source` and `data_date` fields are stripped by the parser before Phase 2 receives them; stripping is logged
25. **Recency gate is enforced in Python** — claims with `data_date` older than threshold are stripped regardless of AI PM's stated reasoning; the AI PM cannot override this
26. **Numeric cross-reference runs post-Phase 2** — Python checks AI PM's numeric claims against cache ground truth; >15% discrepancy logs a hallucination incident and reduces conviction level for that symbol
27. **Three hallucination incidents = override bar** — a symbol with 3+ hallucination incidents in 21 days is barred from AI PM overrides for the next rebalance; this is deterministic Python enforcement, not LLM self-policing
28. **Conviction inflation cap** — >40% of proposals marked "high conviction" triggers automatic downgrade of excess; enforced post-Phase 2
29. **Phase 1 → Phase 2 handoff strips freeform prose** — only structured, sourced fields pass through; prevents Phase 2 from amplifying Phase 1 narrative hallucinations
30. **Temporal context header is injected, not requested** — date/regime/worst-call context is prepended by code, not left to the model to recall correctly
