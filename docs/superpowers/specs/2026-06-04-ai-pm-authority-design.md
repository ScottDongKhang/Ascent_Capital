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
6. **AI PM API cost under $5/year** — $4.74/yr total; rest of system unchanged

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

| Transition | Window | Primary Scoring | Sortino Edge vs Quant | Hit Rate | Profit Factor | Min Evaluated Decisions |
|---|---|---|---|---|---|---|
| 1 → 2 | 21 trading days | 10d outcomes | > 0.20 | ≥ 52% | > 1.2 | ≥ 5 |
| 2 → 3 | 21 trading days | 10d outcomes | > 0.30 | ≥ 55% | > 1.3 | ≥ 8 |
| 3 → 4 | 42 trading days | **63d outcomes** | > 0.40 | ≥ 55% | > 1.3 | ≥ 10 |
| 4 → 5 | 63 trading days | **63d outcomes** | > 0.50 | ≥ 58% | > 1.4 | ≥ 15 |

**Sortino ratio** rewards smooth upward equity curves — penalises downside volatility only.

**Profit factor** = gross winning alpha / gross losing alpha. Must exceed threshold to ensure the AI PM isn't winning frequently but losing big. A profit factor below 1.0 means it loses money in expectation regardless of hit rate.

**Minimum evaluated decisions** prevents statistical luck — the AI PM cannot be promoted until enough override decisions have been scored against actual outcomes. An "evaluated decision" is one where a 10d outcome has been computed.

**Incremental alpha measurement**: override wins and losses are measured as `(ai_weight − quant_weight) × return` — the *delta* contribution only. If the quant had STRL at 7% and the AI PM bumped it to 9%, the AI PM owns only the 2pp of extra weight, not the full 9%. This prevents the AI PM from claiming credit for the quant's signal.

**Fade penalty**: if more than 30% of the AI PM's evaluated decisions are classified as "fades" (positive outcome at 10d, negative at 21d), promotion is blocked regardless of Sortino or profit factor. Consistent fading indicates the AI PM is riding short-term momentum it mistakes for alpha.

**Regime diversity gate** (Level 1→2 and above): the AI PM must show it does not *lose badly* in any observed regime (no more than −0.5% cumulative alpha in any single regime). For Level 2→3 and above, it must show *positive* cumulative alpha in at least one regime other than `calm_bull`, with a minimum of **5 consecutive trading days** observed in that regime. One lucky day does not count.

### Demotion Criteria

All comparisons use **Track D vs Track A★** (pure AI PM signal vs pure quant, no Phase 1 contamination). At Levels 1–2, Track B vs Track A is too diluted to trigger meaningful demotion — the AI PM's actual portfolio weight is too small to be detectable in noise.

- **Soft demotion**: Track D max drawdown exceeds Track A★ by 3pp over the rolling window → drop 1 level. Requires ≥ 10 days of data before this check fires.
- **Hard demotion**: Single day Track D return is 5pp worse than Track A★ → immediate 1-level drop
- **Catastrophic**: Single day Track D return is 10pp worse than Track A★ → revert to Shadow (Level 0)
- **Cool-down after demotion**: 5 trading day lock-out before promotion evaluation resumes.

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

**sleeve_weight_prior is advisory only**: Phase 1 provides regime and sleeve priors to the quant. These are suggestions, not commands. The regime engine's protective adjustments always take precedence. If Phase 1 says "trend 50%" but the regime engine calls crisis (trend → 30%), the regime engine wins. Phase 1 cannot override risk management. This is enforced in the quant pipeline: `sleeve_weight_prior` is clipped to within ±10pp of the regime engine's baseline before being applied.

---

## Alpha Generation — What the AI PM Is Actually Optimizing For

This section defines the core signal quality requirements. Everything else in this spec is governance. Alpha and Sharpe come from here.

### Primary Objective: Sharpe Ratio, Not Return

Every Phase 1 and Phase 2 prompt includes this in the temporal context header:

```
OBJECTIVE: Sharpe ratio, not raw return.
For every position you propose, you must state:
  - Expected 3-month return (with basis)
  - Expected volatility (high/medium/low with reason)
  - What would make you wrong (specific falsifiable condition)

A high-conviction 15% position in a volatile name may hurt Sharpe more
than a moderate-conviction 8% position in a stable name. When in doubt,
choose the lower-volatility expression of the same thesis.
```

The model optimizes for what it is asked to optimize for. Asking for Sharpe explicitly produces different (better) behavior than asking for "best portfolio."

### Sector Thesis Before Stock Selection

Phase 1 must produce a `sector_thesis` before any individual stock picks. This is a required field in the Phase 1 schema — if absent, Phase 2 receives no Phase 1 input and falls back to quant only.

```json
"sector_thesis": [
  {
    "sector": "industrials",
    "view": "overweight",
    "conviction": "high",
    "reason": "Infrastructure bill spending accelerating into 2026 cycle, domestic construction backlog at records",
    "avoid_subsectors": ["China-exposed industrials", "defense primes on margin pressure"],
    "prefer_subsectors": ["domestic construction", "grid infrastructure", "water"],
    "source": "earnings-commentary-2026-Q1-aggregate",
    "data_date": "2026-04-30"
  },
  {
    "sector": "healthcare",
    "view": "underweight",
    "conviction": "medium",
    "reason": "Drug pricing headwinds from IRA implementation, binary FDA risk on pipeline names",
    "avoid_subsectors": ["large-cap pharma", "medical devices near FDA decisions"],
    "prefer_subsectors": ["managed care if crowding=CLEAN"],
    "source": "regulatory-filings-2026-Q1",
    "data_date": "2026-04-15"
  }
]
```

**Why this directly improves Sharpe:** Top-down sector allocation forces diversification across uncorrelated thesis types. Without it, the AI PM could pick 15 stocks that all look good individually but are highly correlated — the portfolio generates return but Sharpe suffers because the positions all move together.

Stock selection in Phase 1 must then be constrained to favored sectors. The AI PM cannot amplify a name in an underweight sector at Level 1 or 2.

### Outcome Scoring Horizons

Override decisions are scored at **5d, 10d, 21d, 63d, and 126d**.

The 10d window is used for promotion gates at Levels 1–2. The **63d window** is primary for Levels 3–5 promotion. This matters because:
- A 10d scoring window teaches the AI PM to make momentum calls (what the quant already does)
- A 63d window teaches it to form and hold theses — which is where its orthogonal edge lives

If a call looks bad at 10d but good at 63d, it is classified as **"early"** not "miss." Early calls count neutral in the promotion gate. Fade calls (good at 10d, bad at 63d) count as losses. This asymmetry rewards thesis quality over short-term prediction.

### The AI PM's Information Edge

The quant sees: price, volume, factor returns, sector labels.  
The AI PM must use what the quant cannot see:

| Signal | Source | What it detects |
|---|---|---|
| Earnings call tone | `earnings_transcripts` | Confidence vs. defensiveness in management commentary |
| 10-K narrative shift | `sec_filings` + `narrative_alpha` | Risk factor language changes, business model pivots |
| Job posting trends | (future) | Hiring acceleration/deceleration as leading indicator |
| Congressional trades | `capitol_trades` | Informed positioning before regulatory events |
| Options flow | `options_scanner` | Institutional directional bets |
| Sector competitor calls | `earnings_transcripts` | Peer-level sector signal (not just your holdings) |

Phase 1 is required to cite at least one non-price source per conviction symbol. A purely price-based Phase 1 thesis is rejected — the quant already has that signal.

### Architecture Transition at Level 4+

At Levels 1–3, the AI PM overrides the quant's portfolio.  
At **Level 4 (Director, 50%)**: the relationship flips. AI PM proposes an independent portfolio from scratch. Quant validates risk (factor exposures, liquidity, sector caps) and clips individual positions but does not change selection. AI PM is the alpha source; quant is the risk guardrail.  
At **Level 5 (CEO, 75%)**: Track D becomes the executed portfolio. Quant runs as pure risk overlay — it can reduce any position by up to 30% for risk reasons but cannot add positions the AI PM didn't propose.

This is the endgame of the AI-native thesis. Not "AI helps quant" — "AI runs the fund, quant keeps it safe."

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
- **Haiku only**: reads perf feedback file + held positions + price moves
- Outputs a lightweight **daily conviction update**: which held names changed and why
- Written to `logs/ai_pm_daily_views.jsonl`
- Does **not** update `merged_weights.json` — portfolio stays frozen until rebalance
- Used for daily view accuracy scoring (2d/5d) and counterfactual log
- Cost: ~$0.005/day

### Rebalance Days (26 days/year)
- **Phase 1**: Sonnet pre-thesis (reads data, forms views) — all rebalances
- **Phase 2**: Sonnet (normal) or **Opus (triggered)** — see smart trigger below
- **Red team**: Sonnet adversarial attack + revision pass — all rebalances
- Produces full portfolio proposal → guardrail check → authority blend → `merged_weights.json`
- Decision logged to `logs/ai_pm_decision_log.jsonl`

### Smart Opus Trigger (~5 rebalances/year)
Phase 2 automatically upgrades from Sonnet → Opus when **any** of the following are true:
- **Regime = crisis** (always — this is the highest-stakes call, Sonnet never decides alone)
- Regime change detected since last rebalance
- Track D divergence from quant > 2% (AI PM signal strongly disagrees with quant)
- Phase 1 proposed 4+ potential overrides (high-complexity decision)
- First rebalance after a demotion (high-stakes recovery moment)

These are the rebalances where judgment quality matters most. Opus fires only when it's earned.

### Annual Cost (AI PM components only — rest of system unchanged)

| Component | Days | Cost/run | Annual |
|---|---|---|---|
| Haiku daily view | 226 | $0.005 | $1.19 |
| Phase 1 Sonnet | 26 | $0.044 | $1.13 |
| Phase 2 Sonnet | ~21 | $0.039 | $0.82 |
| Phase 2 Opus (triggered) | ~5 | $0.195 | $0.98 |
| Red team Sonnet | 26 | $0.024 | $0.62 |
| **Total** | | | **$4.74/yr** |

Budget: $5.00/yr. Headroom: $0.26.

**Graceful degradation on API failure:**
- Phase 2 exception/timeout → use Phase 1 result as single-pass proposal
- Phase 1 also fails → fall back to pure quant, log `ai_pm_fallback: "api_failure"`
- Never block rebalance execution due to AI PM failures — quant always runs independently

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

**Phase 1 accuracy tracking**: the feedback file also scores Phase 1's upstream inputs:
```json
"phase1_accuracy": {
  "regime_assessments": [
    {"date": "2026-06-10", "called": "stressed", "actual_10d": "calm_bull", "correct": false}
  ],
  "regime_accuracy_n": 2,
  "regime_accuracy_rate": 0.50,
  "sleeve_prior_value": +0.0012,
  "sleeve_prior_n": 2,
  "sleeve_prior_confidence": "low"
}
```
A Phase 1 regime call is "correct" if the regime engine agrees within 10 trading days. Sleeve prior value = Track A vs Track A★ cumulative return (how much did Phase 1's priors help the quant?). If `sleeve_prior_value` is consistently negative, Phase 1 is hurting the quant — the spec allows the AI PM's Phase 1 sleeve priors to be disabled via a flag in `earned_authority.json`.

**Override scoring excludes debate modifications**: if the debate layer modifies a position that the AI PM overrode, the AI PM's override is scored against the **pre-debate-adjusted** position, not the final executed weight. Debate is a separate layer with its own accountability. An AI PM decision cannot be held responsible for what debate did to it. Debate modifications are logged in the decision log under `debate_modification`.

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

| Track | Source | What It Measures | Primary use |
|---|---|---|---|
| A — Quant + Phase 1 | Quant weights after Phase 1 sleeve priors applied, before Phase 2 blend | What quant does with Phase 1 context but no Phase 2 override | Phase 2 override value |
| A★ — Pure Quant | Quant weights using default regime weights only, zero AI PM input | True no-AI-PM baseline | Total AI PM value (Phase 1 + Phase 2) |
| B — Actual | Alpaca `last_equity` | What actually happened | Execution reality |
| C — SPY | `prices_live.parquet` | Market benchmark | Absolute performance |
| D — Pure AI PM | `ai_pm_proposed` weights at 100%, normalized | AI PM signal quality independent of dilution | Signal quality at any level |

**Critical: Track A already contains AI PM influence.** Phase 1 Sonnet feeds `sleeve_weight_prior` into the quant before it runs. So Track A is not a clean "no AI PM" baseline — it measures only the incremental value of Phase 2 overrides. Track A★ is the true no-AI-PM baseline, computed by running the quant's alpha stack with default regime weights, ignoring any Phase 1 priors.

**Primary metric by level:**
- **Levels 1–2**: use Track D vs Track A★ — at 5–15% AI weight, Track B vs Track A is pure noise (0.05pp signal in 0.5pp daily vol). Track D isolates signal quality regardless of dilution.
- **Levels 3–5**: Track B vs Track A becomes meaningful (30–75% weight). Both Track D and Track B vs Track A are relevant.

**Track A★ implementation**: on each rebalance day, log what the quant's `merged_weights` would be using only default regime weights (no Phase 1 `sleeve_weight_prior`). This requires storing the pre-Phase-1 quant output. One snapshot, no extra API cost.

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

## GitHub Pages Dashboard Integration

The dashboard lives in `docs/index.html`, auto-generated by `scripts/generate_performance_page.py` and pushed on every `run_all_agents.py` run.

### What exists today (do not break)
- `_earned_authority_html(auth)` — shows old 4-phase tracker + shadow returns
- `_thesis_html(thesis)` — latest AI PM thesis card with overrides
- Capital allocation doughnut (from latest debate verdict)
- Debate accordion (`_debate_html`)
- `load_earned_authority()` — reads `data_cache/earned_authority.json`
- `load_latest_thesis()` — reads `outputs/ai_pm_theses/*.json`

### Changes required in `scripts/generate_performance_page.py`

**New data loaders (add at top of "Extra local data loaders" section):**

```python
def load_counterfactual() -> list[dict]:
    """Load logs/counterfactual_daily.jsonl — Track A★/A/B/C/D daily returns."""
    path = Path("logs/counterfactual_daily.jsonl")
    if not path.exists(): return []
    rows = []
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except: pass
    return rows

def load_perf_feedback() -> dict:
    """Load data_cache/ai_pm_perf_feedback.json — gates, metrics, confidence."""
    path = Path("data_cache/ai_pm_perf_feedback.json")
    if not path.exists(): return {}
    try: return json.loads(path.read_text())
    except: return {}

def load_ai_pm_decisions() -> list[dict]:
    """Load logs/ai_pm_decision_log.jsonl — per-rebalance override records."""
    path = Path("logs/ai_pm_decision_log.jsonl")
    if not path.exists(): return []
    rows = []
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except: pass
    return sorted(rows, key=lambda x: x.get("date",""))
```

**Rewrite `_earned_authority_html(auth, feedback)`:**

Replace old 4-phase tracker with 5-level career ladder. New inputs: `auth` (from `earned_authority.json`) + `feedback` (from `ai_pm_perf_feedback.json`).

Show:
- 5 level dots (Shadow → Analyst → Associate → Manager → Director → CEO) with current level highlighted
- Progress bar: days at current level / evaluation window
- Current title + weight badge: "Analyst — 5% authority — Day 3 of 21"
- Sortino edge vs threshold: "+0.08 (need +0.20)"
- If `stuck_alert: true` → show orange warning banner
- If `in_cooldown: true` → show cooldown countdown

**New `_promotion_gates_html(feedback)`:**

7-gate checklist. Each gate shows pass (green ✓) or fail (red ✗) with current value vs threshold:
```
✓ Hit rate:     52% ≥ 52%
✗ Sortino edge: +0.08 < 0.20 needed
✗ Profit factor: 1.1 < 1.2 needed
✗ Min decisions: 3 < 5 needed
✓ Fade rate:    20% ≤ 30%
✓ Regime gate:  no bad regime
✓ Cooldown:     not active
```

**New `_counterfactual_chart_html(cfdata)`:**

Multi-line equity chart using Chart.js (already used for the main equity curve). Four lines starting at 100 on AI PM live date:
- Track A★ — grey dashed: "Pure Quant (no AI PM)"
- Track B — green solid: "Actual Portfolio"
- Track C — blue dashed: "SPY"
- Track D — orange solid: "Pure AI PM Signal"

Below chart, one summary line: *"AI PM signal quality (D−A★): +0.44pp since live. Actual portfolio at Level 1 (5% weight): +0.07pp measurable impact."*

**New `_override_scorecard_html(decisions, feedback)`:**

Small table of last 5 rebalance overrides with outcomes:
```
Date       Symbol  Type     AI Weight  Quant Weight  +5d    +10d   +21d  Result
2026-06-10  STRL   amplify   9.0%       7.1%         +1.8%  +3.1%  +4.4%  WIN
2026-05-27  HUM    hold      7.1%       7.1%         +2.1%  −27.6%  ?    MISS
```
Plus summary stats: win rate, avg incremental alpha (10d), fade rate.

### Integration in `build_html()` function

Add new sections to the existing AI Intelligence card (`ai-section` div):

```python
# New loaders called at top of build_html or main():
cfdata    = load_counterfactual()
feedback  = load_perf_feedback()
decisions = load_ai_pm_decisions()

# New HTML sections:
gates_html       = _promotion_gates_html(feedback)
cf_chart_html    = _counterfactual_chart_html(cfdata)
scorecard_html   = _override_scorecard_html(decisions, feedback)

# Rewrite call:
authority_html = _earned_authority_html(auth, feedback)  # add feedback param
```

The existing `_thesis_html`, `_debate_html`, and allocation doughnut remain unchanged.

### Graceful empty states

All new loaders return empty list/dict if the file doesn't exist (first run before any AI PM data). Each HTML builder checks for empty input and renders a placeholder: *"No AI PM data yet — starts after next rebalance."* This ensures the dashboard never breaks before the new system is running.

---

## Files

### Modified
| File | Change |
|---|---|
| `ascent/strategy/earned_authority.py` | 5-level ladder, Sortino+hit rate criteria, daily evaluation, level guardrails, model selection per level |
| `run_all_agents.py` | Daily AI PM run (Phase 1 on non-rebalance, full on rebalance), quant snapshot before blend, inject perf feedback, decision log write, counterfactual daily scoring |
| `scripts/generate_performance_page.py` | Add `load_counterfactual()`, `load_perf_feedback()`, `load_ai_pm_decisions()`; rewrite `_earned_authority_html()` for 5-level ladder; add `_promotion_gates_html()`, `_counterfactual_chart_html()`, `_override_scorecard_html()`; wire into `build_html()` |

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
31. **Demotion uses Track D vs Track A★** — not Track B vs Track A; diluted blended portfolio is too noisy at low authority levels to trigger meaningful demotion signals
32. **Track A★ snapshot required on every rebalance** — store quant weights with default regime alpha weights (no Phase 1 priors) at each rebalance; idempotent write
33. **Phase 1 accuracy tracked separately** — sleeve_prior_value and regime accuracy scored in feedback file; consistently negative sleeve_prior_value enables a `disable_sleeve_priors` flag
34. **sleeve_weight_prior clipped to ±10pp of regime baseline** — Phase 1 cannot override risk management; regime engine protective adjustments always take precedence
35. **Override scoring excludes debate modifications** — AI PM override scored against pre-debate weight, not final executed weight; debate modifications logged under `debate_modification` in decision log
36. **Crisis regime always triggers Opus** — regardless of budget allocation; a crisis rebalance is never decided by Sonnet alone
37. **Track D vs Track A★ is primary metric at Levels 1–2** — Track B vs Track A is noise below Level 3 (30% weight); dashboard and promotion evaluation use Track D for signal quality at low authority
