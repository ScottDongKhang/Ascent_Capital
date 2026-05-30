# AI-Native Learning System — Design Spec

**Goal:** Make Ascent Capital genuinely smarter every rebalance by combining three compounding learning loops: AI-informed regime detection, empirical sleeve weight learning, and AI self-calibration from its own track record.

**Primary objectives:** AI nativeness, alpha improvement, Sharpe improvement.

---

## What problem this solves

The system currently has three manual gaps:

1. **Regime model fires on price patterns only.** In April 2026, the HMM called "stressed" during a tariff relief rally (VIX-calm, market recovering) and triggered a 30% exposure cut while SPY ran +6.8%. Sonnet, reading actual earnings calls and Fed language, would have known the rally was real. The AI has no voice in regime determination.

2. **Sleeve weights are hardcoded.** We've manually intervened three times to zero out negative-IC sleeves (fundamental, statarb). The system has no mechanism to discover and fix these itself. The next time a sleeve drifts negative, someone has to notice and patch it manually.

3. **The AI PM has no memory of being right or wrong.** Sonnet forms each pre-thesis from scratch with no knowledge of whether its previous market calls played out. A human analyst who never got performance feedback wouldn't improve. Neither does Sonnet.

---

## Architecture: three components

```
Pre-thesis (Sonnet)
  reads: calibration track record        [Component C]
  outputs: market_character
           regime_assessment             [Component A]
           sleeve_weight_prior           [Component B]

Regime Engine
  blends: HMM output + AI regime_assessment at weight α   [Component A]
  output: better regime label → propagates everywhere

build_alpha_stack()
  gets weights: MetaLearner(regime, sleeve_prior)          [Component B]
  fallback: DEFAULT_ALPHA_WEIGHTS_BY_REGIME if unavailable
  IC gate: _get_gated_weights() still runs after [unchanged]

Post-rebalance (next rebalance day)
  updates: MetaLearner posterior from realized IC           [Component B]
  updates: AI calibration accuracy scores                   [Component C]
  α grows: if AI regime calls are proving accurate          [Component A]
```

---

## Component A: AI Regime Advisor

### What it does

The pre-thesis already runs before every rebalance. Add one output field: `regime_assessment` containing a regime label, confidence score (0–1), and one-sentence reasoning.

This blends with the HMM's probability output:
```
blended_prob = (1 - α) × HMM_prob + α × AI_prob
```

Where `α` starts at **0.05** (AI has 5% voice) and grows by 0.03 each time the AI's regime call matches the regime that actually produced better IC outcomes, capped at **0.30**. The AI earns influence by being right, same as the AI PM earns portfolio authority.

### Why the regime signal is the highest-leverage point

Regime drives: sleeve weights, orchestrator allocation, SPY 200MA overlay, VIXY hedge sizing, debate context, earned authority advance/revert. One better input improves the entire stack simultaneously.

### Guardrails

- α never exceeds 0.30 (HMM always has majority vote)
- AI regime label must be one of the valid labels (`calm_bull`, `stressed`, `crisis`, `euphoric`, `uncertain`) — invalid output → ignore, use HMM only
- If pre-thesis fails, α contribution = 0 (fallback to HMM unchanged)
- Every blend logged to `logs/regime_blend_log.jsonl` with HMM prob, AI prob, α, final label

### New files

- `logs/regime_blend_log.jsonl` — audit trail per rebalance

### Modified files

- `agents/ai_pm_agent.py` — add `regime_assessment` to pre-thesis output schema and `PRE_THESIS_TOOLS`
- `ascent/regime/engine.py` — `blend_with_ai(ai_regime_assessment, ai_alpha)` method applied after HMM fit
- `run_all_agents.py` — pass AI regime assessment to engine before stack runs

---

## Component B: IC-Seeded Bayesian Meta-Learner

### What it does

Replaces the hardcoded `DEFAULT_ALPHA_WEIGHTS_BY_REGIME` table with a living posterior that learns which sleeves are working in each regime from observed IC data.

### State

For each (sleeve, regime) pair, maintain:
- `μ` — posterior mean IC estimate
- `σ²` — posterior variance (uncertainty)
- `n` — number of rebalance-level observations

Stored in `data_cache/sleeve_posteriors.json`.

### Initialization (day 1 payoff)

Seed `μ` from the existing `sleeve_ic_log.jsonl` rather than hardcoded theory. On the first run, the meta-learner already knows trend IC = +0.0153, statarb IC = -0.0016, fundamental IC = -0.0078 from 29 days of observed data. Starting from evidence, not guesses.

Initial variance: `σ²₀ = 0.005` — wide enough that 3–4 rebalance observations meaningfully move the posterior.

### Update rule (rebalance-level only — daily is too autocorrelated)

After each rebalance holding period, compute realized IC per sleeve over that window and apply the Gaussian conjugate update:

```
precision_post = 1/σ² + 1/σ²_ε        (σ²_ε = 0.003, observation noise)
μ_post = (μ/σ² + ic_realized/σ²_ε) / precision_post
σ²_post = 1 / precision_post
n_post = n + 1
```

No daily updates — daily IC is too autocorrelated to add meaningful information.

### Weight derivation

```python
raw_w = max(0, μ_s) / σ_s      # IC × precision (zero if IC negative)
kelly_w = normalize(raw_w)      # normalized Kelly weights

# Blend toward regime defaults when data is sparse
α_conf = min(1.0, n_s / 20)    # 0% data-driven at n=0, 100% at n=20
final_w = α_conf × kelly_w + (1 - α_conf) × regime_default_w
```

This is the **"smarter every run"** behavior: run 1 = 100% regime defaults, run 10 = 50/50, run 20+ = fully empirical.

### AI prior injection (this rebalance only)

When pre-thesis outputs `sleeve_weight_prior = {trend: +0.004, statarb: -0.002}`:
- Shift the effective `μ_effective = μ_s + Δ_ai` (bounded: |Δ_ai| ≤ 0.010 IC units)
- This affects weights for this rebalance only — does NOT write to the posterior
- AI opinion influences the current rebalance; empirical IC updates the long-run posterior

### Priority chain (unchanged structure)

```
active_alpha_config.json by_regime     (self-improve, highest priority)
→ [MetaLearner posterior]              (new, replaces static table)
→ DEFAULT_ALPHA_WEIGHTS_BY_REGIME      (fallback if meta-learner unavailable)
→ active_alpha_config.json global
→ DEFAULT_ALPHA_WEIGHTS                (flat fallback)
```

### Guardrails

- Final weights must sum to 1.0 ± 0.02 — renormalize if violated
- No single sleeve > 0.75
- AI prior shifts bounded to ±0.010 IC units per sleeve
- Regime with n < 3 observations: use 95% regime defaults (don't trust sparse posterior)
- IC gate (`_get_gated_weights`) still runs AFTER meta-learner — hard safety net unchanged
- All weight proposals logged to `logs/meta_learner_weights.jsonl`

### New files

- `ascent/alpha/meta_learner.py` — `SleeveMetaLearner` class
- `data_cache/sleeve_posteriors.json` — posterior state (auto-created)
- `logs/meta_learner_weights.jsonl` — audit trail per rebalance

### Modified files

- `ascent/alpha/stack.py` — `_load_active_alpha_weights()` calls meta-learner before regime defaults
- `run_all_agents.py` — post-rebalance: compute realized sleeve IC, call `meta_learner.update_rebalance()`

---

## Component C: AI Calibration

### What it does

Tracks whether the AI PM's market character calls are proving correct, and injects that track record into the next pre-thesis so Sonnet reasons from evidence rather than vibes.

### Prediction logging (every rebalance)

Record to `logs/ai_thesis_outcomes.jsonl`:
```json
{
  "thesis_date": "2026-06-09",
  "regime": "calm_bull",
  "market_character": "momentum_continuation",
  "sleeve_weight_prior": {"trend": 0.004, "statarb": -0.002},
  "realized_ic_leaders": null,     // filled in next rebalance
  "prediction_correct": null       // filled in next rebalance
}
```

### Outcome fill (next rebalance)

After holding period, check which sleeves had positive realized IC. If the top IC sleeve matches what the market_character implied should work → correct. Update `prediction_correct`, compute rolling accuracy per (regime, character_type).

### Context injection (next pre-thesis)

~200 tokens added to the pre-thesis system prompt:
```
Calibration note (calm_bull):
- momentum_continuation calls: 3/5 correct (60%)
- Last miss (2026-05-19): called momentum_continuation but statarb IC 
  turned positive while trend IC declined. Sector rotation was building.
```

Zero extra API cost — injected into the existing pre-thesis call.

### Accuracy → AI regime authority

When AI calibration accuracy for a given regime exceeds 60% over 5+ calls, that contributes to α growth in Component A. The two loops reinforce each other: better calibration → higher AI regime weight → better blended regime.

### New files

- `ascent/strategy/ai_calibration.py` — `log_thesis()`, `update_outcome()`, `get_context()`
- `logs/ai_thesis_outcomes.jsonl` — prediction log

### Modified files

- `agents/ai_pm_agent.py` — pre-thesis reads calibration context, outputs `market_character`
- `run_all_agents.py` — post-rebalance: call `ai_calibration.update_outcome()`

---

## Payoff timeline

| Component | Next run | Month 3 | Month 6+ |
|-----------|----------|---------|---------|
| AI Regime Advisor | Small AI pull on regime (5%) | Meaningful if AI is accurate | AI earning real regime authority (up to 30%) |
| IC Meta-Learner | Seeded from real IC data, near-identical to hardcoded | Auto-discovered patterns, posterior diverging from defaults | Full empirical regime-IC knowledge, never needs manual fixing |
| AI Calibration | Logging starts | 8–10 data points, meaningful context | Full track record, AI reasons from evidence |

---

## What stays completely unchanged

All 4 specialist agents, orchestrator, portfolio construction (MVO + sector constraints), debate layer, execution, kill switch, earned authority progression, monitoring, slippage tracker, approval server, R2R/BM25 memory, weekend pipeline.

---

## Files created / modified summary

**New:**
- `ascent/alpha/meta_learner.py`
- `ascent/strategy/ai_calibration.py`
- `data_cache/sleeve_posteriors.json` (auto-created on first run)
- `logs/meta_learner_weights.jsonl`
- `logs/regime_blend_log.jsonl`
- `logs/ai_thesis_outcomes.jsonl`

**Modified (minimal changes):**
- `agents/ai_pm_agent.py` — new output fields, reads calibration context
- `ascent/regime/engine.py` — AI blend method
- `ascent/alpha/stack.py` — meta-learner in priority chain
- `run_all_agents.py` — post-rebalance feedback loops
