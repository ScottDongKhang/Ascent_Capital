# AI PM Agent — Design Spec
**Date:** 2026-05-16  
**Status:** Approved for implementation

---

## Goal

Replace the "quant engine with LLM advisory" model with a genuinely AI-native fund: an AI Portfolio Manager that does its own research, forms its own portfolio thesis, and earns execution authority through a live track record. The quant pipeline becomes a permanent sanity check (20% floor), not the primary decision-maker.

---

## Decisions

| Question | Decision |
|----------|----------|
| Authority model | AI reasoning is the final call; quant is a sanity check, not a veto |
| Safety | Earned Autonomy — ai_weight starts 0%, scales with rolling track record |
| Tool access | Both raw data + quant signals, labeled and separated |
| Agent scope | 4 specialist agents become tool calls the AI PM can invoke |
| Research loop | Structured 4-phase loop, max 14 tool calls |
| Thesis format | Full investment memo (JSON) |
| Blend method | Weight-average over union + min_weight filter |

---

## Architecture

```
run_all_agents.py
  Step 0–3 (unchanged): 4 agents run in parallel → orchestrator → merged_weights

  Step 4 (new): AI PM Agent
    Phase 1 — Market context      (2 tool calls max)
      get_regime_state()
      get_macro_data()

    Phase 2 — Quant baseline      (4 tool calls max)
      run_quant_agent("us_equities")
      run_quant_agent("macro")
      run_quant_agent("international")
      run_quant_agent("alternatives")

    Phase 3 — Signal research     (up to 6 tool calls, AI chooses which)
      get_sec_signal(symbol)
      get_transcript_signal(symbol)
      get_attribution_history(symbol)
      get_earnings_signal(symbol)
      get_past_verdicts(regime)
      get_factor_exposures(draft_weights)
      get_var_estimate(draft_weights)
      get_sector_concentration(draft_weights)
      get_position_momentum(symbols)

    Phase 4 — Submit              (terminal)
      propose_portfolio(weights, thesis)   ← ends the loop

  Step 5 (new): Risk validation
    pm_risk_validator(ai_portfolio)
    → if violations: use merged_weights at 100%, log reason
    → if ok: proceed to blend

  Step 6 (new): Blend
    earned_authority.blend(ai_portfolio, merged_weights)
    → weight-average over union
    → drop positions < min_weight=0.02, renormalize
    → final_weights

  Step 7 (new): Audit
    record_event("ai_pm_proposal", thesis)

  Step 8+ (unchanged): debate → verdict → execute
```

What stays exactly the same: execution layer, approval gate, TWAP, kill switches, compliance audit trail, debate layer, regime engine, all 4 agent pipelines.

---

## New Files

### `agents/ai_pm_agent.py`

**Responsibilities:**
- Define all 14 tool schemas (Anthropic format)
- Implement all tool executor functions (pure Python, never raise unhandled exceptions)
- Run `tool_completion()` from `ascent/llm/client.py` with `max_tool_calls=14`
- Parse the `propose_portfolio` tool call into `AIPMResult(portfolio: dict, thesis: dict)`
- Fallback: if loop exits without `propose_portfolio`, return `AIPMResult(portfolio={}, thesis={}, fallback=True)`

**Tool executor implementations:**

| Tool | Implementation |
|------|----------------|
| `get_regime_state` | Read `dashboard/regime_signal.json` |
| `get_macro_data` | Read last row of macro parquet cache |
| `run_quant_agent(agent_id)` | Import and call the existing agent's `run()` function; serialize `AgentOutput` as labeled text |
| `get_sec_signal(symbol)` | Read `data_cache/sec_signals.parquet`, filter to symbol + latest date |
| `get_transcript_signal(symbol)` | Read `data_cache/transcript_signals.parquet`, filter |
| `get_attribution_history(symbol)` | Read `logs/attribution_log.jsonl`, last 63 days for symbol |
| `get_earnings_signal(symbol)` | Read `data_cache/earnings_cache.parquet`, filter |
| `get_past_verdicts(regime)` | Read `outputs/debate_log/`, filter last 5 verdicts where regime matches |
| `get_factor_exposures(weights)` | Call `ascent/risk/factor_exposure.py:format_exposure_context()` |
| `get_var_estimate(weights)` | Reuse `debate/agent_tools.py:get_var_estimate()` |
| `get_sector_concentration(weights)` | Reuse `debate/agent_tools.py:get_sector_concentration()` |
| `get_position_momentum(symbols)` | Reuse `debate/agent_tools.py:get_position_momentum()` |
| `propose_portfolio(weights, thesis)` | Store result, signal loop to terminate |

**System prompt (key constraints):**
```
You are the portfolio manager of Ascent Capital, a multi-strategy quantitative fund.

Work through your research in order:
1. Understand the macro environment and regime (Phase 1 tools)
2. Review what the quant models are recommending (Phase 2 tools)
3. Research specific names you are considering — agree or override with explicit reasoning (Phase 3 tools)
4. Submit your final portfolio and investment thesis via propose_portfolio

Rules:
- You must call propose_portfolio before finishing.
- Weights must be positive and will be normalized. Aim for 12–20 positions.
- Every quant override must include a specific reason referencing the signal data.
- If data is unavailable for a symbol, say so — do not fabricate signals.
```

**Model:** `claude-opus-4-6` (from `ascent/llm/client.py:DEFAULT_MODEL`)

---

### `ascent/strategy/earned_authority.py`

**State** (persisted to `data_cache/earned_authority.json`):
```json
{
  "ai_weight": 0.0,
  "phase": 0,
  "phase_start_date": "2026-05-16",
  "ai_returns_21d": [],
  "quant_returns_21d": [],
  "auto_revert_count": 0,
  "last_updated": "2026-05-16"
}
```

**Authority schedule:**

| Phase | ai_weight | Advance condition |
|-------|-----------|------------------|
| 0 | 0.0 (shadow) | 21 returns recorded AND ai_sharpe > quant_sharpe + 0.05 |
| 1 | 0.25 | Another 21 returns AND same condition |
| 2 | 0.50 | Another 21 returns AND same condition |
| 3 | 0.75 | Sustained (max operational weight) |
| cap | 0.80 | Hard cap — quant floor = 0.20 always |

**Auto-revert condition:** At any phase, if `ai_21d_drawdown > quant_21d_drawdown + 0.05` (5 percentage points) → reset `ai_weight=0`, `phase=0`, `auto_revert_count += 1`, clear return buffers, restart 21-day clock.

**Public API:**
```python
def update_authority(ai_daily_return: float, quant_daily_return: float) -> dict:
    """Append daily returns, check advance/revert, save state. Returns current state."""

def blend(ai_portfolio: dict, quant_portfolio: dict) -> dict:
    """
    Weight-average over union of both portfolios.
    final[sym] = ai_weight * ai_portfolio.get(sym, 0) + quant_weight * quant_portfolio.get(sym, 0)
    Drop symbols below min_weight=0.02, renormalize to 1.0.
    Returns final_weights dict.
    """

def get_state() -> dict:
    """Return current state dict (loaded from JSON)."""
```

**Daily return computation:** In `run_all_agents.py`, after P&L is logged:
- `ai_daily_return` = return of AI PM shadow portfolio (tracked in `data_cache/ai_pm_shadow_returns.jsonl`)
- `quant_daily_return` = return of `merged_weights` portfolio (already logged in PnL logs)

During shadow phase (ai_weight=0), AI PM portfolio is tracked hypothetically — positions are computed and stored but no real capital is allocated to them.

---

### `ascent/risk/pm_risk_validator.py`

Pre-blend hard-limit check. Runs on the AI PM's proposed portfolio before any blending.

**Checks:**
1. Weights sum to 1.0 ± 0.01 (after normalization)
2. No single position > 15%
3. No single sector > 40%
4. No name in distressed filter (mom_252d < −0.65) — reads from feature cache
5. At least 5 positions (concentration guard)
6. No position < 0 (no shorting)

**Behavior on violation:** Returns `(ok=False, violations=["..."])`. Caller falls back to quant portfolio at 100% and logs the violations to `logs/ai_pm_log.jsonl`.

```python
def validate(portfolio: dict) -> tuple[bool, list[str]]:
    """Returns (ok, violations). Never raises."""
```

---

### `ascent/strategy/thesis_formatter.py`

Converts raw `propose_portfolio` tool call output into two forms:

1. **Full JSON** saved to `outputs/ai_pm_theses/YYYY-MM-DD-thesis.json` (schema shown in design decisions above)
2. **Plaintext summary** (3–4 sentences) for embedding in monthly investor report PDF

```python
def format_thesis(raw_thesis: dict, as_of_date: date) -> dict:
    """Validate and serialize full investment memo JSON. Fill missing fields with defaults."""

def thesis_to_plaintext(thesis: dict) -> str:
    """Return 3–4 sentence narrative summary for investor reports."""
```

---

## Shadow Period Tracking

During Phase 0 (ai_weight=0), the AI PM still runs every rebalance day and produces a portfolio. That portfolio's hypothetical daily returns are tracked in `data_cache/ai_pm_shadow_returns.jsonl`:

```json
{"date": "2026-05-16", "ai_return": 0.0082, "quant_return": 0.0031, "ai_weight_at_time": 0.0}
```

This is what `earned_authority.update_authority()` reads. The shadow period requires the AI PM to prove itself before getting any real capital — identical in spirit to `SELF_MODIFY_ENABLED=False`.

---

## Wiring Changes to `run_all_agents.py`

```python
# After existing orchestrator step (merged_weights computed):

# New Step 4: AI PM
try:
    ai_pm_result = run_ai_pm(quant_outputs, merged_weights)
    ok, violations = validate_pm_proposal(ai_pm_result.portfolio)
    if ok:
        final_weights = earned_authority.blend(ai_pm_result.portfolio, merged_weights)
    else:
        log.warning("[AIPMAgent] Proposal rejected: %s — using quant 100%%", violations)
        final_weights = merged_weights
    record_event("ai_pm_proposal", {"thesis": ai_pm_result.thesis, "violations": violations})
    thesis_formatter.format_thesis(ai_pm_result.thesis, today)
except Exception as exc:
    log.error("[AIPMAgent] Failed: %s — using quant portfolio", exc)
    final_weights = merged_weights

# New Step 4b: update shadow returns (non-rebalance days too)
earned_authority.update_authority(ai_daily_return, quant_daily_return)

# Existing: debate runs on final_weights (unchanged)
```

---

## Tests (`tests/test_ai_pm_agent.py`) — 16 tests

**Earned authority:**
- `test_shadow_phase_returns_zero_ai_weight` — blend at phase=0 returns pure quant portfolio
- `test_authority_advances_after_21d_edge` — after 21 returns with AI Sharpe > quant+0.05, ai_weight → 0.25
- `test_authority_does_not_advance_without_edge` — 21 returns, no edge → stays at 0.0
- `test_auto_revert_on_drawdown` — AI drawdown > quant+5% → ai_weight=0, phase=0, count increments
- `test_hard_cap_at_0_80` — phase 3 never exceeds 0.80

**Blend:**
- `test_blend_union_of_positions` — positions only in AI PM portfolio get ai_weight × weight
- `test_blend_min_weight_filter` — positions below 0.02 are dropped and renormalized
- `test_blend_renormalizes_to_1` — final weights always sum to 1.0 ± 0.001

**Risk validator:**
- `test_validator_rejects_concentrated_position` — single position > 15% → violation
- `test_validator_rejects_distressed_name` — mom_252d < -0.65 → violation
- `test_validator_rejects_sector_overweight` — single sector > 40% → violation
- `test_validator_accepts_clean_portfolio` — valid portfolio → (True, [])

**Thesis formatter:**
- `test_format_thesis_fills_missing_fields` — missing keys get default values, no KeyError
- `test_thesis_to_plaintext_returns_string` — plaintext non-empty string

**AI PM agent:**
- `test_fallback_on_no_propose_portfolio_call` — loop exits without terminal tool → AIPMResult(portfolio={}, fallback=True)
- `test_tool_executor_never_raises` — all tools called with bad inputs return strings, never raise
