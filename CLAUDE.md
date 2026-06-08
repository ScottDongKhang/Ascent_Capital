# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio → walk-forward → regime → 4 specialist agents → orchestration → AI PM (earned autonomy) → debate → Alpaca paper trading.

**Daily command**: `python run_all_agents.py` (branches on rebalance day)

---

## Environment

Python 3.12.13, venv at `.venv/`. Always use `.venv/bin/python`. API keys via `APIKeys.from_env()`. Config via `get_config()` — never `Config()` directly.

---

## LLM models (`ascent/llm/client.py`)

Import constants from here — never redefine locally:
```python
DEFAULT_MODEL = "claude-opus-4-6"          # AI PM synthesis (Phase 2)
SONNET_MODEL  = "claude-sonnet-4-6"        # debate agents, red team, pre-thesis (Phase 1)
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```

---

## Key layout

```
ascent/           core engine (config, data, features, alpha, portfolio, backtest,
                  research, regime, risk, reporting, execution, monitoring, llm,
                  dashboard, strategy, causal)
agents/           us_equities_agent, macro_agent, international_agent,
                  alternatives_agent, ai_pm_agent, red_team_agent
orchestrator/     central_intelligence.py
debate/           debate_runner, agents, judge
run_all_agents.py daily entrypoint
ascent/main.py    pipeline entrypoint (called by each agent)
data_cache/       parquet caches, earned_authority.json, active_alpha_config.json,
                  ai_prethesis_latest.json, ai_pm_pattern_memory.json
logs/             eod_log.jsonl, multi_agent_run.jsonl, ai_pm_calibration.jsonl
outputs/debate_log/  verdict_YYYY-MM-DD.json
```

---

## Integrity constraints (never violate)

1. No look-ahead bias — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only.
2. No simulated data under live cache names (`prices_live` = Yahoo live only; fallback → `prices_live_fallback_simulated`).
3. Max-weight hard cap via `_water_fill_cap()` with post-condition check.
4. Sector constraint: < 80% coverage → skip caps + warn, never collapse to single name.
5. Debate is advisory only — never writes to alpha, portfolio, or execution modules.
6. New alpha sleeves: update `DEFAULT_ALPHA_WEIGHTS` in BOTH `ascent/alpha/stack.py` AND `ascent/research/self_improve.py`.
7. Fundamental sleeve is disabled (IC-t = −4.75, anti-signal) — do not re-enable without positive IC-t.

---

## Non-obvious gotchas

- **Agent module names**: `agents.us_equities_agent` not `agents.us_equities` — `_agent` suffix required for all four.
- **`RegimeEngine` constructor**: takes `config=dict`, not a `Config` object.
- **`bdate_range(end="today")`**: returns empty on weekends — use explicit weekday rollback.
- **`apply_hedge_overlay`**: must accept both `RegimeSignal` and plain `str`.
- **ML sleeve cache**: must store `feature_names` — XGBoost crashes on shape mismatch if feature set changes between writes.
- **AI PM two-phase**: Phase 1 = `run_ai_pm_prethesis()` uses `SONNET_MODEL`. Phase 2 = `run_ai_pm(prethesis=...)` uses `DEFAULT_MODEL` (Opus). Never swap — Sonnet for breadth, Opus for judgment.
- **`propose_prethesis` vs `propose_portfolio`**: different tools, different result stores. Phase 1 ends with `propose_prethesis`, Phase 2 with `propose_portfolio`.
- **Pre-thesis runs before quant agents** in `run_all_agents.py`. Failure → `prethesis=None` → graceful single-phase fallback.
- **`run_ai_pm(quant_outputs=...)`**: pass `agent_outputs` list or AI PM re-runs all 4 agents (~160s wasted).
- **`ascent/main.py` returns 10-tuple** (adds `_alpha_breakdown`) — `eod_runner.py` and `us_equities_agent.py` must unpack correctly.
- **`_SPARSE_FILL_ZERO`** in alpha stack: must include ALL sparse panels or NaN-drop silently disables those sleeves.
- **PDF generation**: use `reportlab` — `weasyprint` requires system GObject/Pango unavailable here.
- **PDBC↔KMLM correlation (~0.81)**: frequently triggers orchestrator correlation guard → KMLM halved. Expected, not a bug.
- **Kill switches** (pending paper validation ~July 2026): `EVENT_TRADING_ENABLED=False`, `TWAP_ENABLED=False`, `SELF_MODIFY_ENABLED=False`, `LONG_SHORT_ENABLED=False`.

---

## Data / caching

Cache name provenance — never hide: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live`/`macro_simulated`, `profiles` (sector metadata).

Point-in-time joins: always use `as_of_join()` / `as_of_merge()` for cross-dataset alignment.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Current state (as of 2026-06-07)

- **AI PM**: Level 1 (Analyst, 5% authority). Day 1: June 4. Next rebalance June 10. `data_cache/earned_authority.json` is ground truth.
- **Regime**: calm_bull. **Tests**: ~777 passing.
- **WF OOS baseline**: Sharpe 0.483, CAGR +12.61%, true alpha +2.54% vs SPY.
- **Live perf**: +8.8% portfolio vs +15.9% SPY (Apr 1 – Jun 3). Early April defensive drag is the main gap.
- **GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every daily run.

---

## Session log

> Full history archived to `docs/session_log_archive.md`.

### 2026-06-08 (TradingAgents integration — per-ticker memory + instrument identity + StockTwits)
- `memory/ticker_memory.py` (new): per-ticker AI PM decision log, outcome scoring (incremental alpha at 10d/21d), win/miss/fade/early classification. Storage: `memory/ticker_memory.jsonl`.
- `ascent/integrations/stocktwits.py` (new): StockTwits public API crowd sentiment, band classification, stale guard. Logs IC to `logs/stocktwits_ic.jsonl`.
- `agents/ai_pm_agent.py`: `_build_data_grounding()` now includes sector/industry identity from profiles.parquet. Phase 2 prompt injects per-ticker track record + cross-ticker lessons. Both phases receive StockTwits sentiment block.
- `run_all_agents.py`: `_write_decision_log()` records per-ticker AI PM decisions; daily path scores outcomes; StockTwits fetch gated inside `if is_rebalance:`.
- `tests/memory/test_ticker_memory.py` (new): 10 tests. `tests/integrations/test_stocktwits.py` (new): 8 tests. All 18 passing.
- Fixes: `scored=True` on None fetch (now `continue`), atomic write via `.tmp`+rename, zero-alpha → `None` verdict, walrus operator for ticker context, weak test assertion strengthened.
- Nothing left open.

### 2026-06-07 (MiroFish sentiment validation layer + judge → Opus)
- `ascent/integrations/mirofish_client.py` (new): MiroFish REST client — 8-step API flow, 8-min deadline timer, graceful None on any failure.
- `ascent/integrations/analogue_matcher.py` (new): TF-IDF cosine similarity against 25-event `data_cache/mirofish_analogues.json` library; keyword fallback if sklearn unavailable.
- `ascent/integrations/mirofish_calibration.py` (new): `bootstrap_calibration()` + `get_base_rate()` + `record_entry()`. Idempotent bootstrap from analogues library.
- `ascent/integrations/get_mirofish_sentiment.py` (new): full tool executor — alignment score formula, decision rules (amplify >0.70, trim 25% <0.40 + negative base rate), timeout fallback.
- `agents/ai_pm_agent.py`: `get_mirofish_sentiment` wired into `AI_PM_TOOLS` (Phase 2 only, not PRE_THESIS_TOOLS).
- `debate/agents.py`: devil's advocate attacks AI PM thesis when mirofish alignment < 0.50.
- `debate/judge.py`: judge upgraded from `SONNET_MODEL` → `DEFAULT_MODEL` (Opus) per user direction.
- `run_all_agents.py`: forwards `mirofish_sentiment` from AI PM result into `portfolio_state` for debate.
- `data_cache/mirofish_analogues.json` (new): 25 landmark market events (force-added, gitignored dir).
- `data_cache/mirofish_calibration.json` (new): empty initial state, bootstrapped at runtime.
- 19/19 MiroFish tests + 77/77 debate+AI PM tests passing.
- Files: above new + `ascent/integrations/__init__.py`, `agents/ai_pm_agent.py`, `debate/agents.py`, `debate/judge.py`, `run_all_agents.py`.
- Nothing left open.

### 2026-06-07 (debate persona upgrade — Druckenmiller / Burry / Taleb)

The debate layer had a structural flaw: all three LLM agents (bull, bear, devil's advocate) shared the same underlying epistemology — risk/return framing, Gaussian assumptions, generic "concentration risk" arguments. They disagreed on conclusions but asked the same questions. That's not a real risk committee; it's one brain arguing with itself.

This session rewired each agent with a distinct analytical lens borrowed from the `ai-hedge-fund` repo (but adapted to Ascent's data — no `financial_datasets` API, no fundamentals):

**Bull → Druckenmiller**: The question changes from "is this a good portfolio?" to "where is the asymmetry?" Druckenmiller's framework demands the bull quantify the upside/downside ratio explicitly — if the Monte Carlo p95 is 3x the p5 loss, say so. Forces momentum-first arguments and penalizes defensive hedging in strong-regime setups. Identifies positions the adversarial engine rated hard-to-short as structural longs.

**Bear → Burry**: The question changes from "what could go wrong?" to "what is the weakest number?" Burry's framework demands the bear lead with a specific adversarial score, VaR figure, or event-risk flag — not a generic drawdown warning. The Burry persona also introduces an anti-manufacturing constraint: if the adversarial engine found no compelling bear case, say so. Generic bear arguments without hard numbers are dismissed.

**Devil's Advocate → Taleb**: The question changes from "what's the biggest risk?" to "is this portfolio convex or concave?" This is a fundamentally different question. A portfolio can look fine on every traditional metric (positive expected return, no individual position > 10%) and still be structurally concave — more downside than upside in expectation. That's the Taleb insight no generic devil's advocate would ever surface.

New `_section_tail_asymmetry()` function computes this from existing Monte Carlo data:
- Tail ratio = (p95 − p50) / (p50 − p5). Below 1.0 = provably concave.
- Regime entropy check: low entropy (<0.5) triggers "turkey problem" warning — the model is confident but that confidence is itself a fragility signal.
- Injected into devil's advocate context only (bull and bear don't need it; it would dilute their respective lenses).

What the debate layer can now catch that it couldn't before: a portfolio that passes the quant sanity check, has no adversarial flags, and looks clean to the regime specialist — but is structurally concave because the strategy's upside is capped (by max_weight, sector limits) while the downside is fat-tailed. That's a construction-level fragility, not a position-level one. Only a Taleb-framed agent looks for it.

- Files: `debate/agents.py`
- Nothing left open from this session.

### 2026-06-05 (AI PM learning system + hallucination prevention)
- `ascent/strategy/ai_pm_learning.py` (new): daily Sonnet brief, post-mortem (~21d lag), pattern memory (Haiku → `data_cache/ai_pm_pattern_memory.json`). Injected into every Phase 1+2 prompt.
- Hallucination prevention (all 4 in code, not prompts): `_build_data_grounding()`, `_apply_recency_gate_python()`, feedback citation gate in `_tool_propose_portfolio()`, conviction inflation cap.
- AI PM bootstrapped to Level 1 (5% authority). Day 1: June 4.
- Files: `ascent/strategy/ai_pm_learning.py` (new), `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-04 (AI PM Progressive Authority System + quant two-way integration)
- `earned_authority.py` rewrite: 5-level ladder (Shadow→Analyst→Associate→Manager→Director), Sortino-based promotion/demotion, 5-day cooldown, 63-day stuck alert.
- `ai_pm_guardrails.py` (new), `ai_pm_counterfactual.py` (new), `ai_pm_perf_feedback.py` (new).
- `build_alpha_stack(return_breakdown=True)` → signal quality in `AgentOutput.metadata`. Pre-thesis alpha floor in `ascent/main.py`. Phase 1 writes `data_cache/ai_prethesis_latest.json`.
- Honest WF OOS baseline: Sharpe 0.483. Multi-asset/long-short all scored worse — current config is the ceiling.
- Files: `earned_authority.py`, `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py`, `ascent/alpha/stack.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-03 (WF OOS framework + signal fixes)
- WF framework at `ascent/research/wf_framework/`. Fundamental sleeve disabled. KMLM overweight fixed. IC gate tightened (-0.010 → -0.005). `_log_holdings` day_return now from Alpaca `last_equity`.
- Files: `orchestrator/central_intelligence.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `ascent/monitoring/attribution.py`.

### 2026-06-02 (regime hardening)
- Hard crisis override (VIX>30 + SPY 5d<−7%), asymmetric hysteresis (down 0.40 / up 0.70), entropy penalty (entropy<1e-6 → ×0.90).
- Files: `ascent/regime/engine.py`, `ascent/regime/decision.py`, `ascent/regime/types.py`.

### 2026-06-01 (causal intelligence + investor letter)
- `ascent/causal/` module: PC algorithm DAG (`causal_discovery.py`), per-symbol graph builder (`dag_builder.py`), compatibility gates, early-exit tracker (`tracker.py`). Devil's advocate receives causal mechanisms.
- `ascent/reporting/investor_letter.py` (new): Sonnet monthly letter on first trading day of month.
- Files: `ascent/causal/` (new module), `agents/ai_pm_agent.py`, `run_all_agents.py`, `debate/agents.py`, `ascent/reporting/investor_letter.py`.

### 2026-05-31 (anti-hallucination hardening)
- `generate_structured` gains `json_schema` param (wire-level Anthropic enforcement). Amnesia/grounding instructions + evidence schema in `llm_fundamental` and `narrative_alpha`. `_EVIDENCE_RULE` in `debate/agents.py`.

### 2026-05-25 — 2026-05-28 (adversarial intelligence + two-phase AI PM + GitHub Pages)
- Debate redesigned as genuine risk committee: `adversarial_engine.py`, `adversarial_authority.py`, `adversarial_monitor.py`. ONE falsifiable change per rebalance.
- AI PM two-phase: Sonnet pre-thesis (before quant) + Opus synthesis. Crowding gate (`get_crowding_signal`). `momentum_exhaustion` override type replaces valuation abuse.
- GitHub Pages dashboard (`scripts/generate_performance_page.py` + `docs/index.html`). Auto-pushed after every daily run.
