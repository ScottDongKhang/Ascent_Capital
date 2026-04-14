# CLAUDE.md — Ascent Capital

## What this project is

Ascent Capital is a modular Python quant research and trading platform. It ingests market and macro data, builds multi-sleeve alpha, constructs constrained portfolios, evaluates them through walk-forward out-of-sample testing, and executes through Alpaca paper trading. A multi-agent orchestration shell and LLM debate layer sit above the core engine.

This is not a backtest notebook. It is a layered system with distinct jobs at each layer: data, features, alpha, portfolio construction, backtest, walk-forward evaluation, regime modeling, specialist agents, orchestration, debate, execution, monitoring, and Intel reporting.

---

## Repository layout

```
ascent/                     # Core quant engine — do not restructure
  config/                   # settings.py (Config, APIKeys, UniverseConfig, BacktestConfig), types.py (AgentOutput)
  data/                     # ingest (yahoo, fred, polygon, tiingo, simulated), normalize, store (parquet, point_in_time), universe
  features/                 # build_features, feature_defs, targets
  alpha/                    # trend, meanrev, statarb, ml_sleeve, stack (composite combiner)
  portfolio/                # optimizer — top_n_equal_weight, rank_weighted, sector_constrained_weighted, _water_fill_cap
  backtest/                 # engine, costs — ledger-producing backtest
  research/                 # walk_forward_runner, evaluation, splits, leakage_test, adaptive_optimizer, self_improve
  regime/                   # engine, model, features, decision, integration, posture, breaks, types
  risk/                     # correlation_guard (cross-agent 0.70 cap)
  dashboard/                # export_dashboard_data, build_dashboard
  reporting/                # market_memo, ic_brief_generator, blind_spot_detector, debrief, regime_narrative
  execution/                # eod_runner, alpaca_broker, order_engine, kill_switch, run_log, slippage_tracker, approval_server
  monitoring/               # live_vs_backtest, skill_tracker, forward_pnl_tracker, pre_rebalance_checklist, exit_alerts
  llm/                      # client.py — centralized Anthropic API wrapper

agents/                     # Specialist agent wrappers (us_equities, macro, international, alternatives)
orchestrator/               # central_intelligence.py — capital allocation, conviction scoring, correlation guard, crisis veto
debate/                     # debate_runner, agents (bull/bear/devil/regime specialist/quant sanity), judge, outcome_tracker
memory/                     # r2r_interface — R2R ingestion and query wrapper
simulation/                 # mirofish_interface — Monte Carlo scenario simulation

data_cache/                 # Parquet caches (prices_live, macro_live, profiles, ml_model_*.pkl)
                            # Also: active_alpha_config.json, shadow_configs/
dashboard/                  # Generated HTML dashboards, regime_signal.json, regime_labels.csv,
                            # agent_skill_scores.json
outputs/
  20in20/                   # Intel memos, IC briefs, scenario JSONs, HTML reports
  debate_log/               # verdict_YYYY-MM-DD.json, agent_credibility.json
logs/                       # eod_log.jsonl, slippage_log.jsonl, self_improve_log.jsonl,
                            # skill_scores_log.jsonl, multi_agent_run.jsonl, exit_alerts.jsonl
                            # snapshots/{agent_id}_weights_YYYY-MM-DD.json

ascent/main.py              # Core pipeline entrypoint (called by agents)
run_all_agents.py           # Single daily command — branches on rebalance day
run_20in20.py               # Intel/20in20 reporting runner
```

---

## Core runtime flow

**Single command**: `python3 run_all_agents.py`

Behavior branches on whether today is in `rebalance_calendar.csv`:

**Non-rebalance day:**

1. All four specialist agents run **in parallel** (ThreadPoolExecutor, error isolation per agent)
2. Forward PnL cycle — score yesterday's weights against today's returns (single batched yfinance call for all agents)
3. Skill scores updated (63-day rolling Sharpe per agent)
4. Orchestrator merges outputs, applies skill+conviction+regime capital allocation
5. Merged weights written to `execution/merged_weights.json`
6. Run logged to `logs/multi_agent_run.jsonl` — **stop here, no debate, no execution**

**Rebalance day:**
1–4. Same as above, plus pre-rebalance checklist (blocks if failing)
5. Merged weights written
6. Debate layer runs — bull / bear / devil's advocate / regime specialist / quant sanity → judge verdict
7. Verdict gates execution: `proceed` → execute, `reduce_size` → Haiku adjusts weights then execute, `halt_and_review` → log and stop
8. Orders submitted to Alpaca via `eod_runner.py`
9. Post-fill: slippage tracked, run logged

The `ascent/main.py` pipeline flow (called by each agent):
data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export

---

## Alpha stack

Default sleeve weights (`ascent/alpha/stack.py`, `DEFAULT_ALPHA_WEIGHTS`):


| Sleeve         | Weight | Notes                                                                           |
| -------------- | ------ | ------------------------------------------------------------------------------- |
| Trend          | 70%    | Cross-sectional momentum, primary driver                                        |
| Stat-arb       | 15%    | Sector residuals; requires profiles.parquet for sector-awareness                |
| Mean reversion | 5%     | Short-term reversal                                                             |
| ML (XGBoost)   | 10%    | CPCV evaluation (Phase 3a); cached to `data_cache/ml_model_{agent_id}.pkl`      |
| Volatility     | 0%     | Provisioned, disabled                                                           |


Regime adjusts sleeve weights before combining (via `integration.py:regime_adjust_sleeve_weights()`). Stack renormalizes across only the successfully-loaded sleeves. Output is cross-sectional alpha clipped and z-scored per date.

Each agent gets its own ML model cache keyed by `agent_id` — macro agent's model never contaminates equities.

**ML sleeve evaluation (Phase 3a)**: CPCV with C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays. Two reliability guards: disabled if <10/15 folds converge, or if p5 IC Sharpe < 0. Final production model trained on 80% of dates after OOS validation passes. Implemented in `ascent/research/cpcv.py` + `build_ml_alpha_cpcv()` in `ml_sleeve.py`.

**Consistent normalization**: Every sleeve is cross-sectionally z-scored (`_cs_normalize`) before blending. Previously only the volatility sleeve was normalized; trend and meanrev were in percentile rank space (0–1) while statarb was already z-scored, meaning the weighted blend was not a true weighted average. All sleeves now blend at the same scale.

---

## Portfolio construction

`sector_constrained_weighted()` is the live constructor:

- Sector coverage check: if < 80% of candidates have known sectors, skip sector caps and log warning
- Walks ranked alpha list, limits `max_per_sector=1`
- Rank-weights selected names (score = alpha - min + ε)
- Weight cap via `_water_fill_cap()`: iteratively freeze capped names, redistribute to uncapped — convergence in ≤50 iterations
- Final hard clamp + renorm
- Post-condition: weights sum to 1.0 ± tolerance, no position > max_weight

Regime tightens max_weight via `regime_max_weight()`: crisis → 0.08, calm_bull → 0.15.

SPY 200MA overlay: when SPY < 200-day MA, multiply all target weights by 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` business days.

---

## Agents

### US Equities

- **Universe**: 80+ US large/mid cap stocks
- **Alpha**: Full pipeline (trend 70%, statarb 15%, meanrev 5%, ML 10%)
- **Kill switch**: Handled by `eod_runner.py` (not agent-level)
- **Typical output**: 12–20 positions, 2–10% each

### Macro

- **Universe**: TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM (12 instruments)
- **Alpha**: Trend-only (no ML, no statarb — universe too small); top N of 12
- **Regime-aware sizing**: crisis → top_n=3, max_weight=40%; stressed → top_n=4, max_weight=35%; calm_bull/neutral → top_n=5, max_weight=30%. Regime signal computed before weight construction so sizing responds to market state.
- **Weight bounds**: min 5%
- **Regime signal**: GLD 200-day MA (> 1.05×MA = calm_bull, < 0.95×MA = stressed, else neutral)
- **Cache**: `prices_macro.parquet`

### International

- **Universe**: EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA (12 instruments)
- **Region map**: broad_em (EEM/VWO), asia (EWT/AAXJ/EWJ/EWY/INDA), latam (EWZ/EWC), europe (EWG/EWU), developed (EFA). Max 2 per region.
- **USD penalty**: UUP above 50-day MA → 20% alpha reduction on all EM names (EEM, VWO, EWT, AAXJ, EWZ, EWC, EWY, INDA) to limit dollar-risk exposure
- **Weight cap**: `_water_fill_cap()` (iterative, same as alternatives and equities)
- **Loaded conditionally** (ImportError safe)

### Alternatives

- **Universe**: VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL (7 instruments — SVXY removed)
- **SVXY removed**: long vol (VIXY) and short vol (SVXY) in the same universe allowed self-contradiction before the orchestrator's thesis coherence check. The alternatives sleeve is defensive — short vol has no place here.
- **Alpha**: Trend (80%) + low-vol preference (20%)
- **Kill switch**: 12% drawdown halt (tighter than portfolio-level 15%)
- **Weight bounds**: max 35%, min 5%; top 4
- **Loaded conditionally** (ImportError safe)

---

## Orchestrator (`orchestrator/central_intelligence.py`)

### Capital allocation formula

**Step 1 — Base allocation by regime:**

```
calm_bull:  US 60%, macro 15%, intl 15%, alt 10%
stressed:   US 45%, macro 25%, intl 10%, alt 20%
crisis:     US 30%, macro 30%, intl 5%,  alt 35%
```

**Step 2 — Skill-weighted blend (per-agent, not all-or-nothing):**

- Each agent is handled independently — a warming-up agent does not freeze skill allocation for others
- Agents without skill data → use base allocation directly
- Agents with negative `skill_score` → zero allocation
- Agents with positive skill score: `skill_share = agent_score / sum(active_scores)`, blended 50% skill + 50% base

**Step 3 — Conviction bonus (up to 15%):**

- Conviction = fraction of an agent's top holdings (>2% weight) confirmed by ≥1 other agent
- Agents with `conv > 0.3` receive bonus; low-conv agents subsidize
- Formula: `bonus = (conv / total_conv) * CONVICTION_BONUS_MAX` (linear, not squared)
- `CONVICTION_BONUS_MAX = 0.15`

**Step 4 — Correlation guard:**

- 63-day trailing correlation cap at 0.70 across agents
- Violating cross-agent pairs: halve the smaller weight

**Step 5 — Thesis coherence (two layers):**

- Symbol-level: contradictory pairs (e.g., UUP ↔ PDBC/GLD, VIXY ↔ SVXY, TLT ↔ HYG) with >4% weight on both sides → halve smaller
- Factor-level: 12 factor buckets (rates_long, dollar_long, commodities, gold, vol_long/short, em_equity, us_tech, us_defensive, reits, energy, etc.); 6 contradictory factor pairs; >8% agent weight + >6% merged conflict → 40% reduction on smaller factor. Note: GLD is in the `gold` bucket only — not `commodities` — to avoid double-penalization.

**Step 6 — Crisis veto:**

- Trigger: `us_regime == "crisis"`
- Effect: `merged = 0.60 × macro_weights + 0.40 × merged_weights`

---

## Walk-forward runner

`ascent/research/walk_forward_runner.py` exists but is **not exposed as a runnable mode**. It was removed from `ascent/main.py` because a proper WF OOS test of the multi-agent system (four agents + orchestrator + debate + self-improvement) is not feasible as a single-pipeline backtest — the test would only ever cover the single US equities agent and produce misleading results.

The file is retained because `self_improve.py` Phase D will eventually call `walk_forward_pipeline()` as the real evaluator for variant configs. Until then it is unused.

`ascent/main.py` now runs the full pipeline only: `python3 -m ascent.main [--live] [--start DATE] [--end DATE]`

---

## Regime system

HMM with `n_candidates = [2, 3, 4]` — best K chosen by walk-forward CV on OOS likelihood.

**Labels** (`RegimeLabel` enum): `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`

**Decision layer (hysteresis):**

- Enter threshold: 0.55, Exit threshold: 0.35
- Min dwell: 3 days
- Entropy > 0.90 → mark as `uncertain` regardless of best label

**Regime effects downstream:**

- Alpha sleeve weights adjusted (`integration.py:regime_adjust_sleeve_weights()`)
- `max_weight` overrides tightened per regime (crisis → 0.08, calm_bull → 0.15)
- Orchestrator base allocation shifts (see above)
- Crisis veto activates
- Debate agents receive regime context

**Risk multipliers by regime**: calm_bull 1.0, euphoric 0.85, stressed 0.65, crisis 0.40

Signal exported to `dashboard/regime_signal.json` and `dashboard/regime_labels.csv`.

---

## Debate layer

Runs on rebalance days only. Advisory — never writes to alpha, portfolio, or execution modules.

**Sequence (debate_runner.py):**

1. Score pending verdicts (14-day window) → update `agent_credibility.json`
2. Run debriefs for past verdicts → extract lessons
3. Detect blind spots across all past verdicts (via Haiku)
4. Inject blind spot context into portfolio_state
5. Run Monte Carlo scenario simulation → p5/p50/p95 for worst scenarios
6. Run debate agents in sequence (each failure caught and logged)
7. Judge synthesizes → verdict JSON

**Agents (debate/agents.py):**


| Agent             | Model                     | Role                                                                                |
| ----------------- | ------------------------- | ----------------------------------------------------------------------------------- |
| Bull              | claude-opus-4-6           | Strongest case for executing as-is                                                  |
| Bear              | claude-opus-4-6           | Case for reducing risk or waiting                                                   |
| Devil's Advocate  | claude-opus-4-6           | Single most dangerous assumption; uses Monte Carlo numbers                          |
| Regime Specialist | claude-haiku-4-5-20251001 | Regime playbook argument (sizing, factor coverage, defensiveness)                   |
| Quant Sanity      | Pure Python               | Hard numerical checks (position limits, weight sum, sector concentration, turnover) |


**Quant sanity thresholds**: max single position 15%, max sector 40%, positions 5–25, weights sum ±2%, max turnover flag 50%.

**Verdict schema (judge.py):**

```json
{
  "confidence": 0.0–1.0,
  "recommendation": "proceed" | "reduce_size" | "halt_and_review",
  "key_risks": ["..."],
  "reasoning": "..."
}
```

Judge uses claude-opus-4-6, temp 0.3. Defaults to `reduce_size` on parse failure (safe fallback).

**On `reduce_size`**: claude-haiku-4-5-20251001 reads verdict reasoning + key_risks and outputs adjusted `{symbol: weight}` JSON. Validates sum=1.0, max_weight ≤ 15%, no negatives.

Verdicts written to `outputs/debate_log/verdict_YYYY-MM-DD.json`. Credibility per agent per regime in `outputs/debate_log/agent_credibility.json`.

**Credibility wiring**: `load_credibility_context(regime)` and `load_recent_verdict_outcomes(n=5)` from `debate/outcome_tracker.py` are now injected into every agent's user prompt and the judge's synthesis context. This completes the learning loop — past performance of each debater (per regime) influences how the judge weighs future arguments. The loop was previously wired to track outcomes but the credibility data was never fed back into prompts.

---

## Self-improve loop (`ascent/research/self_improve.py`)

Runs weekly (Sunday 6 AM via launchd).

**Current status: Phase B (lightweight heuristic evaluator)**

- Generates N=5 variant configs by perturbing sleeve weights ±10%, renormalized
- Evaluates via heuristic: `base_sharpe (0.518) + diversity_bonus + noise`
  - Diversity bonus: +0.02 if deviation 5–25%, −0.05 if >25%
  - `0.518` is the Phase 5.1 baseline — stale since WF OOS was removed. Phase D TODO: replace with live forward PnL Sharpe from `skill_tracker`
- **Phase D TODO**: Replace heuristic with real walk-forward call (`walk_forward_pipeline(alpha_weights=...)`)

**Shadow promotion logic:**

1. If `best_variant_sharpe - current_sharpe > MIN_SHARPE_EDGE (0.10)` → promote to shadow
2. Shadow monitored for 30 days (expiry = `promoted_date + timedelta(days=30)`)
3. If real OOS Sharpe beats current by >0.10 after 30 days → promote to live
4. Full strategy logic changes require human review before shadow — never auto-promoted

**State files:**

- `data_cache/active_alpha_config.json` — current live weights
- `data_cache/shadow_configs/` — pending 30-day shadows
- `logs/self_improve_log.jsonl` — weekly variant evaluations

---

## Execution (`ascent/execution/eod_runner.py`)

- **Kill switch**: SOFT_WARN 8% drawdown (log, proceed), HARD_STOP 15% (raise KillSwitchTriggered, abort)
- **Alternatives agent kill switch**: 12% (tighter, in `alternatives_agent.py`)
- Kill switch state persisted in `logs/kill_switch_state.json` (survives restarts)
- **Large trade approval**: trades > 2% NAV written to `execution/pending_approvals.json`, poll every 30s, timeout 30 min
- **Slippage tracking**: wait 30s post-fill, compute fill vs signal close prices, write `logs/slippage_log.jsonl`

Config must be loaded via `get_config()`, not direct `Config()` construction.

---

## Data and caching

**Cache names:**

- `prices_live` — true live fetch from Yahoo/Polygon
- `prices_simulated` — synthetic GBM fallback
- `prices_live_fallback_simulated` — simulated fallback used when live fetch fails (name makes provenance explicit)
- `prices_macro` — macro universe (TLT, UUP, etc.)
- `macro_live` / `macro_simulated` — FRED economic data or fallback
- `profiles` — sector/industry metadata (source of truth for sector constraints)

If live fetch fails, write fallback under `prices_live_fallback_simulated` and log clearly. Never let cache name hide data provenance.

Point-in-time joins via `as_of_join()` / `as_of_merge()` — use these for any cross-dataset alignment to preserve causality.

---

## AgentOutput interface (`ascent/config/types.py`)

```python
@dataclass
class AgentOutput:
    agent_id: str
    as_of_date: date
    target_weights: Dict[str, float]    # {symbol: weight}
    regime_signal: Optional[str]        # RegimeLabel string
    alpha_scores: Optional[pd.DataFrame]
    skill_score: Optional[float]        # rolling 63-day OOS Sharpe
    metadata: Dict[str, Any]
```

Orchestrator reads a list of `AgentOutput` objects. Do not pass raw pipeline tuples.

---

## Monitoring

- **Skill tracker**: 63-day rolling Sharpe per agent from PnL logs → `dashboard/agent_skill_scores.json`; status: `active` (≥10 days), `warming_up`, `insufficient_data`
- **Forward PnL tracker**: loads yesterday's weight snapshot, computes today's return, updates agent NAV (base 100,000), saves today's snapshot for tomorrow. All agents' symbols are batch-fetched in a single yfinance call. PnL log schema is unified: `date`, `nav`, `return`, `agent_id` for all agents (legacy `portfolio_value` key still readable).
- **Pre-rebalance checklist**: validates agent outputs vs held positions; blocks execution if `passed=False`
- **Exit alerts**: monitors intraday drawdowns on held positions → `logs/exit_alerts.jsonl`

---

## LLM clients and models

Centralized in `ascent/llm/client.py`:

- `DEFAULT_MODEL = "claude-opus-4-6"` — debate bull/bear/devil/judge
- `HAIKU_MODEL = "claude-haiku-4-5-20251001"` — regime specialist, weight adjustment after `reduce_size`, blind spot detection, skill score summarization
- Anthropic client is a **lazy singleton** — created once on first call, reused for all subsequent calls
- All LLM calls **retry 3× with exponential backoff** (2s, 4s) before raising
- `HAIKU_MODEL` is defined only here — all other files (`debate/agents.py`, `eod_runner.py`, `blind_spot_detector.py`, `regime_narrative.py`, `debrief.py`, `ic_brief_generator.py`, `exit_alerts.py`) import it from this module. Never re-define it locally.

API keys loaded from `.env` via `APIKeys.from_env()`. Never hardcode secrets.

---

## Known integrity constraints — never violate

1. **No look-ahead bias.** Walk-forward runner must use `get_universe_on_date()` on every fold. Regime engine fitted on training slice only (full-sample engine ignored). ML targets not leaked into feature windows.
2. **No simulated data under live cache names.** If live fetch fails, write to `prices_live_fallback_simulated`.
3. **Max-weight hard cap.** Use `_water_fill_cap()` (iterative). Final post-condition check before returning weights.
4. **Sector constraint fallback.** If sector coverage < 80%, skip caps and log warning. Never collapse portfolio to single name.
5. **Walk-forward runner is not a production entrypoint.** `walk_forward_runner.py` is retained for self_improve Phase D only — do not re-expose it as a runnable mode from `ascent/main.py`.
6. **Debate is advisory only.** Debate layer reads Ascent outputs but never writes to alpha, portfolio, or execution modules directly.
7. **Approval layer for large trades.** Any order > 2% NAV must go through `pending_approvals.json` before Alpaca submission.

---

## Debugging protocol

One step at a time. Targeted shell commands. Minimal output. Verify existing logic before proposing any fix. Use `ast.parse` verification after each patch. Never propose a fix without tracing the failure first.

For planning work: use Claude Opus to produce full execution specs, save to project files, then use Sonnet for implementation.

---

## Environment and dev setup

Python 3.12.13 via Homebrew (`/opt/homebrew/bin/python3.12`), venv at `.venv/`. Run all commands with `.venv/bin/python`, not `python3`.

Post-JAMF (Mac Air M4): standard pip install, launchd via `.plist`, GitHub push/pull as sync layer between Air (dev) and Mini (production).

API keys: load via `APIKeys.from_env()` only. No hardcoded secrets in `settings.py` defaults.

---

## Current portfolio (as of April 2026)

Holdings: APD, CAT, CB, EQIX, MPC, MRK, NEE, T, WMT
Next rebalance: April 15, 2026
Live since: April 1, 2026 (Alpaca paper trading)

---

## What is not built yet

- **Self-improve real evaluator (Phase D)**: `self_improve.py` uses a lightweight heuristic. The TODO in the file marks where to swap in a real `walk_forward_pipeline()` call. Until then, shadow promotions are noisy.
- **No WF OOS for the multi-agent system**: A proper out-of-sample test of the full platform (4 agents + orchestrator + debate + self-improvement) is not feasible as a single backtest. Live forward PnL tracking via `forward_pnl_tracker.py` is the real evaluation layer.
- **Zenith / Hermes voice interface**: `zenith/` directory not present in Phase 5.1. Planned for Phase C/D.
- **Live dashboard rendering**: `dashboard/` generates data files but no live UI is wired up.
- **Phase 4 hedge overlay**: blocked until Phases 1-3 live 30+ days (~May 13, 2026).
- **R2R semantic memory**: `memory/r2r_interface.py` is built with R2R API path, but `R2R_API_KEY` not yet configured. Local BM25 fallback is active. Set `R2R_API_KEY` in `.env` and run `scripts/ingest_verdict_history.py` to bootstrap vector memory.

---

## Instructions for Claude (read this every session)

At the end of every session, append a new entry to the Session log below. Format:

```
### YYYY-MM-DD
- [what was built, fixed, or decided — one line per thing]
- [files touched]
- [anything left open or deferred]
```

Keep entries terse. This is a changelog, not prose. If a session changes the system state in a way that makes any section above stale, update that section too before closing.

---

## Session log

### 2026-04-09

- Drafted initial CLAUDE.md covering full architecture, integrity constraints, debug protocol, and dev setup
- Fixed 6 bugs: verdict NameError crash (run_all_agents.py), max_weight cap broken in alternatives_agent, posture computed with empty probs (eod_runner.py), conviction bonus squared in orchestrator, weight tolerance too loose before live orders, max-weight guard off by 1%
- A4 complete: survivorship bias hardening in walk_forward_runner.py — per-fold universe filtering now applied at price filter, FeatureBuilder, alpha computation, regime univ_train, and weight output; thin-universe skip at <5 symbols with warning; end-of-run fold summary (total/succeeded/skipped/failed/avg universe size)
- Redesigned run_all_agents.py: single command, branches on rebalance day — non-rebalance runs agents+orchestrator only (no debate, no execution); rebalance day runs full pipeline including debate and eod_runner; extracted _log_run() helper
- Full codebase audit completed; CLAUDE.md rewritten to reflect accurate constants, agent details, orchestrator formula, debate agent models, self-improve status, and all known gaps
- Files: CLAUDE.md, run_all_agents.py, agents/alternatives_agent.py, ascent/execution/eod_runner.py, orchestrator/central_intelligence.py, ascent/research/walk_forward_runner.py
- Open: self-improve real evaluator (Phase D), Zenith voice interface

### 2026-04-09 (continued)

- Set up environment: Homebrew, Python 3.12.13 venv, all deps installed
- Fixed walk-forward annualisation bug: engine was intersecting rebalance dates with close_prices, simulating only 164 days instead of 1,631 — fixed by forward-filling weights to every trading day before passing to BacktestEngine; real metrics: CAGR 24.5%, Sharpe 1.95
- Identified 6 problems with WF OOS: in-sample IC, missing SPY 200MA overlay, single-agent only (not multi-agent), ML sleeve always skipped, favorable evaluation period, regime model selection disabled
- Removed walk-forward mode from ascent/main.py entirely — not meaningful for a multi-agent system with debate and self-learning; walk_forward_runner.py retained for self_improve Phase D
- Files: ascent/main.py, ascent/research/walk_forward_runner.py, CLAUDE.md

### 2026-04-09 (fixes)

- Parallel agents: replaced sequential agent calls with ThreadPoolExecutor in run_all_agents.py (~4x daily run speedup)
- Fixed shadow expiry bug: was clamping to day 28 of current month instead of adding 30 days — replaced with timedelta(days=30)
- Fixed orchestrator skill allocation freeze: `all(...)` check was blocking skill weighting when any one agent was warming up — replaced with per-agent logic
- Fixed GLD double-penalization: removed GLD from `commodities` FACTOR_BUCKET (it belongs in `gold` only); was triggering two separate factor contradiction reductions
- LLM client: Anthropic client is now a lazy singleton (not re-created per call); all calls retry 3× with 2s/4s exponential backoff
- Centralized HAIKU_MODEL in ascent/llm/client.py; removed local definitions from 7 files (debate/agents.py, eod_runner.py, blind_spot_detector.py, regime_narrative.py, debrief.py, ic_brief_generator.py, exit_alerts.py)
- Forward PnL tracker: batched all agents' yfinance fetches into one call; unified PnL log key names across all agents (legacy key still readable)
- Lazy-loaded SECTOR_MAP in debate/agents.py — no longer reads profiles.parquet at import time
- Fixed judge.py docstring: said "proceed" on parse failure, actually defaults to "reduce_size"
- Documented CURRENT_OOS_SHARPE (0.518) as stale with Phase D TODO comment
- Files: run_all_agents.py, ascent/llm/client.py, ascent/research/self_improve.py, orchestrator/central_intelligence.py, ascent/monitoring/forward_pnl_tracker.py, debate/agents.py, debate/judge.py, ascent/execution/eod_runner.py, ascent/reporting/blind_spot_detector.py, ascent/reporting/regime_narrative.py, ascent/reporting/debrief.py, ascent/reporting/ic_brief_generator.py, ascent/monitoring/exit_alerts.py, CLAUDE.md
- Open: self-improve Phase D real evaluator, Zenith voice interface, live dashboard UI

### 2026-04-12

- Full codebase audit: verified CLAUDE.md accuracy against all key source files (settings.py, alpha stack, regime, agents, orchestrator, llm/client.py, debate)
- Corrected LLM retry wait: documented as "2s/4s" but actual code is `2**attempt` → 1s, 2s
- Identified 8 architectural weaknesses: skill score lag, sector constraint silent failure, debate halt non-persistent, blocking approval gate (crash loses state), naive transaction cost model, ML sleeve in-sample bias (CPCV needed), regime engine no online updates (batch-only), no hedge overlay
- Produced full remediation design across 4 phases with two review passes
- Wrote spec: `docs/superpowers/specs/2026-04-11-system-hardening-design.md`
- Files: CLAUDE.md (this entry), docs/superpowers/specs/2026-04-11-system-hardening-design.md (new)
- Open: implement all 8 fixes — do not start until self-improve/self-learning system design is finalized; spec is ready at the path above

### 2026-04-12 (Phase 1 hardening)

- Phase 1 system hardening complete (3 fixes, 15 new tests, all passing)
- Fix 1a: `skill_tracker.py` now writes `skill_score_as_of` date; orchestrator rejects scores >1 day stale, falls back to base allocation; sequential PnL→skill→orchestrator dependency commented
- Fix 1b: `SectorDataError` exception replaces silent sector fallback in `optimizer.py`; `validate_sector_data()` startup check added to `run_all_agents.py` — aborts before agents spawn if `profiles.parquet` missing or coverage <80%; `--skip-sector-check` flag with audit log
- Fix 1c: `debate_runner.py` writes `execution/halt_state.json` on `halt_and_review`; `check_halt_state()` in `run_all_agents.py` gates rebalance execution — persists across restarts, requires `execution/halt_override.json` to resume
- Files: `ascent/monitoring/skill_tracker.py`, `orchestrator/central_intelligence.py`, `run_all_agents.py`, `ascent/portfolio/optimizer.py`, `debate/debate_runner.py`, `tests/test_phase1_hardening.py`
- Branch: `feature/phase1-hardening` (merged to main this session)
- Open: Phase 2 (async approval gate, Almgren-Chriss cost model)

### 2026-04-12 (AI agent enhancements — plans only)

- Merged `feature/phase1-hardening` to main; resolved merge conflict in `run_all_agents.py` and stash conflict in `optimizer.py` (kept `SectorDataError` + `_normalize_sector` coverage logic)
- Designed three new AI agent features; wrote full TDD implementation plans, no code written yet
- Plan 1 — Catalyst Scanner: `docs/superpowers/plans/2026-04-12-catalyst-scanner.md` — scans earnings/ex-div/FOMC for held positions via yfinance, injects into `portfolio_state["catalyst_context"]` before debate agents run; new file `ascent/reporting/catalyst_scanner.py`
- Plan 2 — Multi-turn Debate: `docs/superpowers/plans/2026-04-12-multi-turn-debate.md` — adds Round 2 rebuttals (agents respond to each other), extends `run_judge()` with `round2_args`; modifies `debate/agents.py`, `debate/judge.py`, `debate/debate_runner.py`
- Plan 3 — Memory-augmented Debate: `docs/superpowers/plans/2026-04-12-memory-augmented-debate.md` — builds `memory/r2r_interface.py` with R2R HTTP path + local BM25 fallback; queries past verdicts before each debate session, injects into `portfolio_state["memory_context"]`; bootstrap script at `scripts/ingest_verdict_history.py`
- Files: CLAUDE.md, docs/superpowers/plans/2026-04-12-catalyst-scanner.md, docs/superpowers/plans/2026-04-12-multi-turn-debate.md, docs/superpowers/plans/2026-04-12-memory-augmented-debate.md
- Open: implement the three plans (subagent-driven recommended); Phase 2 hardening (async approval gate, Almgren-Chriss) also pending

### 2026-04-12 (Phase 2 hardening)

- Fix 2a: async approval gate — replaced blocking `while time.sleep(30)` loop in both `run_eod()` and `run_eod_with_weights()` with `threading.Event`-based `wait_for_approval_async()`; writes `execution/approval_pending.json` before blocking; resume check at top of `run_eod_with_weights()` detects unexpired state on restart and submits persisted trades directly (no duplicate recompute)
- Fix 2b: Almgren-Chriss cost model — new `ascent/execution/cost_model.py`; `estimate()` computes spread + temp + permanent impact; `apply_cost_filter()` blocks HIGH_IMPACT orders (>10% ADV), warns on SPLIT_RECOMMENDED (>5%) and IMPACT_UNKNOWN; `compute_orders()` gains optional `features=` param; `extract_cost_features()` bridge converts FeatureBuilder DataFrame format; `diff_df` marks blocked orders as `action="blocked_high_impact"`
- 20 new tests in `tests/test_phase2_hardening.py`, all 35 tests passing
- Files: `ascent/execution/approval_server.py`, `ascent/execution/eod_runner.py`, `ascent/execution/order_engine.py`, `ascent/execution/cost_model.py` (new), `tests/test_phase2_hardening.py` (new)
- Open: three AI agent plans (catalyst scanner, multi-turn debate, memory-augmented debate); wire `features=` into live `run_eod_with_weights()` call site (currently `features=None` in production — cost filtering inactive until wired)

### 2026-04-13 (Phase 3 hardening)

- Non-rebalance day run completed successfully before coding session: all 4 agents live, regime=stressed, forward PnL logged, merged weights written
- Fix 3a: CPCV for ML sleeve — new `ascent/research/cpcv.py` (`CPCVSplitter` with C(6,2)=15 purged folds, purge=5 bdays, embargo=5 bdays); new `build_ml_alpha_cpcv()` in `ml_sleeve.py` with two guards (n_converged<10 → disabled, p5 Sharpe<0 → disabled); `stack.py` now calls `build_ml_alpha_cpcv` instead of 80/20 split
- Fix 3b: Regime particle filter + emergency refit — new `ascent/regime/particle_filter.py` (`RegimeParticleFilter` SIR algorithm, 500 particles, reinitializes on every batch refit); `check_emergency_refit_triggers()` in `engine.py` (4 triggers: SPY -3%+VIX>30, 200MA cross, SPY/TLT corr flip, break z-score>3.5); `BreakDetector.latest_zscore()` added to `breaks.py`; `regime_refit_every_days` changed 63→5 in `types.py`; `engine.py` gains `check_and_run_emergency_refit()` and `update_particle_filter()` methods
- 23 new tests in `tests/test_phase3_hardening.py`, all 58 total tests passing
- Files: `ascent/research/cpcv.py` (new), `ascent/alpha/ml_sleeve.py`, `ascent/alpha/stack.py`, `ascent/regime/particle_filter.py` (new), `ascent/regime/engine.py`, `ascent/regime/breaks.py`, `ascent/regime/types.py`, `tests/test_phase3_hardening.py` (new)
- Wired cost model: loaded `dollar_volume` from `data_cache/prices_live.parquet`, pivoted to dates×symbols, passed through `extract_cost_features()` into `compute_orders(features=...)` — cost filtering now active on live orders
- Open: three AI agent plans (catalyst scanner, multi-turn debate, memory-augmented debate); Phase 4 (hedge overlay) blocked until Phases 1-3 live 30+ days

### 2026-04-13 (universe cleanup)

- Removed 15 delisted symbols that returned no yfinance data: TIF, FLIR, XLNX, PBCT, CERN, CTXS, TWTR, DRE, FBHS, SIVB, FRC, ATVI, DISH, PARA, WBA
- Added 15 new symbols chosen to fill sector gaps and improve factor signal quality:
  - Financial data/analytics (previously zero coverage): MSCI, MCO, SPGI
  - Cybersecurity (PANW was only name): CRWD
  - Medical devices (ISRG was only name): SYK
  - Momentum signals previously missing: APP (AppLovin), CMG (Chipotle), TMUS (T-Mobile)
  - Quality industrials with pricing power: ITW (Illinois Tool Works), TDG (TransDigm)
  - Healthcare services (no hospital exposure before): HCA
  - Semiconductor equipment (different cycle from chip designers): KLAC
  - Insurance brokerage (defensive, fee-based): AJG
  - Water utility (most defensive utility class): AWK
  - Healthcare REIT (different driver from PLD/EQIX/SPG): WELL
- Universe: 135 symbols, no duplicates
- Files: `ascent/config/settings.py`

### 2026-04-13 (AI agent features — all three implemented)

- Catalyst scanner: new `ascent/reporting/catalyst_scanner.py` — scans earnings/ex-div/FOMC within 21-day window; injected into `portfolio_state["catalyst_context"]`; all debate agents see upcoming binary events in `_build_context()`; 12 tests
- Multi-turn debate: Round 2 rebuttal functions added to `debate/agents.py` (run_bull/bear/devils_advocate/regime_specialist_rebuttal); `run_judge()` extended with `round2_args` param (backward compatible); `debate_runner.py` runs Round 2 after Round 1, passes both to judge; verdict record stores all Round 2 args; 10 tests
- Memory-augmented debate: new `memory/__init__.py` + `memory/r2r_interface.py` — R2R HTTP API with local BM25 keyword fallback over `outputs/debate_log/*.json`; `query_memory()` called before agents run; `ingest_verdict()` called after verdict written; `_build_context()` includes memory context; bootstrap script `scripts/ingest_verdict_history.py`; 13 tests
- 93 total tests passing (35 new this session)
- Files: `ascent/reporting/catalyst_scanner.py` (new), `memory/__init__.py` (new), `memory/r2r_interface.py` (new), `scripts/ingest_verdict_history.py` (new), `debate/agents.py`, `debate/judge.py`, `debate/debate_runner.py`, `tests/test_catalyst_scanner.py` (new), `tests/test_multi_turn_debate.py` (new), `tests/test_memory_interface.py` (new)
- Open: Phase 4 hedge overlay (blocked until Phases 1-3 live 30+ days — earliest ~May 13); self-improve Phase D real evaluator

### 2026-04-14 (Tony Ngo demo — planned, not yet built)

- Context: Tony Ngo (Stanford, Morgan Stanley, Bridger Capital, 20in20 Partners co-founder) met Scott Apr 11 to discuss college list + Ascent Capital; wants to *interact* with the system, not watch a demo; no follow-up sent yet
- Decision: build interactive Streamlit demo (`demo_app.py`) as follow-up artifact — this IS the follow-up
- Planned architecture for `demo_app.py`:
  - Sidebar: regime picker, VIX slider, SPY momentum, portfolio preset, neuroplasticity toggle (ON/OFF), run button
  - Main panel: portfolio snapshot, neuroplasticity card (sleeve weights before/after regime adjustment with real numbers), debate transcript (5 agents + Round 2 rebuttals appearing progressively), judge verdict card
  - Real LLM calls using actual `debate/agents.py` functions with synthetic `portfolio_state` built from UI inputs
  - Demo mode fallback (scenario-aware pre-written arguments if API unavailable)
  - Deploy: `streamlit run demo_app.py` locally; push to Streamlit Community Cloud for shareable link
  - Dark/gold aesthetic matching Ascent Capital brand
- R2R semantic memory: decision to skip for now — only 4 verdicts exist; revisit mid-May when 30+ verdicts accumulated
- Files: nothing written yet — full build is next session
- Open: build `demo_app.py` and send link to Tony

