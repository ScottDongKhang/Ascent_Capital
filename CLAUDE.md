# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio construction → walk-forward evaluation → regime modeling → 4 specialist agents → orchestration → LLM debate → execution via Alpaca paper trading.

---

## Repository layout

```
ascent/
  config/       settings.py (Config, APIKeys, UniverseConfig, BacktestConfig),
                types.py (AgentOutput), us_equity_universe.json
  data/
    ingest/     yahoo, fred, simulated, fundamentals, earnings,
                edgar_listener, capitol_trades, options_scanner,
                sec_filings, earnings_transcripts, reddit_sentiment, google_trends
    store/      parquet_store, point_in_time, timescale.py, schema.sql
    streaming/  alpaca_stream.py (Alpaca WebSocket IEX, 901 symbols)
    validate/   altdata_validator.py (IC gate)
    universe.py, hub.py, normalize.py
  features/     build_features, feature_defs, targets
  alpha/        trend, meanrev, statarb, ml_sleeve, fundamental, earnings,
                llm_fundamental, event_alpha, altdata_alpha, stack
                [analyst, options_flow, insider, short_interest — sparse/zero-filled]
  portfolio/    optimizer, mvo_optimizer, black_litterman, regime_covariance, hedge_overlay
  backtest/     engine, costs
  research/     walk_forward_runner, walk_forward_lightweight, cpcv,
                self_improve, shadow_promoter, factor_proposer,
                factor_discovery/ (feature_templates, leakage_scanner,
                  regime_cpcv_evaluator, llm_suggester, pysr_engine, discovery_runner)
  regime/       engine, model, features, decision, integration, posture,
                breaks, particle_filter, types
  risk/         factor_data, factor_model, covariance_model,
                factor_exposure, factor_constraints
  reporting/    market_memo, ic_brief_generator, blind_spot_detector,
                catalyst_scanner, debrief, regime_narrative, investor_report
  execution/    eod_runner, alpaca_broker, order_engine, kill_switch,
                cost_model, slippage_tracker, approval_server,
                twap_executor, implementation_shortfall, capacity_model,
                intraday_trigger, event_runner, debate_gate
  monitoring/   skill_tracker, forward_pnl_tracker, attribution,
                slippage_ic_feedback, pre_rebalance_checklist,
                counterfactual_tracker, live_nav, alert_system
  llm/          client.py — centralized Anthropic API wrapper
  dashboard/    export_dashboard_data, live_dashboard.py (port 8502)

agents/         us_equities, macro, international, alternatives, event_agent
orchestrator/   central_intelligence.py
debate/         debate_runner, agents, judge, outcome_tracker,
                disagreement_scorer, agent_tools
memory/         r2r_interface (R2R HTTP + BM25 fallback), reflection_agent
simulation/     mirofish_interface
compliance/     audit_trail, performance_report, risk_disclosure, methodology_index
docs/           methodology.md, risk_disclosures.md, superpowers/plans/
scripts/        setup_timescaledb.sh, verify_audit_trail.py, evaluate_hedge.py

data_cache/     prices_live, macro_live, profiles, ml_model_*.pkl,
                active_alpha_config.json, shadow_configs/,
                factor_returns.parquet, factor_loadings.parquet
dashboard/      HTML dashboards, regime_signal.json, regime_labels.csv,
                agent_skill_scores.json, factor_exposures.json, methodology_index.json
outputs/
  debate_log/         verdict_YYYY-MM-DD.json, agent_credibility.json
  investor_reports/   YYYY-MM.pdf monthly PDF
  factor_proposals/   autonomous factor proposals (human review)
  altdata_proposals/  validated alt-data proposals (human review)
logs/           eod_log, slippage_log, self_improve_log, skill_scores_log,
                multi_agent_run, post_debate_portfolio, attribution_log,
                event_trades, capacity_log, intraday_adjustments,
                audit_trail.jsonl, alerts.jsonl,
                snapshots/{agent_id}_weights_YYYY-MM-DD.json

ascent/main.py        core pipeline entrypoint
run_all_agents.py     single daily command — branches on rebalance day
demo_app.py           Streamlit interactive demo (Tony Ngo)
```

---

## Core runtime flow

**Command**: `python3 run_all_agents.py`

**Non-rebalance day**: agents (parallel) → forward PnL → skill scores → orchestrator → write `merged_weights.json` → log. Stop.

**Rebalance day**: same + pre-rebalance checklist → debate → verdict gates execution → Alpaca orders → slippage tracking.

`ascent/main.py` pipeline: data → normalize → regime fit → features → alpha stack → sector-constrained weights → SPY 200MA overlay → backtest → export.

---

## Alpha stack

| Sleeve | Weight | Notes |
|--------|--------|-------|
| Trend | 41% | Cross-sectional momentum; skip-last-month variant (`mom_252d − mom_21d`) at 0.20 sub-weight |
| Stat-arb | 15% | Sector residuals; needs profiles.parquet |
| ML (XGBoost) | 10% | CPCV C(6,2)=15 folds, purge=5 bdays, embargo=5 bdays; 6 features by IC/IR; p5 guard > −0.05 |
| Mean reversion | 5% | Short-term reversal |
| Volatility | 5% | `−(vol_trend_10d / vol_of_vol_21d)`; long names with declining + stable vol |
| Fundamental | 5% | Gross profitability, accruals, asset growth; 45-day filing lag; momentum-neutral |
| Earnings (PEAD) | 5% | Cross-sectional z-score of EPS surprise; OLS momentum-beta residual; 1-bday lag |
| Analyst | 5% | Analyst revision signal; sparse — zero-filled if cache absent |
| LLM Fundamental | 3% | Chicago Booth 6-step CoT via Haiku; cached by (symbol, quarter_end); 45-day filing lag |
| Options Flow | 2% | Options sentiment; sparse — zero-filled if cache absent |
| Insider | 2% | Insider net transaction score; sparse — zero-filled if cache absent |
| Short Interest | 2% | Short squeeze signal; sparse — zero-filled if cache absent |
| Alt Data | 0% | IC-validated alternative data (SEC 10-K, transcripts, Reddit, Google Trends); 0% until first source passes IC gate |

Distressed filter: zeroes alpha for `mom_252d < −0.65` after blending. All sleeves cross-sectionally z-scored before blending. Weights are regime-adaptive via `data_cache/active_alpha_config.json` (updated weekly by self-improve loop). ML sleeve disabled if <10 folds converge or p5 IC Sharpe < −0.05.

---

## Portfolio construction

`sector_constrained_weighted_mvo()` is the primary path (Plan 2): sector pre-screening → Black-Litterman blending (quant prior + LLM views) → cvxpy MVO (objective: `w'α - λΣ - κ‖w-w_prev‖₁`, CLARABEL solver, SCS fallback) → rank-weight fallback if infeasible.

`sector_constrained_weighted()` (rank-weight fallback): coverage check (< 80% → skip sector caps + warn) → rank alpha → `max_per_sector=1` → `_water_fill_cap()` (iterative, ≤50 iterations) → hard clamp + renorm.

BL blending weight from IC IR: IR < 0.30 → tau=0.05, IR < 0.60 → tau=0.10, else tau=0.15.

Regime tightens max_weight: crisis → 0.08, calm_bull → 0.15. SPY 200MA overlay: SPY < 200MA → multiply weights × 0.70.

Config defaults: `top_n=15`, `max_weight=0.10`, `min_weight=0.02`, `rebalance_freq=10` bdays.

---

## Agents

**US Equities**: 901 symbols (S&P 500 + S&P 400, loaded from `ascent/config/us_equity_universe.json`); full alpha stack; max 1 per sector; 12–20 positions.

**Macro**: TLT, IEF, UUP, GLD, PDBC, HYG, LQD, TIP, SGOV, BIL, DBB, KMLM. Trend-only. Regime-sized: crisis top_n=3/40%, stressed top_n=4/35%, else top_n=5/30%. Cache: `prices_macro.parquet`.

**International**: EEM, VWO, EWT, AAXJ, EWJ, EWZ, EWC, EWY, INDA, EWG, EWU, EFA. Max 2 per region. UUP > 50MA → 20% alpha penalty on EM names.

**Alternatives**: VNQ, GLD, PDBC, DBA, IFRA, VIXY, BIL. Trend 80% + low-vol 20%. Kill switch at 12%. Max 35%, min 5%, top 4.

---

## Orchestrator (`orchestrator/central_intelligence.py`)

1. **Base by regime**: calm_bull US60/mac15/intl15/alt10; stressed US45/mac25/intl10/alt20; crisis US30/mac30/intl5/alt35.
2. **Skill blend**: per-agent independently; negative Sharpe → zero; else 50% skill + 50% base.
3. **Conviction bonus**: up to +15% when ≥2 agents share a name (conv > 0.3).
4. **Correlation guard**: 63-day cross-agent cap at 0.70 → halve smaller.
5. **Thesis coherence**: symbol-level contradictions (UUP↔PDBC/GLD, VIXY↔SVXY) + 12 factor buckets / 6 contradictory pairs → 40% reduction.
6. **EM + commodity cap**: hard 20% aggregate on EM+commodity+gold after all blending.
7. **Crisis veto**: us_regime=crisis → merged = 0.60×macro + 0.40×merged.

---

## Regime system

HMM K=2–4 (best via walk-forward CV). Labels: `calm_bull`, `stressed`, `crisis`, `neutral`, `uncertain`. Hysteresis: enter 0.55 / exit 0.35 / min dwell 3d / entropy > 0.90 → uncertain. Particle filter: 500 particles SIR, reinitializes on batch refit. Emergency refit triggers: SPY −3%+VIX>30, 200MA cross, SPY/TLT corr flip, break z-score > 3.5. Refit every 5 days.

---

## Debate layer

Rebalance days only. Advisory — never writes to alpha/portfolio/execution.

**Sequence**: score past verdicts → debrief → blind spot detection → catalyst scan → Monte Carlo sim → run agents → judge verdict.

| Agent | Model | Role |
|-------|-------|------|
| Bull | claude-opus-4-6 | Strongest case for executing |
| Bear | claude-opus-4-6 | Case for reducing risk |
| Devil's Advocate | claude-opus-4-6 | Most dangerous assumption + Monte Carlo tail |
| Regime Specialist | claude-haiku-4-5-20251001 | Sizing playbook for current regime |
| Quant Sanity | Pure Python | Weight sum, max position, concentration, turnover |

Round 2 rebuttals: bull/bear/devil respond to each other before judge synthesizes.

Verdict: `proceed` | `reduce_size` (Haiku adjusts weights) | `halt_and_review` (persists to `execution/halt_state.json`).

---

## Self-improve loop

Weekly (Sunday 6AM). Generates 5 sleeve-weight variants, scores via real multi-fold OOS (`run_lightweight_oos()`). Shadow promotion if edge > 0.05 Sharpe, 30-day monitoring, auto-promoted by `shadow_promoter.py`. Per-regime variants written to `active_alpha_config.json` `by_regime` section. Stack reads live config on every run.

`SELF_MODIFY_ENABLED = False` — exits early until OOS Sharpe is positive for 30 consecutive trading days on a flat config.

## Factor discovery loop

Monthly (first Sunday of each month, wired into `run_all_agents.py`). Two paths: PySR symbolic regression on pre-computed features (Path A) + Haiku suggests JSON template params (Path B). Both evaluated via per-regime Spearman IC with Harvey FDR correction (IC mean ≥ 0.015, IC IR ≥ 0.60, IC positive in every observed regime). Accepted proposals written to `outputs/factor_proposals/` for human review — nothing auto-deploys.

Gate conditions (~July 2026): `SELF_MODIFY_ENABLED = True`, ≥63 days live regime labels, ≥50 slippage fills.

---

## Execution

Kill switch: SOFT_WARN 8% (log+proceed), HARD_STOP 15% (abort). Alternatives: 12%. State in `logs/kill_switch_state.json`. Large trades (> 2% NAV) → `execution/pending_approvals.json`, async wait via `threading.Event`, 30-min timeout. Almgren-Chriss cost model: blocks > 10% ADV, warns > 5% ADV. Post-fill: slippage logged to `logs/slippage_log.jsonl`.

**Plan 5 additions (execution excellence):**
- `twap_executor.py`: TWAP_ENABLED=False kill switch; orders > 5% ADV routed to equal-spaced child limit orders; Almgren-Chriss optimal window sizing
- `implementation_shortfall.py`: IS decomposition — delay cost / market impact / opportunity cost in bps; logged to `slippage_log.jsonl` backwards-compatibly
- `capacity_model.py`: max AUM per sleeve before signal decay from market impact; informational only; weekly log to `logs/capacity_log.jsonl`
- `intraday_trigger.py`: three triggers (SPY −3%+VIX>30, drawdown ≥12%, event urgency on top-5); checked at 12:00 PM and 14:30 PM ET; adjustments logged to `logs/intraday_adjustments.jsonl`

Config: always `get_config()`, never `Config()` directly.

---

## LLM clients

`ascent/llm/client.py`: `DEFAULT_MODEL = "claude-opus-4-6"`, `HAIKU_MODEL = "claude-haiku-4-5-20251001"`. Lazy singleton. Retry 3× with 2s/4s backoff. All files import `HAIKU_MODEL` from here — never redefine locally.

---

## Data / caching

Cache names: `prices_live` (Yahoo live), `prices_simulated` (GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`, `macro_live/simulated`, `profiles` (sector metadata). Never hide data provenance in cache name. Point-in-time joins via `as_of_join()` / `as_of_merge()`.

---

## Demo app (`demo_app.py`)

Streamlit interactive demo for Tony Ngo. Dark/gold aesthetic. Sidebar: regime, VIX, SPY momentum, portfolio preset, round 2 toggle. Main: portfolio snapshot, debate transcript (5 agents + rebuttals), judge verdict.

**Modes**: Live LLM if `ANTHROPIC_API_KEY` available (via `st.secrets` on Streamlit Cloud, `.env` locally); else demo mode with scenario-aware pre-written arguments.

**Deploy**: push repo to GitHub → share.streamlit.io → set `ANTHROPIC_API_KEY` in Streamlit Cloud secrets dashboard → share URL. Key never in repo (`.env` and `secrets.toml` are gitignored).

---

## Integrity constraints

1. No look-ahead bias — walk-forward uses `get_universe_on_date()` per fold; regime fitted on training slice only.
2. No simulated data under live cache names.
3. Max-weight hard cap via `_water_fill_cap()` with post-condition check.
4. Sector constraint: < 80% coverage → skip caps + warn, never collapse to single name.
5. `walk_forward_runner.py` not a production entrypoint — retained for self_improve Phase D only.
6. Debate is advisory only.
7. Approval gate for orders > 2% NAV.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing fixes. `ast.parse` after each patch. Never propose without tracing first. Planning: Opus for specs → Sonnet for implementation.

---

## Environment

Python 3.12.13 Homebrew, venv at `.venv/`. Use `.venv/bin/python`. API keys via `APIKeys.from_env()` only. Mac Air M5, no JAMF restrictions.

---

## Current portfolio (as of May 7, 2026)

Post-rebalance May 5 holdings (40 orders, full liquidation of Apr 15 portfolio):
KMLM 10.9%, IFRA 9.4%, AMKR/FIX/WDC 5.9% each, VICR 5.7%, VRT 5.6%, VAL 5.2%, WCC 5.0%, DBB 4.6%, CNC 4.5%, WFRD 4.3%, PDBC 4.2%, STLD 3.4%, DBA 3.3%, CHRD 3.1%, EWY 2.4%, EWC 2.3%, IRM 1.8%, MUSA/AAXJ/EEM/VWO 1.5% each, VIXY 2.8% hedge.

NAV: ~$104,815 (May 7). Next rebalance: ~May 13, 2026. Live since April 1, 2026.

Skill scores: all agents still warming up (need 63 days). Regime: calm_bull (refitted May 7 after 5-day stale period).

---

## Known bugs — audit status (2026-05-07)

All audits complete through third pass. No outstanding crash risks. Pipeline runs clean.

---

## Build status

All Plans 1–7 complete. 446 tests passing, 1 skipped. Full specs in `docs/superpowers/plans/`.

| Plan | What it adds |
|------|--------------|
| 1 — Factor Risk Model | FF5+UMD rolling OLS, Ledoit-Wolf Σ, factor P&L attribution |
| 2 — Portfolio Construction | cvxpy MVO, Black-Litterman, regime covariance |
| 3 — Event-Driven Architecture | EDGAR 8-K, Capitol Trades, options anomaly; `EVENT_TRADING_ENABLED=False` |
| 4 — Alternative Data Pipeline | SEC 10-K/Q, transcripts, Reddit, Google Trends; IC gate |
| 5 — Execution Excellence | TWAP, IS decomposition, capacity model, intraday triggers (all kill-switched) |
| 6 — Real-Time Infrastructure | TimescaleDB hypertables, Alpaca WebSocket, live dashboard (port 8502), monthly PDF reports |
| 7 — Live Track Record | SHA-256 hash-chain audit trail, GIPS TWR, risk disclosures, methodology doc |

**Operational next steps (not code):** Deploy TimescaleDB (Docker), configure WebSocket stream (`ALPACA_KEY`), transfer real capital to Alpaca live account (~May–June 2026). YC-ready at April 2027 (12-month live track record).

**Kill switches pending paper validation (~July 2026):** `EVENT_TRADING_ENABLED=False`, `TWAP_ENABLED=False`, `SELF_MODIFY_ENABLED=False`.

**R2R semantic memory:** built but `R2R_API_KEY` not configured; BM25 fallback active.

---

## Session log

### 2026-04-09 — 2026-05-09 (foundation + AI-native Tiers 1–3)
All code referenced here is committed and tested. Details in git log.

**Key architectural decisions:**
- Debate is a conditional circuit breaker, not a daily veto — `debate_gate.py` fires only on high-uncertainty signals
- PDBC↔KMLM correlation (0.81) frequently triggers the orchestrator correlation guard — KMLM gets halved; expected recurring behavior
- ML sleeve: trimmed to 6 features by |ICIR|; `_SPARSE_FILL_ZERO` must include all sparse panels or NaN-drop disables sleeve entirely
- Fundamental sleeve: `high_52w_pct` removed (price momentum, not accounting quality); earnings surprise momentum-neutralized via OLS residual before use
- Tier 3 (factor discovery) blocked on gate conditions until ~July 2026: `SELF_MODIFY_ENABLED`, 63d regime labels, MIN_FILLS=50 in slippage IC
- `signal_score` in slippage IC uses price impact proxy — revisit before MIN_FILLS=50 is reached
- Hedge overlay evaluation: correlation with drawdown was +0.26 (should be negative) — regime calibration issue, revisit after more live data

**Significant non-obvious bugs fixed:**
- `RegimeEngine` takes `config=dict`, not a full `Config` object — `run_all_agents.py` must convert
- `bdate_range(end="today")` returns empty on weekends — use explicit weekday rollback
- `apply_hedge_overlay` must accept both `RegimeSignal` and plain `str` — `AgentOutput.regime_signal` is a string
- ML sleeve cache must store `feature_names` — otherwise XGBoost crashes on "Feature shape mismatch" after feature set changes

**Live milestones:** First rebalance Apr 15 (27 orders). Second May 5 (40 orders, full portfolio rotation). NAV $104,815 as of May 7 (−2.53% on VICR −6.8%). FRED retry logic added May 6.

### 2026-05-10 (Plans 1–7 — institutional build-out ✅)
420→446 tests. Key non-obvious implementation details:

- **Plan 1 (Factor Risk):** Devil's advocate is the sole debate agent receiving factor exposure context.
- **Plan 2 (Portfolio Construction):** BL tau scales with IC IR (< 0.30 → 0.05, < 0.60 → 0.10, else 0.15). MVO uses CLARABEL with SCS fallback; diagonal covariance proxy when factor model unavailable.
- **Plan 5 (Execution):** `weasyprint` requires system GObject/Pango — substituted `reportlab` (pure Python) for all PDF generation. Use `reportlab` for any new PDF work.
- **Plan 6 (Real-Time):** TimescaleDB/WebSocket wired but require Docker + `ALPACA_KEY` to activate. All DB calls return False/empty if unavailable — never raise.
- **Plan 7 (Compliance):** SHA-256 hash chain in `compliance/audit_trail.py`; `verify_integrity()` detects tampering. `scripts/verify_audit_trail.py` exits 0/1 (CI-ready).

### 2026-05-16 (AI PM Agent ✅)
- Design spec: `docs/superpowers/specs/2026-05-16-ai-pm-agent-design.md`; plan: `docs/superpowers/plans/2026-05-16-ai-pm-agent.md`
- `ascent/risk/pm_risk_validator.py` — `validate(portfolio) -> (ok, violations)`; position cap 15%, sector cap 40%, distressed filter (mom_252d < −0.65), min 5 positions, no shorts; **negative-weight short-circuit** returns early before other checks (avoids cascade violations from normalization inflating remaining weights)
- `ascent/strategy/earned_authority.py` — `PHASE_WEIGHTS=[0.0, 0.25, 0.50, 0.75]`, `HARD_CAP=0.80`; same-day dedup guard prevents double-appending returns; state persisted to `data_cache/earned_authority.json`; shadow returns to `data_cache/ai_pm_shadow_returns.jsonl`
- `ascent/strategy/thesis_formatter.py` — `format_thesis()` → `outputs/ai_pm_theses/YYYY-MM-DD-thesis.json`; `thesis_to_plaintext()` 3–4 sentences, never raises
- `agents/ai_pm_agent.py` — Opus, 4-phase loop (market context → 4 quant agents → ≤6 signal tools → `propose_portfolio`); 13 tools; `max_tool_calls=14`; all paths anchored to `_REPO_ROOT = Path(__file__).resolve().parents[1]`; **portfolio dict must be injected into thesis JSON explicitly** — Anthropic tool schema keeps `weights` and `thesis` as separate arguments to `propose_portfolio`
- Wired into `run_all_agents.py`: authority update gated on `if ai_portfolio:` to skip zero-fill before first thesis
- **19 new tests; 465 passing, 1 skipped**
- Current state: shadow period, `ai_weight=0.0`; advances to 25% after 21 rebalance days with Sharpe edge > 0.05 over quant
- Open: push to GitHub; TimescaleDB/WebSocket/real capital still operational decisions
