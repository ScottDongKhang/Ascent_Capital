# CLAUDE.md — Ascent Capital

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio →
walk-forward → regime → 4 specialist agents → orchestration → AI PM (earned autonomy) →
debate → Alpaca paper trading.

**Daily command**: `.venv/bin/python run_all_agents.py` (branches on rebalance day)

---

## Read this first

This file holds **durable rules only** — things that are true across sessions and cannot be
derived by reading the code. It deliberately contains **no performance numbers, no dates, and
no line numbers**, because those go stale within days and a stale figure in an always-loaded
file becomes a confidently-wrong belief rather than a harmless typo.

| You need | Read |
|---|---|
| Any performance / risk figure | `CURRENT_VERIFIED_NUMBERS.md` — the only citable source |
| Where code lives, what to grep | `docs/REPO_MAP.md` |
| What happened in past sessions | `docs/session_log_archive.md` |
| Whether this file is still true | `.venv/bin/python scripts/verify_docs.py` |

**`scripts/verify_docs.py` is the guard on this file.** Every mechanically checkable claim
below has a corresponding check. Run it after touching code that this file describes; a red
check means either the code changed or this file lied. Fix one of those two — never the guard.

**Grounding rule.** Every number you report must come from a named artifact you actually read
in this session. If you reconstructed it, say so. A confident wrong number is worse than an
acknowledged gap — this project has published a synthetic drawdown as fact before.

---

## Environment

Python 3.12.13, venv at `.venv/`. Always use `.venv/bin/python`. API keys via
`APIKeys.from_env()`. Config via `get_config()` — never `Config()` directly.

---

## LLM models (`ascent/llm/client.py`)

Import constants from here — never redefine locally:
```python
DEFAULT_MODEL = "claude-opus-5"             # AI PM synthesis (Phase 2)
SONNET_MODEL  = "claude-sonnet-5"           # debate agents, red team, pre-thesis (Phase 1)
HAIKU_MODEL   = "claude-haiku-4-5-20251001" # classifiers, weight adjustment
```

**Claude 5 rules** (migrated from the 4.6 generation):
- **Never index `resp.content[0].text` for a Claude 5 model** — thinking is ON by default, so
  block 0 is usually a thinking block. Use `extract_text` from `ascent/llm/client.py`.
  *Carve-out*: existing `content[0].text` sites are Haiku-only `messages.create` calls where
  block 0 really is the text. They are safe, and the guard keeps them Haiku-only — repointing
  one at Opus or Sonnet trips `content0_haiku_only`.
- **Never pass `temperature` / `top_p` / `top_k`** — 400 on Claude 5. The `temperature` kwarg
  still exists on the wrappers for call-site compatibility and is silently dropped for
  Claude 5 models; steer with prompts instead.
- **Never pass `thinking={"type": "enabled", "budget_tokens": N}`** — 400. Depth is set by
  `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`). Wrapper defaults:
  `chat_completion`/`generate_structured` = medium, `extended_thinking_completion` and
  `tool_completion` = high. The one surviving `budget_tokens` call is the legacy branch inside
  `ascent/llm/client.py`, gated by `_CLAUDE_5_MODELS`; do not add another anywhere else.
- **`max_tokens` caps thinking + visible text together.** The wrappers raise it to
  `_MIN_TOKENS_WITH_THINKING`; direct `messages.create()` callers must leave their own
  headroom or thinking consumes the whole budget and returns no text.
- **Keep thinking enabled in tool loops.** With thinking disabled, Claude 5 can emit a tool
  call as visible text instead of a `tool_use` block — the turn succeeds and the tool
  silently never runs.
- **Echo thinking blocks back unchanged** when continuing a turn (handled inside
  `tool_completion`); dropping or editing them is rejected.
- Haiku 4.5 is current and unchanged — it keeps the legacy parameter path.

---

## Key layout

```
ascent/           core engine: config, data, features, alpha, portfolio, backtest,
                  research, regime, risk, reporting, execution, monitoring, llm,
                  dashboard, strategy, causal, integrations, memory, utils
agents/           us_equities_agent, macro_agent, international_agent,
                  alternatives_agent, ai_pm_agent, red_team_agent, event_agent
orchestrator/     central_intelligence.py
debate/           debate_runner, agents, judge, adversarial_authority,
                  adversarial_engine, adversarial_monitor, agent_tools,
                  disagreement_scorer, outcome_tracker
run_all_agents.py daily entrypoint
ascent/main.py    pipeline entrypoint (called by each agent)
data_cache/       parquet caches, earned_authority.json, active_alpha_config.json,
                  ai_prethesis_latest.json, ai_regime_assessment.json
logs/             eod_log.jsonl, multi_agent_run.jsonl, ai_pm_calibration.jsonl
outputs/debate_log/  verdict_<date>.json, debrief_<date>.json
```

See `docs/REPO_MAP.md` for what to grep for and which files are worth reading whole.

---

## Integrity constraints (never violate)

1. **No look-ahead bias** — walk-forward uses `get_universe_on_date()` per fold; regime fitted
   on the training slice only.
2. **No simulated data under live cache names** — `prices_live` = Yahoo live only; on
   fetch failure the name becomes `prices_live_fallback_simulated`.
3. **Max-weight hard cap** via `_water_fill_cap()` in `ascent/portfolio/optimizer.py`, with a
   post-condition check.
4. **Sector constraint**: under 80% coverage → skip caps and warn. Never collapse to a single
   name.
5. **Debate is advisory only *at the module level*** — nothing under `debate/` writes to alpha,
   portfolio, or execution. The single sanctioned exception: **at most one** judge position
   change is applied by `run_all_agents.py` itself (grep `apply_judge_position_change`), not by
   `debate/`. It takes only the first entry of the verdict's `position_changes` and never
   iterates — the guard enforces both. It applies on the scheduled-rebalance path *and* the
   discovery path, and is bounded by `_JUDGE_MAX_WEIGHT` / `_JUDGE_MIN_WEIGHT` plus earned
   authority (`debate/adversarial_authority.py`, via `allowed_change_pct`), which stays at the
   `low` tier — 1.0pp max per intervention — until a type reaches `MIN_SAMPLE_SUSPEND` scored
   outcomes. Treat this as a bounded, authority-gated exception, not a licence for debate code
   to acquire write access elsewhere.
6. **New alpha sleeves**: update `DEFAULT_ALPHA_WEIGHTS` in BOTH `ascent/alpha/stack.py` AND
   `ascent/research/self_improve.py`. The guard enforces that their key sets match.
7. **Fundamental sleeve is disabled** (measured anti-signal). Do not re-enable without a
   positive, artifact-backed IC-t.

---

## Non-obvious gotchas

- **Agent module names**: `agents.us_equities_agent`, not `agents.us_equities` — the `_agent`
  suffix is required for all of them.
- **`RegimeEngine` constructor** takes `config=dict`, not a `Config` object.
- **`bdate_range(end="today")`** returns empty on weekends — use an explicit weekday rollback.
- **Market dates**: the host runs at UTC+7, so `date.today()` is a day ahead of the US market
  for much of the day. Use `ascent/utils/market_time.py`.
- **`apply_hedge_overlay`** must accept both a `RegimeSignal` and a plain `str`.
- **ML sleeve cache** must store `feature_names` — XGBoost crashes on shape mismatch if the
  feature set changes between writes.
- **AI PM is two-phase**: Phase 1 `run_ai_pm_prethesis()` uses `SONNET_MODEL` (breadth),
  Phase 2 `run_ai_pm(prethesis=...)` uses `DEFAULT_MODEL` (judgment). Never swap them.
- **Pre-thesis runs AFTER the quant agents and the orchestrator merge**, despite some stale
  in-code comments saying otherwise. It writes `data_cache/ai_regime_assessment.json`, which
  is therefore consumed by the *next* run, not the current one. Failure → `prethesis=None` →
  graceful single-phase fallback.
- **`propose_prethesis` vs `propose_portfolio`**: different tools, different result stores.
  Phase 1 ends with the former, Phase 2 with the latter.
- **`run_ai_pm(quant_outputs=...)`**: pass the `agent_outputs` list or AI PM re-runs all four
  agents and wastes minutes.
- **`ascent/main.py` `run_pipeline` returns a 10-tuple** (last element `_alpha_breakdown`) —
  `eod_runner.py` and `us_equities_agent.py` unpack it positionally.
- **`_SPARSE_FILL_ZERO`** in the ML sleeve must include ALL sparse panels, or a NaN-drop
  silently disables those sleeves.
- **PDF generation**: use `reportlab`. `weasyprint` is installed but fails to import here
  (missing system Pango/GObject).
- **PDBC↔KMLM are highly correlated** and frequently trip the orchestrator correlation guard,
  halving KMLM. Expected, not a bug.
- **Kill switches** stay `False` pending paper validation: `EVENT_TRADING_ENABLED`,
  `TWAP_ENABLED`, `SELF_MODIFY_ENABLED`, `LONG_SHORT_ENABLED`. The code comment on
  `LONG_SHORT_ENABLED` carries the authoritative precondition (a minimum number of paper
  rebalances) — trust it over any date written in prose.
- **Dashboard subprocess**: spawn with `cwd=repo_root` and `PYTHONPATH=repo_root`. `ascent`
  is not pip-installed, so running from `scripts/` breaks every import.
- **Same-day Track B is unreliable**: Alpaca 1D bars settle late afternoon PT, so an early
  run sees `equity == last_equity` and records a fake 0.0. Use
  `alpaca_broker.get_portfolio_history()` for settled returns.
- **`loguru` is not installed** — use `import logging`; never `from loguru import logger`.
- **MiroFish on rebalance days**: the LiteLLM proxy (port 4000 → Haiku) must be running.
  OpenRouter leaves `max_tokens` unset → 402 if credits are low.
- **Discovery mini-rebalance is add-only**: `_insert_candidate_weights`, not a full agent
  re-run. Suppressed near a scheduled rebalance via `_is_near_scheduled_rebalance`.
- **`run_eod_with_weights()` silently no-ops on non-rebalance days** — pass `force=True` on
  discovery / mini-rebalance paths.
- **The AI PM decision log only gets entries on scheduled rebalances.** Off-calendar discovery
  days run the daily-view path, not Phase 2 — a missing entry is expected, not a bug.
- **`reduce_size` cannot reliably reduce size.** When the Haiku adjustment trims too few
  positions, the fallback force-trims the largest positions and renormalizes to 1.0 — which
  sells exactly the hedges the judge argued to protect. Diagnose transmission before
  concluding the AI PM's judgment was bad.
- **Wide-format caches carry their date in a `date` COLUMN, and every consumer restores the
  index itself.** `prices_macro` / `prices_international` / `prices_alternatives` are wide
  (one column per symbol, no id column). `save_parquet` now converts a `DatetimeIndex` input
  into a `date` column up front — reusing the existing `id_cols` / calendar-day-dedup
  machinery — because its `pd.concat(..., ignore_index=True)` and `to_parquet(index=False)`
  otherwise dropped all date information on every save. `load_parquet` is deliberately
  unchanged (it is generic, used by every cache) and returns that `date` column as-is on a
  `RangeIndex`, so **each caller must do `df.set_index("date")` itself** — and `.sort_index()`
  too if it slices positionally. Known call sites, all fixed:
  `agents/macro_agent.py`, `agents/international_agent.py`, `agents/alternatives_agent.py`
  (two per file: the fresh-cache read and the stale-cache fallback),
  `ascent/analyst/proof_audit/run.py::_load_agent_price_matrix`, and
  `ascent/risk/correlation_guard.py::_load_combined_prices` (sorts — its window is
  `returns.iloc[-63:]`). Add the restore to any new consumer.
  **Still true:** the three on-disk caches written before this fix are corrupt (dateless
  `RangeIndex`, 176k/151k/150k rows for 9-13 symbols) and cannot be repaired in place —
  deleting and re-fetching them is a separate planned data operation, not yet done.

---

## Data / caching

Cache-name provenance — never obscure it: `prices_live` (Yahoo live), `prices_simulated`
(GBM), `prices_live_fallback_simulated` (live-fetch failure), `prices_macro`,
`macro_live` / `macro_simulated`, `profiles` (sector metadata).

Point-in-time joins: always use `as_of_join()` / `as_of_merge()` from
`ascent/data/store/point_in_time.py` for cross-dataset alignment.

**`prices_live` duplicate rows have recurred twice.** `save_parquet` dedups on a
`_calendar_day_key`, the live read path additionally defends itself with
`~index.duplicated(keep="last")` in `ascent/main.py`, and first-write dedup plus an
evening-stamp rollover were added after the second recurrence. **Do not treat the cache as
clean on trust** — a previous cleanup had already been reported as holding when it was not.
Measure it: `scripts/reconcile_numbers.py` reports the live duplicate count in its
data-integrity section. This matters most for backtests, because the walk-forward framework
does **not** dedupe on read; only the live pipeline does.

---

## Debugging protocol

One step at a time. Verify existing logic before proposing a fix. Never propose without
tracing first. `ast.parse` after each patch. Planning: Opus for specs → Sonnet for
implementation.

Prefer a test that reproduces the bug before the fix. Note that a long-standing "known-buggy,
don't cite" number is often a *failing test that was right all along* — check whether one
already exists before writing a new one.

---

## Rebalance recap (required after every rebalance)

After **any** run of `run_all_agents.py` that submits orders, write a four-part recap
unprompted. This is the primary interface to what the system decided; a status line is not
sufficient.

1. **Reasoning behind the decision.** Read the day's `outputs/debate_log/verdict_<date>.json`
   (`verdict.reasoning`, `verdict.key_risks`) and the adversarial intervention. Explain why
   *these* trades, which argument won which exchange, and what the judge explicitly declined
   to do. **Then verify execution matched the reasoning** — the `reduce_size` fallback has
   sold the exact positions the judge argued to protect. Quote the reasoning; don't paraphrase
   it away.
2. **Things to watch for.** Live catalysts (FOMC, earnings, ex-div), positions on thin ice,
   guards that nearly fired, data sources that were down, anything that changes the next read.
3. **Performance since the last rebalance.** Portfolio vs SPY over the window, with
   per-position attribution when a few names dominate. Use
   `alpaca_broker.get_portfolio_history()` (settled bars) — never same-day
   `equity − last_equity`, which reads a fake 0.0 before late afternoon PT.
4. **Why performance looks like that.** The causal story, separating sizing and structure from
   stock selection from things outside the model's control.

Ground every number in a real artifact and flag anything reconstructed.

---

## Current state

Live state is **not recorded here** — it goes stale faster than this file is edited. Read:

- `CURRENT_VERIFIED_NUMBERS.md` — performance, risk, AI PM authority level, and what is
  explicitly *not* verifiable. This file wins any disagreement.
- `dashboard/regime_labels.csv` — the current regime label. Note that
  `dashboard/regime_signal.json` and `data_cache/ai_regime_assessment.json` can lag it, and
  `run_all_agents.py` reads its smart-model trigger from the JSON, so a stale label there
  changes behaviour.
- `rebalance_calendar.csv` — the next scheduled rebalance.
- `logs/eod_log.jsonl` — what actually ran, including `catch_up` runs after an outage.

Structural facts that do not go stale: the raw-return lag versus SPY is by design (defensive
non-equity sleeves, a 200MA cut, and a vol-target overlay all cost beta in an equity-only
bull); walk-forward OOS confirms positive risk-adjusted alpha with a negative
walk-forward-efficiency figure that must be disclosed as overfit.

**GitHub Pages**: `https://scottdongkhang.github.io/Ascent_Capital` — auto-updated after every
daily run, which means an unretracted wrong number there is published, not just recorded.

---

## Session log

Session history lives in `docs/session_log_archive.md`, newest first. Append there, not here.
