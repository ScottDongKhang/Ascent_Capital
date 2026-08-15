# Git Checkpoints

This file documents checkpoint tags marking major milestones in the strip-down/rebuild effort
(proof audit, production bugfixes, and the target-architecture rewrite), 2026-08-13/14.

## Checkpoints

| Tag | Commit (short) | Description |
|---|---|---|
| `checkpoint-1-proof-audit` | `193f445` | Proof audit scorer built: guards degenerate signals, flags duplicate agent scores, adds row-level reason disclosure |
| `checkpoint-2-proof-audit-extension` | `3e824e7` | Proof audit extension: corrects false reasons, discloses agent cache fallback, wires real data sources (earnings, fundamentals, insider, altdata) |
| `checkpoint-3-production-bugfixes` | `db0890a` | Production bugfixes: fixes signal_date normalization, save_parquet DatetimeIndex preservation, correlation_guard index sorting; documents cache contract |
| `checkpoint-4-cache-repair-final` | `1cb932c` | Cache repair + final scorecard: deleted corrupted macro/international/alternatives price caches, re-fetched live data, regenerated scorecard (KEEP=2 CUT=12 INSUFFICIENT_DATA=9) |
| `checkpoint-5-target-architecture` | `19c3240` | Target architecture rebuild: alpha stack reduced to meanrev+statarb (2 of 15 sleeves), only `us_equities_agent` allocates live capital, regime/hedge overlays removed, and every unproven live-write path (judge position-change, AI-PM earned-authority blend, and the previously-unmeasured falsifier trim, found during final review) made advisory-only — detected, logged, and recorded for future measurement, but no longer touching live capital |

## How to Use

**Inspect a checkpoint:**
```bash
git show checkpoint-1-proof-audit          # View commit details
git checkout checkpoint-3-production-bugfixes  # Check out the specific commit
```

**Compare against a checkpoint:**
```bash
git diff checkpoint-2-proof-audit-extension..main   # See all changes since this checkpoint
git log checkpoint-1-proof-audit..main --oneline    # See all commits after this checkpoint
```

## Operational Context

Live trading (Alpaca paper trading via `com.ascentcapital.eod` and `.heartbeat` launchd jobs) was **paused before this sub-project work started**. As of 2026-08-15 the validation blocker below is **resolved** — a real walk-forward run on the rebuilt 2-sleeve stack now produces a plausible strategy signature (Sharpe 0.415, Hit Rate 52.3%, CAGR +10.20%, benchmark beta 0.947). This does not itself authorize resuming live trading; that decision is the controller's/human's to make after reviewing the numbers below, not an automatic consequence of the blocker clearing.

## RESOLVED (2026-08-15): validation blocker — root cause was a `prices_live` cache corruption, not the alpha-weight bug

The 2026-08-14 blocker (below, kept for history) turned out to be real but **not** the cause of the bad backtest numbers it was originally attached to. After that fix was confirmed correct (`889c600`, `31d49ee`) and re-validation still showed the same implausible results, a second investigation found the actual cause: `data_cache/prices_live.parquet` contained ~88,841 legacy phantom duplicate-day rows (non-midnight timestamps, sparse ~54/938-symbol coverage) alongside the ~1,650 real trading-day rows. `ascent/research/walk_forward_runner.py` pivoted on these without normalizing, fragmenting the date axis to ~3,298 entries; `pct_change().fillna(0)` in `ascent/backtest/engine.py` then silently deleted real price moves on every phantom date, crushing volatility and hit-rate and zeroing the benchmark's beta/CAGR — the actual tell that gave it away.

Fixed in commits `b628885` (diagnosis — 99.94% of phantom rows turned out to be the *only* record for 49 macro/ETF symbols + 6 more pre-2026, so a blanket drop would have destroyed real data), `9fd74ea` (repair — merged real/phantom rows per (symbol, trading-day) via `_calendar_day_key`, cache now 1,519,291 rows / 0 non-midnight / pivot shape (1650, 938), backed up at `data_cache/prices_live.parquet.pre_phantom_repair_bak`), `9f145fc` (hardening — `validate_cache()` and `pivot_prices()` now detect/guard against this corruption class), `e36442d` (re-validation — the numbers above), `0d1496e` (final-review fix wave — extended `reconcile_numbers.py`'s data-integrity check to the same corruption class, since it had been reporting `dup_rows: 0` on the fully-corrupt cache; aligned `pivot_prices()`'s merge key with `_calendar_day_key` so a future recurrence fails loudly again instead of silently). Full spec: `docs/superpowers/specs/2026-08-15-prices-live-phantom-row-fix-design.md`. Full comparison report: `outputs/wf_results/vc-task-4-post-phantom-fix-report.md`.

**Known residual, not blocking**: `ascent/analyst/proof_audit/run.py::_dedupe_prices_by_calendar_day()` still dedups on plain `.dt.normalize()`, not `_calendar_day_key` — same corruption class, one call site this branch didn't touch. Consumes the now-repaired (clean) cache today, so not currently wrong, but would mis-merge a future recurrence the same way `pivot_prices()` used to. Worth a small follow-up fix.

**Also surfaced, not blocking, needs its own follow-up**: the phantom-row diagnosis found ~49-55 symbols (macro/ETF instruments plus a handful of equities: `ACI BK CBOE L OMC PVH ZBH` among the 901-name `us_equities` universe) whose most recent `prices_live` data is dated 2026-07-24 while the rest of the cache runs to 2026-07-28 — no live path currently writes them. Not caused by this repair; the diagnosis (`outputs/wf_results/phantom-row-diagnosis-2026-08-15.md`) is what surfaced it.

## BLOCKER (2026-08-14, RESOLVED — history only): the rebuild may not be in effect at runtime — validation failed

Sub-project 4's real walk-forward validation run surfaced a critical bug: `ascent/alpha/stack.py::_load_active_alpha_weights()` reads `data_cache/active_alpha_config.json` (stale, dated `2026-05-02`, 3+ months before the rebuild) **in preference to** the code-level `DEFAULT_ALPHA_WEIGHTS`, and `_get_gated_weights()` reads a stale `logs/sleeve_ic_log.jsonl` and zeros sleeves (including `meanrev`, one of the only 2 surviving sleeves) using frozen, non-rolling IC values, redistributing freed weight to `trend` — hardcoded, even though `trend` is no longer part of the live weight set.

Confirmed via a real run: every one of 330 walk-forward folds showed the alpha stack loading 11 sleeves, not the intended 2, and `meanrev` zeroed out on every fold by a static stale value. The resulting backtest (Sharpe -0.43, Hit Rate 4.6%) is not valid evidence and should not be cited. This bug was real and is fixed (`6628948`, `31d49ee`, confirmed correct by `889c600`) — but, per the RESOLVED section above, it was NOT the cause of the bad backtest; the `prices_live` cache corruption was.

Full findings: `outputs/wf_results/vc-task-2-validation-report.md`. Memory:
`alpha-weights-runtime-override-not-fixed.md`.

## Known follow-ups (parked, non-blocking, from checkpoint-5's final review)

- `_apply_falsifier_trim`'s suspension gate (`debate/adversarial_authority.py`) now blocks the *measurement record*, not a trade — if it ever reaches 30 scored outcomes and suspends, the ladder can never demonstrate recovery. Dormant today (`n_scored=0`).
- Discovery mini-rebalance path (`run_all_agents.py`, `_trigger_mini_rebalance`): `portfolio_state` is referenced outside the `try` that defines it — if debate's imports fail, this raises and aborts the mini-rebalance instead of proceeding add-only. Fails safe (no order submitted) but is a one-line fix (`portfolio_state = None` init) worth landing.
- Two stale-comment nits: `run_all_agents.py:1249` still describes the falsifier trim as live action; `CLAUDE.md`'s correlation-guard gotcha attributes `check_cross_agent_correlation` to the wrong file.
