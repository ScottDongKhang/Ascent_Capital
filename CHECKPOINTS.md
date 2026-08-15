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

Live trading (Alpaca paper trading via `com.ascentcapital.eod` and `.heartbeat` launchd jobs) was **paused before this sub-project work started** and remains paused as of 2026-08-14. **Do not resume it** — see the blocker below.

## BLOCKER (2026-08-14): the rebuild may not be in effect at runtime — validation failed

Sub-project 4's real walk-forward validation run surfaced a critical, unresolved bug: `ascent/alpha/stack.py::_load_active_alpha_weights()` reads `data_cache/active_alpha_config.json` (stale, dated `2026-05-02`, 3+ months before the rebuild) **in preference to** the code-level `DEFAULT_ALPHA_WEIGHTS`, and `_get_gated_weights()` reads a stale `logs/sleeve_ic_log.jsonl` and zeros sleeves (including `meanrev`, one of the only 2 surviving sleeves) using frozen, non-rolling IC values, redistributing freed weight to `trend` — hardcoded, even though `trend` is no longer part of the live weight set.

Confirmed via a real run: every one of 330 walk-forward folds showed the alpha stack loading 11 sleeves, not the intended 2, and `meanrev` zeroed out on every fold by a static stale value. The resulting backtest (Sharpe -0.43, Hit Rate 4.6%) is not valid evidence and should not be cited. **This function is not backtest-only — it's what the live pipeline calls too**, so it is NOT confirmed that resuming live trading right now would actually run the intended `meanrev`/`statarb` stack.

Full findings: `outputs/wf_results/vc-task-2-validation-report.md`. Memory:
`alpha-weights-runtime-override-not-fixed.md`. **Fix required before any further validation
attempt or cutover decision**: reset/delete the two stale files (or make both functions reject
a config/log predating the rebuild), and decide what "redistribute to trend" should do now that
trend isn't live.

## Known follow-ups (parked, non-blocking, from checkpoint-5's final review)

- `_apply_falsifier_trim`'s suspension gate (`debate/adversarial_authority.py`) now blocks the *measurement record*, not a trade — if it ever reaches 30 scored outcomes and suspends, the ladder can never demonstrate recovery. Dormant today (`n_scored=0`).
- Discovery mini-rebalance path (`run_all_agents.py`, `_trigger_mini_rebalance`): `portfolio_state` is referenced outside the `try` that defines it — if debate's imports fail, this raises and aborts the mini-rebalance instead of proceeding add-only. Fails safe (no order submitted) but is a one-line fix (`portfolio_state = None` init) worth landing.
- Two stale-comment nits: `run_all_agents.py:1249` still describes the falsifier trim as live action; `CLAUDE.md`'s correlation-guard gotcha attributes `check_cross_agent_correlation` to the wrong file.
