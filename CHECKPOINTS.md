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

Live trading (Alpaca paper trading via `com.ascentcapital.eod` and `.heartbeat` launchd jobs) was **paused before this sub-project work started** and remains paused as of `checkpoint-5-target-architecture`. Resume only after sub-project 3 (subagent-driven rebuild of anything requiring deeper work) and sub-project 4 (validation + cutover decision).

## Known follow-ups (parked, non-blocking, from checkpoint-5's final review)

- `_apply_falsifier_trim`'s suspension gate (`debate/adversarial_authority.py`) now blocks the *measurement record*, not a trade — if it ever reaches 30 scored outcomes and suspends, the ladder can never demonstrate recovery. Dormant today (`n_scored=0`).
- Discovery mini-rebalance path (`run_all_agents.py`, `_trigger_mini_rebalance`): `portfolio_state` is referenced outside the `try` that defines it — if debate's imports fail, this raises and aborts the mini-rebalance instead of proceeding add-only. Fails safe (no order submitted) but is a one-line fix (`portfolio_state = None` init) worth landing.
- Two stale-comment nits: `run_all_agents.py:1249` still describes the falsifier trim as live action; `CLAUDE.md`'s correlation-guard gotcha attributes `check_cross_agent_correlation` to the wrong file.
