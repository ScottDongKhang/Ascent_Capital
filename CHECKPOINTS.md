# Git Checkpoints

This file documents the four checkpoint tags created on 2026-08-13, marking major milestones in the proof-audit and production-bugfix sub-projects.

## Checkpoints

| Tag | Commit (short) | Description |
|---|---|---|
| `checkpoint-1-proof-audit` | `193f445` | Proof audit scorer built: guards degenerate signals, flags duplicate agent scores, adds row-level reason disclosure |
| `checkpoint-2-proof-audit-extension` | `3e824e7` | Proof audit extension: corrects false reasons, discloses agent cache fallback, wires real data sources (earnings, fundamentals, insider, altdata) |
| `checkpoint-3-production-bugfixes` | `db0890a` | Production bugfixes: fixes signal_date normalization, save_parquet DatetimeIndex preservation, correlation_guard index sorting; documents cache contract |
| `checkpoint-4-cache-repair-final` | `1cb932c` | Cache repair + final scorecard: deleted corrupted macro/international/alternatives price caches, re-fetched live data, regenerated scorecard (KEEP=2 CUT=12 INSUFFICIENT_DATA=9) |

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

Live trading (Alpaca paper trading via `com.ascentcapital.eod` and `.heartbeat` launchd jobs) was **paused before this sub-project work started** and remains paused as of `checkpoint-4-cache-repair-final`. Resume only after verifying the scorecard and new cache integrity against live trading conditions.
