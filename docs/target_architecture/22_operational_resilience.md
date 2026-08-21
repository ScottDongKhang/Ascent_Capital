# Operational Resilience — Disaster Recovery, Key-Person Risk, and the Single Machine

Scoped to fill exactly the gap `17_red_team_findings.md` §3 named as missing from
all 21 prior files: disaster recovery, key-person risk, credential/secrets
monitoring, broker/LLM-provider outage handling, and physical single-machine
failure. Follows the same 5-layer format as `01`-`06`, but — per
`20_cost_timeline_reality_check.md`'s finding that this is a part-time solo
project — this is **not** a multi-person ops department. There is one role
(the Operator) wearing five hats, and the design goal is a system that fails
loud and recoverable, not a system with 24/7 coverage. Where a real fund would
staff a NOC, this document proposes the cheapest thing a solo operator will
actually keep doing: a daily automated health-check that lands somewhere it
gets read.

## What already exists — read this before building anything

The single biggest finding of this document, found by reading the code rather
than assuming a green field: **liveness monitoring already exists and is
better-built than `17` gave it credit for, but it is currently not running.**

- `scripts/heartbeat_check.py` (443 lines, stdlib-only by design, comment at
  the top explains why: it exists because the pipeline "silently stopped
  running for 27 days" once already, and a watchdog built from the same
  dependency stack as the thing it watches can be broken by the same
  failure). It reads `logs/eod_log.jsonl` and `rebalance_calendar.csv`,
  computes trading-day-aware staleness (`WARN_MISSED_DAYS = 2`,
  `CRITICAL_MISSED_DAYS = 3`), writes `logs/liveness.json`, and fires a
  direct alert (ntfy.sh if `NTFY_TOPIC` is set, always also a local macOS
  notification via `osascript` as a no-config fallback) — see
  `send_direct_alert()` at line ~294.
- `ascent/monitoring/alert_system.py` provides `send_alert()` with a 4-hour
  dedup window and a `VALID_ALERT_TYPES` set that already includes
  `"liveness"` and `"system_alive"` — `heartbeat_check.py` hands off to it
  when `ascent` is importable, and always also fires the direct channel
  regardless, so a broken `ascent` environment can't take the watchdog down
  with it.
- `scripts/com.ascentcapital.heartbeat.plist` is a launchd job specified to
  run every 6 hours via `StartInterval` (deliberately not
  `StartCalendarInterval`, per the plist's own comment, because calendar
  jobs are skipped entirely — not caught up — if the Mac is asleep at the
  scheduled instant; interval jobs fire promptly on wake instead).
  `scripts/com.ascentcapital.eod.plist` is the corresponding job for the
  daily pipeline itself.

**But as of this audit, neither plist is loaded.** Both files sit in
`~/Library/LaunchAgents/` (confirmed on disk), but `launchctl list` shows
only `com.ascent.mirofish` and `com.ascent.litellm` running — not
`com.ascentcapital.heartbeat` or `com.ascentcapital.eod`. Consistent with
that: `logs/liveness.json` was last written 2026-08-12, reporting `CRITICAL`
— `missed_days: 12`, last logged pipeline run 2026-07-27, one missed
scheduled rebalance (2026-08-05) — and has not been updated since, meaning
the watchdog itself has now been silent for over a week on top of the
outage it was already reporting. `logs/eod_log.jsonl`'s own last entry is
also 2026-07-27. (Some of this gap may be the deliberate trading hold from
2026-08-15 noted in session memory — but the watchdog going silent on
2026-08-12, three days *before* that hold decision, is not explained by it.)
**Layer 5 item 1 below is therefore not "build a watchdog" — it is "load the
watchdog that already exists," which is a five-minute `launchctl load`, not
a weekend project.** This is exactly the kind of gap `18`'s "log
self-healing events, don't trust that a fix is still in effect" principle
warns about — a mechanism can be correctly designed and still silently
stop running.

What's genuinely absent, confirmed by direct search, not just "not written
here":
- **No backup of `data_cache/` or `logs/` anywhere.** Both directories are
  git-ignored (`.gitignore` lines 4 and 14) and neither is tracked — `git ls-files
  data_cache` returns 2 files, `git ls-files logs` returns 49, out of a live
  `data_cache/` that is 922MB. A single machine failure loses the full price
  cache history, `earned_authority.json`, `kill_switch_state.json`,
  `alert_state.json`, and every log — recoverable in principle (caches can be
  re-fetched, most state files default safely to a clean state — see Layer 5
  item 2) but with zero current mechanism to do so systematically.
- **No credential validation or expiry check anywhere.**
  `ascent/config/settings.py::APIKeys.from_env()` (lines 27-46) reads each
  environment variable with `os.getenv(...).strip()` and falls back to an
  empty-string default — it never checks that a key is present, well-formed,
  or still valid; a blank or revoked key produces `APIKeys(alpaca_key="", ...)`
  silently, and the first sign of trouble is whatever exception the Alpaca or
  Anthropic SDK happens to raise deep in the pipeline. This is the "pure
  net-new gap" `17` flagged, confirmed by reading the actual function.
- **No documented recovery procedure** for a mid-run crash with open orders,
  no secrets-expiry monitor, no second-broker contingency, no LLM-provider
  outage fallback distinct from the existing missing-prethesis
  data-availability fallback.

## Layer 1 — Department Mandate & Authority Boundaries

**Mandate**: keep the system observably alive, keep its state recoverable
after any single-machine or single-vendor failure, and make credential/
provider failures loud instead of silent — for one person, on one laptop,
part-time.

**Explicitly not the mandate**: this is not a business-continuity plan for a
firm with clients, a NOC, or SLA obligations. No role here has authority to
place trades or change risk thresholds — Operational Resilience is a
read-and-alert layer over the existing pipeline (analogous to how `05`'s
Reconciliation Analyst only halts, never resizes). The only "write" actions
in this document are (a) persisting a recovery checkpoint, and (b) tripping
the *existing* halt/kill-switch machinery — never inventing new
order-affecting logic.

**Authority boundaries**:
- Can halt the next run (by setting `requires_override: true` in the
  existing halt-state file, same mechanism `05`'s Reconciliation Analyst and
  the kill switch already use) when a health check fails.
- Cannot resume trading — only the Operator, explicitly, per the existing
  `--reset-kill-switch` / `execution/halt_override.json` pattern (see `06`'s
  corrected characterization of `check_halt_state()`).
- Cannot modify risk thresholds, alpha weights, or position sizing.
- Escalates everything to one place: the Operator's phone, via the same
  ntfy.sh/macOS-notification channel `heartbeat_check.py` already has.

**Who checks this layer's own decisions** (the question `17` §2 raised about
every "independent" role in `01`-`06`): nobody, structurally — same honest
answer as the rest of the doc set. The mitigant is the same one `18`
proposes for the ritual layer: every automated recovery action (a cache
re-fetch, a state-file reset to default) gets logged to a file the Operator
is expected to actually read (Layer 3 item 5), not silently trusted.

## Layer 2 — Roles

One human, five monitoring functions, mapped onto existing or trivially-new
code rather than new organizational roles:

| Role | Who/what | New or existing |
|---|---|---|
| **Liveness Monitor** | `scripts/heartbeat_check.py` + its launchd job | Existing — needs to be *loaded*, not built |
| **Recovery-Point Keeper** | New: `scripts/recovery_snapshot.py` | New, ~1 evening, modeled on `kill_switch.py`'s state-persistence pattern |
| **Credential Health Checker** | New: `scripts/credential_check.py` | New, ~1 evening |
| **Backup Custodian** | New: a `launchd`-scheduled `rsync`/`tar` job | New, ~1 evening, no code — a shell script + plist |
| **Daily Digest** (the one ritual piece) | New: extends `heartbeat_check.py`'s existing `--alive-ping` concept | New, ~1 evening — the "cheap thing that actually gets checked" |

No LLM-backed role here. Per `19`'s finding that most department roles are
deterministic threshold checks, this department is *entirely* (a) — every
decision below is `if condition: alert`. Giving any of this an LLM call
would add cost and a new failure mode (the LLM provider itself) to the layer
whose job is to detect provider failures.

## Layer 3 — Per-Role Decision Logic

### Liveness Monitor (existing — activation, not construction)
- **Load the two plists.** `launchctl load ~/Library/LaunchAgents/com.ascentcapital.heartbeat.plist`
  and the `.eod.plist` counterpart. Verify with `launchctl list | grep
  ascentcapital` — the same command used in this audit to discover they
  weren't loaded.
- **Existing thresholds, unchanged**: `WARN_MISSED_DAYS = 2` trading days →
  local + push notification, no halt. `CRITICAL_MISSED_DAYS = 3` → same
  alert channel, `STATUS_CRITICAL`. Neither currently *halts* the pipeline —
  it only alerts (`heartbeat_check.py` is read-only with respect to
  `HALT_STATE_PATH`). This document adds one new trigger only:
  **the watchdog itself going stale is a distinct, higher-severity
  condition than what it watches** (see Recovery-Point Keeper below) — a
  known-broken-once failure mode already, discovered during this audit.

### Recovery-Point Keeper (new)
- Modeled directly on `ascent/execution/kill_switch.py`'s state-persistence
  pattern: `_load_state()` reads a small JSON file with a safe default if
  missing, `_save_state()` writes it back after every mutation, and
  `KS_STATE_PATH = logs/kill_switch_state.json` survives process restarts.
  The same shape works for a recovery checkpoint.
- New file `logs/recovery_checkpoint.json`, written at the *end* of every
  successful `run_all_agents.py` invocation (one call, right before the
  process exits normally — the one place in the codebase that already knows
  "this run fully succeeded"). Contents: `{run_date, last_good_positions:
  <alpaca get_positions() snapshot>, kill_switch_state_hash, eod_log_line_offset,
  earned_authority_state_hash, generated_at}`.
- **Trigger**: on process start, `run_all_agents.py` compares the current
  Alpaca live book (`get_positions()`) against `last_good_positions` in the
  checkpoint. If they disagree by more than the existing Reconciliation
  Analyst-style tolerance (`05`'s **> $500 notional or > 5% of target
  weight**, reused rather than reinvented), that is exactly the "crashed
  mid-run with open orders in flight" scenario `17` named — log it and
  refuse to auto-execute, same fail-closed posture as `_catch_up_guard`.
- This is deliberately **not** a full state-machine recovery system — it's
  one JSON file, one comparison, reusing an existing pattern and an existing
  tolerance. A solo operator can maintain "compare two numbers on startup";
  they cannot maintain a distributed consensus recovery protocol, and
  nothing here should look like one.

### Credential Health Checker (new)
- New file `scripts/credential_check.py`, stdlib-plus-SDK (unlike
  `heartbeat_check.py` this can safely import `ascent` — it only needs to
  run once a day, not survive the same failure class as the daily pipeline,
  since its output is informational, not the last line of defense).
- Checks, each independently, never raising past a caught exception:
  1. **Presence**: every field in `APIKeys.from_env()` that the active
     pipeline actually uses (`alpaca_key`, `alpaca_secret`, and whichever
     LLM key `ascent/llm/client.py` resolves) is non-empty. This alone closes
     the gap found in Layer "what already exists" above — `from_env()` never
     checks this today.
  2. **Liveness**: one cheap, side-effect-free call per provider —
     Alpaca's `get_account()` (already used elsewhere, e.g. Credit/
     Counterparty Risk in `01`) and a minimal Anthropic call (or, cheaper,
     the SDK's key-format validation without a network round-trip, to avoid
     spending real tokens on a health check). A 401/403 here is exactly the
     "silent API key expiry/rejection" failure mode `17` §5 called the
     single most probable real-world failure for this kind of system.
  3. **Billing-adjacent signals**: CLAUDE.md already documents one instance
     of this exact failure class (MiroFish/OpenRouter 402 on low credits) —
     the checker greps `logs/*.log` and `logs/eod_log.jsonl` from the last
     24h for `402`, `401`, `insufficient_quota`, `rate_limit` substrings as a
     cheap secondary signal, since a real billing failure often surfaces in
     application logs before or instead of a clean exception.
- **Trigger**: any failure in (1) or (2) → CRITICAL alert via the same
  `send_direct_alert()` heartbeat already uses (import and reuse, don't
  reimplement). This is intentionally the loudest alert in the whole
  document, because per `17` it's also the most probable one to actually
  fire.
- Runs once daily, scheduled independently of the trading pipeline (its own
  tiny plist, `StartInterval` again, not tied to market hours) — it should
  be able to report "your Alpaca key died" on a Saturday, not wait for
  Monday's rebalance to discover it.

### Backup Custodian (new — no new Python)
- A ~10-line shell script, `scripts/backup_state.sh`, run daily via its own
  `StartInterval` plist: `tar`s `data_cache/*.json` (the small,
  hand-maintained state files — `earned_authority.json`,
  `kill_switch_state.json`, `alert_state.json`, `recovery_checkpoint.json`
  from above), `logs/*.jsonl`, and `rebalance_calendar.csv` into a
  timestamped archive under `~/Backups/ascent-capital/`, then prunes
  anything older than 30 days.
- **Deliberately excludes** the large parquet price caches — those are
  re-fetchable from Yahoo/Polygon/Tiingo and backing up 900MB+ daily on a
  solo operator's laptop is not a good trade of disk and attention for
  something reconstructible. This is the honest, realistic-for-one-person
  scope `20`'s framing calls for: back up the ~10MB of *irreplaceable*
  decision-history state, not the multi-hundred-MB of *replaceable* market
  data.
- **Off-site**: the same script, once local backup succeeds, does one
  `rsync` to a second location (an external drive, a personal cloud-synced
  folder, or — cheapest — a private git remote for just the small JSON/JSONL
  files, since they compress well and a private repo is free). No new
  infrastructure to pay for or maintain; pick whichever the Operator already
  has running for other things, per `18`'s principle that a ritual only
  survives if it rides on something already habitual.
- **Trigger**: if the backup step itself fails 2 days running, that's a
  liveness-style alert too (reuse the same channel, same dedup).

### Daily Digest (the one ritual, per `18`'s design principle)
- `18` §4's core rule: automate *surfacing* information fully, keep the
  moment of human judgment manual. Applied here: a **daily digest is fully
  automated** (it's information, not judgment) — the human gate is that it
  has to land somewhere the Operator will actually see it that day, not
  that they have to write it.
- Extends `heartbeat_check.py`'s existing `--alive-ping` concept into a
  richer one-paragraph daily message (not a new report format, one more
  field on the existing notification): liveness status, credential-check
  result, backup status, and kill-switch/halt state, sent via the same
  ntfy.sh/macOS channel every morning regardless of whether anything is
  wrong — a **positive** "system alive, N/N checks pass" message has real
  value per `18`'s ritual framing: absence of the digest *is itself* the
  signal something broke (including the digest mechanism itself), the same
  meta-lesson this audit's own discovery (watchdog silent since 2026-08-12)
  demonstrates directly.
- **This is the achievable version of a NOC**, explicitly scoped against
  what `17` §2 rightly flagged as unrealistic: no on-call rotation, no SLA,
  no 24/7 coverage. One message a day, cheap to build (reuses existing
  alert plumbing), cheap to maintain (no new dashboard to keep up), and
  actually likely to get read because it's a phone notification, not a file
  the Operator has to remember to open.

## Layer 4 — Data Contracts

| From → To | Contract |
|---|---|
| `heartbeat_check.py` → `logs/liveness.json` | `{status, as_of, last_run, missed_days, missed_day_dates, missed_rebalances, missed_rebalance_dates, generated_at}` — existing, unchanged |
| `run_all_agents.py` (successful exit) → `logs/recovery_checkpoint.json` | `{run_date, last_good_positions: [{symbol, qty, avg_cost}], kill_switch_state_hash, eod_log_line_offset, earned_authority_state_hash, generated_at}` — new |
| `credential_check.py` → `logs/credential_health.json` | `{date, alpaca: {present, reachable}, anthropic: {present, reachable}, billing_signals: [...], generated_at}` — new |
| `backup_state.sh` → `logs/backup_status.jsonl` | one append-only line per run: `{date, local_ok, offsite_ok, archive_path, size_bytes}` — new |
| All four above → `send_alert()` / `send_direct_alert()` | existing contract in `ascent/monitoring/alert_system.py` and `heartbeat_check.py`; new callers, no schema change |
| Daily Digest → ntfy.sh / macOS notification | existing channel, one new message composition point |

## Layer 5 — Concrete Code Mapping

**Ships this week, zero new code:**
1. `launchctl load ~/Library/LaunchAgents/com.ascentcapital.heartbeat.plist`
   and `com.ascentcapital.eod.plist`. Verify via `launchctl list | grep
   ascentcapital` and check `logs/liveness.json`'s `generated_at` advances
   on the expected 6-hour cadence. This alone closes the largest gap this
   audit found — a fully-built, well-designed watchdog that simply is not
   running.
2. Set `NTFY_TOPIC` in `.env` if not already set, so `send_direct_alert()`
   pushes to a phone, not just a local macOS notification that's invisible
   when the laptop is closed.

**New, ~1 evening each:**
3. `scripts/recovery_snapshot.py` (or a function added at the tail of
   `run_all_agents.py`'s success path) — writes `logs/recovery_checkpoint.json`
   per Layer 4, following `kill_switch.py`'s `_load_state()`/`_save_state()`
   pattern exactly (same safe-default-on-missing-file behavior).
4. Add the startup comparison (current Alpaca book vs. `last_good_positions`)
   near `_catch_up_guard()`'s existing call site in `run_all_agents.py`
   (around line 690-716) — same fail-closed idiom, same
   `HALT_STATE_PATH`/`requires_override` write path Reconciliation Analyst
   in `05` already specifies.
5. `scripts/credential_check.py` — new file, imports `APIKeys.from_env()`
   from `ascent/config/settings.py`, adds the presence/liveness/billing-signal
   checks from Layer 3, calls `ascent.monitoring.alert_system.send_alert()`
   on failure (reuse, not reinvent — `"liveness"` is already a
   `VALID_ALERT_TYPES` entry; add `"credential_health"` alongside it).
6. `scripts/backup_state.sh` + `scripts/com.ascentcapital.backup.plist` —
   shell + plist only, no Python. Follows the existing plist pattern
   (`StartInterval`, `RunAtLoad: true`, stdout/stderr redirected to
   `logs/`) already established by the two existing plists.
7. Extend `heartbeat_check.py`'s `_alive_ping_message()` (or a small new
   `daily_digest.py` that imports and calls it) to compose the richer daily
   message from Layer 3, scheduled once daily rather than every 6 hours.

**Explicitly out of scope, and why** (mirroring `20`'s "what's realistic for
one person" discipline):
- No automated failover to a second broker — `17` correctly names this as a
  standing, unmitigated risk, but building and *testing* a second broker
  integration is multi-weekend work with real correctness risk of its own
  (a live-money integration nobody has run before is a worse failure mode
  than a documented single-broker dependency). Log it as a standing finding,
  same treatment `01`'s Credit/Counterparty Risk role already gives it.
- No automated LLM-provider failover (Opus/Sonnet down mid-Phase-2
  synthesis) beyond what already exists (`prethesis=None` graceful
  single-phase fallback). A hard requirement to *retry across providers*
  mid-synthesis is new infrastructure with its own failure modes; the
  existing graceful degradation (proceed without the prethesis rather than
  hang) is the right-sized answer for a system that already tolerates a
  missing Phase 1 pass.
- No incident-severity taxonomy or formal postmortem template — `17` §3
  flags this as missing, correctly, but a solo operator writing a
  four-section postmortem template for themselves is process theater;
  the existing four-part rebalance recap format in CLAUDE.md, extended with
  one more required line ("what broke and what I changed") when an incident
  occurs, does the same job without new ceremony.
