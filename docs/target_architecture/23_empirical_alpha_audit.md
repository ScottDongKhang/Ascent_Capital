# Empirical Alpha Audit — Real Numbers, Not Estimates

`21_alpha_edge_audit.md` recommended computing real per-sleeve IC as the
first, highest-priority triage step. This document does that computation,
using historical `sleeve_ic_log.jsonl` data recovered from three backup
copies (the live file no longer exists in `logs/` — see §4 for why, which
is a real but different finding than "the gate is broken").

## 1. Source reconciliation

Three backups exist: `logs.bak-2026-06-19-rerun/sleeve_ic_log.jsonl` (a
strict subset of the second file, dropped), `logs/sleeve_ic_log.
pre_winsorize_fix.20260625.bak.jsonl`, and `logs/.pre_delete_backup_2026-
08-14/sleeve_ic_log.jsonl`. Deduped and merged on date (matching how
`_get_gated_weights()` itself reads the log), this gives a **canonical
38-date series, 2026-05-05 → 2026-07-30** — but it is **not methodologically
uniform**. It breaks into four regimes:

- **Regime A (2026-05-05 → 2026-06-05, 21 dates)**: stable, large-n (~1560-
  1590), sane t-stats. **The only trustworthy multi-observation slice.**
- **Regime B (2026-06-08 → 2026-06-24, 12 dates)**: `n` abruptly doubles,
  `ic_t` explodes to ±30-55 — a computation artifact (unwinsorized outliers
  or duplicate-counted symbols), exactly the bug the filename
  `pre_winsorize_fix` documents. **Excluded from headline ranking.**
- **Regime C (2026-06-25 → 2026-07-27, 4 dates)**: post-fix but a much
  smaller (n=32-50), incomparable universe.
- **Regime D (2026-07-30, 1 date)**: n returns to ~1600s with sane
  statistics — a single point, but its sleeve ranking independently
  reproduces Regime A's.

## 2. Computed statistics (Regime A, n=21 daily observations per sleeve)

| Sleeve | Mean IC | IC std dev | IC "IR" (mean/std) | Mean `ic_t` | Max n | Live weight |
|---|---:|---:|---:|---:|---:|---|
| **trend** | **+0.01444** | 0.00103 | **14.07** | 3.47 | 1592 | 0% (dormant) |
| **insider** | **+0.00455** | 0.00087 | **5.24** | 2.35 | 1069 | 0% (dormant) |
| meanrev | +0.00089 | 0.00044 | 2.04 | 0.26 | 1588 | **50% (live)** |
| statarb | -0.00113 | 0.00059 | -1.90 | -0.32 | 1582 | **50% (live)** |
| fundamental | -0.01119 | 0.00712 | -1.57 | -4.66 | 1512 | 0% (disabled) |

Regime D (07-30, single point) reproduces the same ordering: trend
(+0.0058) > insider (+0.0049) > meanrev (+0.0045) > statarb (+0.0023) —
independent cross-validation of the ranking, not just Regime A alone.

## 3. Verdict: is meanrev/statarb the best pair?

**No.** In every clean slice of this data, **the two dormant, zero-weighted
sleeves (trend, insider) measure with higher mean IC and dramatically
higher IC information ratio than either currently-live sleeve.** Trend's
IR (14.1) is 7x statarb's magnitude; insider's IR (5.2) is over 2x
meanrev's. Fundamental is worse than both live sleeves and correctly
disabled per CLAUDE.md constraint #7 — but by the same logic that disabled
fundamental, meanrev and statarb are the **weakest of the surviving,
well-measured sleeves**, not a considered top pair.

**Decay within Regime A**: meanrev IC declines from +0.00167 (05-05) to
+0.00008 (06-05), r=-0.964 with time; statarb declines from -0.00002 to
-0.00216, r=-0.971. Trend improves over the same window (r=+0.982) —
direct contrast. **Caveat**: `n` grows ~1/day across this window,
consistent with an expanding (not rolling) IC window — a near-perfect
r≈-0.97 linear decline over 21 days is as consistent with "cumulative
average converging to a lower true mean" as with genuine day-to-day decay.
This is a real but qualified finding, not a confirmed one. Regime C (late
June/July, different methodology) independently shows meanrev deeply
negative (-0.035, -0.022, -0.032) — directionally consistent with
continued deterioration even on different data.

**Gate simulation**: running `_get_gated_weights()`'s exact logic (5-date
rolling window, -0.005 threshold) against the merged series at the last
available point: **meanrev's rolling-5 mean IC = -0.01403 — would have
been gated to zero.** Statarb's = +0.02566 — would have survived. Had the
log kept accumulating and the gate stayed live, **meanrev, currently half
the live book, would already have been mechanically zeroed weeks ago.**

## 4. Why the live log is missing — corrected finding

An earlier pass in this project characterized the missing
`logs/sleeve_ic_log.jsonl` as evidence the IC-gate mechanism is "currently
inert" — true in effect, but the root cause is **not a bug**. Direct
investigation found:

- `ascent/main.py:63` defines `_log_sleeve_ic()`, called unconditionally
  at `ascent/main.py:633` on every live run reaching that point. The write
  path (`ascent/main.py:158-162`) is intact; no exception is silently
  swallowing writes.
- The file was **deliberately deleted on 2026-08-14** (per `docs/
  superpowers/plans/2026-08-14-alpha-weight-override-fix.md`, backed up
  first to the exact `.pre_delete_backup_2026-08-14` copy this audit used)
  because its contents referenced stale sleeve names from a defunct
  15-sleeve architecture that were corrupting `_get_gated_weights()`'s
  redistribution logic. The actual redistribution bug was fixed for real
  in commits `31d49ee` and `889c600`. The plan's own text states
  rebuilding the log with fresh content was explicitly out of scope — it
  was meant to regenerate organically on the next live run.
- **Live trading has simply been paused since 2026-07-27** — `logs/
  eod_log.jsonl`'s last entry, `logs/liveness.json` reporting `"status":
  "CRITICAL"` with 12 missed days as of 2026-08-12, consistent with the
  project's own prior documented decision to hold live trading after the
  walk-forward blocker cleared. `_log_sleeve_ic()` has had zero
  opportunities to fire since the deletion, independent of any code issue.

**Corrected finding**: the IC gate is not broken. It will resume
functioning automatically the moment live trading resumes, no code change
needed. The data in §2-3 above reflects the **last real measurement period
before trading was paused** — it is not stale in the sense of "outdated
by newer, better data superseding it," it's the most recent real evidence
that exists, and it says the live sleeves were the weakest, declining
performers among the measured set at the time trading stopped.

## 5. Bottom line

Given this data: (a) meanrev and statarb are not empirically the best
available sleeves — dormant trend and insider measure better on every
axis; (b) both live sleeves show real IC erosion toward/through zero
within the one clean month of data available; (c) meanrev specifically
would already be mechanically gated to zero under the system's own rule,
which will resume enforcing this the moment trading resumes; (d) the
50/50 weighting has no empirical support in this data — if forced to
choose only between the two live sleeves, meanrev (positive, declining)
dominates statarb (negative) in Regime A, arguing against an even split.
This does not resolve the deeper "is this beta or edge" question — see
`24_beta_decomposition_analysis.md` for that — but it independently
confirms the alpha audit's core concern with real, computed numbers.
