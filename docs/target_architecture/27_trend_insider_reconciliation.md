# Trend & Insider Reconciliation — Correcting the Earlier Recommendation

`23_empirical_alpha_audit.md` recommended researching the dormant `trend`
and `insider` sleeves as the next priority, based on a recovered
`sleeve_ic_log.jsonl` slice ("Regime A," 21 dates) where both measured
better than the live `meanrev`/`statarb` pair. A follow-up audit found this
recommendation was built on a statistical illusion for `trend`, and an
unresolved-but-not-reversed case for `insider`. This document is the
correction.

## The contradiction

- **Recovered log (Regime A, 2026-05-05→06-05)**: trend IC = +0.01444 (IR
  14.07), insider IC = +0.00455 (IR 5.24) — both beating meanrev/statarb.
- **Formal proof audit** (`outputs/analyst/proof_audit_2026-08-13.json`,
  the actual governance gate that originally cut both sleeves, per commit
  `d14ae24`): trend IC = **-0.01173, p=0.0249, n=1636** — a statistically
  significant *anti-signal*, correctly cut. Insider IC = +0.00700, p=0.1826,
  n=615 — positive but not significant, correctly cut for insufficient
  evidence.

Both analyses cover the same ~6.5-year window (Regime A sits entirely
inside the proof audit's 2020-2026 range) — this rules out "different eras"
as an explanation. The discrepancy is methodological.

## Root cause, verified

1. **Different forward-return horizon.** `ascent/main.py::_log_sleeve_ic`
   (the log) correlates against a **21-trading-day** forward return.
   `ascent/analyst/proof_audit/forward_returns.py` (the proof audit) uses a
   **1-trading-day** forward return. Momentum/trend signals are the
   textbook case where intermediate-horizon momentum can be genuinely
   positive while the identical signal shows short-horizon reversal —
   exactly the pattern found here.
2. **Different correlation statistic.** The log uses whole-panel Pearson
   correlation; the proof audit uses Spearman rank-IC on a top/bottom
   quintile long-short construction, more sensitive to tail behavior.
3. **The log's "21 daily observations" are not 21 independent
   measurements.** Verified directly: within Regime A, sample size `n`
   increases by exactly 1 every date, and `mean_ic` drifts by tiny, smooth
   increments — the signature of an **expanding-window cumulative average**
   re-run from scratch each day over ~1500+ dates of history, not a fresh
   single-day IC. The headline Regime-A numbers (IR 14.07 for trend) are a
   ~1500-day running average sampled 21 times near its tail, dressed up as
   21 independent observations. The true effective sample size is closer
   to 1 than 21 — the apparent IR is a statistical illusion of the logging
   mechanism, not evidence of a stable, high-Sharpe signal. `23`'s own
   decay-analysis caveat about this ("as consistent with cumulative-average
   convergence as with genuine decay") was correct and is confirmed here to
   extend to the ranking finding itself.

No Regime-A-specific data corruption (analogous to the winsorize bug that
corrupted the log's Regime B) was found — Regime A's numbers are internally
consistent, they just answer a different, less rigorous question than the
proof audit does.

## Verdict

**Trend: drop.** Not ambiguous. The proof audit is the correct authority —
larger effective sample, standard methodology, and it's the actual
governance gate. It found a statistically significant anti-signal. The
log's positive number is real but measures something else (21-day-horizon
Pearson correlation, inflated by an expanding-window artifact) and has
never been tested with proof-audit rigor. If the 21-day-horizon momentum
hypothesis is worth pursuing, it needs a real re-run inside the proof-audit
harness with the horizon changed and genuine per-date independence — not
inference from this log.

**Insider: unresolved, not promoted, not killed.** The two analyses don't
actually disagree on sign — both find a positive point estimate. The proof
audit's issue is significance (p=0.18 on n=615), not direction. The log's
Regime A result adds no independent corroborating weight (same
expanding-window problem, and its higher apparent n is likely partly a
density-filter leniency artifact rather than genuinely more valid data).
Correct action: leave dormant as-is; "more insider_transactions density →
rerun proof audit" is the real path to a verdict, not this log.

## Correction to prior documents

- `23_empirical_alpha_audit.md` §3's ranking claim ("trend/insider measure
  better than live sleeves") is superseded for `trend` specifically — read
  that section's trend finding as methodologically confounded, not as
  standing evidence for promotion.
- `25_ic_memo_alpha_sleeve_review.md` and the README's "highest-priority
  next step" language, which pointed toward "the sleeves that already
  exist, are zero-weighted, and measure better" as the next research
  target, should be read narrowed to: **insider only**, and even there,
  only as "collect more data and rerun the formal audit," not "promote
  based on existing evidence." The core recommendation in those documents
  — pause and re-underwrite the current meanrev/statarb weighting rather
  than continuing to build governance infrastructure around it — is
  unaffected and still stands on its own (beta-hedged Sharpe ≈ -0.10 for
  the live pair, independent of anything about trend/insider).
