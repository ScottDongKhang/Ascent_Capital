# Alpha / Sharpe / AI-Native / Drawdown — Improvement Spec (2026-06-10)

Companion to `2026-06-10-ai-pm-alpha-audit.md` (the audit fixes are prerequisite work, being
implemented separately). This spec selects THREE workstreams chosen to cover all four goals:

| Workstream | Alpha | Sharpe | Drawdown | AI-native |
|---|---|---|---|---|
| A. Vol targeting + research/production parity | ✅ (removes drag) | ✅✅ | ✅✅ | – |
| B. Falsifier enforcement layer | ✅ | ✅ | ✅✅ | ✅✅ |
| C. Risk-aware construction bundle | ✅ (alpha retention) | ✅✅ | ✅ | – |

**Sequencing constraint:** Workstream B touches `run_all_agents.py` in the same regions as the
audit fixes. Land the audit PR first; A and C are disjoint (main.py, optimizer.py, ingest) and
can proceed in parallel.

---

## Workstream A — Volatility targeting in production + research/production parity

Severity of underlying gap: HIGH (newly discovered during this spec's verification)

### Problem

The honest WF OOS baseline (Sharpe 0.483 — the number every decision is benchmarked against)
is produced by `ascent/research/wf_framework/ascent_strategy.py`, which applies **two** exposure
overlays at every rebalance:

1. `_apply_200ma_overlay()` (line ~237): SPY < 200MA → ×0.70, **unconditional** (no VIX check).
2. `_apply_vol_target()` (line ~271): scale weights by `clip(0.15 / realized_spy_vol_21d, 0.25, 1.0)`.

Production `ascent/main.py:705-733` applies **only** the 200MA overlay, and with a VIX>20
confirmation gate the research version doesn't have. Consequences:

- The live book runs **unscaled gross in high-vol environments** that the validated baseline
  would have run at as little as 25% exposure. Live drawdowns are structurally larger than
  anything the WF record predicts.
- The WF record's April-style behavior (MA-only cuts) doesn't match live behavior (VIX-gated
  cuts), so `monitoring/live_vs_backtest` is comparing two different strategies and live-gap
  attribution is unreliable.
- Vol targeting on momentum portfolios is one of the most robust Sharpe improvements in the
  literature precisely because momentum's worst losses cluster in high-vol states; the research
  side already validated it — production just never received it.

### Fix

1. **Port `_apply_vol_target` to production.** In `ascent/main.py`, immediately after the SPY
   200MA filter block (after line ~733), add the same scaling applied to `target_weights`:

   ```python
   # Vol targeting — mirrors wf_framework/ascent_strategy._apply_vol_target
   spy_rets = spy_close.pct_change().dropna()
   realized = spy_rets.rolling(21).std() * np.sqrt(252)
   scale = (TARGET_VOL / realized).clip(VOL_FLOOR, VOL_CAP)        # 0.15 / vol, clip(0.25, 1.0)
   scale_aligned = scale.reindex(target_weights.index, method="ffill").fillna(1.0)
   target_weights = target_weights.mul(scale_aligned, axis=0)
   ```

   Constants from config (`BacktestConfig`): `target_vol=0.15`, `vol_floor=0.25`, `vol_cap=1.0` —
   add to `ascent/config/settings.py` so self_improve can perturb them within safe bounds.
   Un-invested weight is implicit cash (Alpaca holds it; no BIL purchase needed at this stage).

2. **Backport the VIX confirmation into research.** In
   `wf_framework/ascent_strategy._apply_200ma_overlay()`, add the same `VIX > VIX_STRESSED_CONFIRMATION`
   gate production uses (`ascent/main.py:711-722`, threshold imported from `ascent/regime/engine`).
   VIX history is available in the macro cache; where missing, fall back to MA-only and log.

3. **Re-run the full WF baseline** after 1+2 so research and production are the same strategy.
   Expect the headline numbers to move; the new baseline is the honest one. Update CLAUDE.md
   "Current state".

4. **Add a parity test**: a fixture portfolio + price path pushed through both
   `main.py`'s overlay+vol-target code and `ascent_strategy`'s — final weights must match to
   1e-6. This prevents the next silent divergence.

### Files
`ascent/main.py` (overlay block), `ascent/config/settings.py`,
`ascent/research/wf_framework/ascent_strategy.py:_apply_200ma_overlay()`,
`tests/` (new parity test).

### Success metric
- Parity test green.
- Re-run WF baseline; expect Sharpe ≥ 0.483 with max-drawdown reduced (vol targeting's typical
  effect: −15-30% MDD, flat-to-positive CAGR on momentum books).
- Next live high-vol episode (VIX>25): run log shows gross scaled below 1.0.

### Estimated impact
Sharpe +0.05–0.12 and the largest available drawdown reduction of any single change; also makes
every future live-vs-backtest comparison meaningful. The April drag specifically: VIX-gated MA
cut (already live) + continuous vol scaling replaces the binary 30% cliff that caused it.

---

## Workstream B — Falsifier enforcement layer (the AI layer acts daily)

### Problem

The system *collects* falsifiable conditions everywhere and *acts on none of them*:

- `check_early_exits()` (`ascent/causal/tracker.py:128-180`) runs daily as "Gate 4"
  (`run_all_agents.py:1086-1104`) and flags broken causal mechanisms — its only consequence is
  **a line appended to a shadow log**. The position stays at full weight for up to 10 more days.
- Prethesis `what_would_change_my_mind` (per conviction name) — written, sealed, never checked.
- Judge `prediction` ("X underperforms Y over 10 days") — scored after the fact by
  `adversarial_authority`, but the *position* is never revisited when the prediction is failing.
- AI PM thesis `pre_mortem` — written before submission, never monitored.

Between rebalances (10 business days) the fund is unsupervised except for the drawdown kill
switch. This is the gap between "AI that writes memos" and "AI that manages a book."

### Fix

**1. Unified falsifier registry.** New module `ascent/strategy/falsifier_registry.py`.
On each rebalance day (after the AI PM result is final), build
`data_cache/active_falsifiers.json`:

```json
{"as_of": "2026-06-10", "falsifiers": [
  {"id": "...", "symbol": "PK", "source": "prethesis|causal|judge|pre_mortem",
   "kind": "price|macro|news",
   "condition": {"metric": "ret_since_rebalance", "op": "<", "value": -0.07},
   "raw_text": "original falsifier sentence", "expires": "2026-06-24"}
]}
```

- `causal` entries come straight from the predictions log (thresholds already numeric:
  −5% catalyst_imminent / −8% not_yet_priced — reuse `check_early_exits` logic).
- `judge` entries: parse the 10-day prediction's subject symbol; condition = position return
  < benchmark return − 3pp since intervention.
- `prethesis`/`pre_mortem` entries: ONE Haiku call at rebalance converts each
  `what_would_change_my_mind` sentence into either a structured price/macro condition
  (`kind: price|macro`) or a news watch (`kind: news`) with 2-3 keywords. Unparseable → stored
  as `kind: news` with the raw text.

**2. Daily evaluation.** In the non-rebalance branch of `run_all_agents.py` (replacing the
current Gate 4 shadow-write), call `falsifier_registry.check_all(today)`:

- `price`/`macro` conditions evaluated **in code** from `prices_live.parquet` / `macro_live`
  (no LLM, no hallucination surface).
- `news` conditions: one Haiku call total per day, given the day's Exa headlines
  (`_news_context`, already fetched daily) + the watch list → returns fired/not per falsifier.

**3. Bounded action.** A fired falsifier triggers an intra-period trim through the **existing**
mini-rebalance machinery (`_trigger_mini_rebalance`, cooldown via
`_check_mini_rebalance_cooldown`, `run_all_agents.py:1985-2070`), generalized to accept a
trim instruction, not just a discovery buy:

- Action: reduce the position 25% (floor 4%, matching the REDUCE protocol in the AI PM system
  prompt). Never adds, never exits fully, never touches other positions except renormalization.
- Authority-bounded: at AI PM Level 1, max ONE falsifier trim per 5 trading days (share the
  mini-rebalance cooldown); Level 2+ may take two. Suspended if the falsifier-trim 10-day win
  rate drops below 40% (reuse `adversarial_authority` per-type suspension).
- Every trim logged as an intervention with the fired falsifier's `raw_text` and scored at 10
  days exactly like judge interventions; outcomes feed the post-mortem (audit finding 5) so the
  AI PM learns which of its own falsifiers are informative.
- Debate gate NOT required for a 25% trim (it is risk-reducing and bounded); it IS required if
  the same symbol fires twice in one holding period (escalation → full exit proposal).

**4. Kill the dead code.** Remove the shadow-log write at `run_all_agents.py:1092-1102`
(it pollutes `ai_pm_shadow_returns.jsonl` with a different schema than the authority writer uses
— two record types in one file today).

### Files
`ascent/strategy/falsifier_registry.py` (new), `ascent/causal/tracker.py` (expose threshold
logic), `run_all_agents.py` (Gate 4 replacement, mini-rebalance generalization, registry build
on rebalance), `debate/adversarial_authority.py` (new intervention type `falsifier_trim`),
`tests/strategy/test_falsifier_registry.py` (new).

### Success metric
- Registry file exists after next rebalance with ≥1 entry per source type.
- Unit tests: numeric falsifier fires exactly at threshold; news falsifier requires Haiku "fired";
  cooldown enforced; floor respected.
- Within 3 holding periods: ≥1 falsifier trim executed and scored; trims' 10-day alpha vs
  holding unchanged ≥ 0 on average.

### Estimated impact
Primary drawdown tool at the position level: the SATS-type pattern (parabolic name with a −0.70
transcript signal, flagged in the May 27 verdict, then held untouched for 10 days) gets caught
mid-period. Converts ~4 existing write-only artifacts into supervised risk management. This is
also the highest-leverage "AI-native" change available: the AI layer's *judgment* (its own
falsifiers) gains a bounded, scored, earnable lever on the live book every day, not every 10 days.

---

## Workstream C — Risk-aware construction bundle

### Problem

Construction is rank-weighted with a 10% name cap; risk is invisible to it three ways:

1. **No per-name risk sizing.** A 10% slot in WDC (+928% momentum, verdict-flagged) carries
   several times the daily risk of a 10% slot in WMT. The cap constrains *weight*, not *risk*.
2. **Hidden cluster concentration.** Two of three verdicts flagged the same structure (EM
   cluster EWY/EWT/EEM ~17.5% behaving as one factor; earlier the commodity cluster). Sector
   caps miss it because the members carry different sector labels.
3. **21.2% of the live book has no sector label** (verified: VVV, WDC, YETI missing from
   `profiles.parquet`), so the sector-cap constraint and stat-arb sleeve silently degrade —
   and integrity constraint #4's "<80% coverage → skip caps" fallback is closer than it looks.

### Fix

**C1. Inverse-vol tilt (half-strength) in `sector_constrained_weighted()`**
(`ascent/portfolio/optimizer.py:447`). Before the rank-weight/`_water_fill_cap` step, tilt
scores by relative vol:

```python
vol = returns_63d.std() * np.sqrt(252)            # per candidate symbol
tilt = (vol.median() / vol).clip(0.5, 2.0) ** 0.5  # half-tilt: don't fight the momentum alpha
scores = scores * tilt.reindex(scores.index).fillna(1.0)
```

Half-exponent deliberately preserves momentum ranking while shaving the hottest names; the
existing cap/redistribution machinery is untouched. Returns matrix is already available at the
call site in `main.py`. Add `inverse_vol_tilt: bool` to config (so self_improve can A/B it) and
apply the same tilt in `wf_framework/ascent_strategy.py` (parity, per Workstream A).

**C2. Correlation-cluster cap.** New function `enforce_cluster_cap(weights, returns, max_cluster=0.20)`
in `ascent/portfolio/optimizer.py`, called after `sector_constrained_weighted` and before the
hedge overlay:

- Compute `correlation_matrix(returns, window=63)` (exists, `ascent/risk/covariance.py:40`).
- Single-linkage cluster at corr > 0.70; any cluster of ≥2 names with combined weight > 20% is
  scaled down pro-rata to 20%, freed weight redistributed to non-cluster names via the existing
  redistribution pattern. Log the cluster membership and trim.
- Post-condition: no cluster > 20%, weights sum to 1, name cap still holds (re-run
  `_water_fill_cap` if redistribution breached it).

This institutionalizes what the bear agent keeps catching after the fact — at construction
time, before the debate ever sees it.

**C3. Profiles backfill + coverage guard.**
- One-time: fetch sector/industry for all universe symbols missing from `profiles.parquet`
  via `openbb_client.fetch_symbol` metadata / `yf.Ticker(sym).info` (`sector`, `industry`),
  append to the parquet. Immediate effect: VVV, WDC, YETI labeled; book coverage 78.8% → ~100%.
- Permanent: in the universe/ingest path, after any universe change, auto-backfill missing
  profiles (best-effort, never raises) and print a warning whenever live-book coverage < 90%.
  This protects integrity constraint #4 from silently re-degrading.

### Files
`ascent/portfolio/optimizer.py` (`sector_constrained_weighted`, new `enforce_cluster_cap`),
`ascent/main.py` (call site + returns pass-through), `ascent/config/settings.py`,
`ascent/research/wf_framework/ascent_strategy.py` (parity),
`ascent/data/` ingest (profiles backfill), `tests/portfolio/` (cluster cap post-conditions,
tilt preserves cap + sum-to-1).

### Success metric
- WF re-run with C1+C2 on: Sharpe up or flat with MDD down vs the Workstream-A baseline
  (gate: don't ship if Sharpe drops > 0.02).
- Next rebalance log shows cluster report; EM-style cluster ≤ 20%.
- `profiles.parquet` covers 100% of the live book; "unknown sector" disappears from verdicts.

### Estimated impact
C1: classic risk-adjusted improvement, expected +0.03–0.08 Sharpe via vol-drag reduction in the
top slots. C2: caps the exact tail the devil's advocate keeps quantifying (−5 to −7% correlated
liquidation scenarios) — drawdown insurance at near-zero alpha cost. C3: free; restores two
existing mechanisms (sector caps, stat-arb) to full strength.

---

## Explicitly deferred (and why)

- **Long-short / event trading / self-modify** — kill-switch gated until paper validation
  (~July 2026); none address the current bottlenecks.
- **Fundamental sleeve re-enable** — IC-t = −4.75 stands.
- **New data integrations** — the audit showed decisions don't survive the pipeline; signal
  supply is not the constraint.
- **AI PM capital allocation across agents / regime-call blending** — high value but extends AI
  authority; wait until the audit fixes give Track D an honest record (revisit ~2 rebalances
  after audit-fix deployment).

## Suggested order

1. **C3** (one session, pure data fix, immediate risk-visibility win)
2. **A** (one session + WF re-run; re-baselines everything honestly)
3. **C1+C2** (one session, gated on the re-baselined WF run)
4. **B** (after the audit PR lands — shares `run_all_agents.py` real estate and depends on the
   fixed learning loop for outcome scoring)
