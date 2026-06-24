# Reconciled Plan — Text Signal Alpha (eng-review output, 2026-06-23)

> Supersedes the pasted "Build Prompt: Text Signal Alpha Layer." That prompt was
> written against a mental model of the repo that does not match the code. This
> doc is the reconciled, build-actionable version produced by `/plan-eng-review`.
> Workflow per CLAUDE.md: Opus spec → Sonnet implements.

## Decision (Step 0 scope challenge)

**Reconcile to existing infra.** Do NOT build `SECEdgarFetcher`, the three
class-based sleeves, the `DEFAULT_ALPHA_WEIGHTS` overwrite, or the Phase 5 macro
scalar. The repo already has the data layer, the LLM classification, the offline
panel job, and the exposure overlays. The one genuine gap is that the
already-computed **earnings-call transcript tone panel is not wired into the
alpha stack.** Wire that. Defer the rest.

## What already exists (and why the build prompt was rebuilding it)

| Build-prompt component | Already in repo | Status |
|---|---|---|
| `SECEdgarFetcher.get_latest_earnings_call` | `ascent/data/ingest/earnings_transcripts.py` — EDGAR FTS + correct `User-Agent`, retries, `fetch_recent_8k_transcripts` | exists |
| Earnings-tone LLM scoring | `classify_transcript_signal()` + `_compute_combined_signal()` in same file | exists |
| Offline panel build (1-BDay lag, 63d ffill) | `build_transcript_signal_panel()` / `update_transcript_signals()`; run weekly by `ascent/monitoring/weekend_runner.py:146` | exists |
| `get_latest_mda` / MD&A extraction | `ascent/data/ingest/sec_filings.py` — `extract_mda_section()`, `extract_risk_factors_section()` | exists (no offline panel yet) |
| Phase 5 macro exposure scalar | `ascent/portfolio/exposure.py` — `apply_exposure_overlays()` (VIX-confirmed 200MA cut + vol-target), shared with WF framework | exists |
| FRED fetch (prompt said read `ascent/data/fetch/macro.py`) | `ascent/data/ingest/fred.py` — `fetch_all_macro()` | exists (prompt path wrong) |
| Sleeve interface `class.compute(... ) -> Series` | Real sleeves are module fns `xxx_alpha(features: dict) -> DataFrame` wired inline in `ascent/alpha/stack.py` | prompt interface is fictional |
| `_SPARSE_FILL_ZERO` in `stack.py` | It lives in `ascent/alpha/ml_sleeve.py:291` and is an ML feature-fill set, not a sleeve mechanism | prompt wrong file |

## The change (minimal, honest diff)

Wire `load_transcript_signals()` into the stack as one new module-function
sleeve, `earnings_tone`, seeded at IC-gate floor weight, **reading parquet only —
no network/LLM call inside `build_alpha_stack`.**

```
OFFLINE (already runs weekly, unchanged)
  weekend_runner ──▶ fetch_recent_8k_transcripts ──▶ classify_transcript_signal
                          │
                          ▼
              update_transcript_signals  ──▶  data_cache transcript panel (parquet)
                                                 dates × symbols, 1-BDay lag, ffill 63d
                                                            │
HOT PATH (this change)                                      │ load_parquet only
  build_alpha_stack(features) ──▶ earnings_tone_alpha(features) ◀── load_transcript_signals()
                          │            (reindex panel → features grid; empty → skip, NOT zero)
                          ▼
                   _cs_normalize + weighted blend (renormalized over loaded sleeves)
                          │
                   IC gate (_get_gated_weights) can zero it if rolling mean_ic < -0.005
```

### Files touched (4 code + 2 test)
1. **`ascent/alpha/earnings_tone.py`** (new, ~30 lines) — `earnings_tone_alpha(features: dict) -> pd.DataFrame`. Calls `load_transcript_signals()`, reindexes the panel onto the price grid (`features["close"].index/columns`), returns the date×symbol DataFrame. Returns empty `DataFrame()` on absent cache or any exception (logs `debug`/`warning`). No LLM, no network.
2. **`ascent/alpha/stack.py`** — add a `try/except` block mirroring the `narrative` block (`stack.py:391-415`): import `earnings_tone_alpha`, assign `alphas["earnings_tone"]` when non-empty. Add `"earnings_tone": 0.02` to `DEFAULT_ALPHA_WEIGHTS`, taken from `trend` (0.43 → 0.41). Regime variants inherit via `**DEFAULT_ALPHA_WEIGHTS` spread — no per-regime edits needed.
3. **`ascent/research/self_improve.py`** — add `"earnings_tone": 0.02` to its `DEFAULT_ALPHA_WEIGHTS` (integrity constraint #6). **No** `MIN_SLEEVE_WEIGHTS` floor — an unvalidated sleeve must be prunable (same posture as `fundamental`).
4. **`ascent/main.py`** — register `earnings_tone` in `_log_sleeve_ic`'s `sleeve_builders` (one block, mirroring `short_interest`). **Without this the IC gate cannot see the sleeve at any weight** — the spec's "IC-gated" premise was otherwise false. (Pre-existing gap, NOT fixed here: `narrative`, `llm_fundamental`, `volatility`, `ml` are also live-weighted but absent from `_log_sleeve_ic` → flag as separate PR.)
5. *(no `ml_sleeve.py` change — Phase 4 deferred, so `_SPARSE_FILL_ZERO` is untouched.)*
6. **`tests/test_earnings_tone.py`** (new)
7. **`tests/test_alpha_stack_weights.py`** (new)

## Review sections

### 1. Architecture — 1 decision, resolved
- **Weight seed.** New unvalidated LLM-derived sleeve. Evidence base is hostile: `fundamental` disabled (IC-t −4.75), `llm_fundamental` at 0.03, `narrative` at 0.03, both zeroed in the clean WF run. **Resolution: seed `earnings_tone` at 0.02, no floor, freed from `trend`.** The IC gate (`-0.005` threshold) and meta-learner promote/demote it on live evidence. This is the system's own promotion mechanism — use it, don't front-run it with 0.20. *(Maps to: explicit > clever; let the existing gate decide.)*
- Failure scenario: transcript parquet missing/corrupt → `load_transcript_signals()` returns empty → sleeve absent from `loaded`, visible in the `[alpha_stack] loaded=… skipped=…` print. Not a silent zero. Covered by error handling + test.

### 2. Code quality — 1 finding, resolved
- DRY: do **not** re-extract Q&A / re-classify in the sleeve. The panel is already classified upstream. The sleeve is a thin loader + reindex, structurally identical to the `narrative` block. Reuse `_cs_normalize` from `stack.py`. No new normalization code.

### 3. Test review — coverage diagram

```
CODE PATHS                                          TESTS
[+] ascent/alpha/earnings_tone.py
  └── earnings_tone_alpha(features)
      ├── [GAP→fill] panel present → DataFrame on price grid   test_returns_dataframe_on_grid
      ├── [GAP→fill] cache absent → empty DF (NOT zeros)        test_absent_cache_returns_empty
      ├── [GAP→fill] loader raises → empty DF, no propagate     test_loader_exception_degrades
      └── [GAP→fill] no LLM/network call in hot path            test_no_network (monkeypatch loader)
[~] ascent/alpha/stack.py (modified)
  └── build_alpha_stack: earnings_tone block
      ├── [GAP→fill] empty sleeve → skipped, blend renormalizes test_stack_skips_empty_earnings_tone
      └── [→EVAL] classify_transcript_signal prompt (upstream, unchanged this PR — no eval needed)
[~] weights
      ├── [GAP→fill] DEFAULT_ALPHA_WEIGHTS sums to 1.0          test_default_weights_sum_to_one
      ├── [GAP→fill] each regime variant sums ~1.0              test_regime_variants_sum_to_one
      └── [GAP→fill] stack vs self_improve dicts share key      test_self_improve_has_earnings_tone

COVERAGE TARGET: 7/7 new paths tested (100%)
```
No `[→EVAL]` needed: the prompt template (`classify_transcript_signal`) is
unchanged by this PR.

### 4. Performance — no findings
Hot path adds one parquet read + one reindex per build. O(dates×symbols),
negligible vs ML/CPCV. The build prompt's thousands-of-LLM-calls-per-fold risk
is eliminated by construction (offline panel, parquet read only). _No new tasks._

## NOT in scope (deferred, with rationale)
- **MD&A sentiment-shift sleeve** — infra half-exists (`sec_filings.extract_mda_section`) but there is **no offline panel job** like transcripts have. Needs a new weekly ingest + classification + cache before it can be a parquet-read sleeve. Real work; defer to its own PR. → TODO.
- **Job-posting / headcount sleeve** — no infra, noisiest signal, weakest evidence base. Defer. → TODO.
- **Phase 4 ML text features** — depends on stable text panels existing first; reopens the `_SPARSE_FILL_ZERO` / `feature_names` cache-shape risk (CLAUDE.md gotcha). Defer until `earnings_tone` clears the IC gate. → TODO.
- **Phase 5 macro exposure scalar** — `apply_exposure_overlays()` already does VIX/200MA/vol-target, shared with the WF path. A third cut would double-count macro risk and create a live/backtest divergence. Dropped, not deferred.

## Failure modes (new codepath)
| Failure | Test? | Error handling? | User-visible? |
|---|---|---|---|
| transcript parquet absent | yes | empty DF → skip | yes (`skipped=` print) — not silent |
| parquet corrupt / read raises | yes | caught → empty DF | yes (warning log) |
| panel symbols ∉ price universe | yes (reindex) | reindex drops | n/a (clean) |
| panel stale (>63d post-earnings) | n/a | ffill expires → flat → 0 contribution | n/a (by design) |
No critical gaps (no path is both untested and silent).

## Implementation Tasks
Synthesized from this review's findings.

- [ ] **T1 (P1, human: ~1h / CC: ~10min)** — alpha/earnings_tone — Add `earnings_tone_alpha(features)` thin loader sleeve
  - Surfaced by: Architecture/Code-quality — wire existing `load_transcript_signals()` panel; parquet-only, empty→skip
  - Files: `ascent/alpha/earnings_tone.py` (new)
  - Verify: `pytest tests/test_earnings_tone.py -v`
- [ ] **T2 (P1, human: ~30min / CC: ~5min)** — alpha/stack — Register sleeve + seed weight 0.02 from trend
  - Surfaced by: Architecture — mirror `narrative` block at `stack.py:391`; `trend` 0.43→0.41
  - Files: `ascent/alpha/stack.py`
  - Verify: `pytest tests/test_alpha_stack_weights.py -v`; run `python ascent/main.py` end-to-end
- [ ] **T3 (P2, human: ~15min / CC: ~3min)** — research/self_improve — Add matching weight key (constraint #6), no floor
  - Surfaced by: integrity constraint #6 — weight dict must match `stack.py`
  - Files: `ascent/research/self_improve.py`
  - Verify: `test_self_improve_has_earnings_tone`
- [ ] **T4 (P1, human: ~1h / CC: ~10min)** — tests — Cover all 7 new paths in the diagram
  - Surfaced by: Test review — 100% of new paths
  - Files: `tests/test_earnings_tone.py`, `tests/test_alpha_stack_weights.py`
  - Verify: full suite green + `python ascent/main.py` no crash

## Parallelization
Sequential implementation, no parallelization opportunity (T1→T2→T3 share the
alpha stack; T4 follows). One lane.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | SCOPE_REDUCED | 2 issues, 0 critical gaps; build rebuilt 6 existing modules + fictional sleeve interface — reconciled to 1 IC-gated sleeve |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **VERDICT:** ENG CLEARED (scope reduced) — ready to implement T1–T4. Outside voice skipped (codex not installed; subagent suppressed per session no-spawn policy — informational only, does not gate).

NO UNRESOLVED DECISIONS
