# Session Log Archive — Ascent Capital

Full session history, newest first. `CLAUDE.md` carries no session log — it holds
durable rules only, so that its cost is paid once per session and never on stale narrative.

---

### 2026-08-26 (walk-forward OOS accuracy audit — 5 core fixes, 8-item statistical-rigor batch, 8-finding code review; pushed to GitHub)
- **Goal**: audit the canonical walk-forward backtest (the source of the headline Sharpe 0.415) for accuracy from every angle — data quality, look-ahead channels, statistical rigor — fix what's cheap and real, measure and honestly disclose what isn't fixable yet. 10 commits landed on `main` and pushed to `origin/main` (`78b1ddd..0a3234a`).
- **5 core bugs found and fixed** (commit `62c91a8` + merge cleanup):
  1. Live trading (`eod_runner.py`) drew from a wider, non-survivorship-correct universe than the walk-forward backtest was validated on — aligned both to `build_historical_universe(strict=True, sp500_only=True)`.
  2. Universe-filter renormalization could push a surviving position's weight back over `max_weight` after dropped symbols' weight redistributed — re-applies `_water_fill_cap()` as a post-condition.
  3. Walk-Forward Efficiency was never computed by the current framework (`CURRENT_VERIFIED_NUMBERS.md` said "not computable") — added per-fold in-sample Sharpe tracking and `WFE = OOS Sharpe / mean(IS Sharpe)`, with missing-data days excluded via weight redistribution rather than `fillna(0)` (which would have biased the diagnostic toward zero).
  4. `wf_report` JSON's `alpha_overrides` field was a hardcoded literal, not the weights actually used — now threads the resolved weights through.
  5. Reviewed 3 independent, differently-parameterized market-impact cost formulas (`backtest/costs.py`, `execution/cost_model.py`, `execution/capacity_model.py`) — kept separate (they answer different questions) but cross-documented and measured the gap between them.
- **8-item statistical-rigor batch**, each backed by a real paper, researched via SSRN/web search:
  - **Lo (2002)** autocorrelation-adjusted Sharpe — added as an additional metric; measured 0.415 → 0.422 on the real artifact (small, and up not down).
  - **Bailey & López de Prado (2014)** Deflated Sharpe Ratio — new module, `KNOWN_TRIAL_COUNT=8` curated and cited from this project's own trial history (fundamental/trend sleeves killed, universe fix, AI-PM built-and-removed, etc.); honestly reports "not yet computable" against the current canonical artifact rather than fabricating a number.
  - **Harvey/Liu/Zhu (2016)** multiple-testing hurdle (t>3.0 vs conventional t>2.0) — documented as a gap, not invented, since no t-stat is currently logged.
  - **Point-in-time sector-classification gap** — `walk_forward_runner.py` builds its `sector_map` once from the *current* sector cache and reuses it across all 6 years of folds. Measured via `scripts/measure_sector_pit_gap.py`: 142/260 tracked S&P 500 removals are corporate-action-driven business-identity changes, 21 of those still silently resolve to a sector today, ~96% of folds could plausibly be affected. Disclosed, not fixed (needs historical sector data this project doesn't have).
  - **Total-return (split/dividend) price adjustment**, WF-only via the existing `prices_cache_name=` override — real rerun: Sharpe 0.415 → 0.477, CAGR +10.2% → +13.5%, but max drawdown also *worsens* (−45.65% → −47.80%). Window was ~1 month short of canonical (stale staging cache) — not promoted.
  - **Delisting-return credit** — sourced 12 real, cited M&A deal prices (SEC filings/press releases) for acquisitions closing inside the WF window, built and unit-tested the credit mechanism. Real production rerun came back **byte-identical** to canonical — root cause confirmed directly: 0 of the 12 symbols have any rows in `prices_live` at all, since the fetch pipeline only pulls tickers currently in the live universe. A deeper, unscoped data gap, not a bug in the credit logic. Also flagged (not fixed): a `XEC` ticker paired with an unrelated reason string in `REMOVED_STOCKS` — looks like wrong data.
  - CPCV-on-the-canonical-backtest and White's Reality Check/Hansen's SPA test were researched and deliberately deprioritized (expensive; same missing-trial-log problem as DSR).
- **Code review (8 findings, all fixed, commit `0a3234a`)** — two were confirmed regressions in work reported complete earlier the same session: the "universe filter" fix for `walk_forward_lightweight.py` never actually ran (`get_universe_on_date()` returns a `list`, the code called `.empty`/`.columns` on it, `AttributeError` silently swallowed by a bare `except`), and `run_eod_with_weights()` (the discovery/mini-rebalance path) had no universe filter or max-weight re-cap at all, unlike `run_eod()`. Also fixed: a `-inf` sentinel bug in the peer session's Calmar-based self-improve promotion logic that could force a per-regime promotion with zero valid baseline; silent large weight drops with no logging; delisting credit missing from the shadow/self-improve OOS path; a DSR degenerate case silently returned a misleading neutral `0.5` instead of `None`; an off-by-one dropped each fold's first return from Calmar scoring; and the liquidity-scaled cost model was dead code (no production caller passed `volume=`) — wired real volume data into both production paths, real rerun: Sharpe 0.415 → 0.464, every metric improved slightly.
- **Notable incident, self-inflicted and fixed**: mid-session, ran 3 parallel subagents directly in the shared `main` checkout (no worktree isolation) while a peer Claude session ("Investment firm operations research") was also actively committing to the same repo. A cleanup step (`git checkout --`/`rm -f` meant to isolate this session's own work into a worktree) discarded the peer session's uncommitted edits too, since git can't distinguish whose dirty state belongs to whom in a shared working tree. Recovered fully — a `git diff` captured just before the cleanup had the peer's content, and the peer independently reapplied/verified everything from it. **Lesson applied for the rest of the session**: every subsequent multi-agent batch used `git worktree add` for full filesystem isolation, never touching the shared checkout again — landed via `git merge --ff-only` from the worktree, checked clean first each time.
- Nothing in this batch is promoted to `CANONICAL_WF_ARTIFACT` — every candidate finding (total-return, delisting-credit, liquidity-cost) is measured and documented in `CURRENT_VERIFIED_NUMBERS.md` with real before/after numbers, left for the user to decide.
- Open: `CANONICAL_WF_ARTIFACT` promotion decisions (3 candidate findings above); the `XEC` data-quality bug; sourcing real historical sector-classification data for a true PIT fix; CPCV-on-canonical and Reality Check/SPA if the point estimate's fragility ever justifies the compute.

### 2026-08-21 (scheduler pause confirmed intentional — closes the loop from PROJECT_STATE_2026-08-21.md)
- **Goal**: resolve the open question `PROJECT_STATE_2026-08-21.md` raised — is the unloaded launchd scheduler an oversight or a deliberate hold — by asking the user directly, then documenting the answer. Documentation only; no code, config, or launchd state touched.
- **User decision**: keep the scheduler paused. `com.ascentcapital.eod.plist` and `com.ascentcapital.heartbeat.plist` remain unloaded (last run 2026-07-27, 25 days idle as of this entry) — this was not an oversight. This **extends the 2026-08-15 hold decision** (live trading held despite the walk-forward blocker clearing that day); it is the same decision continued, not a new one.
- **`logs/liveness.json` staleness (last regenerated 2026-08-12, reads CRITICAL) is a byproduct of the pause, not a new monitoring failure.** The liveness check has nothing to observe while the scheduler is unloaded, so a stale CRITICAL reading is the correct, honest state of the file — it was left as-is deliberately, not fake-refreshed to mask the pause.
- **No launchd jobs were reloaded, no trade-submitting scripts were run, and `logs/liveness.json` was not modified.** `PROJECT_STATE_2026-08-21.md` is left as-is (a point-in-time report, not a living doc). `CLAUDE.md`'s "Current state" section was checked and needs no edit — it already directs readers to `logs/eod_log.jsonl` for what actually ran rather than asserting a date or number itself, so it already handles this correctly.
- Open: none — this closes the ambiguity the state report flagged. Resuming the scheduler remains a separate, explicit decision the user has not made.

### 2026-07-28 (dev-effectiveness pass: token cost, doc drift guard, three real number bugs)
- **Goal**: lower per-session token cost, reduce hallucination, make CLAUDE.md research reliable.
- **CLAUDE.md 8,003 -> ~3,995 tokens (-51%)**, paid on every session. Session log moved wholly to this archive; the file now carries durable rules only. Removed **every performance number, date, and line-number citation** from it: those go stale in days, and a stale figure in an always-loaded file is a confidently-wrong belief rather than a typo. It now points at `CURRENT_VERIFIED_NUMBERS.md`, `docs/REPO_MAP.md`, and this archive.
- **New `scripts/verify_docs.py` (24 checks)** — the durable mechanism. Every mechanically checkable CLAUDE.md claim has a check; drift fails loudly instead of rotting. Caught 11 stale claims on first run. Checks include: model constants, kill switches all False, `DEFAULT_ALPHA_WEIGHTS` key parity across both files (constraint 6), fundamental sleeve == 0 (constraint 7), at-most-one judge position change (constraint 5, asserted as an invariant not a literal string), no naive `fromtimestamp`, no numbers/line-numbers in CLAUDE.md, every named path resolves, every cited walk-forward Sharpe traces to an artifact.
- **New `scripts/reconcile_numbers.py`** — regenerates the live-book and data-integrity sections of `CURRENT_VERIFIED_NUMBERS.md` from artifacts between generated-block markers (hand-written caveats survive; `--check` fails if stale). The SSOT had gone 5 weeks stale and was losing to CLAUDE.md on its own tiebreak rule. Reports NOT COMPUTABLE with a reason rather than estimating, and shows both SPY methods where they disagree instead of silently picking one.
- **New `ascent/reporting/verified_numbers.py`** — the single path from artifact to a citable number. Raises rather than returning a plausible default.
- **Three real bugs fixed, not just documented:**
  1. **Sortino was double-annualized** (`wf_framework/metrics.py`): `dv` was annualized *and* the numerator was, dividing every result by sqrt(252). This is the "known-buggy, don't cite 0.042" number — and `test_sortino_geq_sharpe_positive` had been failing and correct all along, written off as a pre-existing failure. True value for the canonical run is 0.042*sqrt(252) = 0.67. Fixed + 2 regression tests pinning the scale invariant.
  2. **Sharpe 0.518 matched no artifact in the repo** yet was published as "the rigorous figure" on GitHub Pages, stated unsourced in `docs/methodology.md`, and hardcoded as the `self_improve` promotion baseline (a fabricated number silently setting the bar every variant is promoted against). All three now read the artifact; the published page also discloses WFE -0.65. Published `docs/index.html` corrected in place.
  3. **Vendor epochs converted in host time** on a UTC+7 host, shifting every post-close bar a day forward. Worst site was `alpaca_broker.get_portfolio_history()` — the series the rebalance recap is *required* to use. Wired to `market_date_from_epoch` along with the dashboard generator and two backfill scripts.
- **Counterfactual chart honesty**: `cumulative()` treated a missing day as a 0% return, fabricating flat sessions and letting four curves with 38/45/42/29 days of real data look comparable. Now emits gaps and discloses per-track coverage. Endpoint values are unchanged (multiplying by 1.0 is a no-op) — this fixes the *presentation*, not the numbers; the underlying log still needs rebuilding.
- **README**: the auto-generated LIVE_STATS table and the prose beneath it disagreed on every field. Prose no longer restates any number.
- **New `docs/REPO_MAP.md`** (~8.5k tokens, on-demand not always-loaded): entrypoints, per-package symbols to grep, line-range outlines of the four 1000+ line files, artifact writer/reader map, test map, task-to-file index. All 101 claimed symbols and 125/126 paths independently verified; guarded by `verify_docs.py::repo_map_pointers`.
- **Search pollution was a non-issue** — ripgrep honors `.gitignore`, so `.worktrees/` (8,179 files vs 374 real) never appeared in results. Measured rather than assumed. Added `graphify-out/` (42MB), `.idea/`, `logs.bak-*/` to `.gitignore`.
- Tests: 47 passed across every suite touching these changes; `verify_docs` 24/24; reconciler idempotent. Pre-existing unrelated failures remain: `tests/integrations/test_openbb_client.py` (network), `tests/data/test_new_ingest.py::test_fetch_ff_factors_returns_dataframe` (`_get_obb` attribute), 5 in `tests/test_wf_framework/test_ascent_engine.py`.
- Open: rebuild `logs/counterfactual_daily.jsonl` before citing any track number; `prices_live` duplicate rows have returned (see the generated section 4) so the ingest dedup is not holding; walk-forward framework still does not dedupe on read.

### 2026-07-04 (ascent-agri — first real user feedback, shipped same day)
- **Chris Kornman (The Crown / Royal Coffee education) replied to Scott's outreach**: praise ("quite advanced for a high school student"), will refer people to the site, open invite to chat. His substance — the trade thinks in near-to-long-term trends (not intraday) and Royal is ~arabica-only — was built and deployed same day: "The long view" (full-history KC=F vs 200-day average) + "Growing conditions — Sul de Minas, Brazil" (rainfall anomaly + tmin with frost-risk line) live on the site. 156 tests.
- Threaded reply drafted in Gmail (accepts video-call offer; no URLs in body — Gmail API rewrites bare URLs into google.com/url tracking redirects, discovered when Scott flagged the sent emails looked scammy; fresh-compose checklist at outreach-drafts/00-SEND-CHECKLIST.md is the fix).
- Outreach round 1 fully sent 7/3 (Nguyen auto-ack promises human reply; DCN via form; Cafe Imports quiet). Log: outreach-drafts/log.csv.

### 2026-07-02 (ascent-agri build — separate repo, NO Ascent Capital code touched)
- Built the full backtest pipeline in the standalone `/Users/scott/Downloads/ascent-agri/` CALS artifact per the port audit: regime pkg (model/decision/posture/breaks near-verbatim; features/engine/integration rewritten for coffee + BRL/USD + Central Highlands weather), alpha pkg (trend/meanrev with rolling TS z-scores replacing cross-sectional, vol_sizing, meta_learner, minimal long-only stack), backtest engine/costs, research splits/cpcv/evaluation + single-series walk_forward_runner rewrite, demo.py + demo.ipynb, 64 new tests (120 total pass).
- **Two latent Ascent Capital bugs surfaced during the port (NOT fixed here — Ascent repo untouched):** (1) `ascent/regime/model.py` `_MarkovModel.filtered_probs()` ignores its input and returns TRAIN filtered probabilities — only matters if the Markov backend is ever asked to score new data (equities always uses the HMM backend for K=3); (2) `ascent/backtest/engine.py` weight drift renormalizes by `drifted.sum()` alone, which silently drops the cash bucket for portfolios at fractional gross exposure (fine when weights sum to ~1).
- Data: BRL/USD via Yahoo BRL=X (FRED unreachable from this network — fetcher tries FRED first), Open-Meteo weather for Buon Ma Thuot. Robusta bottleneck unchanged: 1 contract, no DATABENTO_API_KEY available.
- Files: all in ascent-agri repo + this CLAUDE.md entry. Open (ascent-agri): backfill robusta contracts via Databento when key exists; consider porting the two bug fixes back into Ascent Capital.
- **VIETNAMESE GROWER PAGE (2026-07-03, community impact):** `/vi/` on the live site — focused farmer-facing page in Vietnamese (crop stage + stage-weighted weather risk, rainfall anomaly, market state in plain words, farm-gate đồng/kg with VN number formatting via `transmission_line_vi`), generated by the same daily build; EN|VI toggle both pages; footer flags draft translation + invites corrections. Roadmap: native-speaker (family) review — quality control AND essay material. 154 tests.
- **AGRONOMY LAYER (2026-07-03, ag-science spike for CALS):** `ascentagri/agronomy/` — `phenology.py` (Dak Lak robusta crop calendar: flowering Jan–Mar drought-critical, early fruit Apr–May, filling Jun–Sep, harvest Oct–Dec wetness-sensitive; rainfall anomaly → stage-weighted crop stress index) + `economics.py` (USD/VND optional cache, futures→VND/kg farm-gate transmission, honestly labeled futures-equivalent). Site/brief/API now carry crop stage + stress band + farmer-margin line. `docs/APPLICATION-ANGLES.md` = CALS ag-science vs UC econ framings + claims inventory. Key line: "a rainfall z-score is statistics; knowing the same deficit ruins February flowering but speeds November drying is agronomy; measuring whether the market already knows is economics." 153 tests. Outreach emails drafted earlier same session (writing-as-scott skill + 4 researched targets in Gmail drafts, outreach-drafts/ gitignored).
- **DATA PRODUCT LAYER (same session, corporate-adoption kit):** site now emits `api/latest.json` (versioned schema: regime/posture/anomalies/brief) + `api/history.csv` (2,134 days of derived series — labels, risk multipliers, rain anomalies; NO raw market data redistributed). LICENSE added (MIT code / CC BY 4.0 derived data — unlicensed repo would legally block corporate use). `weekly-research.yml` also snapshots GitHub traffic → `data/ledger/traffic.jsonl` (timestamped usage evidence). `docs/OUTREACH.md` = tiered corporate outreach playbook (specialty importers/roasters first, no Reddit) + email template + the evidence checklist gating any "X users depend on my data" claim. 143 tests.
- **DAILY LEDGER LOOP (same session, Ascent daily-agent pattern applied to ascent-agri):** `ascentagri/ledger.py` — append-only public track record (`data/ledger/forecasts.jsonl`, committed to main by the daily workflow): regime call + target exposure recorded BEFORE outcomes, scored later from the ledger's own prices with 1-day execution delay, never edited. Site section "The ledger — the model in public" (chart activates at 10+ scored days). New `weekly-research.yml` (Mondays): re-runs weather study + WF on fresh data, commits artifacts — evaluation refresh, explicitly NOT parameter retuning. `CONTRIBUTING.md` opens the community on-ramp (new regions/crops/composites) with integrity ground rules. 141 tests.
- **RESEARCH LAYER (same session, Ivy-level upgrade):** built `ascentagri/research/weather_study.py` — a-priori event studies with permutation inference + placebo controls (VN dry→robusta, BR dry/cold→arabica, cross-placebos) — plus working paper `docs/research/weather-and-coffee-returns.{md,pdf}` with honest results: BR dry→arabica +5.1pp/5d (n=4, p=.04 uncorrected, disclosed as not surviving Bonferroni), placebo null, VN→robusta untestable (0 events in 17-month robusta span → quantifies the contract-backfill case), and the 2021 frost (+33.8%/5d) absorbed by the event cooldown rule (documented as event-definition risk, not retro-fixed). Site gained a Research section (PDF served) + RSS feed of daily briefs. `docs/ROADMAP-to-Nov-1.md` = Scott's dated critical path (July: contracts grind + GoatCounter + share URL; Aug: robusta re-run + paper v2 + optional JEI submission; Sep: essays + named users; Oct 15 feature freeze). 133 tests pass.
- **SHIPPED PUBLIC (same session, for real users):** created `github.com/ScottDongKhang/ascent-agri` (public) using the keychain credential and deployed the **Robusta Coffee Monitor** to `https://scottdongkhang.github.io/ascent-agri/` — dark-editorial daily page (regime posture + deterministic plain-English brief, price w/ regime shading, Buon Ma Thuot rainfall anomalies, BRL/USD driver, honest methods). Weekday cron (`update-site.yml`, 21:20 UTC) refetches + republishes; fail-safe = no successful fetch → no publish (stale-but-correct). GoatCounter snippet wired to `ascent-agri.goatcounter.com` — **Scott must claim that code (free signup) to see visitor counts.** 124 tests pass.

### 2026-06-24 (scheduled rebalance run + KLAC price-corruption root-cause → trend sleeve un-gated)
- **Ran** `run_all_agents.py` (scheduled rebalance). Verdict PROCEED (conf 0.58); 24 orders submitted, 0 skipped; kill switch OK (DD 0.8%); AI PM Level 1 / 5% / 0 overrides; day attribution +0.57% vs SPY −0.05%.
- **Investigated two anomalies in the run:**
  1. **IC gate zeroed the 70% trend sleeve** (`rolling mean_ic=-0.0641 < -0.005`). ROOT CAUSE: **KLAC** carried a recurring **×10 price error** in `prices_live` (351 up + 351 down implausible moves across 2020–2026; pipeline panel saw 702). Momentum ranks KLAC top after each fake +900% → the ~−90% revert follows → **trend IC driven strongly negative (−0.093, t −56)** while meanrev mirror-inflated (+0.072, t +55). Because logged `mean_ic` is a **cumulative all-history mean**, a bad source-blend ~6/13–15 stepped it negative on **6/15** — gate has zeroed trend every run since (6/15, 6/22 discovery, 6/24 rebalance).
  2. **AI PM force-seal** (both phases): designed fallback fired because the agentic loop exhausted `max_tool_calls` without `propose_portfolio`. Contributors: Falsifier Haiku JSON parse failure + neutral MiroFish (0.50/mixed). Net: force-sealed near quant baseline, 0 overrides — minimal original judgment today. Not corrupting; flagged.
- **FIX (data, surgical + reversible):** backed up `prices_live` → `data_cache/prices_live.pre_klac_fix.20260624-160508.bak.parquet`; fresh-fetched KLAC split-only (auto_adjust=False, 0 anomalies, $11–269 real range) and **replaced KLAC in place** (4,868 blended rows → 1,627 clean). Split-only basis preserved for the other 935 symbols (no TR swap, no discontinuity). **Verified:** trend `mean_ic` −0.093 → **−0.0025 (gate no longer fires)**, meanrev +0.072 → +0.012. The live source returns clean KLAC — corruption was purely old accumulated rows.
- **Deeper bug found, NOT yet fixed (durable class-fixes):** `save_parquet` append+dedup is **silently failing** — `prices_live` holds 3 blended source generations (`yfinance_hub` 2.98M + `yahoo` 1.43M + `yahoo_hub` 58K = 4.46M rows, ~3× bloat) because the tz-normalized dedup key doesn't match across fetches (mixed tz-aware/naive `date` → object dtype → normalization skipped). This is what let the 10× KLAC copy linger and get arbitrarily picked by the pivot. Also: **IC gate has no robustness to single-symbol outliers** — one corrupt symbol can disable a 70% sleeve (no winsorization / breadth requirement).
- Files: `data_cache/prices_live.parquet` (KLAC repaired), backup `.bak.parquet` (rollback). Open: (a) fix `save_parquet` dedup + collapse the 3-generation bloat; (b) harden IC gate vs outliers; (c) AI PM Falsifier JSON / tool-budget; (d) confirm next real run logs trend ungated & positive.
- **FOLLOW-UPS (a)+(b) DONE (TDD):** (a) `save_parquet` dedup now robust to **object/mixed tz-aware+naive `date`** via new `_calendar_day_key()` (was: `pd.concat` of aware+naive → object dtype → `is_datetime64_any_dtype` False → normalize branch skipped → dedup never fired = the 3-generation blend). One-time collapse of existing bloat: **prices_live 4.46M → 1.82M rows, 0 remaining dup (symbol,calday)**, KLAC stays clean; backup `prices_live.pre_dedup.*.bak.parquet`. (b) IC computation now winsorizes the fwd-return target cross-sectionally (1/99 pct, `_winsorize_rows()` in `main.py`) before `corrwith` — one data-error symbol can no longer drag a sleeve below the gate. **Validated:** clean-data gate decisions unchanged (no-op); on the corrupted backup, raw trend gates (−0.0148) but winsorized does NOT (+0.0010). Files: `ascent/data/store/parquet.py`, `ascent/main.py`, `tests/test_parquet_store_dedup.py` (+mixed-tz test), `tests/test_ic_outlier_robustness.py` (new). 6 new tests pass; full data/alpha/IC sweep 328 passed / 3 pre-existing fails (openbb-network, WF fixture IndexError, known-buggy Sortino — all fail on baseline too). Still open: (c) AI PM Falsifier JSON / tool-budget; (d) confirm next real run logs trend ungated.
- **FOLLOW-UP (c) DONE (TDD):** two independent issues from the run (Falsifier JSON + AI PM tool-budget — confirmed *separate*: `build_registry` runs in the rebalance flow, not the AI PM tool loop). (1) Both Haiku calls in `falsifier_registry.py` did `json.loads(text[start:end])` on the whole array → one missing comma OR truncation at `max_tokens` discarded EVERY falsifier and silently dropped the registry to all-news-watches (the run's `Expecting ',' delimiter: line 99 col 73`). New `_parse_json_objects()` scans balanced top-level `{...}` spans and parses each independently (skips malformed, drops truncated tail, survives the rest); `max_tokens` 800→2000 / 500→1500 to cut truncation. (2) `tool_completion` had no final-turn nudge — a model that over-grounds exhausts the budget into `[max iterations reached]` (empty result_store → caller force-seal), which is why AI PM Phase 1 & 2 both force-sealed. Added `_FINAL_TURN_NUDGE` injected into the tool-result turn on the penultimate iteration so the model gets one explicit chance to call its submission tool within budget (fires only when about to exhaust; happy path unchanged). Files: `ascent/strategy/falsifier_registry.py`, `ascent/llm/client.py`, `tests/strategy/test_falsifier_json_parsing.py` (new, 5), `tests/test_tool_completion_nudge.py` (new, 2). Sweep: 220 passed / 1 pre-existing openbb-network fail. Still open: (d) confirm next real run (Jul 8) logs trend ungated & that AI PM seals in-budget (nudge effect is live-only, not offline-testable).

### 2026-06-22 (clean price re-fetch + WF re-run — backtest number RESOLVED, no longer UNDER REVIEW)
- **What was wrong:** the WF OOS figure had been UNDER REVIEW because `prices_live` was corrupted (~59%/92% dup rows from 3 blended sources + 10×-type errors in 12 symbols). Could not cite a number.
- **What was done:** backed up the corrupt cache (`data_cache/_corrupt_backup_20260622-222216/`); re-fetched **all 938 symbols from a single source** (yfinance `auto_adjust=True`, total-return adjusted, 2020→Jun 2026) into staging `data_cache/prices_live_clean_refetch.parquet`. Verified clean: **0 dup (symbol,date) rows, 0 implausible jumps** (worst remaining are real events — GME/CAR/SHC/LUMN). The re-fetch repaired 11 of the 12 corrupt symbols; only **CHRD dropped** (irreparable source-side ticker-reuse history) → 936 symbols. Re-ran WF with `llm_fundamental`/`narrative` zeroed (logged `skipped`), same engine config as the canonical run.
- **New verified number:** **Sharpe 0.41 (independent recompute 0.417), CAGR +10.3% (recompute +10.4%), excess CAGR vs SPY +1.0pp (10.42% − 9.41%, shown), regression alpha +2.24%/yr, max DD −32.9%, beta 0.73, win rate 50.2%**, OOS 2021-01-08→2026-01-14 (1134 days, 21 folds). **WFE −0.65 (overfit — IS optimizer adds no OOS value; disclosed).** Artifact `outputs/wf_results/wf_report_clean_2026-06-22.json` (+ `wf_equity_clean_2026-06-22.csv`).
- **Sanity check (vs both priors):** lands far closer to the original 0.483 prior than the contaminated −0.14 — exactly as expected (identical 1134-day/21-fold window & beta 0.733; cleaning removes dup/spike-inflated trend signal → modest deflation). Even with dividends *added* (total-return close), CAGR *fell* 12.61→10.3%, confirming corruption was inflating returns. Engine **Sortino field is buggy** (0.042; real ≈0.68 — don't cite).
- **Judgment calls flagged:** (1) used total-return adjusted close (original used split-only) — lifts absolute CAGR ~equally for strat & SPY, leaves Sharpe/alpha ~flat; (2) did **not** swap the clean cache into production `prices_live` (would change live momentum signals to adjusted close before the Jun 24 rebalance) — staging only, corrupt cache backed up, live path dedupes on read.
- Docs updated: `CURRENT_VERIFIED_NUMBERS.md` §1 (UNDER REVIEW → VERIFIED), `README.md` WF section (+ corrected the WF-header "CPCV C(6,2)=15" → rolling 21 folds), this `CLAUDE.md`. Files: `data_cache/prices_live_clean_refetch.parquet` (new), `outputs/wf_results/wf_report_clean_2026-06-22.json` (new).
- Open: optionally swap staging cache → production (deliberate, post-Jun-24); verify the ML-sleeve internal CPCV line in the alpha-stack table (separate from the now-corrected WF header).

### 2026-06-22 (data-integrity audit — WF backtest UNDER REVIEW, prices_live corruption found + ingest fix)
- `prices_live` had ~59% dup rows (yahoo+yfinance_hub tz-keyed dedup never collapsed) + 10×-magnitude errors in 12 symbols. WF re-run blew to CAGR 10¹³%; repaired re-run (median-dedup, 27 folds) → Sharpe −0.14 but still source-blended. WF OOS now UNDER REVIEW.
- Live pipeline shielded: `ascent/main.py` dedupes on read (lines 393/745/789); WF framework does not.
- FIX: `ascent/data/store/parquet.py` `save_parquet` normalizes `date` dedup key to calendar day — stops future dup accumulation. Does NOT clean existing rows.
- Off-calendar Jun 15/22 AI PM blanks confirmed NOT a bug — discovery days use daily-view path, not Phase 2. True AI PM participation = 1/1 scheduled rebalances (Jun 10).
- Files: `ascent/data/store/parquet.py`, `tests/test_parquet_store_dedup.py` (new), `CURRENT_VERIFIED_NUMBERS.md` (new), `AUDIT_DATA_INTEGRITY.md`.
- Open: re-fetch `prices_live` clean; re-run WF; verify CPCV fold-count claim in README.

### 2026-06-22 (discovery churn fix — two guards on the off-calendar mini-rebalance)
- ROOT CAUSE: `_trigger_mini_rebalance` re-ran the FULL us_equities agent + orchestrator (32-order rotation for one new candidate) with no rebalance-calendar awareness — fired 2 days before the Jun 24 scheduled rebalance.
- FIX (TDD, 8 tests): (A) `_is_near_scheduled_rebalance(today, window=3, cal_path)` — suppresses discovery within 3 trading days of next scheduled rebalance (fail-open if calendar missing). (B) `_insert_candidate_weights(current, symbol, max_weight)` — ADD-ONLY: candidate gets 1/(n+1) slot, existing book trimmed pro-rata, max-weight capped. Full agent re-run eliminated from mini-rebalance path.
- Files: `run_all_agents.py`, `tests/strategy/test_discovery_guards.py` (new). 81 strategy+portfolio tests pass.
- NOT committed yet. Today's 32-order rotation already filled at Alpaca — left to ride into Jun 24 rebalance.

### 2026-06-20 (AI PM / "no alpha" investigation — measurement repair + self-heal, honest signal now −6pp)
- Investigated "AI PM poor performance" + "Ascent makes no alpha." TWO separate causes, neither what the dashboard implied.
- **No-alpha-vs-SPY is structural**: pure quant (A★ +12.55%) lags SPY (C +16.61%) by ~4pp, actual book lags the SAME ~4pp. Cause = ~22% defensive non-equity sleeves + 200MA cut + 15% vol-target overlay costing beta in an equity-only bull.
- **AI PM measurement broken two ways**: (1) Track A★/A/D recorded null Jun 8–18 (no self-heal analog to `backfill_track_b`). (2) `earned_authority.json` buffer seeded from shadow log (different A★ series + duplicate `0.00124`) → Sortino engine scoring corrupted data.
- FIX (TDD, 3 new tests): `backfill_astar_d()` recomputes null A★/A/D from as-of snapshots + history (idempotent). `rebuild_buffers_from_counterfactual()` reconciles Sortino buffers to log. Both wired into daily run.
- **Honest result**: D−A★ = −6.06pp/22d (was a corrupted −2.89pp/13d). Modestly value-subtracting in calm_bull; n small.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/earned_authority.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_ai_pm_authority.py`.

### 2026-06-19 (Track B trace + fix — the "actual gave up 11pp of quant" gap was a measurement artifact)
- Track B had real data on only 12 of 38 days vs A★ on 21 — only 2 days overlapped. Honest common-window diff was −4.02pp/n=2 (noise). The ~11pp gap was a disjoint-window artifact.
- Root cause: `_log_holdings` computes `day_ret = (equity − last_equity)/last_equity` but Alpaca's 1D bar settles ~17:00 PT — after the 1:45 PM run → fake 0.0 → Track B.
- FIX: `alpaca_broker.get_portfolio_history()` returns settled 1D bars. `backfill_track_b()` replays them over the log (idempotent). 35 rows corrected; B−A★ now −0.42pp/21d (was −4.02pp/n=2).
- Files: `ascent/execution/alpaca_broker.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`.

### 2026-06-19 (daily run — 4 bugs diagnosed/fixed, today's logs reset, clean rerun)
- BUG 1: `_cs_normalize` got `llm_fundamental` sleeve as cross-sectional Series, not date×symbol DataFrame. Fix: broadcast Series across `features["close"]` dates.
- BUG 2: `name 'pd' is not defined` in `_log_holdings` price-fetch block. Fix: local `import pandas as pd`.
- BUG 3: Dashboard subprocess ModuleNotFoundError — `scripts/` as cwd breaks `ascent` import. Fix: `cwd=repo_root` + `PYTHONPATH=repo_root`.
- BUG 4: `_sortino` summed buffer containing None sentinel → TypeError. Fix: filter None in buffers + `_sortino`.
- Files: `ascent/alpha/stack.py`, `run_all_agents.py`, `ascent/strategy/ai_pm_perf_feedback.py`, tests.

### 2026-06-18 (AI PM "−11.63pp alpha destruction" investigation — measurement artifact, not real)
- The "−11.63pp" headline was a measurement artifact: `score_daily` appended unconditionally (duplicates), Track A★/D froze at 0.0 (returned 0.0 when no prices → should be None), NaN poisoning from yfinance trailing all-NaN row, `get_cumulative_returns` compared disjoint windows.
- Honest signal (Track D vs A★, 12 common days): −2.33pp — pure-AI-PM is modestly defensive in calm_bull. n=12, too few to disable on.
- Also fixed: test leak where `test_plan_a.py` wrote to the real counterfactual log via `monkeypatch.chdir` that didn't sandbox the absolute `_REPO` path.
- Files: `ascent/monitoring/ai_pm_counterfactual.py`, `run_all_agents.py`, `tests/test_ai_pm_counterfactual.py`, `tests/test_plan_a.py`.

### 2026-06-17 (AI PM calibration learning loop fix)
- `_compute_calibration_returns()` added: reads `prices_live.parquet`, fills `realized_21d` for log entries ≥21d old. Was silently no-opping since May 18.
- Files: `run_all_agents.py`, `tests/test_calibration_tracker.py`, `tests/agents/test_ai_pm_fallback_fix.py`.

### 2026-06-11 (MiroFish 10-round fix + AI PM Phase 2 force-seal + clean rebalance rerun)
- MiroFish timeout root-caused: status poll can't detect server-side prepare failure; Zep classification stochastic on thin event text; reddit runner idles in wait-for-commands → looks like timeout.
- Fixes: fast-fail via sim-state polling, graph rebuild on 0 entities, `Accept-Language: en` (prevents Chinese reports breaking sentiment parser), rounds-done via `/env-status env_alive` + POST `/stop`.
- AI PM Phase 2 force-seal: direct Anthropic call with `tool_choice={"type":"tool","name":"propose_portfolio"}` if tool loop exhausts. Worked live: first rejection → retry → sealed.
- Clean rebalance rerun (Jun 10): MiroFish alignment_score=0.82, PROCEED 0.62, ai_weight=5%, 31 orders submitted.
- MiroFish env: LiteLLM proxy at port 4000 (→ Haiku) must be running. OpenRouter 402 = `max_tokens` unset → provider defaults to model max (64k) → afford-check fails.
- Files: `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`, `agents/ai_pm_agent.py`, `scripts/generate_performance_page.py`.

### 2026-06-11 (next-phase improvements — all 3 workstreams + repairs)
- **C3 profiles**: `backfill_missing_profiles()` + `check_book_sector_coverage()` in `ascent/data/ingest/supplementary.py`. Live book 100% sector-labeled. Fixed `_get_portfolio_symbols()` — always returned [] (read payload keys, not nested `weights`).
- **A exposure parity**: `ascent/portfolio/exposure.py` — single source of truth for VIX-confirmed 200MA cut + vol targeting (15% target, floor 0.25, cap 1.0). Production + WF both delegate to it.
- **C1+C2 risk construction**: `_apply_inverse_vol_tilt()` (half-strength, clip [0.5,2]); `enforce_cluster_cap()` (corr>0.70, 20% cap, pro-rata redistribution).
- **B falsifier enforcement**: `ascent/strategy/falsifier_registry.py` — registry from prethesis `what_would_change_my_mind` + judge predictions + pre-mortems. Daily Gate 4 runs `check_all()` + `_apply_falsifier_trim()` (25% ONE trim, floor 4%).
- Fixed: `run_eod_with_weights()` silently no-opped on non-rebalance days — discovery paths now pass `force=True`.
- Files: `ascent/portfolio/exposure.py` (new), `ascent/strategy/falsifier_registry.py` (new), `ascent/data/ingest/supplementary.py`, `ascent/main.py`, `ascent/portfolio/optimizer.py`, `ascent/research/wf_framework/ascent_strategy.py`, `ascent/execution/eod_runner.py`, `run_all_agents.py`.

### 2026-06-10 (AI PM alpha audit fixes — all 11 findings implemented)
- `blend()` rewritten as active-weight budget (5pp one-way TE cap, not 5% mixing). `DUST_THRESHOLD=0.005`.
- `score_daily()` returns None (not 0.0) for missing Track D/A★; `update_authority()` skips None.
- Judge can now `conviction_press` (increase); parse failure defaults to `proceed+degraded` not `reduce_size`.
- `_prethesis_universe` defined from holdings + top alpha scores; `directional_stance` required with falsifier.
- Files: `agents/ai_pm_agent.py`, `ascent/monitoring/ai_pm_counterfactual.py`, `ascent/strategy/ai_pm_learning.py`, `ascent/strategy/earned_authority.py`, `debate/adversarial_authority.py`, `debate/judge.py`, `run_all_agents.py`.

### 2026-06-10 (next-phase improvement spec — alpha/Sharpe/AI-native/drawdown)
- KEY FINDING: `wf_framework/ascent_strategy.py` applies vol-target (15%) which production `ascent/main.py` lacked; production's VIX gate absent from research. Live book and validated strategy were different.
- Spec: `docs/superpowers/specs/2026-06-10-alpha-sharpe-ainative-spec.md`.
- Implementation order: C3 → A → C1+C2 → B (all done Jun 11).

### 2026-06-10 (rebalance run — Phase 1 force-seal + MiroFish diagnosis)
- Phase 1 force-seal: direct Anthropic API call with `tool_choice={"type":"tool","name":"propose_prethesis"}`. Phase 1 now seals: 12 conviction names confirmed.
- MiroFish 402: OpenRouter drained (requested 64000 tokens, `max_tokens` unset). Fix: top up credits.
- Rebalance: PROCEED (0.6 confidence), 27 orders submitted.
- Files: `agents/ai_pm_agent.py`, `ascent/integrations/mirofish_client.py`, `ascent/integrations/get_mirofish_sentiment.py`.

### 2026-06-08 (OpenBB integration — hub reliability + CBOE/CFTC/FF alpha data + AI PM live tools)
- `ascent/integrations/openbb_client.py` (new): central adapter (tiingo→yfinance fallback). All OpenBB calls go here.
- `ascent/data/ingest/cboe_options.py`, `cftc_positioning.py`, `famafrench_factors.py` (all new).
- AI PM Phase 2 gains `get_live_options_flow` + `get_cot_positioning` tools.
- Optional env vars: `TIINGO_TOKEN`, `CFTC_APP_TOKEN`.
- Files: `ascent/integrations/openbb_client.py`, `ascent/data/ingest/cboe_options.py`, `cftc_positioning.py`, `famafrench_factors.py`, `ascent/features/feature_defs.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-08 (AutoHedge integration — Exa news, yfinance fundamentals, ticker discovery)
- `ascent/integrations/exa_news.py` (new): Exa API headlines per symbol daily.
- `ascent/strategy/ticker_discovery.py` (new): `run_discovery()` uses HAIKU_MODEL, conviction threshold 0.75.
- `run_all_agents.py`: 5-day mini-rebalance cooldown, `_trigger_mini_rebalance()`, daily Exa fetch.
- `ascent/main.py` + `us_equities_agent.py`: `extra_symbols: list[str] | None` passthrough for in-memory ticker injection.
- Required: `EXA_API_KEY` env var. Gotcha: `loguru` not installed — use `import logging`.
- Files: `ascent/integrations/exa_news.py`, `ascent/strategy/ticker_discovery.py`, `agents/ai_pm_agent.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `run_all_agents.py`.

### 2026-06-08 (TradingAgents integration — per-ticker memory + StockTwits)
- `memory/ticker_memory.py` (new): per-ticker AI PM decision log, outcome scoring at 10d/21d.
- `ascent/integrations/stocktwits.py` (new): crowd sentiment, band classification, IC logging.
- Files: `memory/ticker_memory.py`, `ascent/integrations/stocktwits.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-07 (MiroFish sentiment validation layer + judge → Opus)
- `ascent/integrations/mirofish_client.py` (new): MiroFish REST client, 8-min deadline, graceful None on failure.
- `ascent/integrations/get_mirofish_sentiment.py` (new): alignment score, decision rules (amplify >0.70, trim <0.40).
- Judge upgraded to `DEFAULT_MODEL` (Opus).
- `data_cache/mirofish_analogues.json` (new): 25 landmark market events.
- Files: above + `ascent/integrations/analogue_matcher.py`, `ascent/integrations/mirofish_calibration.py`, `debate/agents.py`, `debate/judge.py`, `run_all_agents.py`.

### 2026-06-07 (debate persona upgrade — Druckenmiller / Burry / Taleb)
- Bull → Druckenmiller: quantify upside/downside asymmetry explicitly (Monte Carlo p95 vs p5).
- Bear → Burry: lead with specific adversarial score or VaR, not generic warnings.
- Devil's Advocate → Taleb: "is this portfolio convex or concave?" `_section_tail_asymmetry()` computes tail ratio (p95−p50)/(p50−p5); entropy check for turkey-problem warning. Injected into devil's advocate context only.
- Files: `debate/agents.py`.

### 2026-06-05 (AI PM learning system + hallucination prevention)
- `ascent/strategy/ai_pm_learning.py` (new): daily Sonnet brief, post-mortem (~21d lag), pattern memory → `data_cache/ai_pm_pattern_memory.json`.
- Hallucination prevention in code: `_build_data_grounding()`, `_apply_recency_gate_python()`, feedback citation gate, conviction inflation cap.
- Files: `ascent/strategy/ai_pm_learning.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-04 (AI PM Progressive Authority System + quant two-way integration)
- `earned_authority.py` rewrite: 5-level ladder (Shadow→Analyst→Associate→Manager→Director), Sortino-based promotion/demotion, 5-day cooldown, 63-day stuck alert.
- `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py` (all new).
- Phase 1 writes `data_cache/ai_prethesis_latest.json`. AI PM bootstrapped at Level 1 (5% authority) from this date.
- Files: `earned_authority.py`, `ai_pm_guardrails.py`, `ai_pm_counterfactual.py`, `ai_pm_perf_feedback.py`, `ascent/alpha/stack.py`, `ascent/main.py`, `agents/us_equities_agent.py`, `agents/ai_pm_agent.py`, `run_all_agents.py`.

### 2026-06-03 (WF OOS framework + signal fixes)
- WF framework at `ascent/research/wf_framework/`. Fundamental sleeve disabled. KMLM overweight fixed. IC gate −0.010 → −0.005.
- Files: `orchestrator/central_intelligence.py`, `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `run_all_agents.py`, `ascent/monitoring/attribution.py`.

### 2026-06-02 (regime hardening)
- Hard crisis override (VIX>30 + SPY 5d<−7%), asymmetric hysteresis (down 0.40 / up 0.70), entropy penalty (entropy<1e-6 → ×0.90).
- Files: `ascent/regime/engine.py`, `ascent/regime/decision.py`, `ascent/regime/types.py`.

### 2026-06-01 (causal intelligence + investor letter)
- `ascent/causal/` module: PC algorithm DAG, per-symbol graph builder, early-exit tracker. Devil's advocate receives causal mechanisms.
- `ascent/reporting/investor_letter.py` (new): Sonnet monthly letter on first trading day of month.
- Files: `ascent/causal/` (new module), `agents/ai_pm_agent.py`, `run_all_agents.py`, `debate/agents.py`, `ascent/reporting/investor_letter.py`.

### 2026-05-31 (anti-hallucination hardening)
- `generate_structured` gains `json_schema` param (wire-level Anthropic enforcement). `_EVIDENCE_RULE` in `debate/agents.py`.

### 2026-05-25 — 2026-05-28 (adversarial intelligence + two-phase AI PM + GitHub Pages)
- Debate redesigned as genuine risk committee: `adversarial_engine.py`, `adversarial_authority.py`, `adversarial_monitor.py`. ONE falsifiable change per rebalance.
- AI PM two-phase: Sonnet pre-thesis (before quant) + Opus synthesis. `momentum_exhaustion` override type.
- GitHub Pages dashboard (`scripts/generate_performance_page.py` + `docs/index.html`). Auto-pushed after every daily run.
