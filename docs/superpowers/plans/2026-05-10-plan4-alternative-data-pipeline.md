# Plan 4 — Alternative Data Pipeline (Tier 4)

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a pipeline that converts unstructured text and behavioral data into numerical alpha signals, validated by the same IC gate used for factor discovery (IC_mean ≥ 0.015, IC_IR ≥ 0.60, positive in every observed regime). Four sources: (A) SEC 10-K/10-Q full-text, (B) earnings call transcripts, (C) Reddit sentiment, (D) Google Trends. Each source is a separate ingest → LLM/NLP → validate → deploy pipeline. No source enters the alpha stack without passing the IC gate.

**Architecture:** Each source writes a standardized signal DataFrame (dates × symbols, float values) to `data_cache/altdata_{source}.parquet`. The validation framework runs each source through per-regime Spearman IC evaluation (same as factor discovery). Accepted sources are registered in `data_cache/active_alpha_config.json` under `"altdata_weights"` and combined by `ascent/alpha/altdata_alpha.py`. The alpha stack reads this combined signal as a single sleeve.

**Tech Stack:** Python 3.12, `praw` (Reddit), `pytrends` (Google Trends), `beautifulsoup4` (transcript scraping), `requests`, existing Haiku client, existing IC evaluation framework from `ascent/research/factor_discovery/regime_cpcv_evaluator.py`.

**Prerequisites:** Plan 1 (Factor Risk Model) helpful but not required. Factor discovery (Tier 3) must be complete — the same IC gate and CPCV evaluator are reused here. `SELF_MODIFY_ENABLED` gate does NOT apply to alternative data — these are new data sources, not strategy modifications.

**⚠ Rate limits and ToS:** (a) SEC EDGAR: max 10 requests/second; (b) Reddit PRAW: 100 requests/10 minutes on free tier; (c) Google Trends: unofficial API, rate-limited — max 1 request/5 seconds; (d) Earnings transcripts: scrape only from sites that permit it in their robots.txt or use SEC EDGAR filing text (which always permits). Do not exceed stated limits.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/data/ingest/sec_filings.py` | 10-K/10-Q downloader + section extractor |
| Create | `ascent/data/ingest/earnings_transcripts.py` | Earnings call transcript fetcher |
| Create | `ascent/data/ingest/reddit_sentiment.py` | PRAW-based Reddit mention + sentiment |
| Create | `ascent/data/ingest/google_trends.py` | pytrends search momentum signal |
| Create | `ascent/alpha/altdata_alpha.py` | IC-weighted signal combiner for all validated sources |
| Create | `ascent/data/validate/altdata_validator.py` | IC gate — same threshold as factor discovery |
| Modify | `ascent/alpha/stack.py` | Add `altdata` sleeve at 0% initially; enable per source as validated |
| Modify | `run_all_agents.py` | Schedule: 10-K on filing, transcripts post-earnings, Reddit/Trends daily |
| Create | `tests/test_altdata_pipeline.py` | Full test suite — 20 tests |

---

## Task 1: SEC Full-Text Pipeline (10-K / 10-Q)

**File:** `ascent/data/ingest/sec_filings.py`

### Steps
- [ ] 1.1 Write `fetch_full_text_filing(cik, filing_type="10-K") -> str`. Uses EDGAR full-text search API (`https://efts.sec.gov/LATEST/search-index?q=&dateRange=custom&category=form-type&forms={filing_type}&hits.hits.total.value=10`). Downloads the primary document (`.htm`). Strips boilerplate (cover page, signatures, exhibits). Returns the text of the Management Discussion & Analysis (MD&A) section plus Risk Factors section — total ≤ 8,000 tokens.
- [ ] 1.2 Write `extract_mda_section(full_text) -> str`. Finds the MD&A section boundary using regex patterns common to SEC filings (`ITEM 7`, `MANAGEMENT'S DISCUSSION`, `ITEM 2` for 10-Q). Falls back to first 4,000 characters if section detection fails.
- [ ] 1.3 Write `classify_filing_signal(mda_text, symbol, period_end) -> dict`. Calls Haiku with a 5-step structured prompt: (1) revenue trajectory (accelerating/decelerating/flat), (2) margin trend (expanding/contracting), (3) management tone (confident/cautious/defensive), (4) liquidity risk (mentions of covenant, going concern, cash burn), (5) forward guidance sentiment (raised/maintained/lowered). Returns `{"revenue_momentum": float(-1 to 1), "margin_trend": float, "tone": float, "liquidity_risk": float(0–1), "guidance": float}`.
- [ ] 1.4 Write `build_sec_signal_panel(symbols, start_date, end_date) -> pd.DataFrame`. For each symbol, fetches 10-K (annually) and 10-Q (quarterly) since `start_date`. Applies 45-day filing lag (same as existing fundamentals). Forward-fills the signal for 90 days (signal decays at quarterly restatement). Pivots to wide format: dates × symbols. Returns a panel with `revenue_momentum` as the primary signal (highest IC historically based on PEAD research).
- [ ] 1.5 Cache to `data_cache/altdata_sec.parquet`. Incremental: only re-fetch filings not already in cache.
- [ ] 1.6 Schedule: `run_all_agents.py` runs `update_sec_signals()` on the first Sunday of each month (reuses the monthly trigger from factor discovery). New 10-K/10-Q filings are also fetched by the event agent (Plan 3) when they appear on EDGAR RSS.

---

## Task 2: Earnings Call Transcript Pipeline

**File:** `ascent/data/ingest/earnings_transcripts.py`

### Steps
- [ ] 2.1 Primary source: EDGAR 8-K Item 2.02 (Results of Operations) often includes prepared remarks. Fetch via the same EDGAR RSS feed used in Plan 3 (event agent). When an 8-K Item 2.02 is detected, download the filing text.
- [ ] 2.2 Write `extract_qa_section(transcript_text) -> tuple[str, str]`. Returns `(prepared_remarks, qa_section)`. QA section starts at `QUESTION AND ANSWER` or `Q&A` boundary in the text. If no QA section found, return `(full_text[:3000], "")`.
- [ ] 2.3 Write `classify_transcript_signal(prepared_remarks, qa_section, symbol) -> dict`. Haiku prompt: (1) prepared remarks tone — confidence score, (2) management defensiveness on analyst questions — count of hedged vs. direct answers, (3) forward-looking language density — ratio of future tense to past tense, (4) mention of specific numbers vs. qualitative language (quantitative management = more credible). Returns `{"tone": float, "defensiveness": float, "forward_confidence": float, "quantitative_ratio": float}`.
- [ ] 2.4 Write `build_transcript_signal_panel(symbols, start_date) -> pd.DataFrame`. Forward-fill 63 days (earnings cycle). 1-business-day lag (transcript available same day as earnings call). Cache to `data_cache/altdata_transcripts.parquet`.
- [ ] 2.5 Combined signal: average `tone` and `forward_confidence`, subtract `defensiveness`. Cross-sectional z-score. This is the input to the IC validator.

---

## Task 3: Reddit Sentiment Pipeline

**File:** `ascent/data/ingest/reddit_sentiment.py`

### Steps
- [ ] 3.1 Install praw: `pip install praw`. Configure with a Reddit API application (free — create at reddit.com/prefs/apps, read-only access, no posting). Credentials stored in `.env` as `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.
- [ ] 3.2 Write `fetch_daily_mentions(symbols, subreddits=("wallstreetbets", "stocks", "investing", "StockMarket"), lookback_hours=24) -> pd.DataFrame`. For each subreddit, search submissions and top-level comments from the last `lookback_hours`. Count mentions per symbol (case-insensitive ticker match with word boundary: `\bAAPL\b`). Returns a DataFrame with columns `[symbol, mention_count, mention_velocity, avg_sentiment]`.
- [ ] 3.3 Sentiment: use `TextBlob` (install: `pip install textblob`) for sentence-level sentiment on the post text. Average polarity score per symbol across all mentions. Fallback: if TextBlob unavailable, sentiment = 0 (mention count still tracked).
- [ ] 3.4 Write `compute_reddit_signal(mentions_df, lookback=21) -> pd.Series`. Computes cross-sectional z-score of `mention_velocity` (change in mentions vs. 21-day average). This is a contrarian signal — high retail excitement is bearish for institutional-quality stocks. Return the z-score negated (short high-mention names). Note: this is a hypothesis; validate with IC gate before deployment.
- [ ] 3.5 Schedule: run daily at 8:00 AM ET (before market open) via the existing launchd plist. Cache to `data_cache/altdata_reddit.parquet`. 1-day lag enforced by construction (previous day's Reddit activity predicts today's price).
- [ ] 3.6 Privacy: do not store individual post text. Only store aggregated counts and sentiment scores. No usernames stored.

---

## Task 4: Google Trends Pipeline

**File:** `ascent/data/ingest/google_trends.py`

### Steps
- [ ] 4.1 Install pytrends: `pip install pytrends`. No API key required — uses unofficial Google Trends API.
- [ ] 4.2 Write `fetch_trends(symbol, lookback_months=12) -> pd.Series`. Queries Google Trends for the symbol as a search term (not as a company name — test which form gives cleaner data for a sample of 10 symbols and use the better form for all). Returns a daily-frequency Series of relative search interest (0–100 normalized).
- [ ] 4.3 Write `build_trends_panel(symbols, lookback_months=12) -> pd.DataFrame`. Fetches trends for all symbols in batches of 5 (pytrends allows multi-term comparison, but inter-batch normalization is unreliable). Use single-term fetch for each symbol; normalize by dividing by max value over lookback. Returns a wide panel (dates × symbols). Apply 1-day lag.
- [ ] 4.4 Rate limit: 1 request per 5 seconds. For 901 symbols, this is ~75 minutes. Run weekly (Sunday 7 AM), not daily.
- [ ] 4.5 Write `compute_trends_signal(trends_panel, lookback=21) -> pd.DataFrame`. Computes search velocity: (current week − 3-week average) / 3-week average. Cross-sectional z-score per date. This captures acceleration of public interest before it fully reflects in price.
- [ ] 4.6 Cache to `data_cache/altdata_trends.parquet`. Incremental update.

---

## Task 5: Validation Framework

**File:** `ascent/data/validate/altdata_validator.py`

### Steps
- [ ] 5.1 Write `validate_altdata_source(signal_panel, prices_df, regime_labels, source_name) -> dict`. Runs the same IC evaluation as `regime_cpcv_evaluator.evaluate_factor_regime_ic()`. The signal panel is a dates × symbols DataFrame (one signal per source). Returns `{"source": str, "ic_mean": float, "ic_ir": float, "ic_min_regime": float, "n_observations": int, "status": "accepted"|"rejected", "reasons": list[str]}`.
- [ ] 5.2 Acceptance gate (identical to factor discovery): IC_mean ≥ 0.015 AND IC_IR ≥ 0.60 AND IC_min_regime > 0.010 AND n_observations ≥ 20.
- [ ] 5.3 Write `run_altdata_validation(sources_dict, prices_df, regime_labels) -> list[dict]`. `sources_dict` maps `{source_name: signal_panel}`. Returns validation results for each source. Accepted sources are written to `outputs/altdata_proposals/` (same pattern as factor proposals — human reviews before deployment).
- [ ] 5.4 Write `register_altdata_source(source_name, initial_weight=0.02) -> None`. Adds the source to `data_cache/active_alpha_config.json` under `"altdata_weights"`. Only called after human review of the validation proposal. Updates the `altdata` sleeve weight in `DEFAULT_ALPHA_WEIGHTS` proportionally.
- [ ] 5.5 Validation runs monthly (same trigger as factor discovery — first Sunday of each month). `run_all_agents.py` calls `run_altdata_validation()` with all cached signal panels.

---

## Task 6: Alternative Data Alpha Combiner

**File:** `ascent/alpha/altdata_alpha.py`

### Steps
- [ ] 6.1 Write `altdata_alpha(features, active_config=None) -> pd.DataFrame`. Reads `"altdata_weights"` from `active_alpha_config.json`. For each registered source with weight > 0, loads the corresponding signal panel from `data_cache/altdata_{source}.parquet`. Cross-sectionally z-scores each panel. Combines as weighted average.
- [ ] 6.2 Returns empty DataFrame if no sources are registered or all weights are zero. Stack skips and renormalizes (same pattern as other sparse sleeves).
- [ ] 6.3 Log which sources contributed and their weights at INFO level each run.
- [ ] 6.4 Wire into `ascent/alpha/stack.py` as `"altdata"` sleeve at initial weight 0.00 (zero until first source validated). The sleeve is present but silent until activation.

---

## Task 7: Tests

**File:** `tests/test_altdata_pipeline.py` — 20 tests

- [ ] `test_extract_mda_section_finds_boundary` — known 8-K text → extracts MD&A correctly
- [ ] `test_classify_filing_signal_returns_required_keys` — Haiku mock → dict has all 5 keys
- [ ] `test_classify_filing_signal_neutral_on_llm_failure` — LLM error → returns zeros, no raise
- [ ] `test_build_sec_signal_panel_applies_45_day_lag` — signal date ≥ period_end + 45 days
- [ ] `test_build_sec_signal_panel_forward_fills_90_days` — non-NaN for 90 days after filing
- [ ] `test_classify_transcript_defensiveness_detected` — text with hedged answers → high defensiveness
- [ ] `test_build_transcript_panel_1_day_lag` — signal date = earnings date + 1 bday
- [ ] `test_fetch_daily_mentions_returns_schema` — mock PRAW → df has symbol/mention_count/avg_sentiment
- [ ] `test_reddit_signal_is_contrarian` — high mentions → negative z-score
- [ ] `test_reddit_handles_no_mentions` — symbol with zero mentions → signal = 0, no NaN
- [ ] `test_build_trends_panel_normalized` — values between 0 and 1
- [ ] `test_compute_trends_signal_velocity` — rising trend → positive z-score
- [ ] `test_validate_source_accepted_on_good_ic` — IC above threshold → status=accepted
- [ ] `test_validate_source_rejected_on_low_ir` — IC_IR below 0.60 → status=rejected
- [ ] `test_validate_source_rejected_on_negative_regime_ic` — negative IC in one regime → rejected
- [ ] `test_validate_source_rejected_on_insufficient_obs` — n_obs < 20 → rejected
- [ ] `test_register_altdata_source_writes_config` — config file updated with new source
- [ ] `test_altdata_alpha_returns_empty_when_no_sources` — no registered sources → empty DataFrame
- [ ] `test_altdata_alpha_combines_weighted` — two sources → weighted average z-score
- [ ] `test_altdata_sleeve_in_stack_at_zero_weight` — stack includes altdata key, weight=0.00

---

## Acceptance Criteria

1. At least one source (SEC 10-K signals recommended first — highest quality data) completes the pipeline end-to-end: ingest → classify → cache → validate → proposal
2. Validation framework correctly rejects a synthetic signal with IC_IR < 0.60 (verified by test)
3. `altdata_alpha` sleeve returns empty DataFrame gracefully when no sources are validated
4. No altdata source enters the alpha stack without a written proposal in `outputs/altdata_proposals/`
5. Rate limits respected for all APIs (no bans or 429 errors in 30 days of operation)
6. All 20 tests passing; full suite passing
