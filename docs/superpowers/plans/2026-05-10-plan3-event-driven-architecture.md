# Plan 3 — Event-Driven Architecture

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a second execution pathway that wakes on triggers — not on a clock — and acts on market-moving information within minutes. Three event sources: (A) SEC EDGAR 8-K filings, (B) congressional trade disclosures via Capitol Trades, (C) options anomaly spikes. Each source has a separate classifier, size limit, and execution path. Event trades are capped at 0.5% NAV each, use limit orders, and are tracked independently with their own post-hoc IC measurement.

**Architecture:** A background polling thread (`agents/event_agent.py`) runs during market hours (9:30–16:00 ET) independently of the 1:45 PM daily runner. When an event fires, it calls `ascent/alpha/event_alpha.py` for classification, then `ascent/execution/event_runner.py` for sizing and submission. All event decisions are logged to `logs/event_trades.jsonl`. The daily runner reads the event log to avoid double-counting existing event-driven positions in its rebalance.

**Tech Stack:** Python 3.12, `requests` (EDGAR RSS), `threading`, existing `ascent/llm/client.py` (Haiku), existing `ascent/execution/order_engine.py`, `alpaca-trade-api`. Capitol Trades: free REST API (`https://efts.house.gov/LATEST/search-index?q=&dateRange=custom`). Options data: Alpaca options API (paper account has access).

**Prerequisites:** None. This plan is independent. However, Plan 5 (Execution Excellence) refines the execution path used here — implement Plan 3 first, upgrade in Plan 5.

**⚠ Gate condition:** Event trades submit real orders to Alpaca. Test entirely in paper mode before any real-money deployment. Add an `EVENT_TRADING_ENABLED = False` kill switch that must be manually enabled.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `agents/event_agent.py` | Background polling loop — EDGAR, Capitol Trades, options |
| Create | `ascent/alpha/event_alpha.py` | Haiku classifier — event → structured signal |
| Create | `ascent/execution/event_runner.py` | Event trade sizing, limit order submission, approval gate |
| Create | `ascent/data/ingest/edgar_listener.py` | EDGAR RSS polling + deduplication |
| Create | `ascent/data/ingest/capitol_trades.py` | Capitol Trades API client |
| Create | `ascent/data/ingest/options_scanner.py` | IV spike and put/call anomaly detection |
| Create | `logs/event_trades.jsonl` | Event trade log — one entry per event (decision + outcome) |
| Modify | `ascent/execution/eod_runner.py` | Read event log — exclude event-held positions from rebalance sizing |
| Modify | `run_all_agents.py` | Start event agent background thread on non-holiday weekdays |
| Create | `tests/test_event_agent.py` | Full test suite — 16 tests |

---

## Task 1: EDGAR RSS Listener

**File:** `ascent/data/ingest/edgar_listener.py`

### Steps
- [ ] 1.1 Write `poll_edgar_rss(filing_types=("8-K", "8-K/A", "10-Q")) -> list[dict]`. Fetches `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={filing_type}&dateb=&owner=include&count=40&output=atom` for each filing type. Parses Atom XML. Returns list of `{"cik": str, "company_name": str, "filing_type": str, "filed_at": datetime, "url": str}`.
- [ ] 1.2 Write `extract_8k_text(filing_url) -> str`. Fetches the 8-K index page, finds the primary document (`.htm` or `.txt`), downloads it, strips HTML tags, returns the first 4,000 characters. Respects SEC fair-use rate limit: max 10 requests/second. Add `time.sleep(0.11)` between requests.
- [ ] 1.3 Write `is_universe_company(company_name, cik) -> bool`. Checks if the filing company is in the Ascent universe. Match by CIK (reliable) if available in a CIK-to-symbol mapping file (`data_cache/cik_to_symbol.json`). Fall back to fuzzy name match using `difflib.SequenceMatcher` with threshold 0.85.
- [ ] 1.4 Write `build_cik_map(symbols) -> dict`. Fetches CIK for each symbol via EDGAR company search API (`https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&dateRange=custom&startdt=2020-01-01`). Saves to `data_cache/cik_to_symbol.json`. Run once on setup; incremental updates when new symbols added.
- [ ] 1.5 Write `get_recent_filings(lookback_minutes=10) -> list[dict]`. Returns only filings filed within the last `lookback_minutes`. Uses `filed_at` timestamp for filtering. Called every 5 minutes by the polling loop.
- [ ] 1.6 Deduplication: maintain `data_cache/seen_filings.json` — a set of `{filing_url}` strings already processed. Never process the same filing twice across restarts.

---

## Task 2: Event Classifier

**File:** `ascent/alpha/event_alpha.py`

### Steps
- [ ] 2.1 Write `classify_event(filing_text, company_name, filing_type) -> dict`. Calls Haiku with a structured prompt. Returns `{"symbol": str, "direction": "buy"|"sell"|"neutral", "conviction": float (0–1), "urgency": "high"|"medium"|"low", "category": str, "rationale": str}`. Use `generate_structured()` from `ascent/llm/client.py` with a JSON schema.
- [ ] 2.2 Prompt engineering: the prompt instructs Haiku to classify only on hard facts (revenue miss/beat vs. expectation, guidance cut/raise, material adverse events, regulatory approval/rejection). It must not speculate on intent. It must output "neutral" when ambiguous. Provide five few-shot examples in the system prompt covering: earnings beat (+conviction 0.8), guidance cut (-conviction 0.7), routine filing (neutral), lawsuit filing (-conviction 0.6), product approval (+conviction 0.9).
- [ ] 2.3 Write `classify_congressional_trade(disclosure) -> dict`. Input: `{"senator": str, "symbol": str, "transaction_type": "purchase"|"sale", "amount_range": str, "filed_date": date}`. Returns `{"symbol": str, "direction": "buy"|"sell", "conviction": 0.4, "urgency": "low", "category": "congressional_trade", "rationale": str}`. Congressional trades always get conviction 0.4 (medium-low, not a timing signal — disclosure lag is 30–45 days). No LLM call needed — rule-based.
- [ ] 2.4 Write `classify_options_anomaly(symbol, iv_zscore, put_call_ratio_zscore) -> dict`. Rule-based (no LLM): IV z-score > 2.5 AND put/call z-score > 2.0 → `{"direction": "sell", "conviction": 0.6, "urgency": "high"}`. IV z-score > 2.5 only → `{"direction": "sell", "conviction": 0.4, "urgency": "medium"}`. Unusual call buying (put/call < −2.0) → `{"direction": "buy", "conviction": 0.5, "urgency": "medium"}`.
- [ ] 2.5 All classifiers: return `{"direction": "neutral"}` on any exception. Never propagate.

---

## Task 3: Capitol Trades Listener

**File:** `ascent/data/ingest/capitol_trades.py`

### Steps
- [ ] 3.1 Write `fetch_recent_disclosures(lookback_hours=4) -> list[dict]`. Queries the House stock disclosure search API (`https://efts.house.gov/LATEST/search-index?q=&dateRange=custom&startdt={yesterday}&enddt={today}`). Returns list of `{"senator": str, "symbol": str, "transaction_type": str, "amount_range": str, "filed_date": date, "transaction_date": date}`. Parse JSON response.
- [ ] 3.2 For Senate disclosures, use the Senate eFD API (`https://efts.senate.gov/LATEST/search-index`). Same structure.
- [ ] 3.3 Filter to universe symbols only. Deduplicate against `data_cache/seen_disclosures.json`.
- [ ] 3.4 Note: disclosure lag means the transaction happened 30–45 days ago. Conviction is capped at 0.4 accordingly (signal is real but not fresh). This is handled in `classify_congressional_trade()`.

---

## Task 4: Options Anomaly Scanner

**File:** `ascent/data/ingest/options_scanner.py`

### Steps
- [ ] 4.1 Write `compute_iv_baseline(symbol, lookback_days=30) -> tuple[float, float]`. Returns `(mean_iv, std_iv)` computed from the last `lookback_days` of IV data. IV data source: Alpaca options chain snapshot API (`/v2/options/snapshots/{symbol}`). Cache IV baselines in `data_cache/iv_baseline.parquet` and update daily.
- [ ] 4.2 Write `scan_options_anomalies(symbols, zscore_threshold=2.5) -> list[dict]`. For each symbol, fetch current IV and put/call ratio from Alpaca. Compute z-scores against baseline. Return list of `{"symbol": str, "iv_zscore": float, "pc_ratio_zscore": float}` for all symbols exceeding threshold.
- [ ] 4.3 Rate limit: Alpaca options API allows 200 requests/minute. Scan the top-200 by ADV from the universe (largest names have most reliable options data). Run every 15 minutes.
- [ ] 4.4 Graceful fallback: if Alpaca options API is unavailable (paper accounts may have restricted access), disable the options scanner and log a one-time warning. Never block the event agent loop.

---

## Task 5: Event Runner (Sizing + Execution)

**File:** `ascent/execution/event_runner.py`

### Steps
- [ ] 5.1 Add `EVENT_TRADING_ENABLED = False` at the top of this file. When False, all event decisions are logged but no orders are submitted. Must be manually set to True after paper-mode validation.
- [ ] 5.2 Write `compute_event_size(conviction, urgency, nav, max_event_pct=0.005) -> float`. Returns a dollar size for the event trade. Base: `conviction × max_event_pct × NAV`. Urgency multiplier: high → 1.0, medium → 0.6, low → 0.3. Result capped at `max_event_pct × NAV` (0.5% NAV hard cap). Minimum threshold: $500 (below this, skip trade to avoid fractional-share complications).
- [ ] 5.3 Write `submit_event_trade(symbol, direction, size_dollars, rationale, event_id) -> dict`. Calls `order_engine.py` with a limit order at bid+1tick (buy) or ask-1tick (sell). Limit order valid for 5 minutes (GTC with 5-minute cancel). Returns fill result or `{"status": "expired"}` if not filled.
- [ ] 5.4 Event trades > 1% NAV go through the approval gate (`execution/pending_approvals.json`) with a 10-minute timeout (shorter than the 30-minute daily trade timeout — event signals decay fast).
- [ ] 5.5 Write `log_event_trade(event_id, classification, size, fill, nav) -> None`. Appends to `logs/event_trades.jsonl`: `{"event_id", "timestamp", "source", "symbol", "direction", "conviction", "urgency", "category", "rationale", "size_dollars", "fill_price", "nav_at_decision", "status"}`.
- [ ] 5.6 Write `compute_event_ic(lookback_days=20) -> dict`. Reads `logs/event_trades.jsonl`, fetches current prices for all event trades, computes Spearman IC between direction (buy=+1, sell=-1) and 5/10/20-day forward returns. Returns `{"ic_5d": float, "ic_10d": float, "ic_20d": float, "n_trades": int}`. Called weekly by `run_all_agents.py` and written to `logs/skill_scores_log.jsonl`.

---

## Task 6: Event Agent Main Loop

**File:** `agents/event_agent.py`

### Steps
- [ ] 6.1 Write `run_event_agent()`. Runs a polling loop with the following schedule: EDGAR poll every 5 minutes, Capitol Trades every 60 minutes, options scan every 15 minutes. Loop exits at 16:10 ET or when a stop event is set.
- [ ] 6.2 Each event source calls its classifier, then calls `event_runner.submit_event_trade()` for non-neutral signals above a conviction threshold (EDGAR: ≥ 0.6, Capitol Trades: ≥ 0.4, options: ≥ 0.5).
- [ ] 6.3 Rate limit events per symbol: max 1 event trade per symbol per day. If a symbol already has an event trade today, skip. This prevents piling into a name from multiple overlapping signals.
- [ ] 6.4 Write `start_event_agent_thread() -> threading.Thread`. Returns a daemon thread running `run_event_agent()`. Called in `run_all_agents.py` on weekdays during market hours.
- [ ] 6.5 The thread is a daemon — it dies when the main process exits. No cleanup needed.

---

## Task 7: Daily Runner Integration

**File:** `ascent/execution/eod_runner.py` (modify) and `run_all_agents.py` (modify)

### Steps
- [ ] 7.1 In `eod_runner.py`: write `get_event_positions_today() -> dict[str, float]`. Reads `logs/event_trades.jsonl`, returns `{symbol: net_direction}` for all event trades submitted today that resulted in fills. Net direction: +1 if net long, -1 if net short.
- [ ] 7.2 In `eod_runner.py`: when computing the daily rebalance trade list, subtract event-held positions before sizing. If the event agent already bought 0.3% AAPL today, the daily rebalance for AAPL buys 0.3% less (or skips if target is < event position).
- [ ] 7.3 In `run_all_agents.py`: at the top of the main loop (before agent runs), call `start_event_agent_thread()` on weekdays between 9:30 and 15:45 ET. The thread is non-blocking.
- [ ] 7.4 Weekly IC computation: add `compute_event_ic()` to the Sunday block in `run_all_agents.py`.

---

## Task 8: Tests

**File:** `tests/test_event_agent.py` — 16 tests

- [ ] `test_edgar_rss_returns_list` — poll returns a list (may be empty if mocked, never raises)
- [ ] `test_edgar_deduplication` — filing processed twice → only one log entry
- [ ] `test_cik_map_build_and_lookup` — known ticker maps to expected CIK
- [ ] `test_is_universe_company_match` — known S&P 500 name matched
- [ ] `test_is_universe_company_no_match` — random string returns False
- [ ] `test_classify_event_earnings_beat` — positive 8-K text → direction=buy, conviction≥0.6
- [ ] `test_classify_event_guidance_cut` — negative text → direction=sell
- [ ] `test_classify_event_neutral_on_ambiguous` — routine administrative filing → neutral
- [ ] `test_classify_event_returns_neutral_on_exception` — LLM failure → neutral, no raise
- [ ] `test_classify_congressional_trade_purchase` — purchase → buy, conviction=0.4
- [ ] `test_classify_options_anomaly_iv_spike_sell` — high IV z-score → sell
- [ ] `test_classify_options_anomaly_call_buying_buy` — unusual call volume → buy
- [ ] `test_compute_event_size_respects_cap` — never exceeds max_event_pct × NAV
- [ ] `test_compute_event_size_minimum_threshold` — below $500 → returns 0
- [ ] `test_event_trading_disabled_no_order_submitted` — EVENT_TRADING_ENABLED=False → no order call
- [ ] `test_compute_event_ic_correct_direction_sign` — correct trades have positive IC

---

## Acceptance Criteria

1. Event agent thread starts and runs without blocking the daily runner
2. EDGAR poll processes at least one 8-K in test mode within 5 minutes during market hours
3. All event decisions logged to `logs/event_trades.jsonl` regardless of whether order is submitted
4. `EVENT_TRADING_ENABLED = False` guard prevents any Alpaca call (verified in test)
5. Daily runner correctly subtracts event positions before sizing rebalance trades
6. All 16 tests passing; full suite passing
7. 30-day paper-mode validation period before `EVENT_TRADING_ENABLED` is set to True
