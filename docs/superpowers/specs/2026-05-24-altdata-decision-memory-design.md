# Alt Data Pipeline + Decision Memory Enrichment
**Date:** 2026-05-24  
**Approach:** Approach B — alt data as immediate AI PM context; IC gate guards alpha stack entry only

---

## Goal

Close the institutional gap on two fronts:
1. **Data moat (C):** Get all four alt data sources (SEC, transcripts, Reddit, Trends) actually flowing with data so the IC gate can eventually pass
2. **Compounding knowledge (B):** Enrich the AI PM's decision memory with alt data context at override time, so the ML conviction model trains on richer features

---

## Architecture

```
python run_all_agents.py  (single command, single launchd job)
  │
  ├── _collect_altdata()                    ← NEW, runs first
  │     ├── SEC: freshness check → re-fetch if >90 days stale  
  │     ├── Transcripts: EDGAR 8-K fetch for portfolio symbols
  │     ├── Reddit: daily, portfolio symbols only (~20)
  │     └── Trends: portfolio daily (~100s), full 901 on Sundays
  │
  ├── regime memory + authority update
  ├── agents (parallel)
  ├── AI PM — gets fresh alt data via existing get_sec_signal / get_transcript_signal tools
  │           + decision memory now records alt data snapshot per override
  ├── debate → verdict
  └── execution / monitoring
```

No new launchd plist. Every source wrapped in `try/except` — one failure never blocks agent run.

---

## Phase 1: Alt Data Collection

### SEC filings (`ascent/data/ingest/sec_filings.py`)

**Bug fix:** `build_sec_signal_panel` classifies 5 signals but only writes `revenue_momentum` to parquet. Fix: write all signals to `data_cache/altdata_sec_detail.json` (keyed by symbol), keep parquet with `revenue_momentum` as alpha signal (so `altdata_alpha.py` is unchanged).

**New — Risk Factors extraction:** Add `extract_risk_factors_section()` alongside existing `extract_mda_section()`. Enhance `classify_filing_signal()` with 2 new signals:
- `risk_trend`: −1.0 (new risks appearing) to +1.0 (existing risks diminishing)
- `guidance_specificity`: 0.0 (vague language) to 1.0 (specific numerical guidance)

Total SEC signals: 7 (revenue_momentum, margin_trend, tone, liquidity_risk, guidance, risk_trend, guidance_specificity)

**New — YoY comparison:** `classify_filing_signal()` receives previous quarter's signals as context. Haiku compares current vs prior → adds `yoy_improvement` float (−1.0 deteriorating / +1.0 improving). Stored in `altdata_sec_detail.json` alongside the 7 primary signals.

**Freshness gate:** `update_sec_signals()` checks last row date in existing parquet before fetching. If last entry < 90 days old, skip. This prevents redundant EDGAR calls on every daily run.

### Earnings transcripts (`ascent/data/ingest/earnings_transcripts.py`)

**Bug fix:** `update_transcript_signals()` expects pre-fetched records but nothing calls EDGAR to get them. `altdata_transcripts.parquet` is currently always empty.

**New — `fetch_recent_8k_transcripts(symbols)`:** Hits EDGAR search for 8-K Item 2.02 (Results of Operations) filings in the last 90 days per symbol. Extracts text, returns `[{symbol, earnings_date, transcript_text}]` list that existing `update_transcript_signals()` already consumes. No changes to the classification or panel-building logic.

### Reddit sentiment (`ascent/data/ingest/reddit_sentiment.py`)

No structural changes. `build_reddit_panel(portfolio_symbols)` called daily. Requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` env vars — returns empty DataFrame silently if absent.

### Google Trends (`ascent/data/ingest/google_trends.py`)

No structural changes. `update_trends_signals(portfolio_symbols)` called daily (~100s). Full 901-symbol sweep (`update_trends_signals(all_symbols)`) runs only on Sundays to avoid 75-minute daily overhead.

### Collection orchestrator (`run_all_agents.py`)

New `_collect_altdata(portfolio_symbols, all_symbols)` function at top of `run_all_agents.py`:

```python
def _collect_altdata(portfolio_symbols, all_symbols):
    today = date.today()
    for name, fn in [
        ("SEC",         lambda: update_sec_signals(all_symbols)),
        ("Transcripts", lambda: _fetch_and_update_transcripts(portfolio_symbols)),
        ("Reddit",      lambda: build_reddit_panel(portfolio_symbols)),
        ("Trends",      lambda: update_trends_signals(
                            all_symbols if today.weekday() == 6  # Sunday
                            else portfolio_symbols)),
    ]:
        try:
            fn()
        except Exception as e:
            log.warning("[AltData] %s collection failed: %s", name, e)
```

Called before any agent runs, after loading config.

---

## Phase 2: Decision Memory Enrichment

### `ascent/memory/decision_memory.py`

`OverrideRecord` gains 4 new optional fields:

```python
@dataclass
class OverrideRecord:
    # ... existing fields unchanged ...
    wedge_21d: Optional[float] = None
    # NEW
    sec_tone: Optional[float] = None
    transcript_sentiment: Optional[float] = None
    reddit_buzz: Optional[float] = None
    trends_direction: Optional[float] = None
```

New private helper `_read_altdata_context(symbol) -> dict` reads from cached parquets/JSON at ingest time. Returns `{sec_tone, transcript_sentiment, reddit_buzz, trends_direction}` with `None` for any missing source. Called automatically inside `ingest_override()` — no caller changes required.

Backward compatible: existing JSONL records without these fields deserialize fine (`OverrideRecord(**row)` uses `Optional` defaults).

---

## Phase 3: ML Conviction Model

### `ascent/strategy/conviction_gate.py`

`MIN_ML_CASES = 30` already defined. When `n_cases >= MIN_ML_CASES`:

**Features (per matured override record):**
- `override_type` → one-hot (5 types)
- `regime` → one-hot (5 regimes)
- `momentum_252d` → float, 0.0 if None
- `sec_tone` → float, 0.0 if None
- `transcript_sentiment` → float, 0.0 if None
- `reddit_buzz` → float, 0.0 if None
- `trends_direction` → float, 0.0 if None

**Target:** `1` if `wedge_21d > 0`, else `0`

**Model:** `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=500)`
- Retrained when `n_cases` increases since last train
- Cached to `data_cache/conviction_model.pkl`
- Falls back to existing rules-based logic if model probability confidence < 0.55 (prevents overconfident early model from overriding sound heuristics)

`evaluate()` signature unchanged. Internal routing:

```python
if n_cases >= MIN_ML_CASES:
    prob = _ml_predict(features)
    if abs(prob - 0.5) >= 0.05:   # confident enough
        return _ml_gate_result(prob, n_cases, wr)
# fall through to existing rules
```

---

## Files Changed

| File | Type | Change |
|------|------|--------|
| `run_all_agents.py` | Modified | Add `_collect_altdata()` block before agent runs |
| `ascent/data/ingest/sec_filings.py` | Modified | Store all 7+1 signals; Risk Factors extractor; YoY; freshness gate |
| `ascent/data/ingest/earnings_transcripts.py` | Modified | Add `fetch_recent_8k_transcripts()` |
| `ascent/memory/decision_memory.py` | Modified | 4 new fields on `OverrideRecord`; `_read_altdata_context()` helper |
| `ascent/strategy/conviction_gate.py` | Modified | ML model training + inference when n≥30 |
| `data_cache/altdata_sec_detail.json` | New | Per-symbol full signal store (all 7+1 signals) |
| `data_cache/conviction_model.pkl` | New | Trained logistic regression gate model |

---

## Integrity Constraints (unchanged)

- Alt data never enters alpha stack until IC gate passes (`altdata_alpha.py` untouched)
- `altdata_validator.py` IC gate logic unchanged — runs weekly Sunday
- ML conviction model falls back to rules if not confident — never blocks a structural override (data_quality, correlation_risk, news_event)
- All sources fail silently — agent run never blocked by alt data failure

---

## Tests Required

- `tests/test_sec_filings.py` — risk factors extraction, YoY comparison, all-7-signals parquet write, freshness gate
- `tests/test_earnings_transcripts.py` — 8-K fetcher returns correct record format
- `tests/test_decision_memory.py` — new fields serialize/deserialize, backward compat with old records, `_read_altdata_context` returns None gracefully when caches absent
- `tests/test_conviction_gate.py` — ML path activates at n=30, falls back when prob < 0.55
