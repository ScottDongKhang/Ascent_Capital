# Alt Data Pipeline + Decision Memory Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get all four alt data sources (SEC, transcripts, Reddit, Trends) flowing as a single integrated collection pass in `run_all_agents.py`, fix two broken pipelines, deepen SEC signal extraction, and enrich the AI PM's decision memory with alt data context so the ML conviction model trains on richer features.

**Architecture:** `run_all_agents.py` gains `_collect_altdata()` that runs before agents. SEC pipeline stores all 7 signals to a detail JSON (currently drops 4 of 5). Transcripts pipeline gains an 8-K EDGAR fetcher (currently always empty). Decision memory records alt data snapshots at override time. Conviction gate adds logistic regression when n≥30 matured cases.

**Tech Stack:** Python 3.12, pandas, sklearn (LogisticRegression), requests, EDGAR EFTS API, existing `ascent/llm/client.py` Haiku wrapper, existing `ascent/data/ingest/` modules.

**Spec:** `docs/superpowers/specs/2026-05-24-altdata-decision-memory-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ascent/data/ingest/sec_filings.py` | Modify | Store all 7 signals; detail JSON; freshness gate; Risk Factors; YoY |
| `ascent/data/ingest/earnings_transcripts.py` | Modify | Add `fetch_recent_8k_transcripts()` + HTTP helper |
| `run_all_agents.py` | Modify | Add `_collect_altdata()` + `_get_portfolio_symbols()` before agent runs |
| `ascent/memory/decision_memory.py` | Modify | 4 new `OverrideRecord` fields; `_read_altdata_context()` auto-fill |
| `ascent/strategy/conviction_gate.py` | Modify | ML model path when n≥30; optional `symbol` param on `evaluate()` |
| `tests/test_altdata_pipeline.py` | Modify | Add tests for detail JSON, freshness gate, Risk Factors, YoY, 8-K fetcher |
| `tests/test_decision_memory.py` | Modify | Add tests for new fields, backward compat, context reader |
| `tests/test_conviction_gate.py` | Modify | Add tests for ML activation, confidence fallback, symbol param |

---

## Task 1: SEC — Store All 5 Signals in Detail JSON + Freshness Gate

**Files:**
- Modify: `ascent/data/ingest/sec_filings.py`
- Modify: `tests/test_altdata_pipeline.py`

**What's broken today:** `build_sec_signal_panel` classifies 5 signals per symbol via Haiku but only writes `revenue_momentum` to the parquet. The other 4 (margin_trend, tone, liquidity_risk, guidance) are silently discarded. Also, `update_sec_signals` re-fetches EDGAR on every daily run even when the cache is fresh (quarterly filings don't change daily).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_altdata_pipeline.py`:

```python
def test_update_sec_signals_writes_detail_json(tmp_path):
    """All 5 classified signals written to altdata_sec_detail.json keyed by symbol."""
    import json
    from unittest.mock import patch
    from ascent.data.ingest.sec_filings import update_sec_signals

    mock_text = "Revenue grew 15%. Margins expanded. Guidance raised."
    mock_signals = {
        "revenue_momentum": 0.7, "margin_trend": 0.5, "tone": 0.6,
        "liquidity_risk": 0.1, "guidance": 0.8,
    }
    detail_path = tmp_path / "altdata_sec_detail.json"
    cache_path  = tmp_path / "altdata_sec.parquet"

    with patch("ascent.data.ingest.sec_filings._get", return_value='{"hits":{"hits":[{"_source":{"biz_location":"http://fake"}}]}}'), \
         patch("ascent.data.ingest.sec_filings.generate_structured", return_value=mock_signals), \
         patch("ascent.data.ingest.sec_filings._CACHE_PATH", cache_path), \
         patch("ascent.data.ingest.sec_filings._DETAIL_PATH", detail_path):
        update_sec_signals(["AAPL"])

    assert detail_path.exists()
    detail = json.loads(detail_path.read_text())
    assert "AAPL" in detail
    for k in ["revenue_momentum", "margin_trend", "tone", "liquidity_risk", "guidance"]:
        assert k in detail["AAPL"]


def test_update_sec_signals_freshness_gate(tmp_path):
    """If cache last row < 90 days ago, skip EDGAR fetch entirely."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from ascent.data.ingest.sec_filings import update_sec_signals

    # Write a fresh parquet with today's date
    fresh = pd.DataFrame({"AAPL": [0.5]}, index=pd.DatetimeIndex([pd.Timestamp.today()]))
    fresh.index.name = "date"
    cache_path = tmp_path / "altdata_sec.parquet"
    fresh.to_parquet(cache_path)

    mock_get = MagicMock()
    with patch("ascent.data.ingest.sec_filings._get", mock_get), \
         patch("ascent.data.ingest.sec_filings._CACHE_PATH", cache_path):
        result = update_sec_signals(["AAPL"])

    mock_get.assert_not_called()
    assert not result.empty


def test_load_sec_detail_returns_empty_dict_when_missing(tmp_path):
    """load_sec_detail returns {} gracefully when detail JSON absent."""
    from unittest.mock import patch
    from ascent.data.ingest.sec_filings import load_sec_detail
    with patch("ascent.data.ingest.sec_filings._DETAIL_PATH", tmp_path / "missing.json"):
        assert load_sec_detail("AAPL") == {}
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_update_sec_signals_writes_detail_json tests/test_altdata_pipeline.py::test_update_sec_signals_freshness_gate tests/test_altdata_pipeline.py::test_load_sec_detail_returns_empty_dict_when_missing -v 2>&1 | tail -20
```

Expected: FAIL — `_DETAIL_PATH`, `load_sec_detail` not defined.

- [ ] **Step 3: Implement — add detail JSON helpers + freshness gate**

In `ascent/data/ingest/sec_filings.py`, add after the existing `_CACHE_PATH` line:

```python
_DETAIL_PATH = Path("data_cache/altdata_sec_detail.json")
_FRESHNESS_DAYS = 90  # re-fetch only if cache older than this
```

Add these two functions after `load_sec_signals()`:

```python
def _update_detail_cache(symbol: str, signals: dict, period_end: date) -> None:
    """Persist full signal dict for a symbol to altdata_sec_detail.json."""
    detail: dict = {}
    if _DETAIL_PATH.exists():
        try:
            detail = json.loads(_DETAIL_PATH.read_text())
        except Exception:
            pass
    detail[symbol] = {
        **signals,
        "period_end": period_end.isoformat(),
        "as_of": date.today().isoformat(),
    }
    _DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DETAIL_PATH.write_text(json.dumps(detail, indent=2))


def load_sec_detail(symbol: str) -> dict:
    """Return full signal dict for symbol from altdata_sec_detail.json, or {}."""
    if _DETAIL_PATH.exists():
        try:
            return json.loads(_DETAIL_PATH.read_text()).get(symbol, {})
        except Exception:
            pass
    return {}
```

- [ ] **Step 4: Fix `build_sec_signal_panel` to call `_update_detail_cache` and store all signals**

Replace the body of `build_sec_signal_panel` with:

```python
def build_sec_signal_panel(
    symbols: list[str],
    start_date: str,
    end_date: Optional[str] = None,
    filing_type: str = "10-K",
) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today().isoformat()

    rows = []
    for sym in symbols:
        try:
            time.sleep(_SEC_DELAY)
            text = fetch_full_text_filing(sym, filing_type=filing_type,
                                          start_date=start_date, end_date=end_date)
            if not text:
                continue
            period_end = date.today() - timedelta(days=45)
            signal_date = period_end + timedelta(days=45)
            sig = classify_filing_signal(text, sym, period_end)
            _update_detail_cache(sym, sig, period_end)          # ← NEW: persist all signals
            rows.append({"date": signal_date, "symbol": sym, **sig})
        except Exception as e:
            log.warning("[SecFilings] build_sec_signal_panel: symbol %s failed: %s", sym, e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    wide = df.pivot(columns="symbol", values="revenue_momentum")
    date_range = pd.bdate_range(start=start_date, end=end_date)
    wide = wide.reindex(date_range)
    wide = wide.ffill(limit=90)
    wide.index.name = "date"
    return wide
```

- [ ] **Step 5: Add freshness gate to `update_sec_signals`**

Replace the existing `update_sec_signals` function with:

```python
def update_sec_signals(symbols: list[str], lookback_months: int = 12) -> pd.DataFrame:
    """Incremental update — skip entirely if cache is < 90 days stale."""
    start_date = (date.today() - timedelta(days=lookback_months * 30)).isoformat()

    existing = pd.DataFrame()
    if _CACHE_PATH.exists():
        try:
            existing = pd.read_parquet(_CACHE_PATH)
            if not existing.empty:
                last_date = existing.index[-1]
                age_days = (pd.Timestamp.today() - last_date).days
                if age_days < _FRESHNESS_DAYS:
                    log.info("[SecFilings] Cache fresh (last=%s, age=%dd) — skipping EDGAR fetch",
                             last_date.date(), age_days)
                    return existing
        except Exception:
            pass

    panel = build_sec_signal_panel(symbols, start_date=start_date)
    if panel.empty:
        return existing

    if not existing.empty:
        combined = pd.concat([existing, panel]).groupby(level=0).last()
    else:
        combined = panel

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_CACHE_PATH)
    log.info("[SecFilings] Updated SEC signal panel: %s", combined.shape)
    return combined
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_update_sec_signals_writes_detail_json tests/test_altdata_pipeline.py::test_update_sec_signals_freshness_gate tests/test_altdata_pipeline.py::test_load_sec_detail_returns_empty_dict_when_missing -v 2>&1 | tail -20
```

Expected: all 3 PASS.

- [ ] **Step 7: Run full test suite to check for regressions**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py -v 2>&1 | tail -30
```

Expected: all existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add ascent/data/ingest/sec_filings.py tests/test_altdata_pipeline.py
git commit -m "feat: SEC pipeline — store all signals in detail JSON, add freshness gate"
```

---

## Task 2: SEC — Risk Factors Extraction + YoY Comparison

**Files:**
- Modify: `ascent/data/ingest/sec_filings.py`
- Modify: `tests/test_altdata_pipeline.py`

**What this adds:** Two new classification signals (`risk_trend`, `guidance_specificity`) extracted from the Risk Factors section. YoY comparison adds `yoy_improvement` by comparing current signals to previous quarter's. Total SEC signals goes from 5 → 8.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_altdata_pipeline.py`:

```python
def test_extract_risk_factors_section_finds_content():
    """ITEM 1A present → returns risk factors text, not empty."""
    from ascent.data.ingest.sec_filings import extract_risk_factors_section
    text = (
        "ITEM 1. BUSINESS\nWe sell widgets.\n"
        "ITEM 1A. RISK FACTORS\n"
        "Tariff risk could materially impact our supply chain.\n"
        "Competition from low-cost providers is increasing.\n"
        "ITEM 1B. UNRESOLVED STAFF COMMENTS\nNone.\n"
    )
    result = extract_risk_factors_section(text)
    assert "Tariff risk" in result
    assert "UNRESOLVED" not in result.upper()


def test_extract_risk_factors_section_returns_empty_when_absent():
    """No ITEM 1A → returns empty string."""
    from ascent.data.ingest.sec_filings import extract_risk_factors_section
    assert extract_risk_factors_section("ITEM 1. BUSINESS\nWe sell widgets.") == ""


def test_classify_filing_signal_returns_7_keys():
    """With risk_factors_text provided → dict has risk_trend and guidance_specificity."""
    from ascent.data.ingest.sec_filings import classify_filing_signal
    mock_result = {
        "revenue_momentum": 0.6, "margin_trend": 0.3, "tone": 0.5,
        "liquidity_risk": 0.1, "guidance": 0.4,
        "risk_trend": 0.2, "guidance_specificity": 0.7,
    }
    with patch("ascent.data.ingest.sec_filings.generate_structured", return_value=mock_result):
        result = classify_filing_signal(
            "Revenue grew 15%.", "AAPL", date.today(),
            risk_factors_text="Supply chain risks are diminishing.",
        )
    assert "risk_trend" in result
    assert "guidance_specificity" in result
    assert result["risk_trend"] == 0.2


def test_classify_filing_signal_yoy_improvement():
    """Passing prev_signals → yoy_improvement field returned."""
    from ascent.data.ingest.sec_filings import classify_filing_signal
    base_signals = {
        "revenue_momentum": 0.6, "margin_trend": 0.3, "tone": 0.5,
        "liquidity_risk": 0.1, "guidance": 0.4,
        "risk_trend": 0.0, "guidance_specificity": 0.5,
    }
    yoy_mock = {"yoy_improvement": 0.6}
    with patch("ascent.data.ingest.sec_filings.generate_structured", side_effect=[base_signals, yoy_mock]):
        prev = {"revenue_momentum": 0.2, "tone": 0.1}
        result = classify_filing_signal(
            "Revenue accelerated strongly.", "AAPL", date.today(),
            prev_signals=prev,
        )
    assert "yoy_improvement" in result
    assert result["yoy_improvement"] == 0.6
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_extract_risk_factors_section_finds_content tests/test_altdata_pipeline.py::test_extract_risk_factors_section_returns_empty_when_absent tests/test_altdata_pipeline.py::test_classify_filing_signal_returns_7_keys tests/test_altdata_pipeline.py::test_classify_filing_signal_yoy_improvement -v 2>&1 | tail -20
```

Expected: FAIL — `extract_risk_factors_section` not defined, `classify_filing_signal` has no `risk_factors_text` or `prev_signals` params.

- [ ] **Step 3: Add Risk Factors extractor**

In `ascent/data/ingest/sec_filings.py`, add after the `_NEXT_SECTION_PATTERNS` list:

```python
_RISK_PATTERNS = [
    r"ITEM\s+1A[\.A]?\s*[\.\-–—]?\s*RISK\s+FACTORS",
    r"RISK\s+FACTORS",
]
_RISK_NEXT_SECTION = [
    r"ITEM\s+1B[\.\s]",
    r"ITEM\s+2[\.\s]",
    r"UNRESOLVED\s+STAFF\s+COMMENTS",
    r"PROPERTIES",
]


def extract_risk_factors_section(full_text: str) -> str:
    """Extract Item 1A Risk Factors section. Returns empty string if absent."""
    upper = full_text.upper()
    start_idx = -1
    for pat in _RISK_PATTERNS:
        m = re.search(pat, upper)
        if m:
            start_idx = m.start()
            break
    if start_idx == -1:
        return ""
    end_idx = len(full_text)
    for pat in _RISK_NEXT_SECTION:
        m = re.search(pat, upper[start_idx + 100:])
        if m:
            candidate = start_idx + 100 + m.start()
            if candidate < end_idx:
                end_idx = candidate
    section = full_text[start_idx:end_idx]
    section = re.sub(r"<[^>]+>", " ", section)
    section = re.sub(r"\s+", " ", section).strip()
    return section[:4000]
```

- [ ] **Step 4: Update `classify_filing_signal` with 7 signals + YoY**

Replace `_CLASSIFY_SYSTEM` and `classify_filing_signal` with:

```python
_CLASSIFY_SYSTEM = """You are a financial analyst extracting structured signals from 10-K/10-Q filings.

Analyze the MD&A and Risk Factors text and return JSON with these seven float fields:
- revenue_momentum: -1.0 (decelerating/declining) to +1.0 (accelerating/growing strongly)
- margin_trend: -1.0 (contracting) to +1.0 (expanding)
- tone: -1.0 (defensive/cautious) to +1.0 (confident/optimistic)
- liquidity_risk: 0.0 (no concern) to 1.0 (severe: covenant breach, going concern, cash burn)
- guidance: -1.0 (lowered) to +1.0 (raised), 0.0 if maintained or absent
- risk_trend: -1.0 (significant new risks appearing) to +1.0 (existing risks diminishing or resolved)
- guidance_specificity: 0.0 (vague qualitative language) to 1.0 (specific numerical targets)

Be conservative — only assign extreme values on clear hard facts."""

_YOY_SYSTEM = """You are comparing two consecutive quarterly SEC filings for the same company.
Return JSON with one field: yoy_improvement (-1.0 = significantly worse, 0.0 = unchanged, +1.0 = significantly better).
Base your assessment on changes in tone, revenue trajectory, margin direction, and guidance."""


def classify_filing_signal(
    mda_text: str,
    symbol: str,
    period_end: date,
    risk_factors_text: str = "",
    prev_signals: Optional[dict] = None,
) -> dict:
    """Classify MD&A + Risk Factors into 7 structured signals via Haiku. Returns zeros on failure."""
    _neutral = {
        "revenue_momentum": 0.0, "margin_trend": 0.0, "tone": 0.0,
        "liquidity_risk": 0.0, "guidance": 0.0,
        "risk_trend": 0.0, "guidance_specificity": 0.0,
    }
    try:
        if generate_structured is None:
            return _neutral.copy()
        schema = {
            "type": "object",
            "properties": {
                "revenue_momentum":   {"type": "number"},
                "margin_trend":       {"type": "number"},
                "tone":               {"type": "number"},
                "liquidity_risk":     {"type": "number"},
                "guidance":           {"type": "number"},
                "risk_trend":         {"type": "number"},
                "guidance_specificity": {"type": "number"},
            },
            "required": ["revenue_momentum", "margin_trend", "tone",
                         "liquidity_risk", "guidance", "risk_trend", "guidance_specificity"],
        }
        prompt = (
            f"Company: {symbol}  Period ending: {period_end}\n\n"
            f"MD&A:\n{mda_text[:2500]}"
        )
        if risk_factors_text:
            prompt += f"\n\nRISK FACTORS:\n{risk_factors_text[:2000]}"

        result = generate_structured(prompt=prompt, system=_CLASSIFY_SYSTEM, schema=schema, model=HAIKU_MODEL)
        if not isinstance(result, dict):
            return _neutral.copy()

        ranges = [
            ("revenue_momentum", -1.0, 1.0), ("margin_trend", -1.0, 1.0),
            ("tone", -1.0, 1.0), ("liquidity_risk", 0.0, 1.0),
            ("guidance", -1.0, 1.0), ("risk_trend", -1.0, 1.0),
            ("guidance_specificity", 0.0, 1.0),
        ]
        out = {k: float(max(lo, min(hi, result.get(k, 0.0)))) for k, lo, hi in ranges}

        # YoY comparison if previous quarter signals provided
        if prev_signals:
            try:
                yoy_schema = {"type": "object", "properties": {"yoy_improvement": {"type": "number"}}, "required": ["yoy_improvement"]}
                prev_str = ", ".join(f"{k}={v:.2f}" for k, v in prev_signals.items())
                curr_str = ", ".join(f"{k}={v:.2f}" for k, v in out.items())
                yoy_prompt = f"Previous quarter: {prev_str}\nCurrent quarter: {curr_str}"
                yoy = generate_structured(prompt=yoy_prompt, system=_YOY_SYSTEM, schema=yoy_schema, model=HAIKU_MODEL)
                if isinstance(yoy, dict):
                    out["yoy_improvement"] = float(max(-1.0, min(1.0, yoy.get("yoy_improvement", 0.0))))
            except Exception:
                out["yoy_improvement"] = 0.0

        return out
    except Exception as e:
        log.warning("[SecFilings] classify_filing_signal failed for %s: %s", symbol, e)
        return _neutral.copy()
```

- [ ] **Step 5: Update `build_sec_signal_panel` to pass Risk Factors text**

Replace the per-symbol block inside `build_sec_signal_panel`:

```python
            text = fetch_full_text_filing(sym, filing_type=filing_type,
                                          start_date=start_date, end_date=end_date)
            if not text:
                continue
            period_end = date.today() - timedelta(days=45)
            signal_date = period_end + timedelta(days=45)
            risk_text = extract_risk_factors_section(text)
            prev_sig = load_sec_detail(sym) or None
            sig = classify_filing_signal(text, sym, period_end,
                                         risk_factors_text=risk_text,
                                         prev_signals=prev_sig if prev_sig else None)
            _update_detail_cache(sym, sig, period_end)
            rows.append({"date": signal_date, "symbol": sym, **sig})
```

- [ ] **Step 6: Run all four new tests**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_extract_risk_factors_section_finds_content tests/test_altdata_pipeline.py::test_extract_risk_factors_section_returns_empty_when_absent tests/test_altdata_pipeline.py::test_classify_filing_signal_returns_7_keys tests/test_altdata_pipeline.py::test_classify_filing_signal_yoy_improvement -v 2>&1 | tail -20
```

Expected: all 4 PASS.

- [ ] **Step 7: Run full suite**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py -v 2>&1 | tail -30
```

Expected: all tests pass (existing 5-key test still passes since those keys are still present).

- [ ] **Step 8: Commit**

```bash
git add ascent/data/ingest/sec_filings.py tests/test_altdata_pipeline.py
git commit -m "feat: SEC pipeline — Risk Factors extraction, YoY comparison, 7-signal classify"
```

---

## Task 3: Transcripts — 8-K EDGAR Fetcher

**Files:**
- Modify: `ascent/data/ingest/earnings_transcripts.py`
- Modify: `tests/test_altdata_pipeline.py`

**What's broken today:** `update_transcript_signals()` expects `[{symbol, earnings_date, transcript_text}]` records but nothing ever fetches them. `altdata_transcripts.parquet` is always empty. `get_transcript_signal` AI PM tool always returns zeros.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_altdata_pipeline.py`:

```python
def test_fetch_recent_8k_transcripts_returns_records():
    """EDGAR returns a hit with Item 2.02 content → list with one record."""
    import json
    from unittest.mock import patch
    from ascent.data.ingest.earnings_transcripts import fetch_recent_8k_transcripts

    hit = {"_source": {
        "biz_location": "http://fake-8k-url",
        "file_date": "2026-04-01",
    }}
    search_resp = json.dumps({"hits": {"hits": [hit]}})
    filing_text = (
        "SECURITIES AND EXCHANGE COMMISSION\n"
        "ITEM 2.02 RESULTS OF OPERATIONS AND FINANCIAL CONDITION\n"
        "Revenue increased 20% year over year. Management raised guidance.\n"
    )

    with patch("ascent.data.ingest.earnings_transcripts._get",
               side_effect=[search_resp, filing_text]):
        records = fetch_recent_8k_transcripts(["AAPL"])

    assert len(records) == 1
    assert records[0]["symbol"] == "AAPL"
    assert "transcript_text" in records[0]
    assert "earnings_date" in records[0]
    assert "Revenue increased" in records[0]["transcript_text"]


def test_fetch_recent_8k_transcripts_empty_on_no_hits():
    """No EDGAR hits → returns empty list, no exception."""
    import json
    from unittest.mock import patch
    from ascent.data.ingest.earnings_transcripts import fetch_recent_8k_transcripts

    with patch("ascent.data.ingest.earnings_transcripts._get",
               return_value=json.dumps({"hits": {"hits": []}})):
        records = fetch_recent_8k_transcripts(["AAPL"])

    assert records == []


def test_fetch_recent_8k_transcripts_skips_no_item_2_02():
    """8-K without Item 2.02 section → symbol skipped, no record."""
    import json
    from unittest.mock import patch
    from ascent.data.ingest.earnings_transcripts import fetch_recent_8k_transcripts

    hit = {"_source": {"biz_location": "http://fake", "file_date": "2026-04-01"}}
    search_resp = json.dumps({"hits": {"hits": [hit]}})
    filing_no_item = "EXHIBIT 99.1\nPress release about unrelated matters."

    with patch("ascent.data.ingest.earnings_transcripts._get",
               side_effect=[search_resp, filing_no_item]):
        records = fetch_recent_8k_transcripts(["AAPL"])

    assert records == []
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_returns_records tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_empty_on_no_hits tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_skips_no_item_2_02 -v 2>&1 | tail -20
```

Expected: FAIL — `fetch_recent_8k_transcripts` and `_get` not defined in transcripts module.

- [ ] **Step 3: Add HTTP helper + 8-K fetcher to `earnings_transcripts.py`**

Add these imports at the top of `ascent/data/ingest/earnings_transcripts.py` (after existing imports):

```python
import json
import time
```

Add these constants and functions after the existing module-level constants:

```python
_EDGAR_8K_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22"
    "&dateRange=custom&category=form-type&forms=8-K"
    "&startdt={start}&enddt={end}"
)
_SEC_DELAY = 0.12
_8K_ITEM_PATTERNS = [
    r"ITEM\s+2\.02",
    r"RESULTS\s+OF\s+OPERATIONS\s+AND\s+FINANCIAL\s+CONDITION",
]


def _get(url: str, retries: int = 3) -> Optional[str]:
    """HTTP GET with retry. Returns None on failure."""
    try:
        import requests
        for attempt in range(retries):
            try:
                r = requests.get(
                    url,
                    headers={"User-Agent": "Ascent Capital research@ascentcap.ai"},
                    timeout=15,
                )
                if r.status_code == 200:
                    return r.text
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
            except Exception as e:
                log.debug("[Transcripts] fetch attempt %d failed: %s", attempt + 1, e)
                time.sleep(0.5)
    except Exception as e:
        log.warning("[Transcripts] requests not available: %s", e)
    return None


def fetch_recent_8k_transcripts(
    symbols: list[str],
    lookback_days: int = 90,
) -> list[dict]:
    """
    Fetch recent 8-K Item 2.02 (Results of Operations) filings from EDGAR.
    Returns list of {symbol, earnings_date, transcript_text} for downstream
    update_transcript_signals().
    """
    records = []
    start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    end_date = date.today().isoformat()

    for symbol in symbols:
        try:
            time.sleep(_SEC_DELAY)
            url = _EDGAR_8K_SEARCH.format(symbol=symbol, start=start_date, end=end_date)
            raw = _get(url)
            if not raw:
                continue
            hits = json.loads(raw).get("hits", {}).get("hits", [])
            if not hits:
                continue
            src = hits[0].get("_source", {})
            doc_url = src.get("biz_location") or src.get("_id") or ""
            if not doc_url:
                continue
            time.sleep(_SEC_DELAY)
            text = _get(doc_url)
            if not text:
                continue

            # Find Item 2.02 section
            upper = text.upper()
            item_start = -1
            for pat in _8K_ITEM_PATTERNS:
                m = re.search(pat, upper)
                if m:
                    item_start = m.start()
                    break
            if item_start == -1:
                log.debug("[Transcripts] No Item 2.02 found in 8-K for %s", symbol)
                continue

            excerpt = text[item_start:item_start + 6000]
            excerpt = re.sub(r"<[^>]+>", " ", excerpt)
            excerpt = re.sub(r"\s+", " ", excerpt).strip()

            filed_str = src.get("file_date") or src.get("display_date_filed") or end_date
            try:
                earnings_date = date.fromisoformat(filed_str[:10])
            except Exception:
                earnings_date = date.today()

            records.append({
                "symbol": symbol,
                "earnings_date": earnings_date,
                "transcript_text": excerpt,
            })
        except Exception as e:
            log.warning("[Transcripts] 8-K fetch failed for %s: %s", symbol, e)

    return records
```

- [ ] **Step 4: Run the three new tests**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_returns_records tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_empty_on_no_hits tests/test_altdata_pipeline.py::test_fetch_recent_8k_transcripts_skips_no_item_2_02 -v 2>&1 | tail -20
```

Expected: all 3 PASS.

- [ ] **Step 5: Run full altdata suite**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py -v 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ascent/data/ingest/earnings_transcripts.py tests/test_altdata_pipeline.py
git commit -m "feat: transcripts pipeline — add EDGAR 8-K Item 2.02 fetcher"
```

---

## Task 4: Collection Orchestrator — Wire into `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py`
- Modify: `tests/test_altdata_pipeline.py`

**What this does:** Adds `_get_portfolio_symbols()` and `_collect_altdata()` to `run_all_agents.py`. Called once per daily run, right after `run_hub()` completes, before agents start. Each source wrapped in `try/except` — nothing can block the agent run.

- [ ] **Step 1: Write failing test**

Add to `tests/test_altdata_pipeline.py`:

```python
def test_collect_altdata_completes_when_all_sources_fail():
    """All four sources raise exceptions → _collect_altdata completes without raising."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from unittest.mock import patch
    # Import the function we're about to add
    from run_all_agents import _collect_altdata

    with patch("ascent.data.ingest.sec_filings.update_sec_signals", side_effect=Exception("SEC down")), \
         patch("ascent.data.ingest.earnings_transcripts.fetch_recent_8k_transcripts", side_effect=Exception("EDGAR down")), \
         patch("ascent.data.ingest.earnings_transcripts.update_transcript_signals", side_effect=Exception("write fail")), \
         patch("ascent.data.ingest.reddit_sentiment.build_reddit_panel", side_effect=Exception("no reddit key")), \
         patch("ascent.data.ingest.google_trends.update_trends_signals", side_effect=Exception("rate limited")):
        # Must not raise
        _collect_altdata(portfolio_symbols=["AAPL", "MSFT"], all_symbols=["AAPL", "MSFT", "GOOGL"])
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_collect_altdata_completes_when_all_sources_fail -v 2>&1 | tail -10
```

Expected: FAIL — `_collect_altdata` not importable from `run_all_agents`.

- [ ] **Step 3: Add `_get_portfolio_symbols` and `_collect_altdata` to `run_all_agents.py`**

Add after the existing `check_halt_state` function (around line 218), before `_fill_wedge_and_decision_outcomes`:

```python
def _get_portfolio_symbols() -> list[str]:
    """Return symbols with nonzero weight in the current merged portfolio."""
    try:
        p = Path("execution/merged_weights.json")
        if p.exists():
            weights = json.loads(p.read_text())
            return [s for s, w in weights.items()
                    if isinstance(w, (int, float)) and w > 0]
    except Exception:
        pass
    return []


def _collect_altdata(portfolio_symbols: list, all_symbols: list) -> None:
    """
    Run all four alt data collection sources before agents start.
    Each source is independently wrapped — one failure never blocks the rest.
    """
    from ascent.data.ingest.sec_filings import update_sec_signals
    from ascent.data.ingest.earnings_transcripts import (
        fetch_recent_8k_transcripts, update_transcript_signals,
    )
    from ascent.data.ingest.reddit_sentiment import build_reddit_panel
    from ascent.data.ingest.google_trends import update_trends_signals

    today = date.today()
    is_sunday = today.weekday() == 6
    targets = portfolio_symbols if portfolio_symbols else all_symbols[:50]

    sources = [
        ("SEC",         lambda: update_sec_signals(all_symbols)),
        ("Transcripts", lambda: update_transcript_signals(
                            fetch_recent_8k_transcripts(targets))),
        ("Reddit",      lambda: build_reddit_panel(targets)),
        ("Trends",      lambda: update_trends_signals(
                            all_symbols if is_sunday else targets)),
    ]
    for name, fn in sources:
        try:
            print(f"[AltData] Collecting {name}...")
            fn()
            print(f"[AltData] {name} done")
        except Exception as e:
            print(f"[AltData] {name} failed (non-fatal): {e}")
```

- [ ] **Step 4: Wire the call into `main()` after `run_hub()`**

In `run_all_agents.py`, find the block ending with:
```python
    if hub_manifest.get("status") != "ok":
        print(f"[Hub] WARNING: hub failed ...")
```

Add immediately after it (before the `# ── Import agents` comment):

```python
    # ── Alt data collection (runs before agents; each source fails silently) ──
    _collect_altdata(
        portfolio_symbols=_get_portfolio_symbols(),
        all_symbols=us_symbols,
    )
```

- [ ] **Step 5: Run the test**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py::test_collect_altdata_completes_when_all_sources_fail -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Run full altdata suite**

```bash
.venv/bin/python -m pytest tests/test_altdata_pipeline.py -v 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add run_all_agents.py tests/test_altdata_pipeline.py
git commit -m "feat: wire _collect_altdata() into run_all_agents.py before agent runs"
```

---

## Task 5: Decision Memory — Alt Data Fields on OverrideRecord

**Files:**
- Modify: `ascent/memory/decision_memory.py`
- Modify: `tests/test_decision_memory.py`

**What this does:** `OverrideRecord` gains 4 new optional fields. `ingest_override()` auto-fills them by reading from cached parquets at ingest time. Old JSONL records without these fields still load correctly (all fields are `Optional` with `None` defaults).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_decision_memory.py`:

```python
def test_ingest_override_stores_altdata_fields(tmp_path):
    """With mocked caches, new altdata fields are written to the JSONL record."""
    import json
    import pandas as pd
    from unittest.mock import patch
    from ascent.memory.decision_memory import ingest_override

    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    detail_path = tmp_path / "altdata_sec_detail.json"
    detail_path.write_text(json.dumps({"AAPL": {"tone": 0.65, "as_of": "2026-05-24"}}))

    trends_df = pd.DataFrame({"AAPL": [0.42]},
                              index=pd.DatetimeIndex(["2026-05-24"]))
    trends_df.index.name = "date"
    trends_path = tmp_path / "altdata_trends.parquet"
    trends_df.to_parquet(trends_path)

    with patch("ascent.memory.decision_memory._REPO_ROOT", tmp_path):
        ingest_override(
            rebalance_date="2026-05-24",
            symbol="AAPL",
            override_type="valuation",
            regime="calm_bull",
            ai_action="REDUCED from 10% to 5%",
            ai_weight=0.05,
            quant_weight=0.10,
            log_path=log_path,
        )

    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["sec_tone"] == 0.65
    assert rows[0]["trends_direction"] == pytest.approx(0.42, abs=0.01)


def test_ingest_override_backward_compat_old_records(tmp_path):
    """Old records missing altdata fields deserialize without KeyError."""
    import json
    from ascent.memory.decision_memory import query, OverrideRecord

    old_record = {
        "entry_id": "2026-04-01_AAPL",
        "rebalance_date": "2026-04-01",
        "symbol": "AAPL",
        "override_type": "valuation",
        "regime": "calm_bull",
        "ai_action": "REDUCED",
        "ai_weight": 0.05,
        "quant_weight": 0.10,
        "weight_delta": -0.05,
        "momentum_252d": 0.30,
        "wedge_21d": 0.02,
    }
    log_path = tmp_path / "logs" / "decision_memory.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(json.dumps(old_record) + "\n")

    records = query(log_path=log_path)
    assert len(records) == 1
    r = records[0]
    assert r.sec_tone is None
    assert r.transcript_sentiment is None
    assert r.reddit_buzz is None
    assert r.trends_direction is None


def test_read_altdata_context_returns_all_none_when_no_caches(tmp_path):
    """No cache files → all four fields are None, no exception."""
    from unittest.mock import patch
    from ascent.memory.decision_memory import _read_altdata_context
    with patch("ascent.memory.decision_memory._REPO_ROOT", tmp_path):
        ctx = _read_altdata_context("AAPL")
    assert ctx == {
        "sec_tone": None,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": None,
    }
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_decision_memory.py::test_ingest_override_stores_altdata_fields tests/test_decision_memory.py::test_ingest_override_backward_compat_old_records tests/test_decision_memory.py::test_read_altdata_context_returns_all_none_when_no_caches -v 2>&1 | tail -20
```

Expected: FAIL — `_read_altdata_context` not defined; `OverrideRecord` has no new fields.

- [ ] **Step 3: Add import and 4 new fields to `OverrideRecord`**

In `ascent/memory/decision_memory.py`, add `import pandas as pd` to the imports block.

Update `OverrideRecord` to:

```python
@dataclass
class OverrideRecord:
    entry_id: str
    rebalance_date: str
    symbol: str
    override_type: str
    regime: str
    ai_action: str
    ai_weight: float
    quant_weight: float
    weight_delta: float
    momentum_252d: Optional[float] = None
    wedge_21d: Optional[float] = None
    # Alt data snapshot at override time — for ML conviction model training
    sec_tone: Optional[float] = None
    transcript_sentiment: Optional[float] = None
    reddit_buzz: Optional[float] = None
    trends_direction: Optional[float] = None
```

- [ ] **Step 4: Add `_read_altdata_context` helper**

Add after the `OverrideRecord` dataclass definition:

```python
def _read_altdata_context(symbol: str) -> dict:
    """Read current alt data signals for a symbol from cached files. Returns None for absent sources."""
    ctx: dict = {
        "sec_tone": None,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": None,
    }
    try:
        import pandas as _pd
        import json as _json

        detail_path = _REPO_ROOT / "data_cache" / "altdata_sec_detail.json"
        if detail_path.exists():
            detail = _json.loads(detail_path.read_text())
            ctx["sec_tone"] = detail.get(symbol, {}).get("tone")
    except Exception:
        pass

    for key, fname in [
        ("transcript_sentiment", "altdata_transcripts.parquet"),
        ("reddit_buzz",          "altdata_reddit.parquet"),
        ("trends_direction",     "altdata_trends.parquet"),
    ]:
        try:
            import pandas as _pd
            p = _REPO_ROOT / "data_cache" / fname
            if p.exists():
                df = _pd.read_parquet(p)
                if symbol in df.columns:
                    col = df[symbol].dropna()
                    if not col.empty:
                        ctx[key] = float(col.iloc[-1])
        except Exception:
            pass

    return ctx
```

- [ ] **Step 5: Update `ingest_override` to auto-fill alt data context**

In `ingest_override`, replace the `record = OverrideRecord(...)` block with:

```python
    altdata_ctx = _read_altdata_context(symbol)
    record = OverrideRecord(
        entry_id=entry_id,
        rebalance_date=rebalance_date,
        symbol=symbol,
        override_type=override_type,
        regime=regime,
        ai_action=ai_action,
        ai_weight=ai_weight,
        quant_weight=quant_weight,
        weight_delta=ai_weight - quant_weight,
        momentum_252d=momentum_252d,
        wedge_21d=None,
        sec_tone=altdata_ctx["sec_tone"],
        transcript_sentiment=altdata_ctx["transcript_sentiment"],
        reddit_buzz=altdata_ctx["reddit_buzz"],
        trends_direction=altdata_ctx["trends_direction"],
    )
```

- [ ] **Step 6: Run the three new tests**

```bash
.venv/bin/python -m pytest tests/test_decision_memory.py::test_ingest_override_stores_altdata_fields tests/test_decision_memory.py::test_ingest_override_backward_compat_old_records tests/test_decision_memory.py::test_read_altdata_context_returns_all_none_when_no_caches -v 2>&1 | tail -20
```

Expected: all 3 PASS.

- [ ] **Step 7: Run full decision memory suite**

```bash
.venv/bin/python -m pytest tests/test_decision_memory.py -v 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add ascent/memory/decision_memory.py tests/test_decision_memory.py
git commit -m "feat: decision memory — record alt data context per override for ML training"
```

---

## Task 6: ML Conviction Model

**Files:**
- Modify: `ascent/strategy/conviction_gate.py`
- Modify: `tests/test_conviction_gate.py`

**What this does:** When `n_matured_cases >= 30`, trains a `LogisticRegression` on enriched decision memory features and uses it in `evaluate()`. Falls back to existing rules if model confidence < 0.55 (|prob − 0.5| < 0.05). Adds optional `symbol` param to `evaluate()` so the AI PM can pass context for ML feature lookup at inference time. Model cached to `data_cache/conviction_model.pkl`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_conviction_gate.py`. First update `_make_record` helper to include new fields:

```python
def _make_record(i, override_type, regime, wedge, sec_tone=None, trends_direction=None):
    return {
        "entry_id": f"2026-04-{i:02d}_{override_type[:3]}",
        "rebalance_date": f"2026-04-{i:02d}",
        "symbol": "X",
        "override_type": override_type,
        "regime": regime,
        "ai_action": "REDUCED",
        "ai_weight": 0.05,
        "quant_weight": 0.10,
        "weight_delta": -0.05,
        "momentum_252d": None,
        "wedge_21d": wedge,
        "sec_tone": sec_tone,
        "transcript_sentiment": None,
        "reddit_buzz": None,
        "trends_direction": trends_direction,
    }
```

Then add new tests:

```python
def test_evaluate_accepts_symbol_param(tmp_path):
    """symbol kwarg accepted without error — backward compat with no symbol."""
    from ascent.strategy.conviction_gate import evaluate
    log_path = _write_memory_log(tmp_path, [])
    result = evaluate("data_quality", "calm_bull", log_path=log_path, symbol="AAPL")
    assert result.proceed is True


def test_ml_model_not_triggered_below_30_cases(tmp_path):
    """25 matured valuation records → rules path, not ML."""
    from unittest.mock import patch
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.01 if i % 2 == 0 else -0.01)
               for i in range(1, 26)]
    log_path = _write_memory_log(tmp_path, records)

    mock_lr = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock
    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=None) as mock_ml:
        result = evaluate("valuation", "calm_bull", log_path=log_path)
        mock_ml.assert_called_once()
    # Should have gone through rules path
    assert result.confidence in ("strong", "proceed", "caution", "block")


def test_ml_model_activates_at_30_cases_high_confidence(tmp_path):
    """30 matured valuation records + mock ML returning 0.80 prob → STRONG gate."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.02) for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.20, 0.80]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    assert result.proceed is True
    assert result.confidence == "strong"
    assert result.size_multiplier == 1.0


def test_ml_model_falls_back_when_not_confident(tmp_path):
    """Model returns prob=0.53 (|0.53−0.5|=0.03 < 0.05) → falls back to rules."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.01 if i <= 20 else -0.01)
               for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.47, 0.53]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    # Rules-based result (mixed ~50/50 record)
    assert result.confidence in ("proceed", "caution", "block")


def test_ml_model_blocks_on_low_probability(tmp_path):
    """Model returns prob=0.25 → BLOCKED."""
    from unittest.mock import patch, MagicMock
    from ascent.strategy.conviction_gate import evaluate

    records = [_make_record(i, "valuation", "calm_bull", 0.02) for i in range(1, 31)]
    log_path = _write_memory_log(tmp_path, records)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.75, 0.25]]

    with patch("ascent.strategy.conviction_gate._get_ml_model", return_value=mock_model):
        result = evaluate("valuation", "calm_bull", log_path=log_path)

    assert result.proceed is False
    assert result.confidence == "block"
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_conviction_gate.py::test_evaluate_accepts_symbol_param tests/test_conviction_gate.py::test_ml_model_not_triggered_below_30_cases tests/test_conviction_gate.py::test_ml_model_activates_at_30_cases_high_confidence tests/test_conviction_gate.py::test_ml_model_falls_back_when_not_confident tests/test_conviction_gate.py::test_ml_model_blocks_on_low_probability -v 2>&1 | tail -20
```

Expected: FAIL — `symbol` param not accepted, `_get_ml_model` not defined.

- [ ] **Step 3: Add feature building and model helpers to `conviction_gate.py`**

Add these imports at the top of `ascent/strategy/conviction_gate.py`:

```python
import pickle
from pathlib import Path
```

Add these constants and functions after the existing module-level constants (`MIN_ML_CASES`, `BLOCK_WIN_RATE`, etc.):

```python
_MODEL_PATH = Path("data_cache/conviction_model.pkl")
_OVERRIDE_TYPES = ["data_quality", "regime_macro", "news_event", "correlation_risk", "valuation"]
_REGIMES       = ["calm_bull", "stressed", "crisis", "neutral", "uncertain"]

# In-process cache so we don't retrain on every call
_model_cache: dict = {"model": None, "n_trained": 0}


def _build_feature_vector(record) -> list:
    ot_vec  = [1.0 if record.override_type == t else 0.0 for t in _OVERRIDE_TYPES]
    reg_vec = [1.0 if record.regime == r else 0.0 for r in _REGIMES]
    num = [
        record.momentum_252d or 0.0,
        record.sec_tone or 0.0,
        record.transcript_sentiment or 0.0,
        record.reddit_buzz or 0.0,
        record.trends_direction or 0.0,
    ]
    return ot_vec + reg_vec + num


def _get_ml_model(matured_records):
    """Return trained LogisticRegression if n >= MIN_ML_CASES, else None."""
    n = len(matured_records)
    if n < MIN_ML_CASES:
        return None

    if _model_cache["model"] is not None and _model_cache["n_trained"] == n:
        return _model_cache["model"]

    if _MODEL_PATH.exists():
        try:
            with open(_MODEL_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_trained") == n:
                _model_cache["model"] = cached["model"]
                _model_cache["n_trained"] = n
                return _model_cache["model"]
        except Exception:
            pass

    try:
        from sklearn.linear_model import LogisticRegression
        X = [_build_feature_vector(r) for r in matured_records]
        y = [1 if r.wedge_21d > 0 else 0 for r in matured_records]
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X, y)
        _model_cache["model"] = model
        _model_cache["n_trained"] = n
        try:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump({"model": model, "n_trained": n}, f)
        except Exception:
            pass
        return model
    except Exception as exc:
        log.warning("[ConvictionGate] ML model training failed: %s", exc)
        return None
```

- [ ] **Step 4: Update `evaluate()` signature and add ML routing**

Replace the existing `evaluate()` function signature and add the ML block before the existing structural-always-approve check:

```python
def evaluate(
    override_type: str,
    regime: str,
    calibration_ic: Optional[float] = None,
    log_path: Optional[Path] = None,
    symbol: Optional[str] = None,
) -> GateResult:
    from ascent.memory.decision_memory import get_statistics, query, OverrideRecord

    kwargs = {} if log_path is None else {"log_path": log_path}
    stats = get_statistics(override_type=override_type, regime=regime, **kwargs)
    n     = stats["n_cases"]
    wr    = stats["win_rate"]
    aw    = stats["avg_wedge"]

    # ML path — only for non-structural override types
    if override_type not in ("data_quality", "correlation_risk", "news_event"):
        all_matured = [r for r in query(override_type=None, regime=None, n=1000, **kwargs)
                       if r.wedge_21d is not None]
        model = _get_ml_model(all_matured)
        if model is not None:
            from ascent.memory.decision_memory import _read_altdata_context
            ctx = _read_altdata_context(symbol) if symbol else {
                "sec_tone": None, "transcript_sentiment": None,
                "reddit_buzz": None, "trends_direction": None,
            }
            probe = OverrideRecord(
                entry_id="probe", rebalance_date="", symbol=symbol or "",
                override_type=override_type, regime=regime,
                ai_action="", ai_weight=0.0, quant_weight=0.0, weight_delta=0.0,
                momentum_252d=None, wedge_21d=None,
                sec_tone=ctx["sec_tone"],
                transcript_sentiment=ctx["transcript_sentiment"],
                reddit_buzz=ctx["reddit_buzz"],
                trends_direction=ctx["trends_direction"],
            )
            prob = float(model.predict_proba([_build_feature_vector(probe)])[0][1])
            n_total = len(all_matured)
            if abs(prob - 0.5) >= 0.05:
                if prob >= 0.65:
                    return GateResult(
                        proceed=True, size_multiplier=1.0, confidence="strong",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases)",
                        n_cases=n, win_rate=wr,
                    )
                elif prob >= 0.50:
                    return GateResult(
                        proceed=True, size_multiplier=0.80, confidence="proceed",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases)",
                        n_cases=n, win_rate=wr,
                    )
                else:
                    return GateResult(
                        proceed=False, size_multiplier=0.0, confidence="block",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases) — blocked",
                        n_cases=n, win_rate=wr,
                    )
            # else: model not confident enough, fall through to rules

    # ── Existing rules-based logic (unchanged below) ──────────────────────────
    if override_type in ("data_quality", "correlation_risk", "news_event"):
        return GateResult(
            proceed=True, size_multiplier=1.0, confidence="proceed",
            reason=f"{override_type} overrides are always approved — structural AI PM edge",
            n_cases=n, win_rate=wr,
        )
    # ... rest of existing rules unchanged
```

- [ ] **Step 5: Run the five new tests**

```bash
.venv/bin/python -m pytest tests/test_conviction_gate.py::test_evaluate_accepts_symbol_param tests/test_conviction_gate.py::test_ml_model_not_triggered_below_30_cases tests/test_conviction_gate.py::test_ml_model_activates_at_30_cases_high_confidence tests/test_conviction_gate.py::test_ml_model_falls_back_when_not_confident tests/test_conviction_gate.py::test_ml_model_blocks_on_low_probability -v 2>&1 | tail -20
```

Expected: all 5 PASS.

- [ ] **Step 6: Run full conviction gate suite**

```bash
.venv/bin/python -m pytest tests/test_conviction_gate.py -v 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: 569+ tests passing, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add ascent/strategy/conviction_gate.py tests/test_conviction_gate.py
git commit -m "feat: ML conviction model — logistic regression on enriched decision memory features"
```

---

## Final Validation

- [ ] **Verify all four alt data caches exist or get created on first run**

```bash
.venv/bin/python -c "
from ascent.data.ingest.sec_filings import load_sec_detail
from ascent.data.ingest.earnings_transcripts import load_transcript_signals
from ascent.data.ingest.reddit_sentiment import load_reddit_signals
from ascent.data.ingest.google_trends import load_trends_signals
print('SEC detail:', load_sec_detail('AAPL'))
print('Transcripts shape:', load_transcript_signals().shape)
print('Reddit shape:', load_reddit_signals().shape)
print('Trends shape:', load_trends_signals().shape)
"
```

- [ ] **Verify decision memory backward compat with existing JSONL records**

```bash
.venv/bin/python -c "
from ascent.memory.decision_memory import query
records = query()
print(f'{len(records)} records loaded')
for r in records[:3]:
    print(r.symbol, r.override_type, r.sec_tone, r.trends_direction)
"
```

Expected: loads without error; `sec_tone` and `trends_direction` are `None` for old records.

- [ ] **Update CLAUDE.md session log**

Append to the session log in `CLAUDE.md`:

```
### 2026-05-24 (alt data pipeline + decision memory enrichment ✅)
- SEC pipeline: stores all 7 signals (was dropping 4 of 5) to altdata_sec_detail.json; adds Risk Factors extraction; YoY comparison; 90-day freshness gate.
- Transcripts pipeline: adds EDGAR 8-K Item 2.02 fetcher (was always returning zeros).
- run_all_agents.py: _collect_altdata() runs all 4 sources before agents; _get_portfolio_symbols() reads merged_weights.json.
- Decision memory: OverrideRecord gains sec_tone/transcript_sentiment/reddit_buzz/trends_direction; auto-filled at ingest via _read_altdata_context().
- Conviction gate: ML logistic regression activates at n≥30 matured cases; falls back to rules if confidence < 55%; symbol param added to evaluate().
- Files: sec_filings.py, earnings_transcripts.py, run_all_agents.py, decision_memory.py, conviction_gate.py, tests/test_altdata_pipeline.py, tests/test_decision_memory.py, tests/test_conviction_gate.py.
```
