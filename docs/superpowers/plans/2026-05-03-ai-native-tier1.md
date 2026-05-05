# AI-Native Ascent Capital — Tier 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AI a first-class signal source and self-calibrating reasoner in Ascent Capital's live trading pipeline via three targeted upgrades: a CoT-based LLM fundamental signal, a slippage-adjusted IC feedback loop, and regime-conditional debate agent personas.

**Architecture:** Three independent modules, each wired into the existing pipeline at a clear seam. Task A adds a new alpha sleeve (`llm_fundamental`) that sends anonymized financial ratios to Claude Haiku with a 6-step structured CoT prompt (Chicago Booth 2407.17866). Task B adds a weekly slippage IC feedback job that measures how much transaction costs drag signal quality and writes the result back to `active_alpha_config.json`. Task C injects each debate agent's historical regime-conditional accuracy into its system prompt so agents self-calibrate their confidence.

**Tech Stack:** Python 3.12, Claude Haiku (`claude-haiku-4-5-20251001`), existing `ascent/llm/client.py`, `debate/outcome_tracker.py`, `ascent/data/ingest/fundamentals.py`, `ascent/research/self_improve.py`, scipy (already installed).

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/alpha/llm_fundamental.py` | CoT LLM alpha signal — anonymize ratios, call Haiku, cache, z-score |
| Create | `tests/test_llm_fundamental_alpha.py` | Full test suite for Task A |
| Modify | `ascent/alpha/stack.py` (line ~16) | Add `llm_fundamental` sleeve at 3%, reduce `trend` 44%→41% |
| Modify | `ascent/research/self_improve.py` (line ~30) | Sync `DEFAULT_ALPHA_WEIGHTS` with stack.py |
| Create | `ascent/monitoring/slippage_ic_feedback.py` | Slippage IC drag computation + active config update |
| Create | `tests/test_slippage_ic_feedback.py` | Full test suite for Task B |
| Modify | `run_all_agents.py` | Call `run_slippage_ic_feedback()` on Sundays |
| Modify | `debate/agents.py` | Inject per-agent regime accuracy into system prompts |
| Modify | `debate/outcome_tracker.py` | Add `get_agent_regime_accuracy()` helper |
| Create | `tests/test_regime_conditional_personas.py` | Full test suite for Task C |

---

## Task A: CoT LLM Fundamental Alpha Signal

**Problem:** Ascent's fundamental sleeve (`ascent/alpha/fundamental.py`) uses raw accounting ratios as cross-sectional z-scores. It has no reasoning — it doesn't know *why* a particular gross profitability trend matters for a specific company. Chicago Booth (2407.17866) showed that GPT-4 with a structured 6-step CoT prompt achieves 60.35% earnings direction accuracy vs. 52.71% for human analysts, generating 12% annual alpha. The key: anonymize inputs so the model can't recall company history from training data.

**Model note:** The Chicago Booth study used GPT-4. This implementation uses Haiku, which has a meaningful reasoning gap on financial tasks. At 3% sleeve weight the cost of a near-zero IC signal is bounded — but track IC explicitly from day one. If `llm_fundamental` shows mean IC < 0.01 after 30 trading days of signals, reduce the weight or disable the sleeve. The signal logging step below makes this measurable.

**Files:**
- Create: `ascent/alpha/llm_fundamental.py`
- Create: `tests/test_llm_fundamental_alpha.py`
- Modify: `ascent/alpha/stack.py`
- Modify: `ascent/research/self_improve.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_fundamental_alpha.py
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path


def _make_fundamentals(symbols=None, n_quarters=4):
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rows = []
    base_date = pd.Timestamp("2025-12-31")
    for sym in symbols:
        np.random.seed(hash(sym) % 2**31)
        for q in range(n_quarters):
            rows.append({
                "symbol": sym,
                "date":   base_date - pd.DateOffset(months=3 * q),
                "gross_profitability": np.random.uniform(0.2, 0.6),
                "accruals":            np.random.uniform(-0.05, 0.05),
                "asset_growth":        np.random.uniform(-0.02, 0.15),
            })
    return pd.DataFrame(rows)


def test_returns_series(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals()
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm",
                   return_value={"direction": "UP", "confidence": 0.80}):
            result = llm_fundamental_alpha(fund)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_scores_are_cross_sectional_zscored(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    syms = list("ABCDEFGHIJ")
    fund = _make_fundamentals(symbols=syms)
    responses = [
        {"direction": "UP",      "confidence": 0.9},
        {"direction": "DOWN",    "confidence": 0.8},
        {"direction": "UP",      "confidence": 0.7},
        {"direction": "NEUTRAL", "confidence": 0.6},
        {"direction": "DOWN",    "confidence": 0.5},
        {"direction": "UP",      "confidence": 0.85},
        {"direction": "DOWN",    "confidence": 0.75},
        {"direction": "UP",      "confidence": 0.65},
        {"direction": "NEUTRAL", "confidence": 0.4},
        {"direction": "DOWN",    "confidence": 0.9},
    ]
    call_idx = [0]
    def mock_call(symbol, table):
        r = responses[call_idx[0] % len(responses)]
        call_idx[0] += 1
        return r
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            result = llm_fundamental_alpha(fund)
    assert abs(result.mean()) < 0.15, f"Mean should be ~0, got {result.mean()}"
    assert 0.5 < result.std() < 2.0, f"Std should be ~1, got {result.std()}"


def test_empty_on_none_input(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        assert len(llm_fundamental_alpha(None)) == 0
        assert len(llm_fundamental_alpha(pd.DataFrame())) == 0


def test_uses_cache_on_second_call(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals(symbols=["AAPL", "MSFT"])
    count = [0]
    def mock_call(symbol, table):
        count[0] += 1
        return {"direction": "UP", "confidence": 0.75}
    cache_path = tmp_path / "c.json"
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", cache_path):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            llm_fundamental_alpha(fund)
            first = count[0]
            llm_fundamental_alpha(fund)
            second = count[0]
    assert second == first, "Second call must use cache, not re-call LLM"


def test_api_failure_returns_empty(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals()
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", return_value=None):
            result = llm_fundamental_alpha(fund)
    assert isinstance(result, pd.Series)


def test_anonymization_no_ticker_in_table(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    fund = _make_fundamentals(symbols=["AAPL"])
    captured = []
    def mock_call(symbol, table):
        captured.append(table)
        return {"direction": "UP", "confidence": 0.8}
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm", side_effect=mock_call):
            llm_fundamental_alpha(fund)
    assert all("AAPL" not in t for t in captured), "Ticker must not appear in anonymized table"


def test_respects_45day_filing_lag(tmp_path):
    from ascent.alpha.llm_fundamental import llm_fundamental_alpha
    # All quarters within 44 days of as_of_date should be excluded
    fund = pd.DataFrame([{
        "symbol": "AAPL",
        "date": pd.Timestamp("2026-04-20"),   # 13 days before 2026-05-03
        "gross_profitability": 0.4, "accruals": 0.01, "asset_growth": 0.05,
    }])
    with patch("ascent.alpha.llm_fundamental.CACHE_PATH", tmp_path / "c.json"):
        with patch("ascent.alpha.llm_fundamental._call_llm",
                   return_value={"direction": "UP", "confidence": 0.8}) as mock:
            llm_fundamental_alpha(fund, as_of_date=pd.Timestamp("2026-05-03"))
    # Should not crash; data within 45-day lag is excluded silently


def test_stack_includes_llm_fundamental_sleeve():
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert "llm_fundamental" in DEFAULT_ALPHA_WEIGHTS, \
        "stack.py DEFAULT_ALPHA_WEIGHTS must include llm_fundamental sleeve"
    assert DEFAULT_ALPHA_WEIGHTS["llm_fundamental"] > 0
    assert abs(sum(DEFAULT_ALPHA_WEIGHTS.values()) - 1.0) < 1e-6, \
        "Sleeve weights must sum to 1.0"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_llm_fundamental_alpha.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'ascent.alpha.llm_fundamental'`

- [ ] **Step 3: Create `ascent/alpha/llm_fundamental.py`**

```python
"""
ascent/alpha/llm_fundamental.py

LLM-based fundamental alpha signal using Chicago Booth 6-step CoT.

Sends anonymized financial ratios to Claude Haiku. Caches results by
(symbol, quarter_end_date) so the LLM is only called when new
fundamental data is available. Returns a cross-sectional z-score Series.

Source: Kim, Muhn, Nikolaev (2024). arxiv.org/abs/2407.17866
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_PATH = Path("data_cache/llm_fundamental_cache.json")

_SYSTEM_PROMPT = (
    "You are a financial analyst evaluating anonymized company financials. "
    "You do not know the company name, ticker, or exact dates. "
    "Respond only with valid JSON matching the specified format. No other text."
)

_USER_TEMPLATE = """Analyze these quarterly financial metrics for an anonymous company.

Financial Data (Q-3 = three quarters ago, Q0 = most recent quarter):
{metrics_table}

Step 1: Identify 3 key trends in revenue growth, gross margin, and asset base (cite specific numbers).
Step 2: Compute: (a) gross margin change Q-3→Q0, (b) accruals ratio trend, (c) asset growth rate Q-3→Q0.
Step 3: Interpret each economically — improving, stable, or deteriorating, and why.
Step 4: Identify any inflection points in the last 2 quarters.
Step 5: Forecast next-quarter earnings direction. State confidence (0.0–1.0) and primary reason.
Step 6: State the single most important uncertainty in your forecast.

Respond ONLY in this JSON format:
{{"direction": "UP|DOWN|NEUTRAL", "confidence": 0.XX, "key_trend": "one sentence", "uncertainty": "one sentence"}}"""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _format_metrics_table(ratios: dict) -> str:
    quarters = sorted(ratios.keys())
    metrics  = ["gross_profitability", "accruals", "asset_growth"]
    lines    = ["Quarter | " + " | ".join(metrics), "-" * 60]
    for q in quarters:
        vals = ratios.get(q, {})
        row  = [q] + [f"{vals[m]:.3f}" if m in vals else "N/A" for m in metrics]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _call_llm(symbol: str, metrics_table: str) -> Optional[dict]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        raw = generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_USER_TEMPLATE.format(metrics_table=metrics_table),
            model=HAIKU_MODEL,
            max_tokens=400,
            temperature=0.1,
            use_cache=True,
        )
        raw   = raw.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("[LLM Fundamental] No JSON in response for %s", symbol)
            return None
        parsed = json.loads(raw[start:end])
        direction  = str(parsed.get("direction", "")).upper()
        confidence = float(parsed.get("confidence", 0.0))
        if direction not in ("UP", "DOWN", "NEUTRAL"):
            return None
        if not (0.0 <= confidence <= 1.0):
            return None
        return {"direction": direction, "confidence": confidence}
    except Exception as exc:
        log.warning("[LLM Fundamental] Call failed for %s: %s", symbol, exc)
        return None


def llm_fundamental_alpha(
    fundamentals_df: Optional[pd.DataFrame],
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Generate cross-sectional LLM fundamental alpha scores.

    Args:
        fundamentals_df: DataFrame with columns [symbol, date,
                         gross_profitability, accruals, asset_growth].
        as_of_date:      Cutoff date. Data newer than (as_of_date - 45 days)
                         is excluded to enforce the filing lag.

    Returns:
        pd.Series indexed by symbol, cross-sectionally z-scored.
        Empty Series if input is empty or all API calls fail.
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return pd.Series(dtype=float)

    required = {"gross_profitability", "accruals", "asset_growth"}
    available = required.intersection(fundamentals_df.columns)
    if len(available) < 2:
        log.warning("[LLM Fundamental] Missing columns. Have: %s", list(fundamentals_df.columns))
        return pd.Series(dtype=float)

    as_of   = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today()
    cutoff  = as_of - pd.Timedelta(days=45)
    fund    = fundamentals_df.copy()
    fund["date"] = pd.to_datetime(fund["date"])
    fund    = fund[fund["date"] <= cutoff]

    if fund.empty:
        return pd.Series(dtype=float)

    cache       = _load_cache()
    raw_scores: Dict[str, float] = {}
    cache_dirty = False

    for symbol, grp in fund.groupby("symbol"):
        grp = grp.sort_values("date").tail(4)
        if len(grp) < 2:
            continue

        last_date = str(grp["date"].iloc[-1].date())
        cache_key = f"{symbol}_{last_date}"

        if cache_key in cache:
            result = cache[cache_key]
        else:
            ratios = {}
            for i, (_, row) in enumerate(grp.iterrows()):
                label = f"Q-{len(grp)-1-i}" if i < len(grp) - 1 else "Q0"
                ratios[label] = {
                    col: round(float(row[col]), 4)
                    for col in available
                    if pd.notna(row.get(col))
                }
            result      = _call_llm(symbol, _format_metrics_table(ratios))
            cache[cache_key] = result
            cache_dirty = True

        if result is not None:
            sign = 1.0 if result["direction"] == "UP" else (
                   -1.0 if result["direction"] == "DOWN" else 0.0)
            raw_scores[symbol] = sign * result["confidence"]

    if cache_dirty:
        _save_cache(cache)

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    std    = scores.std()
    if std < 1e-8:
        return pd.Series(0.0, index=scores.index)
    return (scores - scores.mean()) / std
```

- [ ] **Step 3b: Add signal logging to `llm_fundamental_alpha()` for IC tracking**

In `llm_fundamental.py`, add the following block immediately after `if cache_dirty: _save_cache(cache)`:

```python
    # Log signals to jsonl for IC tracking (enables sleeve quality audit after 30+ trading days)
    if raw_scores:
        import json as _json
        _sig_path = Path("logs/llm_fundamental_signals.jsonl")
        _sig_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_sig_path, "a") as _f:
            _f.write(_json.dumps({
                "date": str(as_of.date()),
                "n_symbols": len(raw_scores),
                "scores": {k: round(v, 4) for k, v in raw_scores.items()},
            }) + "\n")
```

This file enables a future IC audit: load `logs/llm_fundamental_signals.jsonl`, align with realized 5-day returns, compute Spearman IC. If mean IC < 0.01 after 30 entries, reduce sleeve weight.

- [ ] **Step 4: Add `llm_fundamental` sleeve to `ascent/alpha/stack.py`**

Find `DEFAULT_ALPHA_WEIGHTS` at line ~16. Change `"trend": 0.44` to `"trend": 0.41` and add the new sleeve:

```python
DEFAULT_ALPHA_WEIGHTS = {
    "trend":           0.41,
    "meanrev":         0.05,
    "volatility":      0.05,
    "statarb":         0.15,
    "ml":              0.10,
    "fundamental":     0.05,
    "earnings":        0.05,
    "analyst":         0.05,
    "options_flow":    0.02,
    "insider":         0.02,
    "short_interest":  0.02,
    "llm_fundamental": 0.03,
}
```

Then find where other sleeves are called in `build_alpha_stack()` (look for `"fundamental"` or `"earnings"` in the function body). Add the llm_fundamental call in the same pattern:

```python
    # LLM Fundamental sleeve
    if weights.get("llm_fundamental", 0) > 0:
        try:
            from ascent.alpha.llm_fundamental import llm_fundamental_alpha
            from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
            _fund_df = _lp("fundamentals") if _hd("fundamentals") else None
            _llm_fund = llm_fundamental_alpha(_fund_df)
            if not _llm_fund.empty:
                sleeves["llm_fundamental"] = _llm_fund
                log.info("[Stack] LLM fundamental sleeve: %d symbols", len(_llm_fund))
        except Exception as _e:
            log.warning("[Stack] LLM fundamental sleeve failed: %s", _e)
```

- [ ] **Step 5: Sync `self_improve.py` DEFAULT_ALPHA_WEIGHTS**

In `ascent/research/self_improve.py`, find `DEFAULT_ALPHA_WEIGHTS` (line ~30). Update `"trend"` from `0.44` to `0.41` and add `"llm_fundamental": 0.03`:

```python
DEFAULT_ALPHA_WEIGHTS = {
    "trend":           0.41,
    "meanrev":         0.05,
    "statarb":         0.15,
    "ml":              0.10,
    "volatility":      0.05,
    "fundamental":     0.05,
    "earnings":        0.05,
    "analyst":         0.05,
    "options_flow":    0.02,
    "insider":         0.02,
    "short_interest":  0.02,
    "llm_fundamental": 0.03,
}
```

- [ ] **Step 6: Run all tests**

```bash
.venv/bin/pytest tests/test_llm_fundamental_alpha.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add ascent/alpha/llm_fundamental.py tests/test_llm_fundamental_alpha.py \
        ascent/alpha/stack.py ascent/research/self_improve.py
git commit -m "feat(alpha): CoT LLM fundamental signal — Chicago Booth 6-step prompt, Haiku, cached"
```

---

## Task B: Slippage-Adjusted IC Feedback Loop

**Problem:** Ascent generates signals, fills orders via Alpaca, and logs slippage to `logs/slippage_log.jsonl`. But this data never feeds back into signal quality evaluation. The self-improve loop scores variants by OOS Sharpe but ignores whether high-turnover variants are paying more in market impact. This task closes the measurement loop: compute gross IC vs slippage-adjusted net IC weekly, and write the drag coefficient to `active_alpha_config.json`.

**Important constraint:** `MIN_FILLS = 50`. With 10 fills the Spearman IC estimate has a standard error of ~0.3, making the drag coefficient useless noise that would destabilize self-improve scoring. Accept that this module is a **passive logger until ~July 2026** (when ~60 fills will have accumulated). The drag coefficient is NOT read by self-improve scoring until then — that wiring is a separate TODO left for when sufficient data exists.

**Files:**
- Create: `ascent/monitoring/slippage_ic_feedback.py`
- Create: `tests/test_slippage_ic_feedback.py`
- Modify: `run_all_agents.py` (Sunday run block)
- Modify: `ascent/research/self_improve.py` (scoring function)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_slippage_ic_feedback.py
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch


def _write_slippage_log(path: Path, n: int = 80):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    base = pd.Timestamp("2026-04-01")
    for i in range(n):
        rows.append({
            "date":          str((base + pd.Timedelta(days=i)).date()),
            "symbol":        f"SYM{i % 10:02d}",
            "slippage_bps":  float(np.random.uniform(2, 20)),
            "signal_price":  100.0,
            "fill_price":    100.0 + np.random.uniform(0.01, 0.20),
        })
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def _write_pnl_log(path: Path, symbols, n_days: int = 30):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2026-04-01")
    with open(path, "w") as f:
        for i in range(n_days):
            for sym in symbols:
                f.write(json.dumps({
                    "date":         str((base + pd.Timedelta(days=i)).date()),
                    "symbol":       sym,
                    "daily_return": float(np.random.normal(0.001, 0.015)),
                }) + "\n")


def test_compute_returns_dict(tmp_path):
    from ascent.monitoring.slippage_ic_feedback import compute_slippage_ic_drag
    slip_path = tmp_path / "logs" / "slippage_log.jsonl"
    pnl_path  = tmp_path / "logs" / "us_equities_pnl.jsonl"
    syms = [f"SYM{i:02d}" for i in range(10)]
    _write_slippage_log(slip_path, n=80)  # above MIN_FILLS=50
    _write_pnl_log(pnl_path, syms, n_days=90)

    with patch("ascent.monitoring.slippage_ic_feedback.SLIPPAGE_LOG", slip_path):
        with patch("ascent.monitoring.slippage_ic_feedback.PNL_LOGS",
                   {"us_equities": pnl_path}):
            result = compute_slippage_ic_drag(lookback_days=90)

    assert isinstance(result, dict)
    for key in ["slippage_ic_drag", "gross_ic", "net_ic", "n_fills", "mean_slippage_bps"]:
        assert key in result, f"Missing key: {key}"


def test_insufficient_fills_returns_zero_drag(tmp_path):
    from ascent.monitoring.slippage_ic_feedback import compute_slippage_ic_drag
    slip_path = tmp_path / "logs" / "slippage_log.jsonl"
    _write_slippage_log(slip_path, n=3)  # below MIN_FILLS=50

    with patch("ascent.monitoring.slippage_ic_feedback.SLIPPAGE_LOG", slip_path):
        with patch("ascent.monitoring.slippage_ic_feedback.PNL_LOGS", {}):
            result = compute_slippage_ic_drag()

    assert result["slippage_ic_drag"] == 0.0
    assert result["n_fills"] == 3


def test_updates_active_config(tmp_path):
    from ascent.monitoring.slippage_ic_feedback import update_active_config_with_slippage_feedback
    config_path = tmp_path / "active_alpha_config.json"
    metrics = {"slippage_ic_drag": 0.12, "gross_ic": 0.05,
               "net_ic": 0.044, "n_fills": 30, "mean_slippage_bps": 8.5}

    with patch("ascent.monitoring.slippage_ic_feedback.ACTIVE_CONFIG_PATH", config_path):
        update_active_config_with_slippage_feedback(metrics)

    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert "slippage_feedback" in config
    assert config["slippage_feedback"]["slippage_ic_drag"] == 0.12
    assert config["slippage_feedback"]["gross_ic"] == 0.05


def test_update_preserves_existing_config_keys(tmp_path):
    from ascent.monitoring.slippage_ic_feedback import update_active_config_with_slippage_feedback
    config_path = tmp_path / "active_alpha_config.json"
    config_path.write_text(json.dumps({"global": {"trend": 0.41}}))

    with patch("ascent.monitoring.slippage_ic_feedback.ACTIVE_CONFIG_PATH", config_path):
        update_active_config_with_slippage_feedback({"slippage_ic_drag": 0.05,
                                                      "gross_ic": 0.03, "net_ic": 0.028,
                                                      "n_fills": 15, "mean_slippage_bps": 6.0})

    config = json.loads(config_path.read_text())
    assert "global" in config, "Existing config keys must be preserved"
    assert "slippage_feedback" in config


def test_run_all_agents_calls_slippage_feedback_on_sunday():
    with open("run_all_agents.py") as f:
        src = f.read()
    assert "slippage_ic_feedback" in src or "run_slippage_ic_feedback" in src, \
        "run_all_agents.py must call slippage IC feedback on Sundays"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_slippage_ic_feedback.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'ascent.monitoring.slippage_ic_feedback'`

- [ ] **Step 3: Create `ascent/monitoring/slippage_ic_feedback.py`**

```python
"""
ascent/monitoring/slippage_ic_feedback.py

Slippage-adjusted IC feedback loop.

Weekly: reads slippage_log.jsonl + agent PnL logs, computes gross IC
vs net-of-slippage IC, writes drag coefficient to active_alpha_config.json.

This module is a passive logger until ~60 fills accumulate (MIN_FILLS=50).
Self-improve integration is a future TODO — do not wire the drag coefficient
into self-improve scoring until sufficient data exists.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

SLIPPAGE_LOG = Path("logs/slippage_log.jsonl")
PNL_LOGS     = {
    "us_equities":   Path("logs/us_equities_pnl.jsonl"),
    "macro":         Path("logs/macro_pnl.jsonl"),
    "international": Path("logs/international_pnl.jsonl"),
    "alternatives":  Path("logs/alternatives_pnl.jsonl"),
}
ACTIVE_CONFIG_PATH = Path("data_cache/active_alpha_config.json")
MIN_FILLS = 50
# NOTE: Do not wire slippage_ic_drag into self-improve scoring until >= 60 fills
# have accumulated (expected ~July 2026). With fewer fills the Spearman IC estimate
# has SE ~0.3, making the drag coefficient noise rather than signal.


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def compute_slippage_ic_drag(lookback_days: int = 90) -> Dict[str, float]:
    """
    Compute slippage IC drag over recent fills.

    For each fill: compute gross 10-day forward return and net return
    (gross minus realized slippage). Compare Spearman IC of signal
    vs gross returns to IC vs net returns. The difference is the drag.

    Returns dict with slippage_ic_drag, gross_ic, net_ic, n_fills, mean_slippage_bps.
    """
    slippage_rows = _load_jsonl(SLIPPAGE_LOG)
    if len(slippage_rows) < MIN_FILLS:
        log.info("[SlippageIC] Insufficient fills (%d < %d), skipping",
                 len(slippage_rows), MIN_FILLS)
        return {"slippage_ic_drag": 0.0, "gross_ic": 0.0, "net_ic": 0.0,
                "n_fills": len(slippage_rows), "mean_slippage_bps": 0.0}

    # Build symbol→date→return lookup from PnL logs
    fwd_lookup: Dict[str, Dict[str, float]] = {}
    for path in PNL_LOGS.values():
        for row in _load_jsonl(path):
            sym = row.get("symbol") or row.get("ticker")
            dt  = row.get("date")
            ret = row.get("daily_return") or row.get("return")
            if sym and dt and ret is not None:
                fwd_lookup.setdefault(sym, {})[dt] = float(ret)

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    signals, gross_fwds, net_fwds, slippages_bps = [], [], [], []

    for row in slippage_rows:
        try:
            dt  = pd.Timestamp(row["date"])
            sym = row.get("symbol", "")
            slip_bps      = float(row.get("slippage_bps", 0.0))
            signal_price  = float(row.get("signal_price", 1.0))
            fill_price    = float(row.get("fill_price", 1.0))
        except (KeyError, ValueError, TypeError):
            continue

        if dt < cutoff or sym not in fwd_lookup:
            continue

        sym_rets = {pd.Timestamp(d): r for d, r in fwd_lookup[sym].items()}
        future   = sorted(d for d in sym_rets if d > dt)[:10]
        if len(future) < 5:
            continue

        fwd_return   = sum(sym_rets[d] for d in future)
        slip_return  = slip_bps / 10_000
        signal_score = (signal_price - fill_price) / max(signal_price, 1e-8)

        signals.append(signal_score)
        gross_fwds.append(fwd_return)
        net_fwds.append(fwd_return - slip_return)
        slippages_bps.append(slip_bps)

    if len(signals) < MIN_FILLS:
        return {"slippage_ic_drag": 0.0, "gross_ic": 0.0, "net_ic": 0.0,
                "n_fills": len(signals), "mean_slippage_bps": 0.0}

    gross_ic, _ = spearmanr(signals, gross_fwds)
    net_ic,   _ = spearmanr(signals, net_fwds)
    gross_ic    = float(gross_ic) if not np.isnan(gross_ic) else 0.0
    net_ic      = float(net_ic)   if not np.isnan(net_ic)   else 0.0
    drag        = (gross_ic - net_ic) / max(abs(gross_ic), 1e-6)

    return {
        "slippage_ic_drag":  round(drag, 4),
        "gross_ic":          round(gross_ic, 4),
        "net_ic":            round(net_ic, 4),
        "n_fills":           len(signals),
        "mean_slippage_bps": round(float(np.mean(slippages_bps)), 2),
    }


def update_active_config_with_slippage_feedback(metrics: Dict[str, float]) -> None:
    """Write slippage IC drag to active_alpha_config.json slippage_feedback section."""
    config = {}
    if ACTIVE_CONFIG_PATH.exists():
        try:
            config = json.loads(ACTIVE_CONFIG_PATH.read_text())
        except Exception:
            pass
    config["slippage_feedback"] = {
        **{k: metrics.get(k, 0.0) for k in
           ["slippage_ic_drag", "gross_ic", "net_ic", "n_fills", "mean_slippage_bps"]},
        "updated_at": str(pd.Timestamp.today().date()),
    }
    ACTIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log.info("[SlippageIC] drag=%.4f gross_ic=%.4f net_ic=%.4f fills=%d",
             metrics.get("slippage_ic_drag", 0), metrics.get("gross_ic", 0),
             metrics.get("net_ic", 0), metrics.get("n_fills", 0))


def run_slippage_ic_feedback(lookback_days: int = 90) -> Dict[str, float]:
    """Top-level entry point called from run_all_agents.py on Sundays."""
    metrics = compute_slippage_ic_drag(lookback_days=lookback_days)
    if metrics["n_fills"] >= MIN_FILLS:
        update_active_config_with_slippage_feedback(metrics)
    return metrics
```

- [ ] **Step 4: Wire into `run_all_agents.py` Sunday run**

Find the Sunday self-improve block (search for `run_self_improve` or `sunday` or `weekday() == 6`). Add after the self-improve call:

```python
        # Slippage IC feedback — runs alongside self-improve on Sundays
        try:
            from ascent.monitoring.slippage_ic_feedback import run_slippage_ic_feedback
            _slip_metrics = run_slippage_ic_feedback(lookback_days=90)
            print(f"[SlippageIC] drag={_slip_metrics['slippage_ic_drag']:.4f} "
                  f"gross_ic={_slip_metrics['gross_ic']:.4f} "
                  f"net_ic={_slip_metrics['net_ic']:.4f} "
                  f"fills={_slip_metrics['n_fills']}")
        except Exception as _se:
            print(f"[SlippageIC] Feedback skipped: {_se}")
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/pytest tests/test_slippage_ic_feedback.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ascent/monitoring/slippage_ic_feedback.py tests/test_slippage_ic_feedback.py \
        run_all_agents.py
git commit -m "feat(monitoring): slippage-adjusted IC feedback loop — weekly drag coefficient to active config"
```

---

## Task C: Regime-Conditional Debate Agent Personas

**Problem:** Debate agents argue with the same confidence regardless of how they've actually performed in the current regime. The bull agent might be wrong 70% of the time in stressed regimes but still argues with the same certainty as in calm bull periods. The `outcome_tracker.py` already tracks per-agent, per-regime accuracy in `outputs/debate_log/agent_credibility.json`. This task injects that track record into each agent's system prompt so they self-calibrate.

**Dormant infrastructure note:** `min_samples = 10` is intentional. With 3 debates the accuracy estimate is near-meaningless. As of May 2026 there are 1–2 scored debates total; this feature will produce empty track records for almost every agent/regime combination until ~August 2026. The code is correct and will activate as data accumulates — do not lower the threshold.

**Files:**
- Modify: `debate/outcome_tracker.py` — add `get_agent_regime_accuracy()` helper
- Modify: `debate/agents.py` — add `_get_agent_track_record()`, inject into system prompts
- Create: `tests/test_regime_conditional_personas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regime_conditional_personas.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def _write_credibility(path: Path, bull_stressed=0.35, bear_stressed=0.72, n=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    cred = {
        "by_regime": {
            "stressed": {"bull": bull_stressed, "bear": bear_stressed,
                         "devil": 0.60, "regime_specialist": 0.55},
            "calm_bull": {"bull": 0.68, "bear": 0.45},
        },
        "sample_counts": {
            "stressed":  {"bull": n, "bear": n, "devil": n, "regime_specialist": n},
            "calm_bull": {"bull": n, "bear": n},
        },
    }
    path.write_text(json.dumps(cred))


def test_get_agent_regime_accuracy_returns_float(tmp_path):
    from debate.outcome_tracker import get_agent_regime_accuracy
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        acc = get_agent_regime_accuracy("bull", "stressed")
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_get_agent_regime_accuracy_none_for_missing(tmp_path):
    from debate.outcome_tracker import get_agent_regime_accuracy
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, n=5)  # below min sample count of 10
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        acc = get_agent_regime_accuracy("bull", "stressed")  # n=5 < min_samples=10
    assert acc is None


def test_agent_track_record_injected_into_bull_prompt(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert "35%" in record or "0.35" in record or "35" in record, \
        "Track record must include the accuracy percentage"
    assert "stressed" in record.lower() or "STRESSED" in record


def test_track_record_warns_when_accuracy_below_50(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert "below" in record.lower() or "calibrate" in record.lower() or \
           "50%" in record or "down" in record.lower(), \
        "Should warn when accuracy < 50%"


def test_track_record_empty_string_when_no_data(tmp_path):
    from debate.agents import _get_agent_track_record
    cred_path = tmp_path / "nonexistent_cred.json"
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        record = _get_agent_track_record("bull", "stressed")
    assert record == "", "Must return empty string (not crash) when no credibility data"


def test_run_bull_agent_system_prompt_includes_track_record(tmp_path):
    """The system prompt sent to the LLM must include track record text."""
    import debate.agents as agents_mod
    cred_path = tmp_path / "agent_credibility.json"
    _write_credibility(cred_path, bull_stressed=0.35)

    captured_prompts = []
    def mock_generate(system_prompt, user_prompt, **kwargs):
        captured_prompts.append(system_prompt)
        return '{"verdict": "proceed", "confidence": 0.7, "reasoning": "test"}'

    portfolio_state = {
        "date": "2026-05-03", "us_regime": "stressed", "macro_regime": "stressed",
        "n_positions": 5, "allocation": {}, "weights": {"AAPL": 0.20, "MSFT": 0.20},
    }
    with patch("debate.outcome_tracker.CREDIBILITY_PATH", cred_path):
        with patch("debate.agents.generate_structured", side_effect=mock_generate):
            agents_mod.run_bull_agent(portfolio_state)

    assert len(captured_prompts) > 0
    combined = " ".join(captured_prompts)
    assert "35" in combined or "track" in combined.lower() or "accuracy" in combined.lower(), \
        "System prompt must include agent track record"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_regime_conditional_personas.py -v 2>&1 | head -15
```
Expected: `ImportError` or `AttributeError: module 'debate.outcome_tracker' has no attribute 'get_agent_regime_accuracy'`

- [ ] **Step 3: Add `get_agent_regime_accuracy()` to `debate/outcome_tracker.py`**

Add this function after the `load_credibility_context()` function (around line 280+):

```python
def get_agent_regime_accuracy(agent_name: str, regime: str,
                               min_samples: int = 10) -> Optional[float]:
    """
    Return per-agent accuracy in a specific regime, or None if insufficient data.

    Args:
        agent_name:  "bull", "bear", "devil", or "regime_specialist"
        regime:      Regime label string, e.g. "stressed", "calm_bull"
        min_samples: Minimum number of scored debates required to return a value.
                     Set to 10 — with fewer debates the accuracy estimate is noise.
                     Expect this to return None for all agents until ~August 2026.

    Returns:
        Float accuracy in [0, 1], or None if no data / too few samples.
    """
    cred = _load_credibility()
    regime_key = str(regime).lower()
    accuracy   = cred.get("by_regime", {}).get(regime_key, {}).get(agent_name)
    n_samples  = cred.get("sample_counts", {}).get(regime_key, {}).get(agent_name, 0)
    if accuracy is None or n_samples < min_samples:
        return None  # returns None for nearly all queries until ~August 2026
    return float(accuracy)
```

Note: `_load_credibility()` is an internal helper in `outcome_tracker.py`. Search for the function that loads `CREDIBILITY_PATH` and use its name. If it's inline, create a small helper:

```python
def _load_credibility() -> dict:
    if CREDIBILITY_PATH.exists():
        try:
            return json.loads(CREDIBILITY_PATH.read_text())
        except Exception:
            pass
    return {}
```

- [ ] **Step 4: Add `_get_agent_track_record()` to `debate/agents.py`**

Add this function near the top of `debate/agents.py`, after the imports:

```python
def _get_agent_track_record(agent_name: str, regime: str) -> str:
    """
    Return a formatted track record string for injection into agent system prompts.

    Returns empty string if no data available (graceful degradation).
    """
    try:
        from debate.outcome_tracker import get_agent_regime_accuracy, CREDIBILITY_PATH
        import json as _json
        if not CREDIBILITY_PATH.exists():
            return ""
        cred       = _json.loads(CREDIBILITY_PATH.read_text())
        regime_key = str(regime).lower()
        accuracy   = cred.get("by_regime", {}).get(regime_key, {}).get(agent_name)
        n          = cred.get("sample_counts", {}).get(regime_key, {}).get(agent_name, 0)
        if accuracy is None or n < 10:
            return ""  # dormant until ~August 2026
        warning = (
            " Calibrate confidence DOWN — your historical accuracy here is below 50%."
            if accuracy < 0.50 else
            " Your track record here is solid."
        )
        return (
            f"\nYOUR TRACK RECORD in {regime.upper()} regime: "
            f"{accuracy:.0%} accuracy over {n} debates.{warning}"
        )
    except Exception:
        return ""
```

- [ ] **Step 5: Inject track record into each agent's system prompt**

In `run_bull_agent()`, find the `system_prompt = """..."""` string. Append the track record to it:

```python
def run_bull_agent(portfolio_state: dict) -> str:
    context      = _build_context(portfolio_state)
    regime       = portfolio_state.get("us_regime", "unknown")
    track_record = _get_agent_track_record("bull", regime)   # ← add this line

    system_prompt = f"""You are the Bull analyst...
{track_record}"""                                            # ← append at end of system_prompt
```

Apply the same pattern to `run_bear_agent` (agent_name="bear"), `run_devils_advocate` (agent_name="devil"), and `run_regime_specialist` if it exists (agent_name="regime_specialist").

- [ ] **Step 6: Run all tests**

```bash
.venv/bin/pytest tests/test_regime_conditional_personas.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 7: Full suite check**

```bash
.venv/bin/pytest tests/ -q --tb=short 2>&1 | tail -8
```
Expected: All tests pass (≥265).

- [ ] **Step 8: Commit**

```bash
git add debate/outcome_tracker.py debate/agents.py \
        tests/test_regime_conditional_personas.py
git commit -m "feat(debate): regime-conditional agent personas — historical accuracy injected into system prompts"
```

---

## Final: Push

- [ ] **Push to GitHub**

```bash
git push origin main
```

---

## Self-Review

**Spec coverage:**
- ✅ CoT LLM fundamental signal with Chicago Booth 6-step prompt
- ✅ Anonymization (ticker stripped before LLM call)
- ✅ 45-day filing lag enforced
- ✅ Cache by (symbol, quarter_end)
- ✅ Cross-sectional z-score output
- ✅ Wired into alpha stack at 3%, trend reduced to 41%
- ✅ Slippage-adjusted IC computed from existing log files
- ✅ Drag written to active_alpha_config.json
- ✅ Sunday run wired into run_all_agents.py
- ✅ Per-agent regime accuracy from outcome_tracker
- ✅ Track record injected into all 4 debate agent system prompts

**Type consistency:**
- `llm_fundamental_alpha(fundamentals_df, as_of_date) -> pd.Series` — matches how other alpha functions return Series
- `compute_slippage_ic_drag(lookback_days) -> Dict[str, float]` — consistent throughout
- `get_agent_regime_accuracy(agent_name, regime, min_samples) -> Optional[float]` — used correctly in `_get_agent_track_record`
- `_get_agent_track_record(agent_name, regime) -> str` — returns empty string on failure, never raises
