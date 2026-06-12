# OpenBB Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate OpenBB as a central data adapter to fix yfinance fragility, add historical CBOE options data (fixing the 1-day-snapshot problem), CFTC COT macro positioning, and Fama-French factor inputs, plus two new AI PM live tools.

**Architecture:** Single adapter `ascent/integrations/openbb_client.py` — all OpenBB calls go through it. Hub calls it for price reliability (tiingo fallback). Three new ingest modules write to new parquet caches. Two new tools added to AI PM Phase 2 loop following exact MiroFish pattern.

**Tech Stack:** `openbb`, `openbb-cboe`, `openbb-yfinance`, `openbb-famafrench`, `openbb-cftc`, `openbb-tiingo`, Python 3.12, existing `.venv/`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ascent/integrations/openbb_client.py` | Create | Central adapter — all OpenBB calls |
| `ascent/data/hub.py` | Modify | Replace `_fetch_symbol` with adapter call |
| `memory/ticker_memory.py` | Modify | Replace `_fetch_return` body with adapter call |
| `ascent/data/ingest/cboe_options.py` | Create | Historical options chains (IV skew, PCR, ATM IV) |
| `ascent/data/ingest/cftc_positioning.py` | Create | CFTC COT weekly speculator positioning |
| `ascent/data/ingest/famafrench_factors.py` | Create | Fama-French 5-factor + momentum daily returns |
| `ascent/features/feature_defs.py` | Modify | Add `factor_loadings()` function + gated features |
| `agents/ai_pm_agent.py` | Modify | Add `get_live_options_flow` + `get_cot_positioning` tools |
| `run_all_agents.py` | Modify | Wire CBOE/CFTC/FF ingest into hub run block |
| `tests/integrations/test_openbb_client.py` | Create | Mocked adapter tests |
| `tests/data/test_new_ingest.py` | Create | Mocked ingest tests |

---

## Task 1: Install and Verify OpenBB

**Files:** none

- [ ] **Step 1: Install packages**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pip install openbb openbb-cboe openbb-yfinance openbb-famafrench openbb-cftc openbb-tiingo
```

- [ ] **Step 2: Verify core import works**

```bash
.venv/bin/python -c "from openbb import obb; print('OpenBB OK:', obb)"
```

Expected: `OpenBB OK: <openbb ...>`

- [ ] **Step 3: Verify key providers are installed**

```bash
.venv/bin/python -c "
from openbb import obb
print('cboe:', 'cboe' in str(obb.derivatives.options.chains.__doc__ or ''))
result = obb.equity.price.historical('AAPL', start_date='2026-01-02', end_date='2026-01-10', provider='yfinance')
df = result.to_dataframe()
print('yfinance prices:', df.shape)
"
```

Expected: no ImportError, prices DataFrame with rows > 0

---

## Task 2: `ascent/integrations/openbb_client.py` — Price Functions

**Files:**
- Create: `ascent/integrations/openbb_client.py`
- Create: `tests/integrations/test_openbb_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/integrations/test_openbb_client.py
from __future__ import annotations
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


def _make_mock_obb_result(symbol: str, dates: list, closes: list) -> MagicMock:
    """Build a mock OBBject that .to_dataframe() returns a price DataFrame."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "close": closes,
        "open": closes,
        "high": closes,
        "low": closes,
        "volume": [1_000_000] * len(closes),
    }).set_index("date")
    mock = MagicMock()
    mock.to_dataframe.return_value = df
    return mock


def test_fetch_symbol_returns_dataframe():
    from ascent.integrations.openbb_client import fetch_symbol
    mock_result = _make_mock_obb_result(
        "AAPL",
        ["2026-01-02", "2026-01-03", "2026-01-06"],
        [220.0, 221.5, 223.0],
    )
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")
    assert result is not None
    assert not result.empty
    assert "close" in result.columns
    assert "symbol" in result.columns


def test_fetch_symbol_falls_back_on_tiingo_failure():
    from ascent.integrations.openbb_client import fetch_symbol
    import requests
    mock_result = _make_mock_obb_result(
        "AAPL",
        ["2026-01-02", "2026-01-03"],
        [220.0, 221.5],
    )
    call_count = {"n": 0}

    def side_effect(symbol, start_date, end_date, provider):
        call_count["n"] += 1
        if provider == "tiingo":
            raise Exception("tiingo failed")
        return mock_result

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = side_effect
        mock_obb_fn.return_value = mock_obb
        result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")

    assert result is not None
    assert call_count["n"] == 2  # tried tiingo, then yfinance


def test_fetch_symbol_returns_none_when_both_fail():
    from ascent.integrations.openbb_client import fetch_symbol
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = Exception("all providers failed")
        mock_obb_fn.return_value = mock_obb
        result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")
    assert result is None


def test_fetch_return_computes_forward_return():
    from ascent.integrations.openbb_client import fetch_return
    mock_result = _make_mock_obb_result(
        "CAT",
        ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07",
         "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13",
         "2026-01-14", "2026-01-15", "2026-01-16"],
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
    )
    with patch("ascent.integrations.openbb_client.fetch_symbol", return_value=mock_result.to_dataframe()):
        result = fetch_return("CAT", "2026-01-02", 10)
    assert result is not None
    assert abs(result - 0.10) < 0.001  # 110/100 - 1 = 10%


def test_fetch_return_returns_none_on_failure():
    from ascent.integrations.openbb_client import fetch_return
    with patch("ascent.integrations.openbb_client.fetch_symbol", return_value=None):
        result = fetch_return("CAT", "2026-01-02", 10)
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/integrations/test_openbb_client.py::test_fetch_symbol_returns_dataframe -v 2>&1 | tail -5
```

Expected: `ImportError` or `ModuleNotFoundError` for `openbb_client`

- [ ] **Step 3: Create `ascent/integrations/openbb_client.py` — price functions**

```python
# ascent/integrations/openbb_client.py
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TIINGO_TOKEN_ENV = "TIINGO_TOKEN"


def _get_obb():
    """Lazy-import OpenBB and set credentials from env."""
    from openbb import obb
    token = os.environ.get(_TIINGO_TOKEN_ENV, "")
    if token:
        try:
            obb.user.credentials.tiingo_token = token
        except Exception:
            pass
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        try:
            obb.user.credentials.fred_api_key = fred_key
        except Exception:
            pass
    cftc_token = os.environ.get("CFTC_APP_TOKEN", "")
    if cftc_token:
        try:
            obb.user.credentials.cftc_app_token = cftc_token
        except Exception:
            pass
    return obb


def _normalize_price_df(df: pd.DataFrame, sym: str, source: str) -> pd.DataFrame:
    """Normalize an OBBject price DataFrame to hub schema."""
    df = df.reset_index() if df.index.name == "date" else df.copy()
    if "date" not in df.columns and df.index.name:
        df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = sym.upper()
    df["source"] = source
    keep = [c for c in ["symbol", "date", "close", "high", "low", "open", "volume", "adj_close", "source"]
            if c in df.columns]
    return df[keep].dropna(subset=["close"])


def fetch_symbol(sym: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for one symbol. Tries Tiingo first (if TIINGO_TOKEN set),
    falls back to yfinance. Returns normalized hub-schema DataFrame or None.
    """
    obb = _get_obb()
    has_tiingo = bool(os.environ.get(_TIINGO_TOKEN_ENV, ""))
    providers = (["tiingo", "yfinance"] if has_tiingo else ["yfinance"])

    for provider in providers:
        try:
            result = obb.equity.price.historical(
                sym, start_date=start, end_date=end, provider=provider
            )
            df = result.to_dataframe()
            if df.empty:
                continue
            normalized = _normalize_price_df(df, sym, f"{provider}_hub")
            if not normalized.empty:
                return normalized
        except Exception as exc:
            log.debug("[OBBClient] %s via %s failed: %s", sym, provider, exc)

    return None


def fetch_return(symbol: str, from_date: str, horizon_days: int) -> Optional[float]:
    """
    Fetch forward return for one symbol: return at from_date + horizon_days business days.
    Returns None on failure.
    """
    try:
        end = (date.fromisoformat(from_date) + timedelta(days=horizon_days + 20)).isoformat()
        df = fetch_symbol(symbol, from_date, end)
        if df is None or df.empty or len(df) < 2:
            return None
        closes = df.sort_values("date")["close"].dropna()
        idx = min(horizon_days, len(closes) - 1)
        return float((closes.iloc[idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[OBBClient] fetch_return %s: %s", symbol, exc)
        return None
```

- [ ] **Step 4: Run price tests**

```bash
.venv/bin/python -m pytest tests/integrations/test_openbb_client.py::test_fetch_symbol_returns_dataframe tests/integrations/test_openbb_client.py::test_fetch_symbol_falls_back_on_tiingo_failure tests/integrations/test_openbb_client.py::test_fetch_symbol_returns_none_when_both_fail tests/integrations/test_openbb_client.py::test_fetch_return_computes_forward_return tests/integrations/test_openbb_client.py::test_fetch_return_returns_none_on_failure -v 2>&1 | tail -10
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add ascent/integrations/openbb_client.py tests/integrations/test_openbb_client.py
git commit -m "feat: openbb_client price adapter with tiingo/yfinance fallback"
```

---

## Task 3: `openbb_client.py` — Data Query Functions

**Files:**
- Modify: `ascent/integrations/openbb_client.py`
- Modify: `tests/integrations/test_openbb_client.py`

- [ ] **Step 1: Add failing tests — append to `tests/integrations/test_openbb_client.py`**

```python
# ---------- Task 3 tests ----------

def test_get_live_macro_returns_dict():
    from ascent.integrations.openbb_client import get_live_macro
    mock_df = pd.DataFrame({"value": [5.33]}, index=pd.to_datetime(["2026-06-06"]))
    mock_df.index.name = "date"

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = mock_df
        mock_obb.economy.fred.series.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_live_macro()

    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_live_macro_falls_back_to_parquet(tmp_path):
    from ascent.integrations.openbb_client import get_live_macro
    # Make a fake macro_live.parquet
    fake = pd.DataFrame({
        "fed_funds_rate": [5.33],
        "treasury_10y": [4.25],
    }, index=pd.to_datetime(["2026-06-06"]))
    fake.index.name = "date"
    cache = tmp_path / "macro_live.parquet"
    fake.to_parquet(cache)

    with patch("ascent.integrations.openbb_client._get_obb", side_effect=Exception("obb down")):
        with patch("ascent.integrations.openbb_client._MACRO_CACHE_PATH", cache):
            result = get_live_macro()

    assert "fed_funds_rate" in result
    assert abs(result["fed_funds_rate"] - 5.33) < 0.01


def test_get_options_snapshot_returns_per_symbol():
    from ascent.integrations.openbb_client import get_options_snapshot
    mock_chain_df = pd.DataFrame({
        "strike": [200.0, 210.0, 220.0, 230.0, 240.0],
        "expiration": ["2026-07-18"] * 5,
        "option_type": ["put", "put", "call", "call", "call"],
        "implied_volatility": [0.28, 0.25, 0.22, 0.24, 0.26],
        "volume": [500, 300, 400, 200, 150],
        "underlying_price": [220.0] * 5,
    })
    mock_result = MagicMock()
    mock_result.to_dataframe.return_value = mock_chain_df

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.derivatives.options.chains.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_options_snapshot(["CAT"])

    assert "CAT" in result
    entry = result["CAT"]
    assert "put_call_ratio" in entry
    assert "atm_iv" in entry
    assert "iv_skew" in entry
    assert entry["put_call_ratio"] >= 0
    assert entry["atm_iv"] > 0


def test_get_options_snapshot_handles_failure():
    from ascent.integrations.openbb_client import get_options_snapshot
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.derivatives.options.chains.side_effect = Exception("cboe down")
        mock_obb_fn.return_value = mock_obb
        result = get_options_snapshot(["CAT"])
    assert "CAT" in result
    assert result["CAT"].get("unavailable") is True


def test_get_cot_snapshot_returns_dict():
    from ascent.integrations.openbb_client import get_cot_snapshot
    mock_cot_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-06"]),
        "noncomm_positions_long_all": [187420],
        "noncomm_positions_short_all": [62180],
        "open_interest_all": [3200000],
    })
    mock_result = MagicMock()
    mock_result.to_dataframe.return_value = mock_cot_df

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.regulators.cftc.cot.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_cot_snapshot()

    assert isinstance(result, dict)
    assert "net_noncommercial_long" in result
    assert "pct_long_noncommercial" in result
    assert "as_of_date" in result


def test_get_cot_snapshot_returns_none_on_failure():
    from ascent.integrations.openbb_client import get_cot_snapshot
    with patch("ascent.integrations.openbb_client._get_obb", side_effect=Exception("cftc down")):
        result = get_cot_snapshot()
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/integrations/test_openbb_client.py::test_get_live_macro_returns_dict -v 2>&1 | tail -5
```

Expected: `AttributeError` or `ImportError` — functions not yet defined

- [ ] **Step 3: Append data query functions to `ascent/integrations/openbb_client.py`**

```python
# ── Macro data ─────────────────────────────────────────────────────────────────

_MACRO_CACHE_PATH = _REPO_ROOT / "data_cache" / "macro_live.parquet"

_FRED_SERIES = {
    "DFF":            "fed_funds_rate",
    "DGS10":          "treasury_10y",
    "DGS2":           "treasury_2y",
    "T10Y2Y":         "yield_spread_10y2y",
    "VIXCLS":         "vix",
    "CPIAUCSL":       "cpi",
    "UNRATE":         "unemployment",
    "DCOILWTICO":     "oil_wti",
    "DEXUSEU":        "usd_eur",
    "BAMLH0A0HYM2":   "hy_spread",
    "BAMLC0A0CM":     "ig_spread",
}


def _macro_from_parquet() -> dict[str, float]:
    """Fallback: read latest row from macro_live.parquet."""
    try:
        if _MACRO_CACHE_PATH.exists():
            df = pd.read_parquet(_MACRO_CACHE_PATH)
            if not df.empty:
                latest = df.sort_index().iloc[-1]
                return {col: float(v) for col, v in latest.items() if pd.notna(v)}
    except Exception as exc:
        log.debug("[OBBClient] parquet macro fallback failed: %s", exc)
    return {}


def get_live_macro() -> dict[str, float]:
    """
    Fetch live macro indicators. Tries FRED via OpenBB; falls back to cached parquet.
    Returns {series_name: latest_value}.
    """
    try:
        obb = _get_obb()
        results: dict[str, float] = {}
        for fred_id, name in _FRED_SERIES.items():
            try:
                df = obb.economy.fred.series(
                    symbol=fred_id, provider="fred"
                ).to_dataframe()
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    val_col = "value" if "value" in df.columns else df.columns[0]
                    v = float(df.sort_index().iloc[-1][val_col])
                    results[name] = v
            except Exception:
                pass
        if results:
            return results
    except Exception as exc:
        log.debug("[OBBClient] live macro failed: %s", exc)
    return _macro_from_parquet()


# ── Options snapshot ───────────────────────────────────────────────────────────

def get_options_snapshot(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch current options chain snapshot for each symbol via CBOE.
    Returns per-symbol dict with keys: put_call_ratio, iv_skew, atm_iv, iv_rank_52w, unavailable.
    """
    obb = _get_obb()
    results: dict[str, dict] = {}

    for sym in symbols:
        sym_upper = sym.upper()
        try:
            chain_df = obb.derivatives.options.chains(
                sym_upper, provider="cboe"
            ).to_dataframe()
            if chain_df.empty:
                results[sym_upper] = {"unavailable": True}
                continue

            chain_df.columns = [c.lower() for c in chain_df.columns]
            chain_df = chain_df.rename(columns={
                "impliedvolatility": "implied_volatility",
                "implied_vol": "implied_volatility",
            })

            spot = float(chain_df["underlying_price"].iloc[0]) if "underlying_price" in chain_df.columns else None
            if spot is None or spot <= 0:
                results[sym_upper] = {"unavailable": True}
                continue

            # PCR: total put volume / total call volume
            puts  = chain_df[chain_df["option_type"].str.lower() == "put"]
            calls = chain_df[chain_df["option_type"].str.lower() == "call"]
            put_vol  = float(puts["volume"].sum()) if "volume" in puts.columns else 0.0
            call_vol = float(calls["volume"].sum()) if "volume" in calls.columns else 1.0
            pcr = round(put_vol / max(call_vol, 1.0), 3)

            # ATM IV: strike closest to spot
            chain_df["moneyness"] = abs(chain_df["strike"] - spot)
            atm_row = chain_df.nsmallest(1, "moneyness")
            atm_iv = float(atm_row["implied_volatility"].iloc[0]) if "implied_volatility" in atm_row.columns else None

            # IV skew: OTM call (strike >= 1.03*spot) IV minus OTM put (strike <= 0.97*spot) IV
            otm_calls = calls[calls["strike"] >= spot * 1.03].nsmallest(1, "moneyness")
            otm_puts  = puts[puts["strike"] <= spot * 0.97].nsmallest(1, "moneyness")
            if not otm_calls.empty and not otm_puts.empty and "implied_volatility" in chain_df.columns:
                call_iv = float(otm_calls["implied_volatility"].iloc[0])
                put_iv  = float(otm_puts["implied_volatility"].iloc[0])
                iv_skew = round(call_iv - put_iv, 4)
            else:
                iv_skew = None

            # iv_rank_52w: computed from stored cboe options cache if available
            iv_rank = _compute_iv_rank(sym_upper, atm_iv)

            results[sym_upper] = {
                "put_call_ratio": pcr,
                "atm_iv": round(atm_iv, 4) if atm_iv else None,
                "iv_skew": iv_skew,
                "iv_rank_52w": iv_rank,
                "unavailable": False,
            }
        except Exception as exc:
            log.debug("[OBBClient] options snapshot %s failed: %s", sym_upper, exc)
            results[sym_upper] = {"unavailable": True}

    return results


def _compute_iv_rank(symbol: str, current_iv: Optional[float]) -> Optional[int]:
    """Compute IV percentile rank vs stored 52w history. Returns None if < 21 days of history."""
    if current_iv is None:
        return None
    try:
        cache = _REPO_ROOT / "data_cache" / "options_flow.parquet"
        if not cache.exists():
            return None
        df = pd.read_parquet(cache)
        df = df[df["symbol"] == symbol] if "symbol" in df.columns else df
        if "atm_iv" not in df.columns or len(df) < 21:
            return None
        history = df["atm_iv"].dropna().tail(252)
        if len(history) < 21:
            return None
        rank = int((history < current_iv).mean() * 100)
        return rank
    except Exception:
        return None


# ── COT positioning ────────────────────────────────────────────────────────────

_SP500_COT_CODE = "13874+"  # S&P 500 Non-Commercial Futures, CME


def get_cot_snapshot() -> Optional[dict]:
    """
    Fetch latest CFTC COT report for S&P 500 e-mini futures.
    Returns dict with net_noncommercial_long, pct_long_noncommercial, as_of_date.
    Returns None on failure.
    """
    try:
        obb = _get_obb()
        df = obb.regulators.cftc.cot(
            code=_SP500_COT_CODE, provider="cftc", limit=2
        ).to_dataframe()

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        latest = df.sort_values("date").iloc[-1] if "date" in df.columns else df.iloc[-1]

        # Column names from CFTC legacy report
        long_col  = next((c for c in df.columns if "noncomm" in c and "long" in c and "spread" not in c), None)
        short_col = next((c for c in df.columns if "noncomm" in c and "short" in c and "spread" not in c), None)
        oi_col    = next((c for c in df.columns if "open_interest" in c or "oi" == c), None)

        if not long_col or not short_col:
            log.warning("[OBBClient] COT column names unexpected: %s", list(df.columns[:10]))
            return None

        net_long  = int(latest[long_col]) - int(latest[short_col])
        oi        = int(latest[oi_col]) if oi_col and pd.notna(latest.get(oi_col)) else None
        pct_long  = round(int(latest[long_col]) / oi * 100, 1) if oi else None
        as_of     = str(latest["date"])[:10] if "date" in df.columns else "unknown"

        return {
            "net_noncommercial_long": net_long,
            "noncomm_long":  int(latest[long_col]),
            "noncomm_short": int(latest[short_col]),
            "pct_long_noncommercial": pct_long,
            "open_interest": oi,
            "as_of_date": as_of,
        }
    except Exception as exc:
        log.warning("[OBBClient] COT snapshot failed: %s", exc)
        return None
```

- [ ] **Step 4: Run all openbb_client tests**

```bash
.venv/bin/python -m pytest tests/integrations/test_openbb_client.py -v 2>&1 | tail -15
```

Expected: 11 PASSED

- [ ] **Step 5: ast.parse verify**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/integrations/openbb_client.py').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add ascent/integrations/openbb_client.py tests/integrations/test_openbb_client.py
git commit -m "feat: openbb_client data query functions (macro, options, COT)"
```

---

## Task 4: Hub + Ticker Memory Reliability

**Files:**
- Modify: `ascent/data/hub.py`
- Modify: `memory/ticker_memory.py`

- [ ] **Step 1: Modify `ascent/data/hub.py` — replace `_fetch_symbol` with adapter call**

Open `ascent/data/hub.py`. Find `_fetch_symbol(sym, start, end)` (around line 82). Replace its body:

```python
def _fetch_symbol(sym: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetch one symbol. Tries OpenBB adapter (tiingo→yfinance fallback).
    Falls back to direct yfinance if adapter unavailable.
    """
    try:
        from ascent.integrations.openbb_client import fetch_symbol as _obb_fetch
        result = _obb_fetch(sym, start, end)
        if result is not None and not result.empty:
            return result
    except Exception as exc:
        log.debug("[Hub] openbb_client unavailable for %s: %s", sym, exc)

    # Direct yfinance fallback (existing behavior)
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        })
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["source"] = "yahoo_hub"
        cols = [c for c in
                ["symbol", "date", "close", "high", "low", "open", "volume", "adj_close", "source"]
                if c in df.columns]
        return df[cols]
    except Exception as e:
        log.warning("[Hub] %s: fetch failed — %s", sym, e)
        return None
```

- [ ] **Step 2: Verify hub.py syntax**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/data/hub.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Modify `memory/ticker_memory.py` — replace `_fetch_return` body**

Open `memory/ticker_memory.py`. Find `_fetch_return(symbol, from_date, horizon)`. Replace its body:

```python
def _fetch_return(symbol: str, from_date: str, horizon: int) -> Optional[float]:
    """Fetch stock return at from_date + horizon trading days. Returns None on failure."""
    try:
        from ascent.integrations.openbb_client import fetch_return as _obb_fetch
        return _obb_fetch(symbol, from_date, horizon)
    except Exception:
        pass
    # Direct yfinance fallback
    try:
        import yfinance as yf
        start = date.fromisoformat(from_date)
        end   = (start + timedelta(days=horizon + 15)).isoformat()
        df    = yf.download(symbol, start=from_date, end=end,
                            auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].squeeze().dropna()
        idx = min(horizon, len(closes) - 1)
        return float((closes.iloc[idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[TickerMemory] _fetch_return %s: %s", symbol, exc)
        return None
```

- [ ] **Step 4: Verify ticker_memory.py syntax**

```bash
.venv/bin/python -c "import ast; ast.parse(open('memory/ticker_memory.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run existing ticker_memory tests**

```bash
.venv/bin/python -m pytest tests/memory/test_ticker_memory.py -v 2>&1 | tail -15
```

Expected: all existing tests still pass (≥10 PASSED)

- [ ] **Step 6: Commit**

```bash
git add ascent/data/hub.py memory/ticker_memory.py
git commit -m "feat: hub + ticker_memory use openbb_client with yfinance fallback"
```

---

## Task 5: `ascent/data/ingest/cboe_options.py` — Historical Options

**Files:**
- Create: `ascent/data/ingest/cboe_options.py`
- Create: `tests/data/test_new_ingest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_new_ingest.py
from __future__ import annotations
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── CBOE options tests ─────────────────────────────────────────────────────────

def test_fetch_cboe_options_returns_dataframe():
    from ascent.data.ingest.cboe_options import fetch_cboe_options_row
    mock_snapshot = {
        "CAT": {
            "put_call_ratio": 0.82,
            "atm_iv": 0.215,
            "iv_skew": 0.031,
            "iv_rank_52w": 34,
            "unavailable": False,
        }
    }
    with patch("ascent.data.ingest.cboe_options.get_options_snapshot", return_value=mock_snapshot):
        result = fetch_cboe_options_row("CAT", "2026-06-06")
    assert result is not None
    assert result["symbol"] == "CAT"
    assert result["date"] == "2026-06-06"
    assert "put_call_ratio" in result
    assert "atm_iv" in result
    assert "iv_skew" in result


def test_fetch_cboe_options_returns_none_when_unavailable():
    from ascent.data.ingest.cboe_options import fetch_cboe_options_row
    mock_snapshot = {"CAT": {"unavailable": True}}
    with patch("ascent.data.ingest.cboe_options.get_options_snapshot", return_value=mock_snapshot):
        result = fetch_cboe_options_row("CAT", "2026-06-06")
    assert result is None


def test_update_options_cache_appends_new_rows(tmp_path):
    from ascent.data.ingest.cboe_options import update_options_cache
    mock_rows = [
        {"symbol": "CAT", "date": "2026-06-06", "put_call_ratio": 0.82,
         "atm_iv": 0.215, "iv_skew": 0.031, "iv_rank_52w": None},
        {"symbol": "MRK", "date": "2026-06-06", "put_call_ratio": 1.12,
         "atm_iv": 0.198, "iv_skew": -0.025, "iv_rank_52w": None},
    ]
    cache_path = tmp_path / "options_flow.parquet"

    with patch("ascent.data.ingest.cboe_options.fetch_cboe_options_row",
               side_effect=lambda sym, dt: next(
                   (r for r in mock_rows if r["symbol"] == sym), None)):
        update_options_cache(["CAT", "MRK"], "2026-06-06", cache_path=cache_path)

    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert len(df) == 2
    assert set(df["symbol"].unique()) == {"CAT", "MRK"}


def test_update_options_cache_does_not_duplicate(tmp_path):
    from ascent.data.ingest.cboe_options import update_options_cache
    existing = pd.DataFrame([{
        "symbol": "CAT", "date": pd.Timestamp("2026-06-06"),
        "put_call_ratio": 0.80, "atm_iv": 0.200, "iv_skew": 0.020, "iv_rank_52w": None,
    }])
    cache_path = tmp_path / "options_flow.parquet"
    existing.to_parquet(cache_path, index=False)

    new_row = {"symbol": "CAT", "date": "2026-06-06", "put_call_ratio": 0.82,
               "atm_iv": 0.215, "iv_skew": 0.031, "iv_rank_52w": None}

    with patch("ascent.data.ingest.cboe_options.fetch_cboe_options_row", return_value=new_row):
        update_options_cache(["CAT"], "2026-06-06", cache_path=cache_path)

    df = pd.read_parquet(cache_path)
    # Should not duplicate — one row per (symbol, date)
    assert len(df[df["symbol"] == "CAT"]) == 1
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_cboe_options_returns_dataframe -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `ascent/data/ingest/cboe_options.py`**

```python
# ascent/data/ingest/cboe_options.py
"""
Historical CBOE options data — IV skew, put/call ratio, ATM IV.
Extends the existing options_flow cache with historical data.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ascent.integrations.openbb_client import get_options_snapshot

log = logging.getLogger(__name__)

_REPO_ROOT  = Path(__file__).resolve().parents[3]
_CACHE_NAME = "options_flow"
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / f"{_CACHE_NAME}.parquet"


def fetch_cboe_options_row(symbol: str, fetch_date: str) -> Optional[dict]:
    """
    Fetch a single CBOE options row for symbol on fetch_date.
    Returns dict with columns matching options_flow schema, or None if unavailable.
    """
    snapshot = get_options_snapshot([symbol])
    entry = snapshot.get(symbol.upper(), {})

    if entry.get("unavailable"):
        log.debug("[CBOEOptions] %s unavailable", symbol)
        return None

    return {
        "symbol":         symbol.upper(),
        "date":           fetch_date,
        "put_call_ratio": entry.get("put_call_ratio"),
        "atm_iv":         entry.get("atm_iv"),
        "iv_skew":        entry.get("iv_skew"),
        "iv_rank_52w":    entry.get("iv_rank_52w"),
        "source":         "cboe",
    }


def update_options_cache(
    symbols: list[str],
    fetch_date: str,
    cache_path: Path = _DEFAULT_CACHE,
) -> int:
    """
    Fetch options data for each symbol and append new rows to the cache.
    Deduplicates on (symbol, date) — existing rows for the same date are not overwritten.
    Returns count of newly added rows.
    """
    # Load existing cache
    existing: pd.DataFrame = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if "date" in existing.columns:
                existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        except Exception as exc:
            log.warning("[CBOEOptions] Could not read cache: %s", exc)

    new_rows = []
    for sym in symbols:
        # Skip if already have this date
        if not existing.empty and "symbol" in existing.columns and "date" in existing.columns:
            dup = existing[(existing["symbol"] == sym.upper()) & (existing["date"] == fetch_date)]
            if not dup.empty:
                log.debug("[CBOEOptions] %s @ %s already in cache, skipping", sym, fetch_date)
                continue

        row = fetch_cboe_options_row(sym, fetch_date)
        if row:
            new_rows.append(row)

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if existing.empty:
        combined = new_df
    else:
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, new_df], ignore_index=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("[CBOEOptions] Added %d new rows to %s", len(new_rows), cache_path.name)
    return len(new_rows)
```

- [ ] **Step 4: Run CBOE tests**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_cboe_options_returns_dataframe tests/data/test_new_ingest.py::test_fetch_cboe_options_returns_none_when_unavailable tests/data/test_new_ingest.py::test_update_options_cache_appends_new_rows tests/data/test_new_ingest.py::test_update_options_cache_does_not_duplicate -v 2>&1 | tail -10
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add ascent/data/ingest/cboe_options.py tests/data/test_new_ingest.py
git commit -m "feat: CBOE historical options ingest (IV skew, PCR, ATM IV)"
```

---

## Task 6: `ascent/data/ingest/cftc_positioning.py` — COT Positioning

**Files:**
- Create: `ascent/data/ingest/cftc_positioning.py`
- Modify: `tests/data/test_new_ingest.py`

- [ ] **Step 1: Append failing CFTC tests to `tests/data/test_new_ingest.py`**

```python
# ── CFTC positioning tests ─────────────────────────────────────────────────────

def test_fetch_cot_returns_dataframe_row():
    from ascent.data.ingest.cftc_positioning import fetch_cot_row
    mock_snapshot = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",
    }
    with patch("ascent.data.ingest.cftc_positioning.get_cot_snapshot", return_value=mock_snapshot):
        result = fetch_cot_row()
    assert result is not None
    assert "net_noncommercial_long" in result
    assert "as_of_date" in result
    assert result["net_noncommercial_long"] == 125240


def test_fetch_cot_returns_none_on_failure():
    from ascent.data.ingest.cftc_positioning import fetch_cot_row
    with patch("ascent.data.ingest.cftc_positioning.get_cot_snapshot", return_value=None):
        result = fetch_cot_row()
    assert result is None


def test_update_cot_cache_appends_row(tmp_path):
    from ascent.data.ingest.cftc_positioning import update_cot_cache
    mock_row = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",
    }
    cache_path = tmp_path / "cftc_positioning.parquet"
    with patch("ascent.data.ingest.cftc_positioning.fetch_cot_row", return_value=mock_row):
        update_cot_cache(cache_path=cache_path)
    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert len(df) == 1
    assert df.iloc[0]["net_noncommercial_long"] == 125240


def test_update_cot_cache_deduplicates(tmp_path):
    from ascent.data.ingest.cftc_positioning import update_cot_cache
    existing = pd.DataFrame([{
        "as_of_date": "2026-06-06",
        "net_noncommercial_long": 120000,
        "noncomm_long": 180000,
        "noncomm_short": 60000,
        "pct_long_noncommercial": 60.0,
        "open_interest": 3000000,
    }])
    cache_path = tmp_path / "cftc_positioning.parquet"
    existing.to_parquet(cache_path, index=False)

    mock_row = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",  # same date
    }
    with patch("ascent.data.ingest.cftc_positioning.fetch_cot_row", return_value=mock_row):
        update_cot_cache(cache_path=cache_path)

    df = pd.read_parquet(cache_path)
    assert len(df) == 1  # not duplicated


def test_get_latest_cot_reads_cache(tmp_path):
    from ascent.data.ingest.cftc_positioning import get_latest_cot
    df = pd.DataFrame([
        {"as_of_date": "2026-05-30", "net_noncommercial_long": 110000,
         "noncomm_long": 170000, "noncomm_short": 60000,
         "pct_long_noncommercial": 56.7, "open_interest": 3000000},
        {"as_of_date": "2026-06-06", "net_noncommercial_long": 125240,
         "noncomm_long": 187420, "noncomm_short": 62180,
         "pct_long_noncommercial": 63.2, "open_interest": 3200000},
    ])
    cache_path = tmp_path / "cftc_positioning.parquet"
    df.to_parquet(cache_path, index=False)
    result = get_latest_cot(cache_path=cache_path)
    assert result is not None
    assert result["as_of_date"] == "2026-06-06"
    assert result["net_noncommercial_long"] == 125240
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_cot_returns_dataframe_row -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `ascent/data/ingest/cftc_positioning.py`**

```python
# ascent/data/ingest/cftc_positioning.py
"""
CFTC Commitments of Traders — S&P 500 e-mini speculator positioning.
Fetches weekly COT report via OpenBB CFTC provider.
Cache: data_cache/cftc_positioning.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ascent.integrations.openbb_client import get_cot_snapshot

log = logging.getLogger(__name__)

_REPO_ROOT    = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "cftc_positioning.parquet"


def fetch_cot_row() -> Optional[dict]:
    """Fetch latest COT row via openbb_client. Returns None on failure."""
    return get_cot_snapshot()


def update_cot_cache(cache_path: Path = _DEFAULT_CACHE) -> bool:
    """
    Fetch latest COT report and append to cache if not already present.
    Deduplicates on as_of_date. Returns True if new row added.
    """
    row = fetch_cot_row()
    if row is None:
        log.warning("[COT] Fetch failed — cache not updated")
        return False

    existing: pd.DataFrame = pd.DataFrame()
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
        except Exception as exc:
            log.warning("[COT] Could not read cache: %s", exc)

    as_of = row.get("as_of_date", "")
    if not existing.empty and "as_of_date" in existing.columns:
        if as_of in existing["as_of_date"].astype(str).values:
            log.debug("[COT] %s already in cache", as_of)
            return False

    new_df = pd.DataFrame([row])
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    log.info("[COT] Added row for %s", as_of)
    return True


def get_latest_cot(cache_path: Path = _DEFAULT_CACHE) -> Optional[dict]:
    """
    Read latest COT row from cache. Returns dict or None if cache absent.
    Used by AI PM tool executor — does NOT fetch live.
    """
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None
        latest = df.sort_values("as_of_date").iloc[-1]
        return latest.to_dict()
    except Exception as exc:
        log.warning("[COT] get_latest_cot failed: %s", exc)
        return None
```

- [ ] **Step 4: Run CFTC tests**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_cot_returns_dataframe_row tests/data/test_new_ingest.py::test_fetch_cot_returns_none_on_failure tests/data/test_new_ingest.py::test_update_cot_cache_appends_row tests/data/test_new_ingest.py::test_update_cot_cache_deduplicates tests/data/test_new_ingest.py::test_get_latest_cot_reads_cache -v 2>&1 | tail -10
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add ascent/data/ingest/cftc_positioning.py tests/data/test_new_ingest.py
git commit -m "feat: CFTC COT positioning ingest for S&P 500 e-mini"
```

---

## Task 7: `ascent/data/ingest/famafrench_factors.py` — FF Factor Returns

**Files:**
- Create: `ascent/data/ingest/famafrench_factors.py`
- Modify: `tests/data/test_new_ingest.py`

- [ ] **Step 1: Append failing FF tests to `tests/data/test_new_ingest.py`**

```python
# ── Fama-French factor tests ───────────────────────────────────────────────────

def test_fetch_ff_factors_returns_dataframe():
    from ascent.data.ingest.famafrench_factors import fetch_ff_factors
    mock_df_5f = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"]),
        "mkt_rf": [0.0082, -0.0031, 0.0055],
        "smb":    [0.0021, 0.0010, -0.0008],
        "hml":    [-0.0015, 0.0020, 0.0011],
        "rmw":    [0.0005, -0.0003, 0.0007],
        "cma":    [0.0003, 0.0001, -0.0002],
        "rf":     [0.0002, 0.0002, 0.0002],
    })
    mock_df_mom = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"]),
        "mom": [0.0032, -0.0041, 0.0018],
    })

    def mock_ff_call(factor, frequency, region, provider):
        result = MagicMock()
        result.to_dataframe.return_value = mock_df_5f if factor == "5_factors" else mock_df_mom
        return result

    with patch("ascent.data.ingest.famafrench_factors._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.famafrench.factors.side_effect = mock_ff_call
        mock_obb_fn.return_value = mock_obb
        df = fetch_ff_factors(start="2026-06-01")

    assert df is not None
    assert not df.empty
    for col in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom"):
        assert col in df.columns


def test_update_ff_cache_writes_parquet(tmp_path):
    from ascent.data.ingest.famafrench_factors import update_ff_cache
    mock_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-04", "2026-06-05"]),
        "mkt_rf": [0.008, -0.003],
        "smb":    [0.002, 0.001],
        "hml":    [-0.001, 0.002],
        "rmw":    [0.001, -0.001],
        "cma":    [0.000,  0.000],
        "mom":    [0.003, -0.004],
    }).set_index("date")
    cache_path = tmp_path / "famafrench_factors.parquet"
    with patch("ascent.data.ingest.famafrench_factors.fetch_ff_factors", return_value=mock_df):
        update_ff_cache(cache_path=cache_path)
    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert "mkt_rf" in df.columns
    assert len(df) == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_ff_factors_returns_dataframe -v 2>&1 | tail -5
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `ascent/data/ingest/famafrench_factors.py`**

```python
# ascent/data/ingest/famafrench_factors.py
"""
Fama-French 5-factor + momentum daily returns.
Cache: data_cache/famafrench_factors.parquet
Used as ML sleeve feature inputs via feature_defs.factor_loadings().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT     = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE = _REPO_ROOT / "data_cache" / "famafrench_factors.parquet"
_DEFAULT_START = "2018-01-01"


def _get_obb():
    from ascent.integrations.openbb_client import _get_obb
    return _get_obb()


def fetch_ff_factors(start: str = _DEFAULT_START) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 5-factor + momentum daily returns for America.
    Returns DataFrame indexed by date with columns: mkt_rf, smb, hml, rmw, cma, mom.
    Returns None on failure.
    """
    try:
        obb = _get_obb()

        df_5f = obb.famafrench.factors(
            factor="5_factors",
            frequency="daily",
            region="america",
            provider="famafrench",
        ).to_dataframe()

        df_mom = obb.famafrench.factors(
            factor="momentum",
            frequency="daily",
            region="america",
            provider="famafrench",
        ).to_dataframe()

        # Normalize both DataFrames
        for df in (df_5f, df_mom):
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)

        # Merge on date index
        keep_5f  = [c for c in ("mkt_rf", "smb", "hml", "rmw", "cma", "rf") if c in df_5f.columns]
        keep_mom = [c for c in ("mom", "wml") if c in df_mom.columns]
        mom_col  = "mom" if "mom" in keep_mom else ("wml" if "wml" in keep_mom else None)

        combined = df_5f[keep_5f].copy()
        if mom_col:
            combined["mom"] = df_mom[mom_col]

        # Divide by 100 if returns appear to be in percentage form
        sample = combined["mkt_rf"].dropna()
        if not sample.empty and abs(sample.iloc[-1]) > 1.0:
            combined = combined / 100.0

        combined = combined[combined.index >= pd.Timestamp(start)]
        combined.dropna(subset=["mkt_rf"], inplace=True)
        return combined

    except Exception as exc:
        log.warning("[FFFactors] fetch failed: %s", exc)
        return None


def update_ff_cache(cache_path: Path = _DEFAULT_CACHE) -> bool:
    """
    Fetch FF factors and write to cache. Merges with existing rows.
    Returns True on success.
    """
    start = _DEFAULT_START
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if not existing.empty:
                existing.index = pd.to_datetime(existing.index)
                last_date = existing.index.max()
                start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        except Exception:
            pass

    df = fetch_ff_factors(start=start)
    if df is None or df.empty:
        log.warning("[FFFactors] No data fetched")
        return False

    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        except Exception:
            combined = df
    else:
        combined = df

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    log.info("[FFFactors] Cache updated — %d rows through %s",
             len(combined), str(combined.index.max())[:10])
    return True
```

- [ ] **Step 4: Run FF tests**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_fetch_ff_factors_returns_dataframe tests/data/test_new_ingest.py::test_update_ff_cache_writes_parquet -v 2>&1 | tail -10
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add ascent/data/ingest/famafrench_factors.py tests/data/test_new_ingest.py
git commit -m "feat: Fama-French 5-factor + momentum daily returns ingest"
```

---

## Task 8: `ascent/features/feature_defs.py` — Factor Loading Features

**Files:**
- Modify: `ascent/features/feature_defs.py`
- Modify: `tests/data/test_new_ingest.py`

- [ ] **Step 1: Append failing feature test**

```python
# ── Factor loadings feature test ───────────────────────────────────────────────

def test_factor_loadings_returns_per_symbol_betas():
    from ascent.features.feature_defs import factor_loadings
    import numpy as np

    dates = pd.date_range("2026-01-02", periods=80, freq="B")
    np.random.seed(42)
    # Simulate 3 symbols with known factor exposure
    factor_df = pd.DataFrame({
        "mkt_rf": np.random.normal(0.0005, 0.01, 80),
        "smb":    np.random.normal(0.0001, 0.005, 80),
        "mom":    np.random.normal(0.0002, 0.007, 80),
    }, index=dates)

    # CAT has positive mkt_rf beta ≈ 1.2
    returns_df = pd.DataFrame({
        "CAT": factor_df["mkt_rf"] * 1.2 + np.random.normal(0, 0.003, 80),
        "MRK": factor_df["mkt_rf"] * 0.8 + np.random.normal(0, 0.003, 80),
    }, index=dates)

    result = factor_loadings(returns_df, factor_df, window=63)

    # Result is a dict of {factor_name: DataFrame(dates x symbols)}
    assert isinstance(result, dict)
    assert "mkt_rf" in result
    beta_df = result["mkt_rf"]
    assert "CAT" in beta_df.columns
    assert "MRK" in beta_df.columns
    # CAT's mkt_rf beta should be higher than MRK's
    cat_beta = beta_df["CAT"].dropna().iloc[-1]
    mrk_beta = beta_df["MRK"].dropna().iloc[-1]
    assert cat_beta > mrk_beta


def test_factor_loadings_returns_empty_dict_when_no_overlap():
    from ascent.features.feature_defs import factor_loadings

    returns_df = pd.DataFrame({"CAT": [0.01, -0.02]},
                               index=pd.date_range("2026-01-02", periods=2, freq="B"))
    factor_df = pd.DataFrame({"mkt_rf": [0.005, -0.003]},
                              index=pd.date_range("2025-01-02", periods=2, freq="B"))

    result = factor_loadings(returns_df, factor_df, window=63)
    # No overlapping dates → empty DataFrames or empty dict
    for df in result.values():
        assert df.dropna().empty
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_factor_loadings_returns_per_symbol_betas -v 2>&1 | tail -5
```

Expected: `ImportError` — `factor_loadings` not yet defined

- [ ] **Step 3: Add `factor_loadings()` to `ascent/features/feature_defs.py`**

Find the end of `feature_defs.py` (after the last function). Append:

```python
# ── Fama-French Factor Loadings ───────────────────────────────────────────────

def factor_loadings(
    returns: pd.DataFrame,
    factor_df: pd.DataFrame,
    window: int = 63,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling factor betas for each symbol in returns.

    Beta_k(t) = rolling_cov(symbol_returns, factor_k, window) / rolling_var(factor_k, window)

    Args:
        returns:    dates × symbols daily return DataFrame
        factor_df: dates × factors daily factor return DataFrame
        window:    rolling window in trading days (default 63 = 3 months)

    Returns:
        {factor_name: DataFrame(dates × symbols)} — one beta per symbol per factor.
        Columns with no valid history are NaN. Gated: returns {} if factor_df is empty.
    """
    if factor_df is None or factor_df.empty:
        return {}

    # Align on common dates
    common = returns.index.intersection(factor_df.index)
    if len(common) < window // 2:
        return {col: pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
                for col in factor_df.columns}

    ret_aligned    = returns.loc[common]
    factor_aligned = factor_df.loc[common]

    result: dict[str, pd.DataFrame] = {}
    for factor_name in factor_aligned.columns:
        factor_series = factor_aligned[factor_name]
        factor_var = factor_series.rolling(window, min_periods=window // 2).var()

        betas: dict[str, pd.Series] = {}
        for sym in ret_aligned.columns:
            cov = ret_aligned[sym].rolling(window, min_periods=window // 2).cov(factor_series)
            betas[sym] = (cov / factor_var.replace(0, float("nan"))).reindex(returns.index)

        result[factor_name] = pd.DataFrame(betas, index=returns.index)

    return result
```

- [ ] **Step 4: Run feature tests**

```bash
.venv/bin/python -m pytest tests/data/test_new_ingest.py::test_factor_loadings_returns_per_symbol_betas tests/data/test_new_ingest.py::test_factor_loadings_returns_empty_dict_when_no_overlap -v 2>&1 | tail -8
```

Expected: 2 PASSED

- [ ] **Step 5: ast.parse verify**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/features/feature_defs.py').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add ascent/features/feature_defs.py tests/data/test_new_ingest.py
git commit -m "feat: factor_loadings() rolling beta feature for Fama-French factors"
```

---

## Task 9: Wire New Ingest into `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py`

- [ ] **Step 1: Find the hub run block**

```bash
grep -n "run_hub\|hub_is_fresh\|from ascent.data.hub" run_all_agents.py | head -10
```

Note the line number of the `run_hub(...)` call — you'll insert after it.

- [ ] **Step 2: Add CBOE / CFTC / FF ingest after the hub run**

Find the block that calls `run_hub(...)`. It looks like:

```python
    manifest = run_hub(start_date=..., end_date=...)
    if manifest["status"] != "ok":
        ...
```

After the `manifest` check block (after the `if manifest["status"] != "ok"` block closes), add:

```python
        # ── New ingest: CBOE options, CFTC COT, Fama-French factors ──────────
        try:
            from ascent.data.ingest.cboe_options import update_options_cache
            _universe_syms = list(collect_all_symbols())
            _n_opts = update_options_cache(_universe_syms, today.isoformat())
            if _n_opts:
                print(f"[Runner] CBOE options: {_n_opts} new rows added")
        except Exception as _opts_e:
            print(f"[Runner] CBOE options ingest skipped: {_opts_e}")

        try:
            from ascent.data.ingest.cftc_positioning import update_cot_cache
            _cot_added = update_cot_cache()
            if _cot_added:
                print("[Runner] CFTC COT: updated")
        except Exception as _cot_e:
            print(f"[Runner] CFTC COT ingest skipped: {_cot_e}")

        try:
            from ascent.data.ingest.famafrench_factors import update_ff_cache
            _ff_ok = update_ff_cache()
            if _ff_ok:
                print("[Runner] Fama-French factors: updated")
        except Exception as _ff_e:
            print(f"[Runner] Fama-French factors ingest skipped: {_ff_e}")
```

- [ ] **Step 3: Verify syntax**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Smoke test the import chain**

```bash
.venv/bin/python -c "
from ascent.data.ingest.cboe_options import update_options_cache
from ascent.data.ingest.cftc_positioning import update_cot_cache
from ascent.data.ingest.famafrench_factors import update_ff_cache
print('All ingest imports OK')
"
```

Expected: `All ingest imports OK`

- [ ] **Step 5: Commit**

```bash
git add run_all_agents.py
git commit -m "feat: wire CBOE/CFTC/FF ingest into daily hub run"
```

---

## Task 10: AI PM `get_live_options_flow` Tool

**Files:**
- Modify: `agents/ai_pm_agent.py`

- [ ] **Step 1: Add tool schema to `AI_PM_TOOLS`**

In `agents/ai_pm_agent.py`, find the `get_mirofish_sentiment` tool entry (around line 508). Insert the following **after** the `get_mirofish_sentiment` entry and **before** `propose_portfolio`:

```python
    {
        "name": "get_live_options_flow",
        "description": (
            "Fetch current options market signals for specific symbols: "
            "put/call ratio (PCR), IV skew direction, and IV rank vs 52-week history. "
            "Use on your top AMPLIFY candidates to check if the options market "
            "confirms or contradicts the thesis. "
            "High PCR (>1.2) = heavy put buying = crowd hedging against your thesis. "
            "Positive IV skew = call-bid = upside being priced in = thesis confirmation. "
            "Negative IV skew = put-bid = crowd protecting against decline = caution. "
            "IV rank > 80th pct = options expensive = expect-the-unexpected event risk priced in. "
            "Call for 1-4 symbols max. Phase 2 only — never pre-thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symbols to check (1-4 max)",
                },
            },
            "required": ["symbols"],
        },
    },
```

- [ ] **Step 2: Add tool executor function**

After `_tool_get_mirofish_sentiment` and before `_tool_propose_prethesis`, add:

```python
def _tool_get_live_options_flow(inputs: dict) -> str:
    """Fetch CBOE options flow signals for AMPLIFY candidates."""
    symbols = [str(s).upper().strip() for s in inputs.get("symbols", []) if s]
    if not symbols:
        return "Error: symbols list is required."
    try:
        from ascent.integrations.openbb_client import get_options_snapshot
        snapshot = get_options_snapshot(symbols)
        lines = ["OPTIONS FLOW SIGNALS", "=" * 44]
        for sym in symbols:
            entry = snapshot.get(sym, {})
            if entry.get("unavailable"):
                lines.append(f"{sym}: unavailable (CBOE fetch failed)")
                continue
            pcr   = entry.get("put_call_ratio")
            skew  = entry.get("iv_skew")
            iv    = entry.get("atm_iv")
            rank  = entry.get("iv_rank_52w")
            pcr_str  = f"PCR={pcr:.2f}" if pcr is not None else "PCR=n/a"
            skew_str = (f"IV_skew={skew:+.3f} ({'call-bid ↑' if skew and skew > 0 else 'put-bid ↓'})"
                        if skew is not None else "IV_skew=n/a")
            rank_str = f"IV_rank={rank}th pct" if rank is not None else "IV_rank=n/a (need 21d history)"
            # Interpretation
            if pcr is not None and pcr > 1.2:
                interp = "CAUTION: heavy put buying — options market hedging against position"
            elif skew is not None and skew > 0.02:
                interp = "CONFIRMS: call-bid skew — options market pricing in upside"
            elif rank is not None and rank > 80:
                interp = "NOTE: IV elevated — event risk is priced in, reduce surprise"
            else:
                interp = "NEUTRAL: no strong options signal"
            lines.append(f"\n{sym}: {pcr_str} | {skew_str} | {rank_str}")
            lines.append(f"  → {interp}")
        return "\n".join(lines)
    except Exception as exc:
        log.warning("[AIPMAgent] get_live_options_flow failed: %s", exc)
        return f"Options flow unavailable: {exc}. Proceed without options signal."
```

- [ ] **Step 3: Add executor to `_make_executor` dispatcher**

In `_make_executor`, find the `_map` dict. After the `"get_mirofish_sentiment"` entry, add:

```python
        "get_live_options_flow":    _tool_get_live_options_flow,
```

- [ ] **Step 4: Add system prompt guidance**

In `_SYSTEM_PROMPT`, find the MiroFish section (the `══ MIROFISH CROWD VALIDATION` block). Immediately before it, add:

```python
"══ OPTIONS FLOW (AMPLIFY picks only) ══\n"
"For your 1-2 AMPLIFY picks, optionally call get_live_options_flow. "
"Use it to check if the options market agrees with your thesis.\n"
"  PCR > 1.2     → heavy put buying → crowd hedging → reduce AMPLIFY weight 10-15%\n"
"  Positive skew → call-bid → confirmation → proceed at full AMPLIFY weight\n"
"  IV rank > 80  → event risk priced in → note in thesis, not a block\n"
"  unavailable   → skip signal, proceed normally\n"
"Call AFTER identifying AMPLIFY candidates, BEFORE propose_portfolio. Phase 2 only.\n\n"
```

- [ ] **Step 5: ast.parse verify**

```bash
.venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Run existing AI PM tests**

```bash
.venv/bin/python -m pytest tests/test_ai_pm_agent.py -v --tb=short 2>&1 | tail -15
```

Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add agents/ai_pm_agent.py
git commit -m "feat: get_live_options_flow tool in AI PM Phase 2 (CBOE PCR + IV skew)"
```

---

## Task 11: AI PM `get_cot_positioning` Tool

**Files:**
- Modify: `agents/ai_pm_agent.py`

- [ ] **Step 1: Add tool schema to `AI_PM_TOOLS`**

After the `get_live_options_flow` entry and before `propose_portfolio`, add:

```python
    {
        "name": "get_cot_positioning",
        "description": (
            "Fetch the latest CFTC Commitments of Traders report for S&P 500 e-mini futures. "
            "Returns speculator (non-commercial) net long positioning and 3-year percentile rank. "
            "Use once per Phase 2 to check if the broad equity market is macro-crowded. "
            "Extreme speculator long (>85th pct) = institutional macro crowding even if individual names are clean. "
            "This is a portfolio-level tail risk signal, not a name-selection signal. "
            "Does not take inputs. Phase 2 only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
```

- [ ] **Step 2: Add tool executor function**

After `_tool_get_live_options_flow`, add:

```python
def _tool_get_cot_positioning(_: dict) -> str:
    """Fetch latest CFTC COT positioning from cache or live."""
    try:
        from ascent.data.ingest.cftc_positioning import get_latest_cot
        from ascent.integrations.openbb_client import get_cot_snapshot
        import pandas as pd
        from pathlib import Path

        # Try cache first (populated by daily hub run)
        cot = get_latest_cot()

        # If cache is absent or stale (>7 days), try live fetch
        if cot is None:
            cot = get_cot_snapshot()

        if cot is None:
            return "COT data unavailable — CFTC fetch failed. Proceed without macro positioning context."

        net_long    = cot.get("net_noncommercial_long", 0)
        pct_long    = cot.get("pct_long_noncommercial")
        as_of       = cot.get("as_of_date", "unknown")
        oi          = cot.get("open_interest")

        # Compute 3-year percentile rank from cache history
        rank_str = "n/a (need more history)"
        try:
            cache = Path(__file__).resolve().parents[1] / "data_cache" / "cftc_positioning.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
                if len(df) >= 13:  # 13 weeks = 1 quarter min
                    history = df["net_noncommercial_long"].dropna().tail(156)  # ~3 years
                    rank = int((history < net_long).mean() * 100)
                    rank_str = f"{rank}th pct (vs {len(history)}w history)"
                    if rank > 85:
                        rank_label = "EXTENDED LONG — macro crowding risk"
                    elif rank < 20:
                        rank_label = "LIGHT — institutions underweight equities"
                    else:
                        rank_label = "neutral"
                    rank_str = f"{rank_str} → {rank_label}"
        except Exception:
            pass

        pct_str = f"{pct_long:.1f}% of open interest" if pct_long else "n/a"
        oi_str  = f"{oi:,}" if oi else "n/a"

        return (
            f"CFTC S&P 500 E-MINI POSITIONING (as of {as_of})\n"
            f"{'=' * 48}\n"
            f"Speculator net long:  {net_long:+,} contracts\n"
            f"Pct long (spec):      {pct_str}\n"
            f"Open interest:        {oi_str}\n"
            f"3-year rank:          {rank_str}\n"
            f"\n→ Use this as a portfolio-level tail risk check, not a name filter.\n"
            f"  >85th pct = apply macro caution to full portfolio sizing."
        )
    except Exception as exc:
        log.warning("[AIPMAgent] get_cot_positioning failed: %s", exc)
        return f"COT positioning unavailable: {exc}. Proceed without macro positioning context."
```

- [ ] **Step 3: Add executor to `_make_executor` dispatcher**

After `"get_live_options_flow"` entry in `_map`, add:

```python
        "get_cot_positioning":      _tool_get_cot_positioning,
```

- [ ] **Step 4: Add system prompt guidance**

In `_SYSTEM_PROMPT`, in the Phase 2 optional tools section (near `get_crowding_signal` guidance), add after the `get_crowding_signal` block:

```python
"• get_cot_positioning — call once to check if the broad equity market is macro-crowded.\n"
"  >85th pct speculator long = apply 10-15% gross exposure discount to full portfolio.\n"
"  Use alongside get_crowding_signal (per-name) for complete positioning picture.\n"
```

- [ ] **Step 5: ast.parse verify**

```bash
.venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add agents/ai_pm_agent.py
git commit -m "feat: get_cot_positioning tool in AI PM Phase 2 (CFTC macro crowding)"
```

---

## Task 12: Full Test Suite Pass

**Files:** none

- [ ] **Step 1: Run all new tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/integrations/test_openbb_client.py tests/data/test_new_ingest.py -v 2>&1 | tail -25
```

Expected: all tests pass (≥17 total)

- [ ] **Step 2: Run existing full test suite for regressions**

```bash
.venv/bin/python -m pytest tests/ -x -q \
  --ignore=tests/integrations/test_openbb_client.py \
  --ignore=tests/data/test_new_ingest.py \
  2>&1 | tail -15
```

Expected: same pass count as before (~777), 0 failures

- [ ] **Step 3: Final integration smoke test**

```bash
.venv/bin/python -c "
from ascent.integrations.openbb_client import fetch_symbol, get_live_macro, get_options_snapshot, get_cot_snapshot
from ascent.data.ingest.cboe_options import update_options_cache
from ascent.data.ingest.cftc_positioning import update_cot_cache, get_latest_cot
from ascent.data.ingest.famafrench_factors import update_ff_cache
from ascent.features.feature_defs import factor_loadings
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: OpenBB integration complete — hub reliability, CBOE/CFTC/FF alpha data, AI PM live tools

- ascent/integrations/openbb_client.py: central adapter (tiingo fallback, macro, options, COT)
- hub.py + ticker_memory.py: use adapter with yfinance fallback
- cboe_options.py: historical IV skew, PCR, ATM IV (fixes options_flow panel gap)
- cftc_positioning.py: S&P 500 e-mini COT speculator positioning
- famafrench_factors.py: 5-factor + momentum daily returns for ML sleeve
- feature_defs.py: factor_loadings() rolling beta computation
- ai_pm_agent.py: get_live_options_flow + get_cot_positioning tools (Phase 2 only)
- run_all_agents.py: CBOE/CFTC/FF ingest in daily hub run

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Central `openbb_client.py` adapter | Tasks 2–3 |
| `fetch_prices` with tiingo→yfinance fallback | Task 2 |
| `hub.py` uses adapter with yfinance fallback | Task 4 |
| `ticker_memory._fetch_return` uses adapter | Task 4 |
| `cboe_options.py` — historical IV skew, PCR, ATM IV, iv_rank_52w | Task 5 |
| IV skew = OTM call (≥1.03×spot) IV − OTM put (≤0.97×spot) IV | Task 3 `get_options_snapshot` |
| `cftc_positioning.py` — COT net speculator long, pct_long, weekly | Task 6 |
| `famafrench_factors.py` — 5-factor + momentum daily | Task 7 |
| `feature_defs.factor_loadings()` rolling beta | Task 8 |
| `run_all_agents.py` — CBOE/CFTC/FF called in hub run | Task 9 |
| `get_live_options_flow` AI PM tool — Phase 2 only | Task 10 |
| `get_cot_positioning` AI PM tool — Phase 2 only | Task 11 |
| `get_macro_data` upgrade | Excluded — `get_live_macro` is available in `openbb_client.py` and can be wired in by Task 10/11 implementer if needed; macro data is refreshed daily by hub so the cache is already fresh during the AI PM call |
| All integrity constraints | Gated imports throughout; no new cache names conflict; Phase 2 only rules maintained |

**Placeholder scan:** No TBDs. All steps contain code or exact commands.

**Type consistency:**
- `fetch_symbol(sym, start, end) -> pd.DataFrame | None` — used in Tasks 2 and 4 ✓
- `fetch_return(symbol, from_date, horizon_days) -> float | None` — used in Tasks 2 and 4 ✓
- `get_options_snapshot(symbols) -> dict[str, dict]` — used in Tasks 3, 5, and 10 ✓
- `get_cot_snapshot() -> dict | None` — used in Tasks 3 and 11 ✓
- `fetch_cboe_options_row(symbol, fetch_date) -> dict | None` — used in Task 5 ✓
- `update_options_cache(symbols, fetch_date, cache_path) -> int` — used in Tasks 5 and 9 ✓
- `update_cot_cache(cache_path) -> bool` — used in Tasks 6 and 9 ✓
- `get_latest_cot(cache_path) -> dict | None` — used in Tasks 6 and 11 ✓
- `factor_loadings(returns, factor_df, window) -> dict[str, pd.DataFrame]` — used in Tasks 8 ✓
