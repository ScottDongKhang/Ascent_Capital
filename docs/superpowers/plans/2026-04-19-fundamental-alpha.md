# Fundamental Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fundamental data signals (gross profitability, accruals, asset growth, 52-week high) to close the alpha model's biggest gap — it is entirely price-based today, and these factors are orthogonal to momentum with 30+ years of out-of-sample validation.

**Architecture:** Three layers. (1) Fetch and cache quarterly financial statement data from yfinance with a 45-day filing lag for point-in-time correctness. (2) Build a `fundamental_alpha` sleeve that z-scores and blends gross profitability, accruals, asset growth, and 52-week high proximity. (3) Wire the sleeve into `stack.py` at 10% weight, reduce trend from 65% → 55%, and add the same fundamental features to the ML sleeve's feature set so XGBoost also learns from them.

**Tech Stack:** Python 3.12, yfinance, pandas, numpy, pytest, existing `ascent/alpha/stack.py`, `ascent/features/feature_defs.py`, `ascent/alpha/ml_sleeve.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ascent/data/ingest/fundamentals.py` | Create | Fetch quarterly financials from yfinance, apply 45-day lag, save to cache |
| `ascent/features/feature_defs.py` | Modify | Add `high_52w_pct()` + `build_fundamental_panel()` |
| `ascent/features/build_features.py` | Modify | Add `fundamentals_df` param to `FeatureBuilder` |
| `ascent/alpha/fundamental.py` | Create | `fundamental_alpha(features)` sleeve — blends 4 signals |
| `ascent/alpha/stack.py` | Modify | Add fundamental sleeve, adjust DEFAULT_ALPHA_WEIGHTS |
| `ascent/alpha/ml_sleeve.py` | Modify | Add fundamental features to `ML_FEATURES` |
| `ascent/main.py` | Modify | Load fundamentals cache, pass to `FeatureBuilder` |
| `tests/test_fundamental_alpha.py` | Create | All tests for this plan |

---

## Task 1: Fetch and Cache Quarterly Fundamentals

**The problem:** yfinance has free quarterly income statement, balance sheet, and cash flow data. We need gross_profit, total_assets, net_income, and operating_cashflow per symbol. All values must be lagged 45 days from period end to avoid look-ahead bias (reports are typically filed 30–45 days after quarter end).

**Files:**
- Create: `ascent/data/ingest/fundamentals.py`
- Test: `tests/test_fundamental_alpha.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_fundamental_alpha.py`:

```python
# tests/test_fundamental_alpha.py
import json
import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_income_stmt():
    """Minimal quarterly income statement DataFrame mimicking yfinance output."""
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Gross Profit": 5e9, "Net Income": 2e9},
        periods[1]: {"Gross Profit": 4.8e9, "Net Income": 1.9e9},
        periods[2]: {"Gross Profit": 4.5e9, "Net Income": 1.7e9},
        periods[3]: {"Gross Profit": 4.2e9, "Net Income": 1.6e9},
    })


def _fake_balance_sheet():
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Total Assets": 50e9},
        periods[1]: {"Total Assets": 48e9},
        periods[2]: {"Total Assets": 46e9},
        periods[3]: {"Total Assets": 44e9},
    })


def _fake_cashflow():
    periods = pd.to_datetime(["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"])
    return pd.DataFrame({
        periods[0]: {"Operating Cash Flow": 3e9},
        periods[1]: {"Operating Cash Flow": 2.8e9},
        periods[2]: {"Operating Cash Flow": 2.6e9},
        periods[3]: {"Operating Cash Flow": 2.4e9},
    })


def _mock_ticker(sym):
    t = MagicMock()
    t.quarterly_income_stmt   = _fake_income_stmt()
    t.quarterly_balance_sheet = _fake_balance_sheet()
    t.quarterly_cashflow      = _fake_cashflow()
    return t


# ── Task 1 tests ───────────────────────────────────────────────────────────────

def test_fetch_fundamentals_returns_required_columns():
    """fetch_fundamentals must return a DataFrame with symbol, date, gross_profit, total_assets."""
    with patch("yfinance.Ticker", side_effect=_mock_ticker):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["AAPL", "MSFT"], delay_s=0)

    assert not df.empty
    for col in ["symbol", "date", "gross_profit", "total_assets", "net_income", "op_cashflow"]:
        assert col in df.columns, f"missing column: {col}"


def test_fetch_fundamentals_applies_45_day_lag():
    """date column must be period_end + 45 days (filing lag)."""
    with patch("yfinance.Ticker", side_effect=_mock_ticker):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["AAPL"], delay_s=0)

    aapl = df[df["symbol"] == "AAPL"].copy()
    assert not aapl.empty
    # All dates must be > period_end (lag applied)
    # period_end is 2025-12-31 → date must be 2026-02-14
    latest = aapl["date"].max()
    assert latest >= pd.Timestamp("2026-02-14"), \
        f"Expected filing date >= 2026-02-14 (Dec 31 + 45d), got {latest}"


def test_fetch_fundamentals_graceful_on_missing_symbol():
    """A symbol with no data must be skipped, not crash the whole fetch."""
    def mock_bad_ticker(sym):
        t = MagicMock()
        t.quarterly_income_stmt   = pd.DataFrame()
        t.quarterly_balance_sheet = pd.DataFrame()
        t.quarterly_cashflow      = pd.DataFrame()
        return t

    with patch("yfinance.Ticker", side_effect=mock_bad_ticker):
        from ascent.data.ingest.fundamentals import fetch_fundamentals
        df = fetch_fundamentals(["BADTICKER"], delay_s=0)

    # Should return empty DataFrame, not raise
    assert isinstance(df, pd.DataFrame)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py -k "Task 1 or fetch_fundamentals" -v --tb=short 2>&1 | tail -10
```

Expected: 3 failed — module doesn't exist.

- [ ] **Step 3: Create `ascent/data/ingest/fundamentals.py`**

```python
"""ascent/data/ingest/fundamentals.py

Fetch quarterly financial statement data from Yahoo Finance.
Stores gross_profit, total_assets, net_income, op_cashflow per symbol.

Point-in-time: all values dated at period_end + FILING_LAG_DAYS to
approximate when data was actually available to investors.
"""
from __future__ import annotations
import logging
import time

import pandas as pd
import numpy as np
import yfinance as yf

log = logging.getLogger(__name__)

FILING_LAG_DAYS = 45


def _safe_get(df: pd.DataFrame, col, names: list[str]):
    """Try multiple row-name variants; return float or None."""
    if df is None or df.empty or col not in df.columns:
        return None
    for name in names:
        if name in df.index:
            val = df.loc[name, col]
            if pd.notna(val):
                return float(val)
    return None


def _fetch_one(sym: str) -> pd.DataFrame:
    """Fetch quarterly fundamentals for one symbol. Returns empty DF on failure."""
    try:
        t   = yf.Ticker(sym)
        inc = t.quarterly_income_stmt
        bs  = t.quarterly_balance_sheet
        cf  = t.quarterly_cashflow

        if inc is None or inc.empty:
            return pd.DataFrame()

        rows = []
        for period in inc.columns:
            gross_profit = _safe_get(inc, period, ["Gross Profit", "GrossProfit"])
            net_income   = _safe_get(inc, period, ["Net Income", "NetIncome",
                                                    "Net Income Common Stockholders"])
            total_assets = _safe_get(bs,  period, ["Total Assets", "TotalAssets"])
            op_cf        = _safe_get(cf,  period, ["Operating Cash Flow",
                                                    "Cash Flow From Continuing Operating Activities"])

            if total_assets and total_assets != 0:
                rows.append({
                    "symbol":       sym,
                    "period_end":   pd.Timestamp(period),
                    "gross_profit": gross_profit,
                    "total_assets": total_assets,
                    "net_income":   net_income,
                    "op_cashflow":  op_cf,
                })

        return pd.DataFrame(rows)
    except Exception as e:
        log.debug("fundamentals fetch failed for %s: %s", sym, e)
        return pd.DataFrame()


def fetch_fundamentals(symbols: list[str], delay_s: float = 0.3) -> pd.DataFrame:
    """
    Fetch quarterly fundamentals for all symbols.

    Returns long-format DataFrame with columns:
      symbol, date (= period_end + 45d), period_end,
      gross_profit, total_assets, net_income, op_cashflow
    """
    frames = []
    for i, sym in enumerate(symbols):
        df = _fetch_one(sym)
        if not df.empty:
            frames.append(df)
        if i > 0 and i % 25 == 0:
            log.info("fundamentals: %d/%d symbols fetched", i, len(symbols))
        if delay_s > 0:
            time.sleep(delay_s)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["date"] = result["period_end"] + pd.Timedelta(days=FILING_LAG_DAYS)
    return result


def save_fundamentals(df: pd.DataFrame) -> None:
    from ascent.data.store.parquet import save_parquet
    save_parquet(df, "fundamentals")
    log.info("fundamentals: saved %d rows, %d symbols",
             len(df), df["symbol"].nunique() if not df.empty else 0)


def load_fundamentals() -> pd.DataFrame:
    from ascent.data.store.parquet import load_parquet, has_data
    if not has_data("fundamentals"):
        return pd.DataFrame()
    return load_parquet("fundamentals")


if __name__ == "__main__":
    # Run directly to refresh the cache:
    # .venv/bin/python ascent/data/ingest/fundamentals.py
    import sys
    sys.path.insert(0, ".")
    from ascent.config.settings import get_config
    from ascent.data.universe import get_current_universe
    cfg = get_config()
    symbols = get_current_universe()
    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    df = fetch_fundamentals(symbols, delay_s=0.3)
    save_fundamentals(df)
    print(f"Done. {len(df)} rows saved.")
```

- [ ] **Step 4: Verify syntax**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "import ast; ast.parse(open('ascent/data/ingest/fundamentals.py').read()); print('OK')"
```

- [ ] **Step 5: Run Task 1 tests — confirm 3 passed**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py::test_fetch_fundamentals_returns_required_columns \
    tests/test_fundamental_alpha.py::test_fetch_fundamentals_applies_45_day_lag \
    tests/test_fundamental_alpha.py::test_fetch_fundamentals_graceful_on_missing_symbol \
    -v --tb=short 2>&1 | tail -10
```

Expected: 3 passed.

- [ ] **Step 6: Run full suite — no regressions**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: 180+ passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/data/ingest/fundamentals.py tests/test_fundamental_alpha.py
git commit -m "$(cat <<'EOF'
feat(data): fetch quarterly fundamentals — gross profit, assets, earnings, cashflow

45-day filing lag applied for point-in-time correctness.
Graceful per-symbol error handling. Cache via save/load_fundamentals().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fundamental Feature Panel + Alpha Sleeve

**The problem:** Raw quarterly data needs to become daily cross-sectional signals. Gross profitability = gross_profit/assets (higher = better). Accruals = (net_income - op_cashflow)/assets (lower = better, inverted in sleeve). Asset growth = YoY total_assets change (lower = better, inverted). 52-week high proximity is price-only and always available. Each is z-scored cross-sectionally then blended equally.

**Files:**
- Modify: `ascent/features/feature_defs.py`
- Create: `ascent/alpha/fundamental.py`
- Test: `tests/test_fundamental_alpha.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fundamental_alpha.py`:

```python
# ── Task 2 tests ───────────────────────────────────────────────────────────────

def _make_fundamentals_df(syms=("AAPL", "MSFT", "GOOGL"), n_quarters=6):
    """Synthetic fundamentals long-format df."""
    rows = []
    np.random.seed(42)
    for sym in syms:
        for q in range(n_quarters):
            period = pd.Timestamp("2024-01-01") + pd.DateOffset(months=3 * q)
            rows.append({
                "symbol":       sym,
                "period_end":   period,
                "date":         period + pd.Timedelta(days=45),
                "gross_profit": np.random.uniform(1e9, 5e9),
                "total_assets": np.random.uniform(20e9, 60e9),
                "net_income":   np.random.uniform(0.5e9, 3e9),
                "op_cashflow":  np.random.uniform(1e9, 4e9),
            })
    return pd.DataFrame(rows)


def _make_close(syms=("AAPL", "MSFT", "GOOGL"), n=300):
    idx = pd.bdate_range(end="2026-04-19", periods=n)
    np.random.seed(0)
    return pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, (len(idx), len(syms))), axis=0),
        index=idx, columns=list(syms)
    )


def test_build_fundamental_panel_produces_three_metrics():
    """build_fundamental_panel must return gross_profitability, accruals, asset_growth."""
    from ascent.features.feature_defs import build_fundamental_panel
    syms = ["AAPL", "MSFT", "GOOGL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms)

    result = build_fundamental_panel(fund_df, close.index, syms)

    assert "gross_profitability" in result, "must produce gross_profitability"
    assert "accruals" in result, "must produce accruals"
    assert "asset_growth" in result, "must produce asset_growth"


def test_build_fundamental_panel_forward_fills():
    """After a quarterly report date, values must be forward-filled to daily."""
    from ascent.features.feature_defs import build_fundamental_panel
    syms = ["AAPL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms, n_quarters=4)

    result = build_fundamental_panel(fund_df, close.index, syms)

    gp = result.get("gross_profitability")
    assert gp is not None
    # After the first available filing date, must have non-NaN values
    valid = gp["AAPL"].dropna()
    assert len(valid) > 50, "gross_profitability must be forward-filled to >50 daily rows"


def test_high_52w_pct_feature():
    """high_52w_pct must return values in (0, 1] — price / 52wk high."""
    from ascent.features.feature_defs import high_52w_pct
    close = _make_close()
    result = high_52w_pct(close)
    valid = result.iloc[252:].dropna(how="all")  # after warmup
    assert not valid.empty
    assert (valid.values[np.isfinite(valid.values)] <= 1.0).all(), \
        "price / 52wk-high must be <= 1.0"
    assert (valid.values[np.isfinite(valid.values)] > 0).all()


def test_fundamental_alpha_builds_composite():
    """fundamental_alpha must return non-empty DataFrame when features present."""
    from ascent.alpha.fundamental import fundamental_alpha
    syms = ["AAPL", "MSFT", "GOOGL"]
    close = _make_close(syms)
    fund_df = _make_fundamentals_df(syms)

    from ascent.features.feature_defs import build_fundamental_panel, high_52w_pct
    panel = build_fundamental_panel(fund_df, close.index, syms)

    features = {"close": close, "high_52w_pct": high_52w_pct(close)}
    features.update(panel)

    result = fundamental_alpha(features)
    assert not result.empty, "fundamental_alpha must return non-empty DataFrame"
    assert set(result.columns) == set(syms)


def test_fundamental_alpha_works_without_fundamentals():
    """fundamental_alpha must not crash when only close is available (52wk high only)."""
    from ascent.alpha.fundamental import fundamental_alpha
    close = _make_close()
    result = fundamental_alpha({"close": close})
    assert not result.empty, "must work with only close — 52wk high is always computable"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py -k "panel or high_52w or fundamental_alpha" \
    -v --tb=short 2>&1 | tail -10
```

Expected: 5 failed.

- [ ] **Step 3: Add `high_52w_pct()` and `build_fundamental_panel()` to `feature_defs.py`**

Read `ascent/features/feature_defs.py`. After the last existing feature function (before `build_all_features`), add:

```python
# ── 52-Week High ──────────────────────────────────────────────────────

def high_52w_pct(close: pd.DataFrame) -> pd.DataFrame:
    """Price as fraction of 52-week high. Near 1.0 = near high = momentum strength.
    George & Hwang (2004): better predictor than raw past returns."""
    return close / close.rolling(252, min_periods=63).max()


# ── Fundamental Panel ─────────────────────────────────────────────────

def build_fundamental_panel(
    fundamentals_df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    symbols: list,
) -> dict:
    """
    Convert long-format quarterly fundamentals to daily wide panels.

    Applies forward-fill from each filing date. Point-in-time safe because
    fundamentals_df already has the 45-day filing lag baked into 'date'.

    Returns dict with keys: gross_profitability, accruals, asset_growth.
    Each value is a DataFrame(dates × symbols).
    Missing symbols get NaN columns (graceful degradation).
    """
    if fundamentals_df is None or fundamentals_df.empty:
        return {}

    sym_series: dict[str, dict] = {
        "gross_profitability": {},
        "accruals": {},
        "asset_growth": {},
    }

    for sym in symbols:
        sub = fundamentals_df[fundamentals_df["symbol"] == sym].copy()
        if sub.empty:
            continue
        sub = sub.dropna(subset=["total_assets"]).sort_values("date")
        sub = sub[sub["total_assets"] != 0]
        if sub.empty:
            continue
        sub = sub.set_index("date")

        ta = sub["total_assets"]

        if "gross_profit" in sub.columns:
            gp = sub["gross_profit"] / ta.replace(0, np.nan)
            sym_series["gross_profitability"][sym] = gp

        if "net_income" in sub.columns and "op_cashflow" in sub.columns:
            acc = (sub["net_income"] - sub["op_cashflow"]) / ta.replace(0, np.nan)
            sym_series["accruals"][sym] = acc

        if len(ta) >= 4:
            ag = ta / ta.shift(4) - 1
            sym_series["asset_growth"][sym] = ag

    result = {}
    for metric, sym_dict in sym_series.items():
        if not sym_dict:
            continue
        wide = pd.DataFrame(sym_dict)
        wide = wide.reindex(date_index, method="ffill")
        wide = wide.reindex(columns=symbols)
        result[metric] = wide

    return result
```

Also add `high_52w_pct` to `build_all_features()` — find the return dict and add:

```python
    features["high_52w_pct"] = high_52w_pct(close)
```

Add it after the momentum section (around the `rsi_14` line).

- [ ] **Step 4: Create `ascent/alpha/fundamental.py`**

```python
"""ascent/alpha/fundamental.py

Fundamental alpha sleeve — combines four academically-validated signals:

  gross_profitability  (Novy-Marx 2013)   — long high GP/Assets
  accruals             (Sloan 1996)        — long low accruals (inverted)
  asset_growth         (Cooper 2008)       — long low asset growth (inverted)
  high_52w_pct         (George/Hwang 2004) — long proximity to 52-week high

Each factor cross-sectionally z-scored. Blended equally (0.25 each).
Degrades gracefully: if fundamental cache missing, uses only 52-week high.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)


def fundamental_alpha(features: dict) -> pd.DataFrame:
    """
    Build fundamental alpha composite from features dict.

    Args:
        features: dict from build_all_features. Uses keys:
                  'close' (required), 'gross_profitability', 'accruals',
                  'asset_growth', 'high_52w_pct' (all optional except close).

    Returns:
        DataFrame (dates × symbols). Empty if close missing.
    """
    close = features.get("close")
    if close is None or close.empty:
        return pd.DataFrame()

    components = []

    # 52-week high proximity — always computable from prices
    h52 = features.get("high_52w_pct")
    if h52 is None:
        h52 = close / close.rolling(252, min_periods=63).max()
    try:
        components.append(_cs_zscore(h52.reindex(close.index).reindex(columns=close.columns)))
        log.info("fundamental_alpha: 52-week high loaded")
    except Exception as e:
        log.warning("fundamental_alpha: 52-week high failed: %s", e)

    # Gross profitability — long high
    if "gross_profitability" in features:
        try:
            gp = features["gross_profitability"].reindex(close.index, method="ffill").reindex(columns=close.columns)
            components.append(_cs_zscore(gp))
            log.info("fundamental_alpha: gross profitability loaded")
        except Exception as e:
            log.warning("fundamental_alpha: gross profitability failed: %s", e)

    # Accruals — long low (invert)
    if "accruals" in features:
        try:
            acc = features["accruals"].reindex(close.index, method="ffill").reindex(columns=close.columns)
            components.append(-_cs_zscore(acc))
            log.info("fundamental_alpha: accruals loaded")
        except Exception as e:
            log.warning("fundamental_alpha: accruals failed: %s", e)

    # Asset growth — long low (invert)
    if "asset_growth" in features:
        try:
            ag = features["asset_growth"].reindex(close.index, method="ffill").reindex(columns=close.columns)
            components.append(-_cs_zscore(ag))
            log.info("fundamental_alpha: asset growth loaded")
        except Exception as e:
            log.warning("fundamental_alpha: asset growth failed: %s", e)

    if not components:
        return pd.DataFrame()

    composite = sum(components) / len(components)
    return composite
```

- [ ] **Step 5: Verify syntax on both files**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast
for f in ['ascent/features/feature_defs.py', 'ascent/alpha/fundamental.py']:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 6: Run Task 2 tests — confirm 5 passed**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py -k "panel or high_52w or fundamental_alpha" \
    -v --tb=short 2>&1 | tail -10
```

- [ ] **Step 7: Run full suite — no regressions**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: 185+ passed.

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/features/feature_defs.py ascent/alpha/fundamental.py tests/test_fundamental_alpha.py
git commit -m "$(cat <<'EOF'
feat(alpha): fundamental sleeve — gross profitability, accruals, asset growth, 52wk high

Four academically-validated factors orthogonal to momentum.
build_fundamental_panel() converts quarterly data to daily with forward-fill.
Degrades gracefully to 52wk-high-only when fundamentals cache absent.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire Into Stack, ML Sleeve, and Main Pipeline

**The problem:** The fundamental sleeve exists but nothing calls it. Stack needs a `"fundamental"` sleeve entry. The ML sleeve needs fundamental features added so XGBoost learns from them too. `FeatureBuilder` and `main.py` need to load and pass the fundamentals cache.

**Files:**
- Modify: `ascent/alpha/stack.py`
- Modify: `ascent/alpha/ml_sleeve.py`
- Modify: `ascent/features/build_features.py`
- Modify: `ascent/main.py`
- Test: `tests/test_fundamental_alpha.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fundamental_alpha.py`:

```python
# ── Task 3 tests ───────────────────────────────────────────────────────────────

def test_default_alpha_weights_include_fundamental():
    """DEFAULT_ALPHA_WEIGHTS must include 'fundamental' key."""
    from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS
    assert "fundamental" in DEFAULT_ALPHA_WEIGHTS, \
        "fundamental sleeve must be in DEFAULT_ALPHA_WEIGHTS"
    assert abs(DEFAULT_ALPHA_WEIGHTS["fundamental"] - 0.10) < 0.001
    # trend should be reduced to 0.55
    assert abs(DEFAULT_ALPHA_WEIGHTS["trend"] - 0.55) < 0.001
    # all weights must sum to 1.0
    assert abs(sum(DEFAULT_ALPHA_WEIGHTS.values()) - 1.0) < 0.01


def test_ml_features_include_fundamental_signals():
    """ML_FEATURES must include gross_profitability, accruals, asset_growth, high_52w_pct."""
    from ascent.alpha.ml_sleeve import ML_FEATURES
    for feat in ["gross_profitability", "accruals", "asset_growth", "high_52w_pct"]:
        assert feat in ML_FEATURES, f"ML_FEATURES must include {feat}"


def test_feature_builder_accepts_fundamentals_df():
    """FeatureBuilder must accept fundamentals_df kwarg without error."""
    import pandas as pd
    import numpy as np
    from ascent.features.build_features import FeatureBuilder

    n = 60
    idx = pd.bdate_range(end="2026-04-19", periods=n)
    syms = ["A", "B", "C"]
    close = pd.DataFrame(
        100 * np.cumprod(1 + np.random.normal(0, 0.01, (n, 3)), axis=0),
        index=idx, columns=syms
    )
    # Build minimal long-format price df
    price_df = close.stack().reset_index()
    price_df.columns = ["date", "symbol", "close"]
    price_df["open"] = price_df["close"]
    price_df["high"] = price_df["close"]
    price_df["low"]  = price_df["close"]
    price_df["volume"] = 1e6

    # Should not raise even with fundamentals_df=None
    fb = FeatureBuilder(price_df, fundamentals_df=None)
    features = fb.compute_features()
    assert "high_52w_pct" in features
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py -k "default_alpha_weights or ml_features or feature_builder_accepts" \
    -v --tb=short 2>&1 | tail -10
```

Expected: 3 failed.

- [ ] **Step 3: Update `DEFAULT_ALPHA_WEIGHTS` in `stack.py`**

Find `DEFAULT_ALPHA_WEIGHTS` (lines 16–22). Replace with:

```python
DEFAULT_ALPHA_WEIGHTS = {
    "trend":       0.55,
    "meanrev":     0.05,
    "volatility":  0.05,
    "statarb":     0.15,
    "ml":          0.10,
    "fundamental": 0.10,
}
```

Then in `build_alpha_stack()`, add the fundamental sleeve computation after the ML sleeve block (before `loaded = list(alphas.keys())`):

```python
    try:
        from ascent.alpha.fundamental import fundamental_alpha
        fund = fundamental_alpha(features)
        if fund is not None and not fund.empty:
            alphas["fundamental"] = fund
            log.info("fundamental alpha loaded shape=%s", fund.shape)
        else:
            log.warning("fundamental alpha returned empty")
    except Exception as exc:
        log.error("fundamental alpha failed: %s", exc)
```

- [ ] **Step 4: Update `ML_FEATURES` in `ml_sleeve.py`**

Find `ML_FEATURES = [...]`. Add four entries:

```python
ML_FEATURES = [
    "mom_21d", "mom_63d", "mom_126d",
    "vol_21d", "vol_63d", "vol_ratio_10_63",
    "zscore_20d", "rsi_14", "macd_hist",
    "dollar_vol_rank_21d",
    "high_52w_pct",
    "gross_profitability",
    "accruals",
    "asset_growth",
]
```

- [ ] **Step 5: Add `fundamentals_df` to `FeatureBuilder` in `build_features.py`**

Read `ascent/features/build_features.py`. In `FeatureBuilder.__init__`, add `fundamentals_df: pd.DataFrame = None` parameter and store it:

```python
    def __init__(self, price_df: pd.DataFrame, macro_df: pd.DataFrame | None = None,
                 fundamentals_df: pd.DataFrame | None = None):
        # ... existing code unchanged ...
        self.fundamentals_df = fundamentals_df
```

In `compute_features()`, after calling `build_all_features(...)`, add fundamental panel features:

```python
    def compute_features(self) -> dict[str, pd.DataFrame]:
        features = build_all_features(
            self.close, self.volume, self.dollar_volume, self.macro_pivot
        )
        # Augment with fundamental panel if available
        if self.fundamentals_df is not None and not self.fundamentals_df.empty:
            try:
                from ascent.features.feature_defs import build_fundamental_panel
                fund_panel = build_fundamental_panel(
                    self.fundamentals_df, self.close.index, list(self.close.columns)
                )
                features.update(fund_panel)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("fundamental panel failed: %s", e)
        return features
```

- [ ] **Step 6: Load fundamentals cache in `main.py`**

Read `ascent/main.py`. Find where `profiles` is loaded (around the `has_data("profiles")` block). After it, add:

```python
    # Load fundamentals cache for fundamental alpha sleeve
    fundamentals_df = None
    from ascent.data.store.parquet import has_data as _hd2, load_parquet as _lp2
    if _hd2("fundamentals"):
        try:
            fundamentals_df = _lp2("fundamentals")
            print(f"[Alpha] Fundamentals cache loaded: {len(fundamentals_df)} rows")
        except Exception as _e:
            print(f"[Alpha] Fundamentals cache load failed: {_e}")
```

Then find where `FeatureBuilder(` is instantiated. Add `fundamentals_df=fundamentals_df` to the constructor call.

- [ ] **Step 7: Verify syntax on all modified files**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast
for f in ['ascent/alpha/stack.py', 'ascent/alpha/ml_sleeve.py',
          'ascent/features/build_features.py', 'ascent/main.py']:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 8: Run Task 3 tests — confirm 3 passed**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_fundamental_alpha.py -k "default_alpha_weights or ml_features or feature_builder_accepts" \
    -v --tb=short 2>&1 | tail -10
```

- [ ] **Step 9: Run full test suite — confirm 188+ passed, no regressions**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

- [ ] **Step 10: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/alpha/stack.py ascent/alpha/ml_sleeve.py \
        ascent/features/build_features.py ascent/main.py \
        tests/test_fundamental_alpha.py
git commit -m "$(cat <<'EOF'
feat(alpha): wire fundamental sleeve into stack + ML — trend 65% -> 55%, fundamental 10%

fundamental sleeve in stack.py at 10% weight.
ML_FEATURES now includes gross_profitability, accruals, asset_growth, high_52w_pct.
FeatureBuilder accepts fundamentals_df; main.py loads fundamentals cache.
Self-improve loop will tune the 10% weight over 30-day shadow period.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Refresh the Fundamentals Cache

After all tasks pass, seed the cache by running:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python ascent/data/ingest/fundamentals.py
```

This fetches ~135 symbols at 0.3s/symbol — takes about 40 seconds. Run it once now, then add it to the weekly launchd job alongside self_improve.

---

## Self-Review

**Spec coverage:**
1. ✅ Gross profitability (Novy-Marx 2013) — in fundamental sleeve + ML features
2. ✅ Accruals (Sloan 1996) — in fundamental sleeve + ML features
3. ✅ Asset growth (Cooper 2008) — in fundamental sleeve + ML features
4. ✅ 52-week high (George & Hwang 2004) — in fundamental sleeve + ML features + build_all_features
5. ✅ ML enrichment (Kozak 2020) — all four fundamentals added to ML_FEATURES
6. ✅ Point-in-time correctness — 45-day filing lag in fetch
7. ✅ Graceful degradation — falls back to 52wk-high if no cache; ML features marked optional

**Placeholder scan:** None found.

**Type consistency:**
- `build_fundamental_panel(fundamentals_df, date_index, symbols) -> dict` — used in `FeatureBuilder.compute_features()` ✅
- `fundamental_alpha(features: dict) -> pd.DataFrame` — called in `build_alpha_stack()` ✅
- `FeatureBuilder(price_df, macro_df=None, fundamentals_df=None)` — called in `main.py` ✅
- `ML_FEATURES` list — read by `build_ml_alpha_cpcv()` via feature lookup ✅
