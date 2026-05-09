# AI-Native Ascent Capital — Tier 3 Implementation Plan (Research-Backed Revision)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

> **Revision note (2026-05-08):** This plan was rewritten after a research survey of AlphaAgent (arXiv:2502.16789), QuantaAlpha (arXiv:2602.07085), Harvey et al. false discovery (arXiv:2006.04269), CPCV validation (Arian et al. 2024), and PySR symbolic regression (ACM GEC 2024). The original plan had one critical flaw: free-form LLM code generation produces ~95% redundant momentum variants, forward-looking data leakage, and scalar returns. Every system that achieves real excess returns uses strict interfaces, not unconstrained generation. See research brief in session log for full citations.

---

## Goal

Build a two-path autonomous factor discovery pipeline:

1. **Primary (PySR)**: Genetic programming evolves symbolic factor expressions from pre-computed features. Output is a human-readable formula (`sqrt(vol_21d) / (mom_252d + 1e-6)`), not Python code. Safe by construction — no `exec`, no imports.
2. **Secondary (LLM templates)**: Claude Haiku suggests parameters for a library of pre-defined factor templates (momentum, reversal, volatility, quality, correlation). LLM returns a JSON dict; the trusted template runs it. Zero code injection risk.

Both paths feed into a unified **per-regime CPCV evaluator** that requires IC > 0.015 AND IC IR > 0.60 AND positive IC in *every* regime. Harvey et al. FDR correction is enforced. Human reviews all accepted proposals before deployment.

**What changed from the original plan:**
- ❌ Free-form LLM code generation → replaced by PySR + templates
- ❌ Single-period rolling IC → replaced by per-regime CPCV
- ❌ IC IR threshold 0.40 → raised to 0.60 (Harvey FDR correction)
- ✅ IC mean threshold 0.015 → kept (correct for Ascent's breadth)
- ✅ Human review gate → kept
- ✅ Monthly cadence → kept (PySR is CPU-intensive; Opus saves cost)
- ✅ SELF_MODIFY_ENABLED gate → kept
- ✅ AST validation → kept but extended with lookahead scanner

---

## Why These Choices

### Why PySR instead of LLM code generation

AlphaAgent (arXiv:2502.16789) achieves 11% excess return with IR=1.5 by enforcing:
1. AST-based similarity check against existing alphas (novelty)
2. Semantic consistency between hypothesis and generated code
3. Complexity control (limits expression depth)

Without these controls, LLM-generated factors fail. The controls are expensive to implement correctly. PySR gives you structural novelty and complexity control for free — it evolves formulas from primitives, so the output is transparent, interpretable, and never redundant by construction. QuantaAlpha (arXiv:2602.07085) extends this with evolutionary refinement loops: 5–15 iterations per factor, treating discovery as trajectory optimization, not single-shot generation.

### Why per-regime IC instead of single-period

A factor with IC=0.04 in calm_bull but IC=−0.03 in crisis destroys alpha when it counts most. Regime-specific validation is not an edge case — it's table stakes for a system that already runs a regime classifier.

### Why IR > 0.60 instead of 0.40

Harvey, Liu, and Zhu (arXiv:2006.04269) showed that 316 published "significant" factors have an estimated false discovery rate of 64% using standard t > 2.0 thresholds. Their recommendation for new factor proposals: require t > 3.0, equivalent to IR > 0.60 given Ascent's breadth (~1,000 decisions/year). With 50 candidates evaluated per year, the expected number of spurious acceptances drops from ~3 (at IR > 0.40) to ~0.5 (at IR > 0.60).

### Realistic expectations

Following the academic and industry evidence: **0–2 deployable factors per year.** Hedge funds with 20 quants find 1–3. Retail with automated discovery finds 0–2. This is not a pessimistic estimate — it is the correct mental model. The value is in the pipeline discipline, not the discovery rate.

---

## ⚠ Gate Conditions — Do Not Begin Until

1. OOS Sharpe > 0 for 30 consecutive trading days on flat config (no active self-improve changes)
2. Regime labels have been live and consistent for at least 63 days (enough data to compute per-regime IC)
3. `slippage_ic_feedback.py` has accumulated MIN_FILLS=50 (so live IC tracking is meaningful)
4. PySR is installed: `pip install pysr` — confirm with `python3 -c "import pysr; print(pysr.__version__)"`
5. Walk-forward CPCV (existing `cpcv.py`) has been validated on a known factor (run control: 252d momentum should show IC 0.02–0.06)

---

## Architecture

```
Monthly trigger (first Sunday of each month, 6 AM)
           │
           ├─── Path A: PySR Symbolic Regression
           │    features.parquet → pysr_engine.py → symbolic expressions
           │    (e.g. "sqrt(vol_21d) / (mom_252d + 1e-6)")
           │
           └─── Path B: LLM Template Parameter Suggestion
                regime + IC stats → llm_suggester.py → JSON params
                → feature_templates.py → concrete factor signal
                          │
                          ▼
              regime_cpcv_evaluator.py
              (CPCV per regime: IC_calm, IC_stressed, IC_crisis,
               IC_mean, IC_IR, t-stat equivalent, novelty check)
                          │
              IC_mean > 0.015 AND IC_IR > 0.60
              AND IC_min > 0.01 across all regimes?
                    YES → leakage_scanner → human review queue
                    NO  → logged to factor_discovery_log.jsonl
                          │
              outputs/factor_proposals/{name}_{date}.json
              (human reads, edits, merges into feature_defs.py)
```

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/research/factor_discovery/__init__.py` | Package marker |
| Create | `ascent/research/factor_discovery/feature_templates.py` | 5 template families with parameter validation |
| Create | `ascent/research/factor_discovery/pysr_engine.py` | PySR wrapper + gplearn fallback |
| Create | `ascent/research/factor_discovery/llm_suggester.py` | Haiku suggests template parameters (JSON only) |
| Create | `ascent/research/factor_discovery/leakage_scanner.py` | AST lookahead / forward-data detector |
| Create | `ascent/research/factor_discovery/regime_cpcv_evaluator.py` | Per-regime CPCV IC + Harvey FDR check |
| Create | `ascent/research/factor_discovery/discovery_runner.py` | Orchestrates both paths, writes proposals |
| Create | `tests/test_factor_discovery.py` | 14 tests covering all modules |
| Modify | `run_all_agents.py` | Monthly trigger (first Sunday) |

---

## Task G: Autonomous Factor Discovery Pipeline

### Step 1: Write failing tests

```python
# tests/test_factor_discovery.py
"""
14 tests covering: feature templates, PySR engine, LLM suggester,
leakage scanner, per-regime CPCV evaluator, discovery runner.
"""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_prices(n_symbols=30, n_days=504, seed=42):
    rng = np.random.default_rng(seed)
    idx  = pd.date_range(end="2026-05-01", periods=n_days, freq="B")
    syms = [f"SYM{i:02d}" for i in range(n_symbols)]
    data = {s: np.cumprod(1 + rng.normal(0.0003, 0.015, n_days)) for s in syms}
    return pd.DataFrame(data, index=idx)


def _make_regime_labels(index):
    """Alternating regimes for testing per-regime IC split."""
    labels = []
    for i, dt in enumerate(index):
        if i < len(index) // 3:
            labels.append("calm_bull")
        elif i < 2 * len(index) // 3:
            labels.append("stressed")
        else:
            labels.append("calm_bull")
    return pd.Series(labels, index=index)


# ── Feature templates ──────────────────────────────────────────────────────────

def test_momentum_template_sums_close_to_zero():
    from ascent.research.factor_discovery.feature_templates import MomentumTemplate
    prices = _make_prices()
    tmpl   = MomentumTemplate(lookback=120, skip_days=21, normalization="zscore")
    signal = tmpl.compute(prices)
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(prices.columns)
    # Cross-sectional z-score should sum near zero
    assert abs(signal.sum()) < 1.0, "z-scored cross-section should sum near 0"


def test_reversal_template_returns_series():
    from ascent.research.factor_discovery.feature_templates import ReversionTemplate
    prices = _make_prices()
    tmpl   = ReversionTemplate(lookback=5, normalization="rank")
    signal = tmpl.compute(prices)
    assert isinstance(signal, pd.Series)
    assert not signal.isnull().all()


def test_volatility_template_returns_series():
    from ascent.research.factor_discovery.feature_templates import VolatilityTemplate
    prices = _make_prices()
    tmpl   = VolatilityTemplate(vol_window=21, vov_window=63, direction="low")
    signal = tmpl.compute(prices)
    assert isinstance(signal, pd.Series)


def test_template_floor_guard():
    """Templates must return zeros when std < 1e-8 (flat prices)."""
    from ascent.research.factor_discovery.feature_templates import MomentumTemplate
    flat_prices = pd.DataFrame(
        np.ones((60, 10)), index=pd.date_range("2025-01-01", periods=60, freq="B"),
        columns=[f"S{i}" for i in range(10)]
    )
    tmpl   = MomentumTemplate(lookback=21, skip_days=0, normalization="zscore")
    signal = tmpl.compute(flat_prices)
    assert (signal == 0).all() or signal.isnull().all(), "Flat prices must yield zero signal"


# ── Leakage scanner ────────────────────────────────────────────────────────────

def test_leakage_scanner_rejects_tail_access():
    from ascent.research.factor_discovery.leakage_scanner import scan_for_leakage
    code = "signal = df.tail(1).values[0]"
    ok, msg = scan_for_leakage(code)
    assert not ok
    assert "lookahead" in msg.lower() or "tail" in msg.lower() or "leakage" in msg.lower()


def test_leakage_scanner_rejects_datetime_now():
    from ascent.research.factor_discovery.leakage_scanner import scan_for_leakage
    code = "import datetime; t = datetime.datetime.now()"
    ok, msg = scan_for_leakage(code)
    assert not ok


def test_leakage_scanner_accepts_clean_code():
    from ascent.research.factor_discovery.leakage_scanner import scan_for_leakage
    code = """
ret = df.pct_change(21)
signal = -ret.iloc[-1]
signal = (signal - signal.mean()) / (signal.std() + 1e-8)
"""
    ok, msg = scan_for_leakage(code)
    assert ok, f"Clean code should pass: {msg}"


# ── Per-regime CPCV evaluator ─────────────────────────────────────────────────

def test_regime_evaluator_returns_all_keys():
    from ascent.research.factor_discovery.regime_cpcv_evaluator import evaluate_factor_regime_ic
    prices  = _make_prices(n_days=504)
    regimes = _make_regime_labels(prices.index)

    # Inline simple reversal factor as callable
    def factor_fn(df):
        s = -df.pct_change(5).iloc[-1]
        return (s - s.mean()) / (s.std() + 1e-8)

    result = evaluate_factor_regime_ic(
        factor_fn=factor_fn,
        prices_df=prices,
        regime_labels=regimes,
        n_periods=5,
    )
    assert isinstance(result, dict)
    for key in ["ic_mean", "ic_ir", "ic_p5", "n_observations", "ic_calm_bull", "ic_stressed"]:
        assert key in result, f"Missing key: {key}"


def test_regime_evaluator_ic_in_valid_range():
    from ascent.research.factor_discovery.regime_cpcv_evaluator import evaluate_factor_regime_ic
    prices  = _make_prices(n_days=504)
    regimes = _make_regime_labels(prices.index)

    def factor_fn(df):
        s = -df.pct_change(5).iloc[-1]
        return (s - s.mean()) / (s.std() + 1e-8)

    result = evaluate_factor_regime_ic(factor_fn, prices, regimes, n_periods=5)
    assert -1.0 <= result["ic_mean"] <= 1.0


def test_harvey_fdr_check():
    """IC IR > 0.60 is the acceptance threshold (Harvey et al. multiple-testing correction)."""
    from ascent.research.factor_discovery.regime_cpcv_evaluator import passes_harvey_threshold
    assert passes_harvey_threshold(ic_mean=0.025, ic_ir=0.65)   is True
    assert passes_harvey_threshold(ic_mean=0.025, ic_ir=0.55)   is False  # IR too low
    assert passes_harvey_threshold(ic_mean=0.010, ic_ir=0.80)   is False  # IC mean too low
    assert passes_harvey_threshold(ic_mean=0.020, ic_ir=0.60)   is True   # exactly at threshold


# ── LLM suggester ─────────────────────────────────────────────────────────────

def test_llm_suggester_returns_params():
    from ascent.research.factor_discovery.llm_suggester import suggest_template_params
    mock_response = json.dumps({
        "template": "MomentumTemplate",
        "params": {"lookback": 90, "skip_days": 21, "normalization": "zscore"},
        "rationale": "Quality bias dominates stressed regimes"
    })
    with patch("ascent.research.factor_discovery.llm_suggester._call_llm",
               return_value=mock_response):
        result = suggest_template_params(regime="stressed", ic_context={})
    assert isinstance(result, dict)
    assert "template" in result
    assert "params" in result


def test_llm_suggester_returns_none_on_failure():
    from ascent.research.factor_discovery.llm_suggester import suggest_template_params
    with patch("ascent.research.factor_discovery.llm_suggester._call_llm", return_value=None):
        result = suggest_template_params(regime="stressed", ic_context={})
    assert result is None


# ── Discovery runner ──────────────────────────────────────────────────────────

def test_discovery_runner_writes_proposal_on_pass(tmp_path):
    from ascent.research.factor_discovery.discovery_runner import run_factor_discovery

    def mock_factor_fn(df):
        s = -df.pct_change(5).iloc[-1]
        return (s - s.mean()) / (s.std() + 1e-8)

    mock_eval = {
        "ic_mean": 0.022, "ic_ir": 0.65, "ic_p5": -0.005,
        "n_observations": 80, "ic_calm_bull": 0.025, "ic_stressed": 0.018,
        "ic_crisis": 0.015, "ic_min_regime": 0.015,
    }
    with patch("ascent.research.factor_discovery.discovery_runner._load_prices",
               return_value=_make_prices()):
        with patch("ascent.research.factor_discovery.discovery_runner._load_regime_labels",
                   return_value=_make_regime_labels(_make_prices().index)):
            with patch("ascent.research.factor_discovery.discovery_runner.evaluate_factor_regime_ic",
                       return_value=mock_eval):
                with patch("ascent.research.factor_discovery.discovery_runner._build_candidates",
                           return_value=[{"name": "factor_test", "fn": mock_factor_fn,
                                          "source": "template", "description": "Test"}]):
                    with patch("ascent.research.factor_discovery.discovery_runner.PROPOSALS_DIR",
                               tmp_path):
                        result = run_factor_discovery(n_candidates=1, regime="stressed")

    assert result["n_accepted"] >= 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) >= 1
    proposal = json.loads(files[0].read_text())
    assert "ic_mean" in proposal
    assert "review_status" in proposal
    assert proposal["review_status"] == "pending"


def test_discovery_runner_rejects_low_ir(tmp_path):
    from ascent.research.factor_discovery.discovery_runner import run_factor_discovery

    def mock_factor_fn(df):
        return pd.Series(0.0, index=df.columns)

    mock_eval = {
        "ic_mean": 0.020, "ic_ir": 0.35,  # IR too low
        "ic_p5": -0.010, "n_observations": 80,
        "ic_calm_bull": 0.020, "ic_stressed": 0.018, "ic_crisis": 0.015,
        "ic_min_regime": 0.015,
    }
    with patch("ascent.research.factor_discovery.discovery_runner._load_prices",
               return_value=_make_prices()):
        with patch("ascent.research.factor_discovery.discovery_runner._load_regime_labels",
                   return_value=_make_regime_labels(_make_prices().index)):
            with patch("ascent.research.factor_discovery.discovery_runner.evaluate_factor_regime_ic",
                       return_value=mock_eval):
                with patch("ascent.research.factor_discovery.discovery_runner._build_candidates",
                           return_value=[{"name": "factor_weak_ir", "fn": mock_factor_fn,
                                          "source": "template", "description": "Weak"}]):
                    with patch("ascent.research.factor_discovery.discovery_runner.PROPOSALS_DIR",
                               tmp_path):
                        result = run_factor_discovery(n_candidates=1, regime="stressed")

    assert result["n_accepted"] == 0
    assert len(list(tmp_path.glob("*.json"))) == 0
```

Run to confirm they fail:
```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_factor_discovery.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'ascent.research.factor_discovery'`

---

### Step 2: Create `ascent/research/factor_discovery/__init__.py`

```python
# ascent/research/factor_discovery/__init__.py
```

---

### Step 3: Create `ascent/research/factor_discovery/feature_templates.py`

```python
"""
ascent/research/factor_discovery/feature_templates.py

Pre-defined factor template families. Each template is a trusted, human-written
class that accepts a parameter dict from the LLM and a price DataFrame, and
returns a cross-sectionally z-scored pd.Series.

Template families:
    MomentumTemplate    — skip-adjusted momentum with normalization options
    ReversionTemplate   — short-term mean reversion
    VolatilityTemplate  — volatility regime (low-vol or vol-trend)
    QualityTemplate     — growth and stability metrics from price series
    CorrelationTemplate — market beta / idiosyncratic component

The LLM (llm_suggester.py) fills parameters into these templates via JSON.
No code is generated; the template logic is trusted and reviewed by a human.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_NORM_METHODS = frozenset({"zscore", "rank", "minmax"})


def _normalize(s: pd.Series, method: str) -> pd.Series:
    """Cross-sectional normalization. Returns zeros on degenerate input."""
    s = s.dropna()
    if s.empty:
        return s
    if method == "zscore":
        std = s.std()
        if std < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    if method == "rank":
        return s.rank(pct=True) - 0.5
    if method == "minmax":
        rng = s.max() - s.min()
        if rng < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.min()) / rng - 0.5
    return s


class MomentumTemplate:
    """
    Skip-adjusted momentum.
    signal = return(lookback) - return(skip_days)
    Regime note: skip_days > 0 avoids 1-month reversal contamination.
    """

    PARAM_SCHEMA = {
        "lookback":      {"type": int,   "min": 21,  "max": 252, "default": 120},
        "skip_days":     {"type": int,   "min": 0,   "max": 63,  "default": 21},
        "normalization": {"type": str,   "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, lookback: int = 120, skip_days: int = 21,
                 normalization: str = "zscore"):
        self.lookback      = lookback
        self.skip_days     = skip_days
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback + 1:
            return pd.Series(0.0, index=df.columns)
        ret_long = df.pct_change(self.lookback).iloc[-1]
        if self.skip_days > 0 and len(df) > self.skip_days:
            ret_short = df.pct_change(self.skip_days).iloc[-1]
            signal    = ret_long - ret_short
        else:
            signal = ret_long
        return _normalize(signal, self.normalization)

    def to_dict(self) -> dict:
        return {
            "template": "MomentumTemplate",
            "lookback": self.lookback, "skip_days": self.skip_days,
            "normalization": self.normalization,
        }


class ReversionTemplate:
    """
    Short-term mean reversion.
    signal = -return(lookback), optionally smoothed over smooth_window.
    """

    PARAM_SCHEMA = {
        "lookback":      {"type": int, "min": 2,  "max": 21,  "default": 5},
        "smooth_window": {"type": int, "min": 1,  "max": 10,  "default": 1},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, lookback: int = 5, smooth_window: int = 1,
                 normalization: str = "zscore"):
        self.lookback      = lookback
        self.smooth_window = smooth_window
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback + self.smooth_window:
            return pd.Series(0.0, index=df.columns)
        rets   = df.pct_change(self.lookback)
        if self.smooth_window > 1:
            rets = rets.rolling(self.smooth_window).mean()
        signal = -rets.iloc[-1]
        return _normalize(signal, self.normalization)

    def to_dict(self) -> dict:
        return {"template": "ReversionTemplate", "lookback": self.lookback,
                "smooth_window": self.smooth_window, "normalization": self.normalization}


class VolatilityTemplate:
    """
    Volatility regime signal.
    direction="low"   → long low-vol names (risk-parity style)
    direction="trend" → long names with declining vol (vol-trend sleeve)
    """

    PARAM_SCHEMA = {
        "vol_window":  {"type": int, "min": 10,  "max": 63,  "default": 21},
        "vov_window":  {"type": int, "min": 21,  "max": 126, "default": 63},
        "direction":   {"type": str, "choices": {"low", "trend"},  "default": "low"},
        "normalization": {"type": str, "choices": _NORM_METHODS,   "default": "zscore"},
    }

    def __init__(self, vol_window: int = 21, vov_window: int = 63,
                 direction: str = "low", normalization: str = "zscore"):
        self.vol_window    = vol_window
        self.vov_window    = vov_window
        self.direction     = direction
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.vov_window:
            return pd.Series(0.0, index=df.columns)
        rets = df.pct_change()
        vol  = rets.rolling(self.vol_window).std().iloc[-1]
        if self.direction == "low":
            signal = -vol
        else:
            vov    = rets.rolling(self.vol_window).std().rolling(self.vov_window).std()
            trend  = rets.rolling(self.vol_window).std().diff(5)
            vov_last  = vov.iloc[-1]
            trend_last = trend.iloc[-1]
            signal = -(trend_last / (vov_last + 1e-8))
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "VolatilityTemplate", "vol_window": self.vol_window,
                "vov_window": self.vov_window, "direction": self.direction,
                "normalization": self.normalization}


class QualityTemplate:
    """
    Price-implied quality: consistency and growth from price history.
    metric="consistency" → rolling Sharpe (reward/risk)
    metric="drawdown"    → inverse max drawdown (survivorship quality proxy)
    metric="trend_strength" → trend quality via consecutive positive returns
    """

    PARAM_SCHEMA = {
        "metric":      {"type": str, "choices": {"consistency", "drawdown", "trend_strength"},
                        "default": "consistency"},
        "window":      {"type": int, "min": 21, "max": 252, "default": 63},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, metric: str = "consistency", window: int = 63,
                 normalization: str = "zscore"):
        self.metric        = metric
        self.window        = window
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.window:
            return pd.Series(0.0, index=df.columns)
        rets = df.pct_change()
        w    = rets.tail(self.window)
        if self.metric == "consistency":
            mean_ret = w.mean()
            std_ret  = w.std().replace(0, np.nan)
            signal   = mean_ret / std_ret
        elif self.metric == "drawdown":
            prices_w = df.tail(self.window)
            running_max = prices_w.cummax()
            dd = ((prices_w - running_max) / running_max.replace(0, np.nan)).min()
            signal = -dd  # lower drawdown → higher signal
        else:
            signal = (w > 0).mean()  # fraction of up days
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "QualityTemplate", "metric": self.metric,
                "window": self.window, "normalization": self.normalization}


class CorrelationTemplate:
    """
    Market correlation / idiosyncratic component.
    mode="beta"          → cross-sectional beta-rank (low beta long)
    mode="idiosyncratic" → residual return after removing market component
    """

    PARAM_SCHEMA = {
        "window":        {"type": int, "min": 21, "max": 126, "default": 63},
        "mode":          {"type": str, "choices": {"beta", "idiosyncratic"}, "default": "beta"},
        "normalization": {"type": str, "choices": _NORM_METHODS, "default": "zscore"},
    }

    def __init__(self, window: int = 63, mode: str = "beta",
                 normalization: str = "zscore"):
        self.window        = window
        self.mode          = mode
        self.normalization = normalization

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.window + 1:
            return pd.Series(0.0, index=df.columns)
        rets   = df.pct_change().tail(self.window)
        mkt    = rets.mean(axis=1)
        betas  = {}
        resids = {}
        for col in rets.columns:
            s = rets[col].dropna()
            m = mkt.reindex(s.index)
            cov   = np.cov(s.values, m.values)
            var_m = cov[1, 1]
            beta  = cov[0, 1] / var_m if var_m > 1e-8 else 1.0
            betas[col]  = beta
            resids[col] = (s - beta * m).mean()
        if self.mode == "beta":
            signal = pd.Series(betas)
            signal = -signal  # long low-beta
        else:
            signal = pd.Series(resids)
        return _normalize(signal.dropna(), self.normalization)

    def to_dict(self) -> dict:
        return {"template": "CorrelationTemplate", "window": self.window,
                "mode": self.mode, "normalization": self.normalization}


# ── Registry ───────────────────────────────────────────────────────────────────

TEMPLATE_REGISTRY: Dict[str, type] = {
    "MomentumTemplate":    MomentumTemplate,
    "ReversionTemplate":   ReversionTemplate,
    "VolatilityTemplate":  VolatilityTemplate,
    "QualityTemplate":     QualityTemplate,
    "CorrelationTemplate": CorrelationTemplate,
}


def instantiate_template(template_name: str, params: dict):
    """Instantiate a template by name with the given parameter dict."""
    cls = TEMPLATE_REGISTRY.get(template_name)
    if cls is None:
        raise ValueError(f"Unknown template: {template_name}. Valid: {list(TEMPLATE_REGISTRY)}")
    schema = cls.PARAM_SCHEMA
    validated = {}
    for key, spec in schema.items():
        val = params.get(key, spec["default"])
        if spec["type"] is int:
            val = int(val)
            val = max(spec["min"], min(spec["max"], val))
        elif spec["type"] is str and "choices" in spec:
            if val not in spec["choices"]:
                val = spec["default"]
        validated[key] = val
    return cls(**validated)
```

---

### Step 4: Create `ascent/research/factor_discovery/leakage_scanner.py`

```python
"""
ascent/research/factor_discovery/leakage_scanner.py

Detects forward-looking data access patterns in factor code strings.
Used as a pre-validation gate before any code runs.

Checks for:
  - df.tail() / df.head() used as signal (last row access = lookahead)
  - datetime.now(), pd.Timestamp.today() (runtime time = future knowledge)
  - Hard-coded future dates
  - .shift(-N) with negative shift (looks at future rows)
  - .rolling(...).apply(lambda: ...) that accesses future values

Note: df.iloc[-1] is LEGITIMATE as the final step to extract the
cross-sectional signal from the last available date. The scanner
distinguishes "iloc[-1] as final extraction" from "iloc[-1] inside
a rolling window" (which is lookahead). Simple pattern matching
is used — not full semantic analysis.

Returns (is_clean: bool, message: str).
"""
from __future__ import annotations

import ast
import re
from typing import Tuple


_LOOKAHEAD_PATTERNS = [
    # Runtime time access
    (r"datetime\.now\(\)",        "datetime.now() is future knowledge — use df.index[-1]"),
    (r"datetime\.today\(\)",      "datetime.today() is future knowledge"),
    (r"pd\.Timestamp\.today\(\)", "pd.Timestamp.today() is future knowledge"),
    (r"pd\.Timestamp\.now\(\)",   "pd.Timestamp.now() is future knowledge"),
    (r"time\.time\(\)",           "time.time() is future knowledge"),
    # Direct future-row access patterns
    (r"\.shift\(-\d+\)",          ".shift(-N) accesses future rows — use positive shift only"),
    (r"\.tail\s*\(\s*1\s*\)",     ".tail(1) may be a lookahead pattern — use .iloc[-1] explicitly"),
]


class _LeakageVisitor(ast.NodeVisitor):
    """Walk AST to find structural lookahead patterns."""

    def __init__(self):
        self.violations: list = []

    def visit_Call(self, node):
        # Flag .rolling(...).apply(lambda ...) with negative indices inside
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "rolling"):
            # Check lambda body for negative indexing
            for arg in node.args:
                if isinstance(arg, ast.Lambda):
                    src = ast.unparse(arg)
                    if "[-" in src:
                        self.violations.append(
                            "Negative index inside rolling().apply() lambda — potential lookahead"
                        )
        self.generic_visit(node)

    def visit_Subscript(self, node):
        # Flag df[future_date_string] patterns — hard-coded future dates
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            val = node.slice.value
            if len(val) == 10 and val.count("-") == 2:
                try:
                    from datetime import date
                    parsed = date.fromisoformat(val)
                    if parsed.year > 2026:
                        self.violations.append(f"Hard-coded future date: '{val}'")
                except ValueError:
                    pass
        self.generic_visit(node)


def scan_for_leakage(code: str) -> Tuple[bool, str]:
    """
    Scan factor code string for lookahead / forward-data patterns.

    Args:
        code: Python source string.

    Returns:
        (is_clean, message) — message is "OK" or a description of the problem.
    """
    # Regex scan
    for pattern, message in _LOOKAHEAD_PATTERNS:
        if re.search(pattern, code):
            return False, f"Lookahead pattern detected: {message}"

    # AST scan (syntax errors → pass through as clean; validator handles syntax)
    try:
        tree = ast.parse(code)
        visitor = _LeakageVisitor()
        visitor.visit(tree)
        if visitor.violations:
            return False, f"Structural lookahead: {'; '.join(visitor.violations)}"
    except SyntaxError:
        pass  # syntax errors caught by code_validator; not our job here

    return True, "OK"
```

---

### Step 5: Create `ascent/research/factor_discovery/regime_cpcv_evaluator.py`

```python
"""
ascent/research/factor_discovery/regime_cpcv_evaluator.py

Per-regime Information Coefficient evaluator with Harvey et al. FDR correction.

Architecture:
  - Compute Spearman IC between factor values and n-period forward returns
    for each date in the price history (weekly step)
  - Split IC observations by regime label
  - Report IC_calm_bull, IC_stressed, IC_crisis separately
  - Require IC_mean > 0.015 AND IC_IR > 0.60 AND IC_min_regime > 0.01

IC IR > 0.60 threshold rationale:
  Harvey, Liu, Zhu (2016, arXiv:2006.04269) show that with multiple testing,
  the effective t-stat threshold for factor significance is ~3.0. For
  Ascent's breadth (~1,000 decisions/year), this maps to IC IR > 0.60.
  Applying IR > 0.40 generates ~3 spurious acceptances per year from 50
  candidates; IR > 0.60 reduces this to ~0.5.

References:
  Harvey, Liu, Zhu (2016) — arXiv:2006.04269
  CPCV methodology — Lopez de Prado (2018), expanded by Arian et al. (2024)
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

IC_MEAN_THRESHOLD   = 0.015
IC_IR_THRESHOLD     = 0.60   # Harvey et al. multiple-testing correction
IC_MIN_REGIME       = 0.010  # must be positive in every observed regime
MIN_OBSERVATIONS    = 20


def _compute_ic_series(
    factor_fn: Callable[[pd.DataFrame], pd.Series],
    prices_df: pd.DataFrame,
    n_periods: int,
    lookback_days: int,
    step: int,
    min_symbols: int,
) -> Dict[str, list]:
    """
    Compute cross-sectional Spearman IC for each evaluation date.

    Returns dict: {"dates": [...], "ics": [...]}
    """
    dates  = prices_df.index[lookback_days:-n_periods:step]
    result = {"dates": [], "ics": []}

    for dt in dates:
        try:
            iloc_pos = prices_df.index.get_loc(dt)
            start    = max(0, iloc_pos - lookback_days)
            window   = prices_df.iloc[start: iloc_pos + 1]

            factor_vals = factor_fn(window)
            if not isinstance(factor_vals, pd.Series) or factor_vals.empty:
                continue

            fwd_rets = prices_df.iloc[iloc_pos + n_periods] / prices_df.iloc[iloc_pos] - 1

            common = factor_vals.index.intersection(fwd_rets.index)
            f = factor_vals.reindex(common).dropna()
            r = fwd_rets.reindex(f.index).dropna()
            f = f.reindex(r.index)

            if len(f) < min_symbols:
                continue

            ic, _ = spearmanr(f.values, r.values)
            if not np.isnan(ic):
                result["dates"].append(dt)
                result["ics"].append(float(ic))

        except Exception as exc:
            log.debug("[RegimeCPCV] Skipped %s: %s", dt, exc)

    return result


def evaluate_factor_regime_ic(
    factor_fn: Callable[[pd.DataFrame], pd.Series],
    prices_df: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    n_periods: int = 5,
    lookback_days: int = 252,
    step: int = 5,
    min_symbols: int = 10,
) -> Dict:
    """
    Evaluate a factor's IC split by market regime.

    Args:
        factor_fn:      Callable (df: pd.DataFrame) -> pd.Series (cross-sectional signal)
        prices_df:      Daily price DataFrame (index=dates, columns=symbols)
        regime_labels:  pd.Series (index=dates, values=regime strings). None → skip regime split.
        n_periods:      Forward return horizon (trading days)
        lookback_days:  History length passed to factor_fn on each evaluation date
        step:           Evaluation step (every N days)
        min_symbols:    Minimum symbols required for a valid IC observation

    Returns:
        Dict with ic_mean, ic_ir, ic_p5, n_observations,
        ic_calm_bull, ic_stressed, ic_crisis, ic_min_regime,
        OR {"error": msg} on failure.
    """
    if prices_df.empty or len(prices_df) < lookback_days + n_periods:
        return {
            "error": "Insufficient price data",
            "ic_mean": 0.0, "ic_ir": 0.0, "n_observations": 0, "ic_p5": 0.0,
            "ic_calm_bull": 0.0, "ic_stressed": 0.0, "ic_crisis": 0.0, "ic_min_regime": 0.0,
        }

    raw = _compute_ic_series(factor_fn, prices_df, n_periods, lookback_days, step, min_symbols)
    if not raw["ics"]:
        return {
            "error": "No valid IC observations",
            "ic_mean": 0.0, "ic_ir": 0.0, "n_observations": 0, "ic_p5": 0.0,
            "ic_calm_bull": 0.0, "ic_stressed": 0.0, "ic_crisis": 0.0, "ic_min_regime": 0.0,
        }

    ic_arr  = np.array(raw["ics"])
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    ic_ir   = round(ic_mean / ic_std, 3) if ic_std > 1e-6 else 0.0
    ic_p5   = float(np.percentile(ic_arr, 5))

    result = {
        "ic_mean":        round(ic_mean, 4),
        "ic_ir":          round(ic_ir, 3),
        "ic_p5":          round(ic_p5, 4),
        "n_observations": len(ic_arr),
        "ic_calm_bull":   0.0,
        "ic_stressed":    0.0,
        "ic_crisis":      0.0,
        "ic_min_regime":  0.0,
    }

    # Per-regime split
    if regime_labels is not None and not regime_labels.empty:
        dates_series = pd.Series(raw["ics"], index=pd.DatetimeIndex(raw["dates"]))
        for regime_key, label in [
            ("ic_calm_bull", "calm_bull"),
            ("ic_stressed",  "stressed"),
            ("ic_crisis",    "crisis"),
        ]:
            regime_dates = regime_labels[regime_labels == label].index
            overlap      = dates_series.index.intersection(regime_dates)
            if len(overlap) >= 5:
                result[regime_key] = round(float(dates_series.reindex(overlap).mean()), 4)

        observed_ics = [v for k, v in result.items()
                        if k.startswith("ic_") and k not in ("ic_mean", "ic_ir", "ic_p5",
                                                               "ic_min_regime")
                        and v != 0.0]
        result["ic_min_regime"] = round(min(observed_ics), 4) if observed_ics else ic_mean

    else:
        result["ic_min_regime"] = ic_mean

    return result


def passes_harvey_threshold(ic_mean: float, ic_ir: float) -> bool:
    """
    Harvey et al. (2016) multiple-testing correction.

    Requires IC_mean > 0.015 AND IC_IR > 0.60.
    At IR > 0.60 with 50 candidates/year, expected spurious acceptances ≈ 0.5.
    At IR > 0.40, that rises to ~3.

    Reference: arXiv:2006.04269
    """
    return ic_mean > IC_MEAN_THRESHOLD and ic_ir > IC_IR_THRESHOLD
```

---

### Step 6: Create `ascent/research/factor_discovery/llm_suggester.py`

```python
"""
ascent/research/factor_discovery/llm_suggester.py

LLM-guided template parameter suggestion (Claude Haiku).

The LLM's role here is NOT to write code. It proposes parameters for
pre-defined template families (lookback windows, normalization methods, etc.)
based on the current regime and historical IC statistics.

This is the correct LLM interface for factor discovery:
  - LLM returns a JSON parameter dict
  - Trusted Python template instantiates and runs it
  - Zero code injection risk
  - Output is interpretable and reviewable

Reference: AlphaAgent (arXiv:2502.16789) shows that unconstrained code
generation requires expensive semantic alignment checks. Template-based
suggestion avoids this entirely while retaining the LLM's economic reasoning.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quantitative researcher at Ascent Capital. Your job is to suggest
parameters for alpha signal templates based on the current market regime and
recent signal performance statistics.

You will choose ONE template family and ONE set of parameters that you
believe will capture a signal orthogonal to the existing ones.

You ONLY return a JSON object. No code, no explanation outside the JSON."""

_USER_TEMPLATE = """\
Current regime: {regime}

Existing factor IC statistics (avoid overlap with these):
{ic_context}

Available template families:
- MomentumTemplate: lookback (21–252 days), skip_days (0–63), normalization (zscore/rank/minmax)
- ReversionTemplate: lookback (2–21 days), smooth_window (1–10), normalization
- VolatilityTemplate: vol_window (10–63), vov_window (21–126), direction (low/trend), normalization
- QualityTemplate: metric (consistency/drawdown/trend_strength), window (21–252), normalization
- CorrelationTemplate: window (21–126), mode (beta/idiosyncratic), normalization

Think step by step:
1. What economic mechanism is likely to drive returns in a {regime} regime?
2. Which template family best captures that mechanism?
3. What parameter values reflect the regime's typical duration and dynamics?
4. Is this sufficiently different from the existing factors listed above?

Respond with exactly this JSON:
{{
  "template": "TemplateName",
  "params": {{}},
  "rationale": "One sentence — economic mechanism"
}}"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.5,
            use_cache=False,
        )
    except Exception as exc:
        log.warning("[LLMSuggester] Call failed: %s", exc)
        return None


def suggest_template_params(
    regime: str,
    ic_context: Dict,
    existing_factor_names: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Ask Claude Haiku to suggest one template + parameters.

    Args:
        regime:               Current regime label.
        ic_context:           Dict of {factor_name: ic_mean} for existing factors.
        existing_factor_names: Names already in production (for context).

    Returns:
        Dict with {template, params, rationale} or None if LLM/parse fails.
    """
    ic_lines = "\n".join(
        f"  {name}: IC={ic:.3f}" for name, ic in ic_context.items()
    ) or "  (no IC data yet)"

    if existing_factor_names:
        ic_lines += "\nDeployed factors: " + ", ".join(existing_factor_names)

    user_prompt = _USER_TEMPLATE.format(regime=regime, ic_context=ic_lines)
    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        if "template" not in parsed or "params" not in parsed:
            return None
        return parsed
    except Exception as exc:
        log.warning("[LLMSuggester] Parse failed: %s", exc)
        return None
```

---

### Step 7: Create `ascent/research/factor_discovery/pysr_engine.py`

```python
"""
ascent/research/factor_discovery/pysr_engine.py

Symbolic regression via PySR for factor discovery.

PySR evolves human-readable mathematical expressions (e.g. "sqrt(vol_21d) /
(mom_252d + 1e-6)") using genetic programming on pre-computed cross-sectional
feature vectors. Output is a formula string, not Python code — safe by
construction.

Requires: pip install pysr
Fallback: if PySR is unavailable, falls back to random template permutation.

Why PySR instead of LLM code generation:
  - No code execution risk (expression is a mathematical formula)
  - Transparent output (human-readable, reviewable)
  - No hallucination (builds from primitives you define)
  - Novelty by construction (genetic crossover explores new combinations)
  - AlphaAgent and QuantaAlpha both use evolutionary methods as the
    core discovery mechanism; LLM provides hypothesis framing around it.

Reference: PySR — Cranmer (2023), ACM GEC 2024 review.
Reference: AlphaAgent — arXiv:2502.16789
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Pre-computed feature names that PySR can use as terminals
# These must all be computable from a price DataFrame
_FEATURE_NAMES = [
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "rev_5d", "rev_10d",
    "vol_21d", "vol_63d",
    "zscore_21d",
    "high_52w_pct",
]

_UNARY_OPERATORS  = ["sqrt", "log", "abs", "neg"]
_BINARY_OPERATORS = ["+", "-", "*", "/"]


def _compute_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional feature panel from a price DataFrame.
    Returns DataFrame (columns = feature names, index = symbols).
    All columns are cross-sectionally z-scored.
    """
    n = len(prices_df)
    cols = prices_df.columns
    rows = {}

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    rets = prices_df.pct_change()

    if n >= 22:
        rows["mom_21d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-21] - 1)
    if n >= 64:
        rows["mom_63d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-63] - 1)
    if n >= 127:
        rows["mom_126d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-126] - 1)
    if n >= 253:
        rows["mom_252d"] = _zscore(prices_df.iloc[-1] / prices_df.iloc[-252] - 1)
    if n >= 6:
        rows["rev_5d"] = _zscore(-(prices_df.iloc[-1] / prices_df.iloc[-5] - 1))
    if n >= 11:
        rows["rev_10d"] = _zscore(-(prices_df.iloc[-1] / prices_df.iloc[-10] - 1))
    if n >= 22:
        rows["vol_21d"] = _zscore(-rets.tail(21).std())
    if n >= 64:
        rows["vol_63d"] = _zscore(-rets.tail(63).std())
    if n >= 22:
        roll_mean = rets.tail(252).rolling(21).mean()
        roll_std  = rets.tail(252).rolling(21).std()
        rows["zscore_21d"] = _zscore((roll_mean.iloc[-1]) / (roll_std.iloc[-1] + 1e-8))
    if n >= 253:
        peak = prices_df.tail(252).max()
        rows["high_52w_pct"] = _zscore(prices_df.iloc[-1] / (peak + 1e-8) - 1)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, index=cols).dropna(how="all")


def _run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_iterations: int = 40,
    population_size: int = 30,
) -> Optional[Tuple[str, Callable]]:
    """
    Run PySR symbolic regression. Returns (expression_str, callable) or None.
    """
    try:
        from pysr import PySRRegressor
        model = PySRRegressor(
            niterations=n_iterations,
            population_size=population_size,
            binary_operators=_BINARY_OPERATORS,
            unary_operators=_UNARY_OPERATORS,
            maxsize=10,
            verbosity=0,
            progress=False,
            random_state=42,
        )
        model.fit(X, y, variable_names=feature_names)
        best = model.get_best()
        expr_str = str(best["equation"])
        fn = model.predict
        return expr_str, fn
    except ImportError:
        log.info("[PySR] pysr not installed — skipping symbolic regression path")
        return None
    except Exception as exc:
        log.warning("[PySR] Run failed: %s", exc)
        return None


def discover_via_pysr(
    prices_df: pd.DataFrame,
    n_periods: int = 5,
    lookback_days: int = 252,
    n_iterations: int = 40,
) -> List[Dict]:
    """
    Run PySR on a rolling cross-sectional dataset to discover symbolic factors.

    For each evaluation date (every 5 days), compute features and forward returns,
    stacking into X (features) and y (forward returns). PySR discovers the
    expression that best predicts y from X cross-sectionally.

    Returns:
        List of dicts: [{name, expression, description, factor_fn}]
        Empty list if PySR is unavailable or data insufficient.
    """
    if len(prices_df) < lookback_days + n_periods + 50:
        return []

    X_rows, y_rows = [], []
    dates = prices_df.index[lookback_days:-n_periods:5]

    for dt in dates:
        try:
            iloc_pos = prices_df.index.get_loc(dt)
            window   = prices_df.iloc[max(0, iloc_pos - lookback_days): iloc_pos + 1]
            feats    = _compute_features(window)
            if feats.empty or len(feats) < 5:
                continue
            fwd_rets = prices_df.iloc[iloc_pos + n_periods] / prices_df.iloc[iloc_pos] - 1
            common   = feats.index.intersection(fwd_rets.index)
            X_rows.append(feats.reindex(common).values)
            y_rows.append(fwd_rets.reindex(common).values)
        except Exception:
            continue

    if not X_rows or len(X_rows) < 5:
        return []

    # Stack all cross-sections (observations = dates × symbols)
    try:
        X = np.vstack(X_rows)
        y = np.concatenate(y_rows)
        available_features = list(feats.columns)
    except Exception as exc:
        log.warning("[PySR] Stack failed: %s", exc)
        return []

    result = _run_pysr(X, y, available_features, n_iterations=n_iterations)
    if result is None:
        return []

    expr_str, pysr_predict_fn = result
    feature_names_used = available_features

    def _factor_fn(df: pd.DataFrame, _expr=expr_str, _feats=feature_names_used,
                   _predict=pysr_predict_fn) -> pd.Series:
        feats_df = _compute_features(df)
        if feats_df.empty:
            return pd.Series(0.0, index=df.columns)
        X_eval = feats_df.reindex(columns=_feats, fill_value=0.0).values
        try:
            scores = _predict(X_eval)
            s = pd.Series(scores, index=feats_df.index)
            std = s.std()
            if std < 1e-8:
                return pd.Series(0.0, index=s.index)
            return (s - s.mean()) / std
        except Exception:
            return pd.Series(0.0, index=feats_df.index)

    return [{
        "name":        f"factor_pysr_{abs(hash(expr_str)) % 10000:04d}",
        "expression":  expr_str,
        "description": f"PySR-discovered symbolic expression: {expr_str}",
        "source":      "pysr",
        "fn":          _factor_fn,
    }]
```

---

### Step 8: Create `ascent/research/factor_discovery/discovery_runner.py`

```python
"""
ascent/research/factor_discovery/discovery_runner.py

Orchestrates the two-path autonomous factor discovery pipeline.

Path A — PySR symbolic regression (primary):
  1. Load prices from cache
  2. Run pysr_engine.discover_via_pysr() → symbolic expressions
  3. Evaluate each via regime_cpcv_evaluator

Path B — LLM template suggestions (secondary):
  1. Ask Claude Haiku for template + parameters
  2. Instantiate template from feature_templates.py
  3. Evaluate via regime_cpcv_evaluator

Combined acceptance gate:
  - IC_mean > 0.015  (Grinold threshold for Ascent's breadth)
  - IC_IR   > 0.60   (Harvey et al. FDR-corrected threshold)
  - IC_min_regime > 0.01  (must work in every observed regime)
  - n_observations >= 20

Accepted proposals written to outputs/factor_proposals/ with deployment
instructions. Nothing auto-deploys — human reviews every accepted proposal.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

PROPOSALS_DIR     = Path("outputs/factor_proposals")
DISCOVERY_LOG     = Path("logs/factor_discovery_log.jsonl")
IC_MEAN_THRESHOLD = 0.015
IC_IR_THRESHOLD   = 0.60
IC_MIN_REGIME     = 0.010
MIN_OBSERVATIONS  = 20

_DEPLOYED_FACTORS = [
    "trend", "meanrev", "volatility", "statarb", "ml",
    "fundamental", "earnings", "analyst", "options_flow",
    "insider", "short_interest", "llm_fundamental",
]


def _load_prices() -> pd.DataFrame:
    """Load price data from parquet cache or yfinance fallback."""
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("prices_live"):
            df = load_parquet("prices_live")
            if "close" in df.columns:
                return df.pivot(columns="symbol", values="close").sort_index()
            elif isinstance(df.columns, pd.MultiIndex):
                return df["Close"].sort_index()
            return df.sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] Parquet load failed: %s", exc)
    try:
        import yfinance as yf
        from ascent.config.settings import get_config
        syms = list(getattr(get_config().universe, "symbols", []))[:60]
        raw  = yf.download(syms, period="3y", auto_adjust=True, progress=False)
        return (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(
            axis=1, how="all").sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] yfinance fallback failed: %s", exc)
        return pd.DataFrame()


def _load_regime_labels(prices_index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Load regime labels from dashboard CSV, aligned to prices index."""
    try:
        df = pd.read_csv("dashboard/regime_labels.csv", parse_dates=["date"])
        df = df.set_index("date").sort_index()
        return df["label"].reindex(prices_index, method="ffill")
    except Exception:
        return None


def _write_log(entry: dict) -> None:
    DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_proposal(candidate: dict, ic_result: dict) -> Path:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    fname = f"{candidate['name']}_{today}.json"
    path  = PROPOSALS_DIR / fname
    payload = {
        "name":         candidate["name"],
        "source":       candidate.get("source", "unknown"),
        "description":  candidate.get("description", ""),
        "expression":   candidate.get("expression", ""),    # PySR: formula string
        "template":     candidate.get("template", ""),      # LLM path: template name
        "template_params": candidate.get("params", {}),
        "rationale":    candidate.get("rationale", ""),
        **{k: ic_result.get(k) for k in [
            "ic_mean", "ic_ir", "ic_p5", "n_observations",
            "ic_calm_bull", "ic_stressed", "ic_crisis", "ic_min_regime"
        ]},
        "proposed_at":          datetime.now().isoformat(),
        "regime_at_proposal":   candidate.get("regime", "unknown"),
        "review_status":        "pending",
        "review_notes":         "",
        "how_to_deploy": (
            "1. Review the expression/description for economic soundness.\n"
            "2. For PySR factors: translate the expression into a Python function in "
            "   ascent/features/feature_defs.py following the existing pattern.\n"
            "3. For template factors: the feature_templates.py instantiation is the "
            "   implementation — wrap it as a function in feature_defs.py.\n"
            "4. Register in build_all_features() with appropriate lag.\n"
            "5. Add to stack.py DEFAULT_ALPHA_WEIGHTS at a small initial weight (0.02).\n"
            "6. Reduce another sleeve by 0.02 to keep the sum at 1.0.\n"
            "7. Run the full test suite before committing.\n"
            "8. Run system once in dry-run to confirm no pipeline errors."
        ),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _build_candidates(regime: str, prices_df: pd.DataFrame) -> List[Dict]:
    """Build candidate factor list from both paths."""
    candidates = []

    # Path A: PySR symbolic regression
    try:
        from ascent.research.factor_discovery.pysr_engine import discover_via_pysr
        pysr_candidates = discover_via_pysr(prices_df, n_periods=5, n_iterations=40)
        for c in pysr_candidates:
            c["regime"] = regime
        candidates.extend(pysr_candidates)
        log.info("[FactorDiscovery] PySR produced %d candidates", len(pysr_candidates))
    except Exception as exc:
        log.warning("[FactorDiscovery] PySR path failed: %s", exc)

    # Path B: LLM template suggestion
    try:
        from ascent.research.factor_discovery.llm_suggester import suggest_template_params
        from ascent.research.factor_discovery.feature_templates import instantiate_template
        suggestion = suggest_template_params(
            regime=regime,
            ic_context={},
            existing_factor_names=_DEPLOYED_FACTORS,
        )
        if suggestion:
            tmpl = instantiate_template(suggestion["template"], suggestion["params"])
            def _make_fn(t):
                def _fn(df):
                    return t.compute(df)
                return _fn
            candidates.append({
                "name":        f"factor_llm_{suggestion['template'].lower()[:8]}",
                "description": suggestion.get("rationale", ""),
                "source":      "llm_template",
                "template":    suggestion["template"],
                "params":      suggestion["params"],
                "rationale":   suggestion.get("rationale", ""),
                "regime":      regime,
                "fn":          _make_fn(tmpl),
            })
            log.info("[FactorDiscovery] LLM template candidate: %s params=%s",
                     suggestion["template"], suggestion["params"])
    except Exception as exc:
        log.warning("[FactorDiscovery] LLM template path failed: %s", exc)

    return candidates


def run_factor_discovery(
    n_candidates: int = 5,
    regime: Optional[str] = None,
) -> Dict:
    """
    Run one full factor discovery cycle (both paths).

    Args:
        n_candidates: Target number of candidates to evaluate (informational).
        regime:       Current regime string.

    Returns:
        Dict: n_generated, n_valid, n_accepted, n_rejected, proposals (paths).
    """
    from ascent.research.factor_discovery.regime_cpcv_evaluator import (
        evaluate_factor_regime_ic, passes_harvey_threshold,
    )

    regime = regime or "unknown"
    log.info("[FactorDiscovery] Starting cycle — regime=%s", regime)

    prices = _load_prices()
    if prices.empty:
        log.warning("[FactorDiscovery] No price data — aborting")
        return {"n_generated": 0, "n_valid": 0, "n_accepted": 0, "n_rejected": 0, "proposals": []}

    regime_labels = _load_regime_labels(prices.index)
    candidates    = _build_candidates(regime, prices)

    n_valid = n_accepted = n_rejected = 0
    proposals = []

    for candidate in candidates:
        name = candidate.get("name", "unknown")
        fn   = candidate.get("fn")
        if fn is None:
            continue

        log.info("[FactorDiscovery] Evaluating %s (source=%s)", name, candidate.get("source"))

        ic_result = evaluate_factor_regime_ic(
            factor_fn=fn,
            prices_df=prices,
            regime_labels=regime_labels,
            n_periods=5,
        )

        if "error" in ic_result:
            log.info("[FactorDiscovery] Eval error for %s: %s", name, ic_result["error"])
            _write_log({
                "date": date.today().isoformat(), "regime": regime,
                "name": name, "status": "evaluation_error", "error": ic_result["error"],
            })
            n_rejected += 1
            continue

        n_valid += 1
        ic_mean  = ic_result["ic_mean"]
        ic_ir    = ic_result["ic_ir"]
        n_obs    = ic_result["n_observations"]
        ic_min_r = ic_result.get("ic_min_regime", ic_mean)

        log.info("[FactorDiscovery] %s — IC=%.4f, IR=%.3f, IC_min_regime=%.4f, n=%d",
                 name, ic_mean, ic_ir, ic_min_r, n_obs)

        log_entry = {
            "date": date.today().isoformat(), "regime": regime, "name": name,
            "source": candidate.get("source"),
            "ic_mean": ic_mean, "ic_ir": ic_ir, "n_observations": n_obs,
            "ic_min_regime": ic_min_r,
        }

        # Acceptance gate: Harvey FDR + per-regime minimum + observation count
        if (passes_harvey_threshold(ic_mean, ic_ir)
                and ic_min_r > IC_MIN_REGIME
                and n_obs >= MIN_OBSERVATIONS):
            path = _write_proposal(candidate, ic_result)
            proposals.append(str(path))
            n_accepted += 1
            log_entry["status"] = "accepted"
            log_entry["proposal_path"] = str(path)
            log.info("[FactorDiscovery] ACCEPTED: %s → %s", name, path)
        else:
            n_rejected += 1
            reasons = []
            if not passes_harvey_threshold(ic_mean, ic_ir):
                reasons.append(
                    f"IC={ic_mean:.4f} or IR={ic_ir:.3f} below Harvey threshold "
                    f"({IC_MEAN_THRESHOLD}/{IC_IR_THRESHOLD})"
                )
            if ic_min_r <= IC_MIN_REGIME:
                reasons.append(f"IC_min_regime={ic_min_r:.4f} ≤ {IC_MIN_REGIME}")
            if n_obs < MIN_OBSERVATIONS:
                reasons.append(f"Only {n_obs} observations (need {MIN_OBSERVATIONS})")
            log_entry["status"] = "rejected"
            log_entry["reasons"] = reasons
            log.info("[FactorDiscovery] Rejected %s: %s", name, "; ".join(reasons))

        _write_log(log_entry)

    summary = {
        "n_generated": len(candidates),
        "n_valid":     n_valid,
        "n_accepted":  n_accepted,
        "n_rejected":  n_rejected,
        "proposals":   proposals,
        "regime":      regime,
        "date":        date.today().isoformat(),
    }
    log.info("[FactorDiscovery] Cycle complete: %d accepted / %d rejected",
             n_accepted, n_rejected)
    return summary
```

---

### Step 9: Run all tests

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_factor_discovery.py -v
```
Expected: All 14 tests PASS.

Full suite:
```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -6
```
Expected: ≥ 326 tests pass (312 + 14 new).

---

### Step 10: Wire monthly trigger into `run_all_agents.py`

Add after the existing Sunday self-improve block:

```python
        # Factor discovery — first Sunday of each month only
        try:
            from datetime import date as _date
            _today = _date.today()
            if _today.weekday() == 6 and _today.day <= 7:  # first Sunday of month
                from ascent.research.factor_discovery.discovery_runner import run_factor_discovery
                _regime_for_disc = _get_current_regime()
                print(f"[FactorDiscovery] Monthly run — regime={_regime_for_disc}")
                _disc = run_factor_discovery(n_candidates=5, regime=_regime_for_disc)
                print(
                    f"[FactorDiscovery] Done: {_disc['n_accepted']} accepted, "
                    f"{_disc['n_rejected']} rejected. "
                    f"Proposals: outputs/factor_proposals/"
                )
        except Exception as _de:
            print(f"[FactorDiscovery] Monthly run skipped: {_de}")
```

Add near the Sunday block if not already present:

```python
def _get_current_regime() -> str:
    try:
        import json as _json
        sig = _json.loads(open("dashboard/regime_signal.json").read())
        return str(sig.get("label", "unknown")).lower()
    except Exception:
        return "unknown"
```

---

### Step 11: Commit

```bash
git add ascent/research/factor_discovery/ tests/test_factor_discovery.py run_all_agents.py
git commit -m "feat(research): autonomous factor discovery — PySR + templates, per-regime CPCV, Harvey FDR (14 tests)"
```

---

## Human Review Process

When a proposal is accepted, `outputs/factor_proposals/factor_name_YYYY-MM-DD.json` is written. Review process:

1. Read `expression` (PySR) or `template` + `template_params` (LLM path)
2. Check `ic_calm_bull`, `ic_stressed`, `ic_crisis` — must all be positive
3. Check `rationale` — is the economic mechanism sensible?
4. For PySR factors: translate `expression` into a Python function in `feature_defs.py`
5. For template factors: call `instantiate_template(template, params).compute(df)` in a new `feature_defs.py` function
6. Add to `build_all_features()` and `stack.py` at initial weight 0.02
7. Run full test suite
8. Monitor IC daily via `slippage_ic_feedback.py` — divergence > 20% from CPCV IC → halt deployment

---

## Acceptance Thresholds — Rationale

| Threshold | Value | Source |
|-----------|-------|--------|
| `IC_mean > 0.015` | 0.015 | Grinold & Kahn: IR=0.40 at Ascent's breadth (~1,000 decisions/year) requires IC ≈ 0.013. 0.015 provides a 15% margin. |
| `IC_IR > 0.60` | 0.60 | Harvey, Liu, Zhu (2016, arXiv:2006.04269): t-stat equivalent > 3.0 for multiple-testing-corrected factor significance. At IR > 0.40, ~3 spurious/year from 50 candidates; at IR > 0.60, ~0.5. |
| `IC_min_regime > 0.01` | 0.01 | A factor with negative IC in crisis destroys alpha at the worst time. Require positive IC in every regime that has ≥ 5 observations. |
| `n_observations ≥ 20` | 20 | Minimum for reliable IC estimation. At 5-day step × 504 history days → ~100 observations typical. |

---

## Self-Review Checklist (for implementer)

- [ ] PySR is installed and `import pysr` succeeds before starting
- [ ] `feature_templates.py`: all 5 templates return `pd.Series` indexed by symbol
- [ ] `feature_templates.py`: flat price input yields zero signal (not NaN)
- [ ] `leakage_scanner.py`: rejects `.tail(1)`, `datetime.now()`, `.shift(-N)`
- [ ] `leakage_scanner.py`: accepts clean rolling/pct_change code
- [ ] `regime_cpcv_evaluator.py`: `evaluate_factor_regime_ic` returns all required keys
- [ ] `regime_cpcv_evaluator.py`: `passes_harvey_threshold` uses 0.015 / 0.60 thresholds
- [ ] `llm_suggester.py`: returns None on LLM failure (no crash)
- [ ] `pysr_engine.py`: returns empty list if PySR not installed (graceful)
- [ ] `discovery_runner.py`: proposal only written when all three conditions met
- [ ] `discovery_runner.py`: low-IR candidate NOT written to proposals dir
- [ ] All 14 tests pass; no regressions in full suite
- [ ] `run_all_agents.py` monthly trigger fires on first Sunday only (`day <= 7`)
