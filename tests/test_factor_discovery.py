"""
tests/test_factor_discovery.py
14 tests: feature templates, PySR engine, LLM suggester,
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
    assert passes_harvey_threshold(ic_mean=0.025, ic_ir=0.55)   is False
    assert passes_harvey_threshold(ic_mean=0.010, ic_ir=0.80)   is False
    assert passes_harvey_threshold(ic_mean=0.020, ic_ir=0.60)   is True


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
        "ic_mean": 0.020, "ic_ir": 0.35,
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
