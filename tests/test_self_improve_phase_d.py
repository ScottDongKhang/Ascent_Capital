# tests/test_self_improve_phase_d.py
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import json


def _make_price_cache(tmp_path, n_days=300, n_syms=20):
    """Create a minimal prices_live parquet file for testing."""
    np.random.seed(42)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_syms)] + ["SPY"]
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)
    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    return prices


def test_lightweight_oos_returns_sharpe(tmp_path, monkeypatch):
    """run_lightweight_oos must return a dict with 'sharpe' and 'turnover'."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)
    (tmp_path / "data_cache").mkdir(exist_ok=True)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos

    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.0}},
        n_days=63,
    )
    assert isinstance(result, dict)
    assert "sharpe" in result
    assert "turnover" in result
    assert isinstance(result["sharpe"], float)


def test_lightweight_oos_is_deterministic(tmp_path, monkeypatch):
    """Same config must return same Sharpe every run — no noise."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos

    cfg = {"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                              "statarb": 0.15, "ml": 0.10, "volatility": 0.05}}
    r1 = run_lightweight_oos(cfg, n_days=63)
    r2 = run_lightweight_oos(cfg, n_days=63)
    assert abs(r1["sharpe"] - r2["sharpe"]) < 0.001, "evaluate_variant must be deterministic"


def test_evaluate_variant_uses_real_oos(tmp_path, monkeypatch):
    """evaluate_variant must call run_lightweight_oos, not return noise."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "agent_skill_scores.json").write_text(
        json.dumps({"us_equities": {"sharpe": 0.55, "status": "active", "n_days": 63}})
    )

    from ascent.research.self_improve import evaluate_variant

    cfg = {"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                              "statarb": 0.15, "ml": 0.10, "volatility": 0.0}}
    s1 = evaluate_variant(cfg)
    s2 = evaluate_variant(cfg)
    assert abs(s1 - s2) < 0.001, "Must be deterministic — no random noise"
    assert isinstance(s1, float)
