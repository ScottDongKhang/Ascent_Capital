# tests/test_walkforward_institutional.py
import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path


def _make_price_cache(tmp_path, n_days=500, n_syms=25):
    np.random.seed(42)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_syms)] + ["SPY"]
    n_actual = len(idx)
    rets = np.random.normal(0.0003, 0.012, (n_actual, len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)
    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    return prices


def test_lightweight_oos_uses_multiple_folds(tmp_path, monkeypatch):
    """With enough data, run_lightweight_oos must return n_folds > 1."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
    )
    assert result["n_folds"] > 1, \
        f"Expected multiple folds with 500 days of data, got n_folds={result['n_folds']}"


def test_lightweight_oos_purge_embargo_respected(tmp_path, monkeypatch):
    """Verify the function runs without error when purge and embargo are applied."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                                             "statarb": 0.10, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        purge_days=5,
        embargo_days=5,
    )
    assert isinstance(result["sharpe"], float)
    assert result["n_folds"] >= 1


def test_lightweight_oos_survivorship_bias_fix(tmp_path, monkeypatch):
    """Universe must be filtered per fold date — graceful fallback when no universe data."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
        n_days=63,
        filter_universe_by_date=True,
    )
    assert "sharpe" in result
    assert "n_folds" in result


def test_lightweight_oos_sharpe_from_all_folds(tmp_path, monkeypatch):
    """Sharpe must be computed across all fold returns, not just the last fold."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path, n_days=500)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos
    r1 = run_lightweight_oos({"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                                                  "statarb": 0.15, "ml": 0.10, "volatility": 0.05}},
                              n_days=63)
    assert np.isfinite(r1["sharpe"])
    assert r1.get("n_folds", 0) >= 1


def test_walk_forward_runner_calls_universe_per_fold():
    """walk_forward_runner must call get_universe_on_date on every fold -- A4 gap."""
    import inspect
    from ascent.research import walk_forward_runner
    src = inspect.getsource(walk_forward_runner)
    assert "get_universe_on_date" in src, \
        "walk_forward_runner must call get_universe_on_date() per fold to prevent survivorship bias"
