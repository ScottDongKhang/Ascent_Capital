# tests/test_self_evolving_alpha.py
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date


def _make_features(n=60, syms=5):
    """Minimal features dict for build_alpha_stack tests."""
    idx = pd.bdate_range(end="2026-04-18", periods=n)
    cols = [f"S{i}" for i in range(syms)]
    np.random.seed(0)
    price = pd.DataFrame(100 * np.cumprod(1 + np.random.normal(0, 0.01, (n, syms)), axis=0),
                         index=idx, columns=cols)
    return {
        "close":      price,
        "returns_1d": price.pct_change().fillna(0),
        "mom_21d":    price.pct_change(21).fillna(0),
        "mom_63d":    price.pct_change(63).fillna(0),
        "vol_21d":    price.pct_change().rolling(21).std().fillna(0.01),
    }


def _write_active_config(tmp_path: Path, global_weights: dict, by_regime: dict = None):
    config = {"global": global_weights}
    if by_regime:
        config["by_regime"] = by_regime
    config["updated_at"] = "2026-04-18"
    p = tmp_path / "data_cache" / "active_alpha_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config))
    return p


def test_stack_uses_active_config_when_present(tmp_path, monkeypatch):
    """build_alpha_stack must use active_alpha_config.json when it exists."""
    monkeypatch.chdir(tmp_path)
    custom = {"trend": 0.80, "meanrev": 0.05, "statarb": 0.10, "ml": 0.05, "volatility": 0.0}
    _write_active_config(tmp_path, custom)

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights()
    assert abs(loaded.get("trend", 0) - 0.80) < 0.001, \
        "stack must load trend=0.80 from active_alpha_config.json"


def test_stack_falls_back_to_defaults_when_no_config(tmp_path, monkeypatch):
    """Without active_alpha_config.json, stack uses DEFAULT_ALPHA_WEIGHTS."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir(exist_ok=True)

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights()
    assert abs(loaded.get("trend", 0) - 0.65) < 0.001, \
        "without config file, trend should be default 0.65"


def test_stack_uses_regime_weights_when_regime_in_config(tmp_path, monkeypatch):
    """When active config has by_regime and regime matches, use those weights."""
    monkeypatch.chdir(tmp_path)
    global_w = {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05}
    stressed_w = {"trend": 0.45, "meanrev": 0.05, "statarb": 0.30, "ml": 0.15, "volatility": 0.05}
    _write_active_config(tmp_path, global_w, by_regime={"stressed": stressed_w})

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights(regime="stressed")
    assert abs(loaded.get("trend", 0) - 0.45) < 0.001, \
        "stressed regime config must give trend=0.45"
    assert abs(loaded.get("statarb", 0) - 0.30) < 0.001


def test_stack_falls_back_to_global_for_unknown_regime(tmp_path, monkeypatch):
    """For an unknown regime, fall back to global weights in active config."""
    monkeypatch.chdir(tmp_path)
    global_w = {"trend": 0.70, "meanrev": 0.05, "statarb": 0.10, "ml": 0.10, "volatility": 0.05}
    _write_active_config(tmp_path, global_w, by_regime={})

    import importlib
    import ascent.alpha.stack as stack_mod
    importlib.reload(stack_mod)

    loaded = stack_mod._load_active_alpha_weights(regime="euphoric")
    assert abs(loaded.get("trend", 0) - 0.70) < 0.001
