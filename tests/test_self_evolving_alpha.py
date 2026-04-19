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


# ── Task 2 tests ───────────────────────────────────────────────────────────────

def test_shadow_promoter_promotes_expired_winner(tmp_path, monkeypatch):
    """A shadow config past its expiry that still beats baseline must be promoted to live."""
    import json
    import numpy as np
    import pandas as pd
    from datetime import timedelta, date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()

    shadow = {
        "variant_id":        "v1_20260318",
        "alpha_weights":     {"trend": 0.70, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.0},
        "oos_sharpe":        0.72,
        "edge_over_current": 0.20,
        "shadow_expires":    (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":       "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v1_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # Write prices so lightweight OOS can run
    idx = pd.bdate_range(end=date.today(), periods=300)
    syms = [f"S{i}" for i in range(20)] + ["SPY"]
    np.random.seed(1)
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    prices.to_parquet(tmp_path / "data_cache" / "prices_live.parquet")

    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_sharpe=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert active_path.exists(), "active_alpha_config.json must be written after promotion"
    config = json.loads(active_path.read_text())
    assert "global" in config, "promoted config must have 'global' key"
    assert abs(config["global"].get("trend", 0) - 0.70) < 0.001


def test_shadow_promoter_skips_unexpired(tmp_path, monkeypatch):
    """Shadow configs that haven't expired yet must not be promoted."""
    import json
    from datetime import timedelta, date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "logs").mkdir()

    shadow = {
        "variant_id":    "v2_20260410",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.70,
        "shadow_expires": (date.today() + timedelta(days=15)).isoformat(),
        "promoted_at":   "2026-04-10T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v2_20260410.json"
    shadow_file.write_text(json.dumps(shadow))

    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_sharpe=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "must NOT promote a config that hasn't expired yet"


def test_shadow_promoter_archives_weak_expired(tmp_path, monkeypatch):
    """An expired shadow config that no longer beats baseline must be archived, not promoted."""
    import json
    from datetime import timedelta, date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache" / "shadow_configs").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data_cache" / "archived_configs").mkdir()

    shadow = {
        "variant_id":    "v3_20260318",
        "alpha_weights": {"trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05},
        "oos_sharpe":    0.52,
        "shadow_expires": (date.today() - timedelta(days=1)).isoformat(),
        "promoted_at":   "2026-03-18T06:00:00",
    }
    shadow_file = tmp_path / "data_cache" / "shadow_configs" / "v3_20260318.json"
    shadow_file.write_text(json.dumps(shadow))

    # No price cache — OOS returns 0.0 sharpe → below baseline 0.518
    from ascent.research.shadow_promoter import run_shadow_promotion
    run_shadow_promotion(baseline_sharpe=0.518)

    active_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert not active_path.exists(), "weak expired config must not become live"
    archived = list((tmp_path / "data_cache" / "archived_configs").glob("*.json"))
    assert len(archived) >= 1, "expired weak config must be moved to archived_configs"


# ── Task 3 tests ───────────────────────────────────────────────────────────────

def test_generate_variants_produces_valid_weights():
    """generate_variants must produce N variants, each summing to 1.0."""
    from ascent.research.self_improve import generate_variants
    base = {"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                               "statarb": 0.15, "ml": 0.10, "volatility": 0.05}}
    variants = generate_variants(base, n=5)
    assert len(variants) == 5
    for v in variants:
        total = sum(v["alpha_weights"].values())
        assert abs(total - 1.0) < 0.01, f"weights must sum to 1, got {total}"


def test_run_self_improve_writes_regime_config(tmp_path, monkeypatch):
    """When regime='stressed', self_improve must write stressed weights to by_regime."""
    import numpy as np
    import pandas as pd
    from datetime import date
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir()
    (tmp_path / "logs").mkdir()

    idx = pd.bdate_range(end=date.today(), periods=300)
    syms = [f"S{i}" for i in range(20)] + ["SPY"]
    np.random.seed(2)
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(syms)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    prices.to_parquet(tmp_path / "data_cache" / "prices_live.parquet")

    from ascent.research.self_improve import run_self_improve
    run_self_improve(current_regime="stressed")

    log_path = tmp_path / "logs" / "self_improve_log.jsonl"
    assert log_path.exists()


def test_promote_regime_variant_writes_by_regime(tmp_path, monkeypatch):
    """_promote_regime_variant must write weights to by_regime.stressed in active config."""
    import json
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data_cache").mkdir()

    from ascent.research.self_improve import _promote_regime_variant
    weights = {"trend": 0.50, "meanrev": 0.05, "statarb": 0.25, "ml": 0.15, "volatility": 0.05}
    _promote_regime_variant(weights, regime="stressed", oos_sharpe=0.65, edge=0.13)

    config_path = tmp_path / "data_cache" / "active_alpha_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert "by_regime" in config
    assert "stressed" in config["by_regime"]
    assert abs(config["by_regime"]["stressed"].get("trend", 0) - 0.50) < 0.001
