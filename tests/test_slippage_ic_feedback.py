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
