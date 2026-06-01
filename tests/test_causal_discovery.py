# tests/test_causal_discovery.py
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_synthetic_data():
    """5 nodes, 100 observations — clear A→B and C→D causal links via construction."""
    rng = np.random.default_rng(42)
    n = 100
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.2, n)   # A causes B
    c = rng.normal(0, 1, n)
    d = 0.7 * c + rng.normal(0, 0.3, n)   # C causes D
    e = rng.normal(0, 1, n)
    return np.column_stack([a, b, c, d, e])


def test_run_pc_returns_dag_schema():
    """run_pc must return a dict with 'nodes', 'edges', 'active_transmission_chains'."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    assert "nodes" in result
    assert "edges" in result
    assert "active_transmission_chains" in result
    assert result["nodes"] == node_names
    assert isinstance(result["edges"], list)
    assert isinstance(result["active_transmission_chains"], list)


def test_run_pc_edges_have_required_fields():
    """Each edge must have 'from', 'to', 'strength', 'direction' fields."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    for edge in result["edges"]:
        assert "from" in edge, f"Edge missing 'from': {edge}"
        assert "to" in edge,   f"Edge missing 'to': {edge}"
        assert edge["strength"] in ("strong", "moderate", "weak")
        assert edge["direction"] in ("positive", "negative")


def test_run_pc_no_self_loops():
    """PC algorithm must not produce self-loop edges."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    for edge in result["edges"]:
        assert edge["from"] != edge["to"], f"Self-loop detected: {edge}"


def test_run_pc_no_duplicate_edges():
    """Each directed edge should appear at most once."""
    from ascent.causal.causal_discovery import run_pc

    data = _make_synthetic_data()
    node_names = ["fed_rate", "hy_spread", "vix", "unemployment", "xlf"]
    result = run_pc(data, node_names, alpha=0.05)

    pairs = [(e["from"], e["to"]) for e in result["edges"]]
    assert len(pairs) == len(set(pairs)), "Duplicate edges in DAG output"


def test_discover_macro_dag_writes_json(tmp_path):
    """discover_macro_dag must write a valid JSON file to the given path."""
    import pandas as pd
    from ascent.causal.causal_discovery import discover_macro_dag

    dates = pd.date_range("2024-01-05", periods=100, freq="W-FRI")
    macro_df = pd.DataFrame({
        "fed_rate": 5.25 + 0.01 * np.random.default_rng(0).standard_normal(100),
        "hy_spread": 3.5 + 0.1 * np.random.default_rng(1).standard_normal(100),
        "vix": 15 + np.random.default_rng(2).standard_normal(100),
        "unemployment": 4.0 + 0.01 * np.random.default_rng(3).standard_normal(100),
    }, index=dates)
    sector_df = pd.DataFrame({
        "XLF": 0.001 * np.random.default_rng(4).standard_normal(100),
        "XLK": 0.001 * np.random.default_rng(5).standard_normal(100),
        "XLV": 0.001 * np.random.default_rng(6).standard_normal(100),
        "XLE": 0.001 * np.random.default_rng(7).standard_normal(100),
        "XLP": 0.001 * np.random.default_rng(8).standard_normal(100),
    }, index=dates)
    out_path = tmp_path / "macro_causal_dag.json"

    discover_macro_dag(
        macro_df=macro_df,
        sector_df=sector_df,
        regime="calm_bull",
        output_path=out_path,
    )

    assert out_path.exists()
    dag = json.loads(out_path.read_text())
    assert "as_of" in dag
    assert "regime" in dag
    assert dag["regime"] == "calm_bull"
    assert len(dag["nodes"]) >= 5
