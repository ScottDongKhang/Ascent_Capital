# tests/test_ai_pm_prethesis_causal.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch


def test_aiprethesis_has_causal_mechanisms_field():
    """AIPreThesis must have a causal_mechanisms field (list, default empty)."""
    from agents.ai_pm_agent import AIPreThesis
    pt = AIPreThesis(
        macro_view="rates falling",
        regime_interpretation="calm_bull",
        high_conviction_names=[{"symbol": "WDC", "thesis": "NAND recovery"}],
        names_to_avoid=[],
        sector_tilts=[],
    )
    assert hasattr(pt, "causal_mechanisms"), "AIPreThesis missing causal_mechanisms field"
    assert isinstance(pt.causal_mechanisms, list)


def test_get_causal_graph_tool_in_pre_thesis_tools():
    """get_causal_graph must be registered in PRE_THESIS_TOOLS."""
    from agents.ai_pm_agent import PRE_THESIS_TOOLS
    names = {t["name"] for t in PRE_THESIS_TOOLS}
    assert "get_causal_graph" in names, \
        "get_causal_graph tool must be in PRE_THESIS_TOOLS"


def test_assemble_causal_mechanisms_gate1_filters_valuation(tmp_path):
    """
    _assemble_causal_mechanisms must apply Gate 1 (compatibility):
    valuation mechanism in calm_bull is excluded.
    """
    from ascent.config.types import CausalMechanism
    import agents.ai_pm_agent as mod

    graph = {
        "symbol": "WDC", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [
            {
                "mechanism": "NAND recovery → margins",
                "intervention": "IF nand +15%",
                "falsification_condition": "IF margin < 38%",
                "horizon_days": 63,
                "timing": "catalyst_imminent",
                "mechanism_type": "supply_demand_inflection",  # allowed in calm_bull
            },
            {
                "mechanism": "DCF compression",
                "intervention": "IF rates fall",
                "falsification_condition": "IF rate > 5%",
                "horizon_days": 42,
                "timing": "not_yet_priced",
                "mechanism_type": "valuation",   # blocked in calm_bull
            },
        ],
    }
    (tmp_path / "WDC_2026-03-31.json").write_text(json.dumps(graph))

    with patch("ascent.causal.dag_builder.DEFAULT_CACHE_DIR", tmp_path):
        with patch("ascent.causal.dag_builder.get_quarter_end", return_value="2026-03-31"):
            result = mod._assemble_causal_mechanisms(
                high_conviction_symbols=["WDC"],
                regime="calm_bull",
                cache_dir=tmp_path,
            )

    assert len(result) == 1, f"Expected 1 (valuation blocked), got {len(result)}"
    assert isinstance(result[0], CausalMechanism)
    assert result[0].symbol == "WDC"
    assert result[0].mechanism_type == "supply_demand_inflection"


def test_assemble_causal_mechanisms_gate2_filters_priced_in(tmp_path):
    """
    _assemble_causal_mechanisms must apply Gate 2 (priced_in filter):
    mechanisms with timing=='priced_in' are excluded regardless of compatibility.
    """
    import agents.ai_pm_agent as mod

    graph = {
        "symbol": "AAPL", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [
            {
                "mechanism": "Already rallied",
                "intervention": "IF eps beats",
                "falsification_condition": "IF price flat",
                "horizon_days": 21,
                "timing": "priced_in",  # excluded by gate 2
                "mechanism_type": "momentum_catalyst",  # compatible with calm_bull
            },
        ],
    }
    (tmp_path / "AAPL_2026-03-31.json").write_text(json.dumps(graph))

    with patch("ascent.causal.dag_builder.DEFAULT_CACHE_DIR", tmp_path):
        with patch("ascent.causal.dag_builder.get_quarter_end", return_value="2026-03-31"):
            result = mod._assemble_causal_mechanisms(
                high_conviction_symbols=["AAPL"],
                regime="calm_bull",
                cache_dir=tmp_path,
            )

    assert len(result) == 0, "priced_in mechanism should be excluded by Gate 2"


def test_build_velocity_context_ranks_catalyst_imminent_first(tmp_path):
    """_build_velocity_context must rank catalyst_imminent above not_yet_priced."""
    import agents.ai_pm_agent as mod

    graph_a = {
        "symbol": "AAPL", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [{"mechanism": "m_a", "intervention": "i", "falsification_condition": "f",
                        "horizon_days": 21, "timing": "not_yet_priced",
                        "mechanism_type": "momentum_catalyst"}],
    }
    graph_b = {
        "symbol": "WDC", "quarter_end": "2026-03-31", "built_at": "2026-06-01",
        "mechanisms": [{"mechanism": "m_b", "intervention": "i2", "falsification_condition": "f2",
                        "horizon_days": 63, "timing": "catalyst_imminent",
                        "mechanism_type": "supply_demand_inflection"}],
    }
    (tmp_path / "AAPL_2026-03-31.json").write_text(json.dumps(graph_a))
    (tmp_path / "WDC_2026-03-31.json").write_text(json.dumps(graph_b))

    with patch("ascent.causal.dag_builder.DEFAULT_CACHE_DIR", tmp_path):
        with patch("ascent.causal.dag_builder.get_quarter_end", return_value="2026-03-31"):
            lines = mod._build_velocity_context(
                symbols=["AAPL", "WDC"],
                regime="calm_bull",
                cache_dir=tmp_path,
            )

    text = "\n".join(lines)
    assert "WDC" in text
    assert "AAPL" in text
    wdc_pos = text.index("WDC")
    aapl_pos = text.index("AAPL")
    assert wdc_pos < aapl_pos, "catalyst_imminent (WDC) must rank before not_yet_priced (AAPL)"
