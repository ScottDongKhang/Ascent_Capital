# tests/test_dag_builder.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


_FAKE_GRAPH = {
    "symbol": "WDC",
    "quarter_end": "2026-03-31",
    "built_at": "2026-06-01",
    "mechanisms": [
        {
            "mechanism": "NAND oversupply correction → gross margin expansion → EPS rerating",
            "intervention": "IF NAND spot price +15% from trough THEN WDC gross margin > 40%",
            "falsification_condition": "IF WDC Q3 gross margin < 38%, thesis broken",
            "horizon_days": 63,
            "timing": "catalyst_imminent",
            "mechanism_type": "supply_demand_inflection",
        }
    ],
}


def _write_fake_cache(cache_dir: Path, symbol: str, quarter_end: str, graph: dict) -> None:
    """Pre-write a cache file so tests can exercise cache-hit paths without LLM."""
    (cache_dir / f"{symbol}_{quarter_end}.json").write_text(json.dumps(graph))


def test_build_graph_cache_hit_skips_llm(tmp_path):
    """When cache file exists, build_graph returns it without calling the LLM."""
    from ascent.causal.dag_builder import build_graph

    _write_fake_cache(tmp_path, "WDC", "2026-03-31", _FAKE_GRAPH)

    call_count = [0]

    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        return json.dumps({"mechanisms": []})

    with patch("ascent.causal.dag_builder.generate_structured", side_effect=mock_generate):
        result = build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    assert call_count[0] == 0, "LLM must not be called when cache file exists"
    assert result["mechanisms"][0]["mechanism"] == _FAKE_GRAPH["mechanisms"][0]["mechanism"]


def test_build_graph_cache_file_path(tmp_path):
    """build_graph must write the cache to {cache_dir}/{symbol}_{quarter_end}.json."""
    from ascent.causal.dag_builder import build_graph

    with patch("ascent.causal.dag_builder.generate_structured",
               return_value=json.dumps(_FAKE_GRAPH)):
        build_graph("AAPL", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    expected = tmp_path / "AAPL_2026-03-31.json"
    assert expected.exists(), f"Cache file not found at {expected}"


def test_build_graph_mechanism_schema(tmp_path):
    """Each mechanism in cached result must have all required fields."""
    from ascent.causal.dag_builder import build_graph

    _write_fake_cache(tmp_path, "WDC", "2026-03-31", _FAKE_GRAPH)
    result = build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    m = result["mechanisms"][0]
    for field in ("mechanism", "intervention", "falsification_condition",
                  "horizon_days", "timing", "mechanism_type"):
        assert field in m, f"Mechanism missing required field: {field}"
    assert m["timing"] in ("priced_in", "not_yet_priced", "catalyst_imminent")
    assert m["mechanism_type"] in (
        "momentum_catalyst", "quality_defensive", "macro_hedge",
        "mean_reversion", "valuation", "supply_demand_inflection",
    )


def test_build_graph_returns_schema(tmp_path):
    """build_graph must return a dict with symbol, quarter_end, built_at, mechanisms."""
    from ascent.causal.dag_builder import build_graph

    _write_fake_cache(tmp_path, "WDC", "2026-03-31", _FAKE_GRAPH)
    result = build_graph("WDC", "2026-03-31", "f", "t", "s", cache_dir=tmp_path)

    assert result["symbol"] == "WDC"
    assert result["quarter_end"] == "2026-03-31"
    assert "built_at" in result
    assert isinstance(result["mechanisms"], list)
    assert len(result["mechanisms"]) == 1


def test_load_or_build_respects_cache_hit(tmp_path):
    """load_or_build returns cached data without LLM call if cache file exists."""
    from ascent.causal.dag_builder import load_or_build

    cached = {
        "symbol": "AAPL", "quarter_end": "2026-03-31",
        "built_at": "2026-05-01",
        "mechanisms": [{"mechanism": "cached", "intervention": "i",
                        "falsification_condition": "f", "horizon_days": 21,
                        "timing": "not_yet_priced", "mechanism_type": "momentum_catalyst"}]
    }
    _write_fake_cache(tmp_path, "AAPL", "2026-03-31", cached)

    result = load_or_build("AAPL", "2026-03-31", cache_dir=tmp_path)

    assert result["mechanisms"][0]["mechanism"] == "cached"


def test_load_or_build_returns_empty_when_no_cache(tmp_path):
    """load_or_build returns empty mechanisms when no cache file exists."""
    from ascent.causal.dag_builder import load_or_build

    result = load_or_build("NOBODY", "2026-03-31", cache_dir=tmp_path)

    assert result["mechanisms"] == []
    assert result["symbol"] == "NOBODY"
