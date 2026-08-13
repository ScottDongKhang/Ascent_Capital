"""One canonical name per question. Five artifacts answered one question before."""
import dataclasses
import json

import pytest

from ascent.analyst.catalog import registry


EXPECTED = {
    "counterfactual.track_astar",
    "counterfactual.track_a",
    "counterfactual.track_b",
    "counterfactual.track_c",
    "counterfactual.track_d",
}


def test_five_tracks_registered():
    assert EXPECTED.issubset(set(registry.names()))


def test_every_descriptor_is_complete():
    for name in registry.names():
        s = registry.describe(name)
        assert s.name == name
        assert s.description
        assert s.index_kind == "market_trading_day"
        assert s.coverage
        assert s.provenance


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        registry.describe("counterfactual.does_not_exist")


def test_names_are_unique_per_source_column():
    seen = {}
    for name in registry.names():
        s = registry.describe(name)
        key = (str(s.source), s.column)
        assert key not in seen, f"{name} and {seen[key]} both address {key}"
        seen[key] = name


def test_load_returns_dated_floats(tmp_path, monkeypatch):
    log = tmp_path / "counterfactual_daily.jsonl"
    log.write_text(
        json.dumps({"date": "2026-06-11", "track_b_return": 0.002}) + "\n"
        + json.dumps({"date": "2026-06-10", "track_b_return": 0.001}) + "\n"
        + json.dumps({"date": "2026-06-12", "track_b_return": None}) + "\n"
    )
    monkeypatch.setitem(
        registry.SERIES, "counterfactual.track_b",
        dataclasses.replace(registry.SERIES["counterfactual.track_b"], source=log),
    )
    out = registry.load("counterfactual.track_b")

    assert list(out.index) == sorted(out.index), "index must be sorted"
    assert len(out) == 2, "null values are dropped"
    assert out.iloc[0] == pytest.approx(0.001)


def test_real_sources_resolve():
    """Every registered source file must exist in the repo as it stands."""
    missing = [n for n in registry.names() if not registry.describe(n).source.exists()]
    assert not missing, f"unresolvable sources: {missing}"
