"""Smoke tests for ascent/research/hypothesis_registry.py."""

from ascent.research import hypothesis_registry as hr


def test_record_and_find_rejected(tmp_path, monkeypatch):
    registry_path = tmp_path / "hypothesis_registry.jsonl"
    monkeypatch.setattr(hr, "REGISTRY_PATH", registry_path)

    config = {"meanrev": 0.4, "statarb": 0.6}
    hr.record_verdict(
        variant_config=config,
        variant_id="v1_test",
        oos_sharpe=0.1,
        edge=-0.05,
        promoted=False,
        reason="below edge threshold",
    )

    prior = hr.was_previously_rejected(config)
    assert prior is not None
    assert prior["variant_id"] == "v1_test"
    assert prior["promoted"] is False


def test_promoted_variant_is_not_flagged_rejected(tmp_path, monkeypatch):
    registry_path = tmp_path / "hypothesis_registry.jsonl"
    monkeypatch.setattr(hr, "REGISTRY_PATH", registry_path)

    config = {"meanrev": 0.55, "statarb": 0.45}
    hr.record_verdict(
        variant_config=config,
        variant_id="v2_test",
        oos_sharpe=0.9,
        edge=0.2,
        promoted=True,
    )

    assert hr.was_previously_rejected(config) is None


def test_config_hash_stable_across_key_order(tmp_path):
    config_a = {"meanrev": 0.5, "statarb": 0.5}
    config_b = {"statarb": 0.5, "meanrev": 0.5}
    assert hr._config_hash(config_a) == hr._config_hash(config_b)


def test_no_prior_record_returns_none(tmp_path, monkeypatch):
    registry_path = tmp_path / "does_not_exist.jsonl"
    monkeypatch.setattr(hr, "REGISTRY_PATH", registry_path)

    assert hr.was_previously_rejected({"meanrev": 0.5, "statarb": 0.5}) is None
