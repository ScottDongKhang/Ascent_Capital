# tests/strategy/test_falsifier_registry.py
"""
Falsifier enforcement layer: registry build, code-path condition evaluation,
fire-once semantics, judge falsifiers, trim bookkeeping.

All LLM calls are mocked — the registry must work (and fail safe) without them.
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import ascent.strategy.falsifier_registry as fr


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Redirect registry + caches into tmp; disable LLM."""
    monkeypatch.setattr(fr, "REGISTRY_PATH", tmp_path / "active_falsifiers.json")
    # Default: Haiku structuring unavailable → news fallback
    monkeypatch.setattr(
        fr, "_structure_with_haiku",
        lambda items, today: [
            {"id": f"{src}-{sym}-{i}", "symbol": sym, "source": src,
             "kind": "news", "condition": {"keywords": []}, "raw_text": txt[:300]}
            for i, (sym, txt, src) in enumerate(items)
        ],
    )
    return tmp_path


def _price_panel(symbols, n=30, start="2026-06-01", drops=None):
    """Flat prices, with optional {symbol: total_return} applied linearly."""
    dates = pd.bdate_range(start, periods=n)
    data = {}
    for s in symbols:
        path = np.full(n, 100.0)
        if drops and s in drops:
            path = 100.0 * np.linspace(1.0, 1.0 + drops[s], n)
        data[s] = path
    return pd.DataFrame(data, index=dates)


class TestBuildRegistry:
    def test_build_from_prethesis_and_premortem(self, tmp_registry):
        today = date(2026, 6, 10)
        prethesis = {"high_conviction_names": [
            {"symbol": "CAT", "what_would_change_my_mind": "Infrastructure orders fall"},
            {"symbol": "WMT", "thesis": "no falsifier given"},  # skipped
        ]}
        thesis = {"pre_mortem": "Rate spike hits all cyclicals at once"}
        n = fr.build_registry(today, prethesis_raw=prethesis, thesis=thesis)
        assert n == 2
        reg = json.loads(fr.REGISTRY_PATH.read_text())
        assert reg["as_of"] == "2026-06-10"
        symbols = {f["symbol"] for f in reg["falsifiers"]}
        assert symbols == {"CAT", "__PORTFOLIO__"}
        assert all(not f["fired"] for f in reg["falsifiers"])

    def test_build_replaces_previous_registry(self, tmp_registry):
        today = date(2026, 6, 10)
        fr.build_registry(today, prethesis_raw={"high_conviction_names": [
            {"symbol": "OLD", "what_would_change_my_mind": "x"}]})
        fr.build_registry(today, prethesis_raw={"high_conviction_names": [
            {"symbol": "NEW", "what_would_change_my_mind": "y"}]})
        reg = json.loads(fr.REGISTRY_PATH.read_text())
        assert {f["symbol"] for f in reg["falsifiers"]} == {"NEW"}

    def test_empty_inputs_zero_entries(self, tmp_registry):
        assert fr.build_registry(date(2026, 6, 10)) == 0


class TestCheckAll:
    def _seed(self, entries, as_of="2026-06-01"):
        fr._save_registry({"as_of": as_of, "falsifiers": entries})

    def test_price_condition_fires(self, tmp_registry, monkeypatch):
        self._seed([{
            "id": "p1", "symbol": "PK", "source": "prethesis", "kind": "price",
            "condition": {"metric": "ret_since_rebalance", "op": "<",
                          "value": -0.07, "since": "2026-06-01"},
            "raw_text": "PK drops 7%", "expires": "2026-07-01",
            "fired": False, "trimmed": False,
        }])
        panel = _price_panel(["PK", "SPY"], drops={"PK": -0.10})
        monkeypatch.setattr(fr, "_load_close_panel", lambda: panel)
        fired = fr.check_all(date(2026, 6, 20))
        assert len(fired) == 1 and fired[0]["symbol"] == "PK"
        assert fired[0]["fired_value"] == pytest.approx(-0.10, abs=0.01)

    def test_price_condition_not_met(self, tmp_registry, monkeypatch):
        self._seed([{
            "id": "p1", "symbol": "PK", "source": "prethesis", "kind": "price",
            "condition": {"metric": "ret_since_rebalance", "op": "<",
                          "value": -0.07, "since": "2026-06-01"},
            "raw_text": "PK drops 7%", "expires": "2026-07-01",
            "fired": False, "trimmed": False,
        }])
        panel = _price_panel(["PK"], drops={"PK": -0.02})
        monkeypatch.setattr(fr, "_load_close_panel", lambda: panel)
        assert fr.check_all(date(2026, 6, 20)) == []

    def test_fires_only_once(self, tmp_registry, monkeypatch):
        self._seed([{
            "id": "p1", "symbol": "PK", "source": "prethesis", "kind": "price",
            "condition": {"metric": "ret_since_rebalance", "op": "<",
                          "value": -0.07, "since": "2026-06-01"},
            "raw_text": "PK drops 7%", "expires": "2026-07-01",
            "fired": False, "trimmed": False,
        }])
        panel = _price_panel(["PK"], drops={"PK": -0.10})
        monkeypatch.setattr(fr, "_load_close_panel", lambda: panel)
        assert len(fr.check_all(date(2026, 6, 20))) == 1
        assert fr.check_all(date(2026, 6, 21)) == []  # already fired

    def test_relative_price_judge_condition(self, tmp_registry, monkeypatch):
        fr.add_judge_falsifier("EWY", "EWY underperforms SPY", date(2026, 6, 1))
        panel = _price_panel(["EWY", "SPY"], drops={"EWY": -0.06, "SPY": 0.02})
        monkeypatch.setattr(fr, "_load_close_panel", lambda: panel)
        fired = fr.check_all(date(2026, 6, 10))  # within the 14-day expiry window
        assert len(fired) == 1
        assert fired[0]["source"] == "judge"
        assert fired[0]["fired_value"] < -0.03

    def test_expired_falsifier_ignored(self, tmp_registry, monkeypatch):
        self._seed([{
            "id": "p1", "symbol": "PK", "source": "prethesis", "kind": "price",
            "condition": {"metric": "ret_since_rebalance", "op": "<",
                          "value": -0.07, "since": "2026-06-01"},
            "raw_text": "PK drops 7%", "expires": "2026-06-05",  # expired
            "fired": False, "trimmed": False,
        }])
        panel = _price_panel(["PK"], drops={"PK": -0.20})
        monkeypatch.setattr(fr, "_load_close_panel", lambda: panel)
        assert fr.check_all(date(2026, 6, 20)) == []

    def test_macro_condition(self, tmp_registry, monkeypatch):
        self._seed([{
            "id": "m1", "symbol": "__PORTFOLIO__", "source": "pre_mortem",
            "kind": "macro",
            "condition": {"metric": "vix", "op": ">", "value": 30.0},
            "raw_text": "VIX above 30", "expires": "2026-07-01",
            "fired": False, "trimmed": False,
        }])
        monkeypatch.setattr(fr, "_load_close_panel", lambda: None)
        monkeypatch.setattr(fr, "_latest_macro", lambda m: 34.5)
        fired = fr.check_all(date(2026, 6, 20))
        assert len(fired) == 1 and fired[0]["fired_value"] == 34.5

    def test_no_llm_news_does_not_fire(self, tmp_registry, monkeypatch):
        """News conditions need the LLM; without it they must stay silent."""
        self._seed([{
            "id": "n1", "symbol": "CAT", "source": "prethesis", "kind": "news",
            "condition": {"keywords": ["guidance cut"]},
            "raw_text": "CAT cuts guidance", "expires": "2026-07-01",
            "fired": False, "trimmed": False,
        }])
        monkeypatch.setattr(fr, "_load_close_panel", lambda: None)
        monkeypatch.setattr(fr, "_check_news_batch", lambda b, n: [])
        assert fr.check_all(date(2026, 6, 20),
                            news_context={"CAT": ["CAT cuts FY guidance"]}) == []

    def test_mark_trimmed(self, tmp_registry):
        self._seed([{
            "id": "p1", "symbol": "PK", "source": "prethesis", "kind": "price",
            "condition": {}, "raw_text": "", "expires": "2026-07-01",
            "fired": True, "trimmed": False,
        }])
        fr.mark_trimmed("p1")
        reg = json.loads(fr.REGISTRY_PATH.read_text())
        assert reg["falsifiers"][0]["trimmed"] is True


class TestHelpers:
    def test_ret_since(self):
        panel = _price_panel(["AAA"], drops={"AAA": -0.10})
        r = fr._ret_since(panel, "AAA", "2026-06-01")
        assert r == pytest.approx(-0.10, abs=0.01)
        assert fr._ret_since(panel, "MISSING", "2026-06-01") is None
        assert fr._ret_since(None, "AAA", "2026-06-01") is None

    def test_compare(self):
        assert fr._compare(-0.08, "<", -0.07)
        assert not fr._compare(-0.05, "<", -0.07)
        assert fr._compare(35.0, ">", 30.0)
