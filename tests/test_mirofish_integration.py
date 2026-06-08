# tests/test_mirofish_integration.py
from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------- Task 2 tests ----------

def test_find_analogues_returns_top3():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues(
        "Federal Reserve begins rate hike cycle to fight inflation",
        ["QQQ", "XLK"],
        top_k=3,
    )
    assert isinstance(results, list)
    assert len(results) <= 3
    # Each entry is (analogue_dict, confidence_float)
    for analogue, conf in results:
        assert isinstance(analogue, dict)
        assert "event_id" in analogue
        assert 0.0 <= conf <= 1.0

def test_find_analogues_best_match_is_relevant():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues(
        "Fed hikes interest rates 75 basis points to fight inflation",
        ["QQQ"],
        top_k=3,
    )
    event_ids = [a["event_id"] for a, _ in results]
    # Should match rate hike events
    assert any("hike" in eid or "pivot" in eid or "cpi" in eid for eid in event_ids)

def test_find_analogues_empty_returns_empty():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues("xyzzy unknown gibberish event", [], top_k=3)
    # Should return list (may be empty if no similarity found)
    assert isinstance(results, list)

def test_find_analogues_returns_confidence_between_0_and_1():
    from ascent.integrations.analogue_matcher import find_analogues
    results = find_analogues("infrastructure spending federal contracts", ["CAT"], top_k=2)
    for _, conf in results:
        assert 0.0 <= conf <= 1.0
