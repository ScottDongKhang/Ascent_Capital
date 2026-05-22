import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_fundamentals():
    rows = []
    for sym in ["AAPL", "MSFT"]:
        for q in ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]:
            rows.append({
                "symbol": sym, "date": q,
                "gross_profit": 100.0, "total_assets": 500.0,
                "net_income": 50.0, "op_cashflow": 60.0,
            })
    return pd.DataFrame(rows)


def test_seed_writes_cache(tmp_path):
    from scripts.seed_llm_cache import seed_cache

    fundamentals = _make_fundamentals()
    cache_path = tmp_path / "llm_fundamental_cache.json"

    mock_result = {"direction": "UP", "confidence": 0.8, "key_trend": "strong", "uncertainty": "rates"}

    with patch("ascent.alpha.llm_fundamental._call_llm", return_value=mock_result):
        seed_cache(fundamentals, cache_path=cache_path)

    assert cache_path.exists()
    cache = json.loads(cache_path.read_text())
    assert len(cache) >= 2  # at least one entry per symbol


def test_seed_handles_no_fundamentals(tmp_path):
    from scripts.seed_llm_cache import seed_cache

    cache_path = tmp_path / "llm_fundamental_cache.json"
    seed_cache(pd.DataFrame(), cache_path=cache_path)  # must not raise
    # No cache written when no data
    assert not cache_path.exists()
