# tests/data/test_new_ingest.py
from __future__ import annotations
import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── CBOE options tests ─────────────────────────────────────────────────────────

def test_fetch_cboe_options_returns_dataframe():
    from ascent.data.ingest.cboe_options import fetch_cboe_options_row
    mock_snapshot = {
        "CAT": {
            "put_call_ratio": 0.82,
            "atm_iv": 0.215,
            "iv_skew": 0.031,
            "iv_rank_52w": 34,
            "unavailable": False,
        }
    }
    with patch("ascent.data.ingest.cboe_options.get_options_snapshot", return_value=mock_snapshot):
        result = fetch_cboe_options_row("CAT", "2026-06-06")
    assert result is not None
    assert result["symbol"] == "CAT"
    assert result["date"] == "2026-06-06"
    assert "put_call_ratio" in result
    assert "atm_iv" in result
    assert "iv_skew" in result


def test_fetch_cboe_options_returns_none_when_unavailable():
    from ascent.data.ingest.cboe_options import fetch_cboe_options_row
    mock_snapshot = {"CAT": {"unavailable": True}}
    with patch("ascent.data.ingest.cboe_options.get_options_snapshot", return_value=mock_snapshot):
        result = fetch_cboe_options_row("CAT", "2026-06-06")
    assert result is None


def test_update_options_cache_appends_new_rows(tmp_path):
    from ascent.data.ingest.cboe_options import update_options_cache
    mock_rows = [
        {"symbol": "CAT", "date": "2026-06-06", "put_call_ratio": 0.82,
         "atm_iv": 0.215, "iv_skew": 0.031, "iv_rank_52w": None, "source": "cboe"},
        {"symbol": "MRK", "date": "2026-06-06", "put_call_ratio": 1.12,
         "atm_iv": 0.198, "iv_skew": -0.025, "iv_rank_52w": None, "source": "cboe"},
    ]
    cache_path = tmp_path / "options_flow.parquet"

    with patch("ascent.data.ingest.cboe_options.fetch_cboe_options_row",
               side_effect=lambda sym, dt: next(
                   (r for r in mock_rows if r["symbol"] == sym), None)):
        update_options_cache(["CAT", "MRK"], "2026-06-06", cache_path=cache_path)

    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert len(df) == 2
    assert set(df["symbol"].unique()) == {"CAT", "MRK"}


def test_update_options_cache_does_not_duplicate(tmp_path):
    from ascent.data.ingest.cboe_options import update_options_cache
    existing = pd.DataFrame([{
        "symbol": "CAT", "date": pd.Timestamp("2026-06-06"),
        "put_call_ratio": 0.80, "atm_iv": 0.200, "iv_skew": 0.020,
        "iv_rank_52w": None, "source": "cboe",
    }])
    cache_path = tmp_path / "options_flow.parquet"
    existing.to_parquet(cache_path, index=False)

    new_row = {"symbol": "CAT", "date": "2026-06-06", "put_call_ratio": 0.82,
               "atm_iv": 0.215, "iv_skew": 0.031, "iv_rank_52w": None, "source": "cboe"}

    with patch("ascent.data.ingest.cboe_options.fetch_cboe_options_row", return_value=new_row):
        update_options_cache(["CAT"], "2026-06-06", cache_path=cache_path)

    df = pd.read_parquet(cache_path)
    assert len(df[df["symbol"] == "CAT"]) == 1
