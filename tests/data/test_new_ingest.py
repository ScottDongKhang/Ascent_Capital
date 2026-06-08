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


# ── CFTC positioning tests ─────────────────────────────────────────────────────

def test_fetch_cot_returns_dataframe_row():
    from ascent.data.ingest.cftc_positioning import fetch_cot_row
    mock_snapshot = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",
    }
    with patch("ascent.data.ingest.cftc_positioning.get_cot_snapshot", return_value=mock_snapshot):
        result = fetch_cot_row()
    assert result is not None
    assert "net_noncommercial_long" in result
    assert "as_of_date" in result
    assert result["net_noncommercial_long"] == 125240


def test_fetch_cot_returns_none_on_failure():
    from ascent.data.ingest.cftc_positioning import fetch_cot_row
    with patch("ascent.data.ingest.cftc_positioning.get_cot_snapshot", return_value=None):
        result = fetch_cot_row()
    assert result is None


def test_update_cot_cache_appends_row(tmp_path):
    from ascent.data.ingest.cftc_positioning import update_cot_cache
    mock_row = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",
    }
    cache_path = tmp_path / "cftc_positioning.parquet"
    with patch("ascent.data.ingest.cftc_positioning.fetch_cot_row", return_value=mock_row):
        update_cot_cache(cache_path=cache_path)
    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert len(df) == 1
    assert df.iloc[0]["net_noncommercial_long"] == 125240


def test_update_cot_cache_deduplicates(tmp_path):
    from ascent.data.ingest.cftc_positioning import update_cot_cache
    existing = pd.DataFrame([{
        "as_of_date": "2026-06-06",
        "net_noncommercial_long": 120000,
        "noncomm_long": 180000,
        "noncomm_short": 60000,
        "pct_long_noncommercial": 60.0,
        "open_interest": 3000000,
    }])
    cache_path = tmp_path / "cftc_positioning.parquet"
    existing.to_parquet(cache_path, index=False)

    mock_row = {
        "net_noncommercial_long": 125240,
        "noncomm_long": 187420,
        "noncomm_short": 62180,
        "pct_long_noncommercial": 63.2,
        "open_interest": 3200000,
        "as_of_date": "2026-06-06",
    }
    with patch("ascent.data.ingest.cftc_positioning.fetch_cot_row", return_value=mock_row):
        update_cot_cache(cache_path=cache_path)

    df = pd.read_parquet(cache_path)
    assert len(df) == 1


def test_get_latest_cot_reads_cache(tmp_path):
    from ascent.data.ingest.cftc_positioning import get_latest_cot
    df = pd.DataFrame([
        {"as_of_date": "2026-05-30", "net_noncommercial_long": 110000,
         "noncomm_long": 170000, "noncomm_short": 60000,
         "pct_long_noncommercial": 56.7, "open_interest": 3000000},
        {"as_of_date": "2026-06-06", "net_noncommercial_long": 125240,
         "noncomm_long": 187420, "noncomm_short": 62180,
         "pct_long_noncommercial": 63.2, "open_interest": 3200000},
    ])
    cache_path = tmp_path / "cftc_positioning.parquet"
    df.to_parquet(cache_path, index=False)
    result = get_latest_cot(cache_path=cache_path)
    assert result is not None
    assert result["as_of_date"] == "2026-06-06"
    assert result["net_noncommercial_long"] == 125240
