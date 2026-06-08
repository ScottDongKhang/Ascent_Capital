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


# ── Fama-French factor tests ───────────────────────────────────────────────────

def test_fetch_ff_factors_returns_dataframe():
    from ascent.data.ingest.famafrench_factors import fetch_ff_factors
    import numpy as np

    mock_df_5f = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"]),
        "mkt_rf": [0.0082, -0.0031, 0.0055],
        "smb":    [0.0021, 0.0010, -0.0008],
        "hml":    [-0.0015, 0.0020, 0.0011],
        "rmw":    [0.0005, -0.0003, 0.0007],
        "cma":    [0.0003, 0.0001, -0.0002],
        "rf":     [0.0002, 0.0002, 0.0002],
    })
    mock_df_mom = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-04", "2026-06-05", "2026-06-06"]),
        "mom": [0.0032, -0.0041, 0.0018],
    })

    def mock_ff_call(*args, **kwargs):
        factor = kwargs.get("factor", args[0] if args else "5_factors")
        result = MagicMock()
        result.to_dataframe.return_value = (mock_df_5f if factor == "5_factors" else mock_df_mom).copy()
        return result

    with patch("ascent.data.ingest.famafrench_factors._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.famafrench.factors.side_effect = mock_ff_call
        mock_obb_fn.return_value = mock_obb
        df = fetch_ff_factors(start="2026-06-01")

    assert df is not None
    assert not df.empty
    for col in ("mkt_rf", "smb", "hml", "rmw", "cma", "mom"):
        assert col in df.columns


def test_update_ff_cache_writes_parquet(tmp_path):
    from ascent.data.ingest.famafrench_factors import update_ff_cache
    mock_df = pd.DataFrame({
        "mkt_rf": [0.008, -0.003],
        "smb":    [0.002, 0.001],
        "hml":    [-0.001, 0.002],
        "rmw":    [0.001, -0.001],
        "cma":    [0.000,  0.000],
        "mom":    [0.003, -0.004],
    }, index=pd.date_range("2026-06-04", periods=2, freq="B"))
    mock_df.index.name = "date"
    cache_path = tmp_path / "famafrench_factors.parquet"
    with patch("ascent.data.ingest.famafrench_factors.fetch_ff_factors", return_value=mock_df):
        update_ff_cache(cache_path=cache_path)
    assert cache_path.exists()
    df = pd.read_parquet(cache_path)
    assert "mkt_rf" in df.columns
    assert len(df) == 2


# ── Factor loadings feature tests ──────────────────────────────────────────────

def test_factor_loadings_returns_per_symbol_betas():
    from ascent.features.feature_defs import factor_loadings
    import numpy as np

    dates = pd.date_range("2026-01-02", periods=80, freq="B")
    np.random.seed(42)
    factor_df = pd.DataFrame({
        "mkt_rf": np.random.normal(0.0005, 0.01, 80),
        "smb":    np.random.normal(0.0001, 0.005, 80),
        "mom":    np.random.normal(0.0002, 0.007, 80),
    }, index=dates)

    returns_df = pd.DataFrame({
        "CAT": factor_df["mkt_rf"] * 1.2 + np.random.normal(0, 0.003, 80),
        "MRK": factor_df["mkt_rf"] * 0.8 + np.random.normal(0, 0.003, 80),
    }, index=dates)

    result = factor_loadings(returns_df, factor_df, window=63)

    assert isinstance(result, dict)
    assert "mkt_rf" in result
    beta_df = result["mkt_rf"]
    assert "CAT" in beta_df.columns
    assert "MRK" in beta_df.columns
    cat_beta = beta_df["CAT"].dropna().iloc[-1]
    mrk_beta = beta_df["MRK"].dropna().iloc[-1]
    assert cat_beta > mrk_beta


def test_factor_loadings_returns_empty_when_no_data():
    from ascent.features.feature_defs import factor_loadings

    returns_df = pd.DataFrame({"CAT": [0.01, -0.02]},
                               index=pd.date_range("2026-01-02", periods=2, freq="B"))
    factor_df = pd.DataFrame({"mkt_rf": [0.005, -0.003]},
                              index=pd.date_range("2025-01-02", periods=2, freq="B"))

    result = factor_loadings(returns_df, factor_df, window=63)
    for df in result.values():
        assert df.dropna().empty
