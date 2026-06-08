# tests/integrations/test_openbb_client.py
from __future__ import annotations
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


def _make_mock_obb_result(symbol: str, dates: list, closes: list) -> MagicMock:
    """Build a mock OBBject that .to_dataframe() returns a price DataFrame."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "close": closes,
        "open": closes,
        "high": closes,
        "low": closes,
        "volume": [1_000_000] * len(closes),
    }).set_index("date")
    mock = MagicMock()
    mock.to_dataframe.return_value = df
    return mock


def test_fetch_symbol_returns_dataframe():
    from ascent.integrations.openbb_client import fetch_symbol
    mock_result = _make_mock_obb_result(
        "AAPL",
        ["2026-01-02", "2026-01-03", "2026-01-06"],
        [220.0, 221.5, 223.0],
    )
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")
    assert result is not None
    assert not result.empty
    assert "close" in result.columns
    assert "symbol" in result.columns


def test_fetch_symbol_falls_back_on_tiingo_failure():
    from ascent.integrations.openbb_client import fetch_symbol
    mock_result = _make_mock_obb_result(
        "AAPL",
        ["2026-01-02", "2026-01-03"],
        [220.0, 221.5],
    )
    call_count = {"n": 0}

    def side_effect(symbol, start_date, end_date, provider):
        call_count["n"] += 1
        if provider == "tiingo":
            raise Exception("tiingo failed")
        return mock_result

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = side_effect
        mock_obb_fn.return_value = mock_obb
        with patch.dict("os.environ", {"TIINGO_TOKEN": "fake_token"}):
            result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")

    assert result is not None
    assert call_count["n"] == 2  # tried tiingo, then yfinance


def test_fetch_symbol_returns_none_when_both_fail():
    from ascent.integrations.openbb_client import fetch_symbol
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = Exception("all providers failed")
        mock_obb_fn.return_value = mock_obb
        result = fetch_symbol("AAPL", "2026-01-02", "2026-01-06")
    assert result is None


def test_fetch_return_computes_forward_return():
    from ascent.integrations.openbb_client import fetch_return
    mock_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07",
                                 "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13",
                                 "2026-01-14", "2026-01-15", "2026-01-16"]),
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "symbol": ["CAT"] * 11,
        "source": ["yfinance_hub"] * 11,
    })
    with patch("ascent.integrations.openbb_client.fetch_symbol", return_value=mock_df):
        result = fetch_return("CAT", "2026-01-02", 10)
    assert result is not None
    assert abs(result - 0.10) < 0.001  # 110/100 - 1 = 10%


def test_fetch_return_returns_none_on_failure():
    from ascent.integrations.openbb_client import fetch_return
    with patch("ascent.integrations.openbb_client.fetch_symbol", return_value=None):
        result = fetch_return("CAT", "2026-01-02", 10)
    assert result is None


# ---------- Task 3 tests ----------

def test_get_live_macro_returns_dict():
    from ascent.integrations.openbb_client import get_live_macro
    mock_df = pd.DataFrame({"value": [5.33]}, index=pd.to_datetime(["2026-06-06"]))
    mock_df.index.name = "date"

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dataframe.return_value = mock_df
        mock_obb.economy.fred.series.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_live_macro()

    assert isinstance(result, dict)
    assert len(result) > 0


def test_get_live_macro_falls_back_to_parquet(tmp_path):
    from ascent.integrations.openbb_client import get_live_macro
    fake = pd.DataFrame({
        "fed_funds_rate": [5.33],
        "treasury_10y": [4.25],
    }, index=pd.to_datetime(["2026-06-06"]))
    fake.index.name = "date"
    cache = tmp_path / "macro_live.parquet"
    fake.to_parquet(cache)

    with patch("ascent.integrations.openbb_client._get_obb", side_effect=Exception("obb down")):
        with patch("ascent.integrations.openbb_client._MACRO_CACHE_PATH", cache):
            result = get_live_macro()

    assert "fed_funds_rate" in result
    assert abs(result["fed_funds_rate"] - 5.33) < 0.01


def test_get_options_snapshot_returns_per_symbol():
    from ascent.integrations.openbb_client import get_options_snapshot
    mock_chain_df = pd.DataFrame({
        "strike": [200.0, 210.0, 220.0, 230.0, 240.0],
        "expiration": ["2026-07-18"] * 5,
        "option_type": ["put", "put", "call", "call", "call"],
        "implied_volatility": [0.28, 0.25, 0.22, 0.24, 0.26],
        "volume": [500, 300, 400, 200, 150],
        "underlying_price": [220.0] * 5,
    })
    mock_result = MagicMock()
    mock_result.to_dataframe.return_value = mock_chain_df

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.derivatives.options.chains.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_options_snapshot(["CAT"])

    assert "CAT" in result
    entry = result["CAT"]
    assert "put_call_ratio" in entry
    assert "atm_iv" in entry
    assert "iv_skew" in entry
    assert entry["put_call_ratio"] >= 0
    assert entry["atm_iv"] > 0


def test_get_options_snapshot_handles_failure():
    from ascent.integrations.openbb_client import get_options_snapshot
    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.derivatives.options.chains.side_effect = Exception("cboe down")
        mock_obb_fn.return_value = mock_obb
        result = get_options_snapshot(["CAT"])
    assert "CAT" in result
    assert result["CAT"].get("unavailable") is True


def test_get_cot_snapshot_returns_dict():
    from ascent.integrations.openbb_client import get_cot_snapshot
    mock_cot_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-06"]),
        "noncomm_positions_long_all": [187420],
        "noncomm_positions_short_all": [62180],
        "open_interest_all": [3200000],
    })
    mock_result = MagicMock()
    mock_result.to_dataframe.return_value = mock_cot_df

    with patch("ascent.integrations.openbb_client._get_obb") as mock_obb_fn:
        mock_obb = MagicMock()
        mock_obb.regulators.cftc.cot.return_value = mock_result
        mock_obb_fn.return_value = mock_obb
        result = get_cot_snapshot()

    assert isinstance(result, dict)
    assert "net_noncommercial_long" in result
    assert "pct_long_noncommercial" in result
    assert "as_of_date" in result


def test_get_cot_snapshot_returns_none_on_failure():
    from ascent.integrations.openbb_client import get_cot_snapshot
    with patch("ascent.integrations.openbb_client._get_obb", side_effect=Exception("cftc down")):
        result = get_cot_snapshot()
    assert result is None
