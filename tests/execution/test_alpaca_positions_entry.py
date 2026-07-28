# tests/execution/test_alpaca_positions_entry.py
"""
get_positions() must surface entry price so the stop-loss can evaluate
drawdown-from-entry on the live book.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from ascent.execution import alpaca_broker as ab

_FAKE = [
    {"symbol": "ALGM", "qty": "66.51392", "market_value": "3062.97",
     "current_price": "46.05", "avg_entry_price": "66.370003",
     "unrealized_plpc": "-0.30646"},
    {"symbol": "TLT", "qty": "108.481866", "market_value": "9090.78",
     "current_price": "83.80", "avg_entry_price": "87.41",
     "unrealized_plpc": "-0.04130"},
]


def _mock_get(payload):
    """Patch the inline requests.get used by get_positions()."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return patch.object(ab.requests, "get", return_value=resp)


def test_positions_include_entry_price_and_plpc():
    with _mock_get(_FAKE):
        df = ab.get_positions()
    assert {"avg_entry_price", "unrealized_plpc"} <= set(df.columns)
    row = df.set_index("symbol").loc["ALGM"]
    assert float(row["avg_entry_price"]) == 66.370003
    assert float(row["unrealized_plpc"]) < 0


def test_existing_columns_are_unchanged():
    with _mock_get(_FAKE):
        df = ab.get_positions()
    for col in ("symbol", "qty", "market_value", "current_price", "weight"):
        assert col in df.columns


def test_missing_entry_price_becomes_nan_not_an_error():
    partial = [{"symbol": "X", "qty": "1", "market_value": "10",
                "current_price": "10"}]
    with _mock_get(partial):
        df = ab.get_positions()
    assert pd.isna(df.set_index("symbol").loc["X", "avg_entry_price"])


def test_empty_book_still_exposes_the_new_columns():
    """
    get_positions() early-returns a hardcoded column list when the account
    holds nothing. The stop-loss checks `"avg_entry_price" in pos.columns`,
    so that branch must carry the new columns as well.
    """
    with _mock_get([]):
        df = ab.get_positions()
    assert df.empty
    assert {"avg_entry_price", "unrealized_plpc"} <= set(df.columns)
