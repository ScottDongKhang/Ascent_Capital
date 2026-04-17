# tests/test_plan_a.py
import json
from pathlib import Path
from datetime import date
from unittest.mock import patch

def test_pnl_entry_includes_spy():
    """Every PnL log entry must include spy_return and alpha fields."""
    from ascent.monitoring.forward_pnl_tracker import _fetch_latest_returns

    with patch("yfinance.download") as mock_dl:
        import pandas as pd
        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        data = pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 204.0], "SPY": [500.0, 502.5]},
            index=dates,
        )
        mock_dl.return_value = pd.concat({"Close": data}, axis=1)
        result = _fetch_latest_returns(["AAPL", "MSFT", "SPY"])

    assert "SPY" in result, "SPY must be in returned dict"
    assert abs(result["SPY"] - 0.005) < 0.0001

def test_holdings_log_has_benchmark(tmp_path, monkeypatch):
    """_log_holdings must write spy_return and alpha_vs_spy."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from unittest.mock import MagicMock, patch
    import pandas as pd

    mock_pos = pd.DataFrame({
        "symbol": ["AAPL"], "qty": [10.0],
        "market_value": [1000.0], "current_price": [100.0], "weight": [1.0]
    })
    mock_acct = {"equity": "101000", "last_equity": "100000", "cash": "0"}

    with patch("ascent.execution.alpaca_broker.get_positions", return_value=mock_pos), \
         patch("ascent.execution.alpaca_broker.get_account", return_value=mock_acct), \
         patch("yfinance.download") as mock_dl:

        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        spy_data = pd.DataFrame({"SPY": [500.0, 502.5]}, index=dates)
        mock_dl.return_value = pd.concat({"Close": spy_data}, axis=1)

        from run_all_agents import _log_holdings
        _log_holdings(date(2026, 4, 16))

    entry = json.loads((tmp_path / "logs" / "holdings_log.jsonl").read_text().strip())
    assert "spy_return" in entry, "holdings_log must have spy_return"
    assert "alpha_vs_spy" in entry, "holdings_log must have alpha_vs_spy"
    assert abs(entry["spy_return"] - 0.005) < 0.001
