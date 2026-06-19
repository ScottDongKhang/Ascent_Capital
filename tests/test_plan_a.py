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

    # The counterfactual DAILY_LOG is an absolute _REPO-based path computed at
    # import time, so monkeypatch.chdir does NOT sandbox it — redirect it to
    # tmp_path explicitly or _log_holdings pollutes the real production log.
    with patch("ascent.execution.alpaca_broker.get_positions", return_value=mock_pos), \
         patch("ascent.execution.alpaca_broker.get_account", return_value=mock_acct), \
         patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG",
               tmp_path / "logs" / "counterfactual_daily.jsonl"), \
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

def test_us_equities_pnl_has_own_log():
    """US equities PnL must go to us_equities_pnl.jsonl, not eod_log.jsonl."""
    from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
    assert str(PNL_LOGS["us_equities"]) == "logs/us_equities_pnl.jsonl", \
        "us_equities must use its own PnL log, not eod_log.jsonl"

def test_attribution_produces_report(tmp_path, monkeypatch):
    """attribution report must return top contributors, drags, and alpha vs SPY."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from unittest.mock import patch
    import pandas as pd
    from datetime import date

    positions = [
        {"symbol": "MPWR", "weight": 0.064, "market_value": 6222.0, "current_price": 1402.0, "qty": 4.4},
        {"symbol": "EWY",  "weight": 0.112, "market_value": 10968.0, "current_price": 147.0, "qty": 74.4},
        {"symbol": "MRK",  "weight": 0.024, "market_value": 2385.0,  "current_price": 115.0, "qty": 20.6},
    ]
    mock_returns = {"MPWR": 0.037, "EWY": 0.018, "MRK": -0.021, "SPY": 0.003}

    with patch("yfinance.download") as mock_dl:
        dates = pd.date_range("2026-04-14", periods=2, freq="B")
        price_data = {s: [100.0, 100.0*(1+r)] for s, r in mock_returns.items()}
        df = pd.DataFrame(price_data, index=dates)
        mock_dl.return_value = pd.concat({"Close": df}, axis=1)

        from ascent.monitoring.attribution import run_attribution
        report = run_attribution(positions, date(2026, 4, 16))

    assert "portfolio_return" in report
    assert "spy_return" in report
    assert "alpha_vs_spy" in report
    assert "top_contributors" in report
    assert "top_drags" in report
    assert len(report["top_contributors"]) >= 1
    assert len(report["top_drags"]) >= 1
    assert report["top_contributors"][0]["symbol"] == "MPWR"
    assert report["top_drags"][0]["symbol"] == "MRK"
