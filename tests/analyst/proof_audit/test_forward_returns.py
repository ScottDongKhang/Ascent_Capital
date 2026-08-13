"""Forward returns must be strictly next-day -- never same-day, never look back."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.forward_returns import (
    eligible_dates,
    forward_return_matrix,
)


def _toy_prices():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"AAA": [100, 102, 101, 105, 110], "BBB": [50, 49, 51, 52, 53]},
        index=dates,
    )


def test_forward_return_is_next_day_not_same_day():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    expected_day0_aaa = (102 - 100) / 100
    assert fwd.iloc[0]["AAA"] == pytest.approx(expected_day0_aaa)


def test_last_row_is_nan_no_lookahead():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    assert fwd.iloc[-1].isna().all()


def test_index_and_columns_match_input():
    prices = _toy_prices()
    fwd = forward_return_matrix(prices)
    assert list(fwd.index) == list(prices.index)
    assert list(fwd.columns) == list(prices.columns)


def test_eligible_dates_excludes_final_date(monkeypatch):
    prices = _toy_prices()

    def fake_universe(date, universe_df=None):
        return ["AAA", "BBB"] * 15  # 30 symbols, always eligible

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date", fake_universe
    )
    dates = eligible_dates(prices, min_universe_size=20)
    assert prices.index[-1] not in dates
    assert len(dates) == 4


def test_eligible_dates_respects_min_universe_size(monkeypatch):
    prices = _toy_prices()

    def fake_universe(date, universe_df=None):
        return ["AAA"]  # only 1 symbol -- never eligible at threshold 20

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date", fake_universe
    )
    dates = eligible_dates(prices, min_universe_size=20)
    assert dates == []
