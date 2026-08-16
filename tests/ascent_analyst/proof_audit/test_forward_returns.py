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


def test_eligible_dates_handles_tz_aware_index_against_real_universe():
    """Regression test for the tz-aware vs tz-naive TypeError seen on real prices_live data.

    prices_live is pivoted into a DataFrame whose DatetimeIndex is tz-aware
    (America/New_York), e.g. Timestamp('2024-01-02 19:00:00-0500'). The real (unpatched)
    get_universe_on_date compares against tz-naive start_date/end_date columns built by
    build_historical_universe(). Before the fix, this raised:
        TypeError: Invalid comparison between dtype=datetime64[us] and Timestamp
    eligible_dates() must coerce/tz-strip before calling get_universe_on_date so this
    comparison succeeds -- no monkeypatch here, this exercises the real production function.
    """
    dates = pd.date_range(
        "2024-01-02 19:00:00", periods=5, freq="D", tz="America/New_York"
    )
    prices = pd.DataFrame(
        {"AAA": [100, 102, 101, 105, 110], "BBB": [50, 49, 51, 52, 53]},
        index=dates,
    )

    # Must not raise -- this is the real bug: no monkeypatch on get_universe_on_date.
    out = eligible_dates(prices, min_universe_size=1)

    # 2024-01-02 through 2024-01-05 (index[:-1]) are all within the historical universe
    # window (UNIVERSE_START predates 2024, end_date 2099-12-31 for active names), so with
    # min_universe_size=1 every candidate date should be eligible.
    assert len(out) == 4
    assert prices.index[-1] not in out
