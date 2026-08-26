"""
Tests for ascent/research/walk_forward_runner.py::apply_delisting_terminal_credit
and the ascent/data/universe.py::DELISTING_TERMINAL_TERMS data it reads.

Uses small synthetic price panels -- not the production price cache -- so these
tests are fast and self-contained. Covers:
  - a cash-deal symbol gets the correct credited return
  - a stock-deal symbol's return correctly incorporates the acquirer's price
  - the position stops contributing to subsequent folds (no double-count/leak)
  - a fold that predates the deal is unaffected (no look-ahead)
"""
import pandas as pd
import pytest

from ascent.data.universe import DELISTING_TERMINAL_TERMS
from ascent.research.walk_forward_runner import apply_delisting_terminal_credit


def _dates(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


@pytest.fixture
def cash_symbol():
    """Pick a real cash-deal symbol from DELISTING_TERMINAL_TERMS (e.g. JNPR)."""
    for sym, terms in DELISTING_TERMINAL_TERMS.items():
        if terms["deal_type"] == "cash":
            return sym, terms
    pytest.fail("no cash-deal symbol found in DELISTING_TERMINAL_TERMS")


@pytest.fixture
def stock_symbol():
    """Pick a real stock-deal symbol from DELISTING_TERMINAL_TERMS (e.g. PXD/XOM)."""
    for sym, terms in DELISTING_TERMINAL_TERMS.items():
        if terms["deal_type"] == "stock":
            return sym, terms
    pytest.fail("no stock-deal symbol found in DELISTING_TERMINAL_TERMS")


def test_cash_deal_symbol_credited_correct_terminal_value(cash_symbol):
    sym, terms = cash_symbol
    idx = _dates(30)
    closure_day = pd.Timestamp(terms["removed_date"])
    # closure_day may not be a business day in our synthetic index -- build the
    # index so it definitely spans the closure date, using bdate_range anchored
    # around it instead.
    idx = pd.bdate_range(closure_day - pd.Timedelta(days=20), periods=40)

    last_price = 55.0
    close = pd.DataFrame(index=idx, columns=[sym, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[sym] = float("nan")
    # Real price data exists up through the last trading day before closure,
    # then goes NaN (as if the symbol simply stopped being fetchable).
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]  # leave the closure day itself NaN
    close.loc[idx <= last_real_date, sym] = last_price

    credited_close, _ = apply_delisting_terminal_credit(close, None)

    # The nearest trading day <= removed_date should now carry the cash terminal value.
    nearest = idx[idx <= closure_day].max()
    assert credited_close.loc[nearest, sym] == pytest.approx(terms["cash_amount"])

    # And the implied return from last real price to terminal value is correct.
    day_ret = credited_close.loc[nearest, sym] / last_price - 1.0
    expected_ret = terms["cash_amount"] / last_price - 1.0
    assert day_ret == pytest.approx(expected_ret)


def test_stock_deal_symbol_incorporates_acquirer_price(stock_symbol):
    sym, terms = stock_symbol
    acquirer = terms["acquirer_symbol"]
    closure_day = pd.Timestamp(terms["removed_date"])
    close_date  = pd.Timestamp(terms["close_date"])
    idx = pd.bdate_range(min(closure_day, close_date) - pd.Timedelta(days=25), periods=50)

    last_price     = 40.0
    acquirer_price = 110.0

    close = pd.DataFrame(index=idx, columns=[sym, acquirer, "SPY"], dtype=float)
    close["SPY"]     = 100.0
    close[acquirer]  = acquirer_price
    close[sym]       = float("nan")
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]
    close.loc[idx <= last_real_date, sym] = last_price

    credited_close, _ = apply_delisting_terminal_credit(close, None)

    nearest = idx[idx <= closure_day].max()
    expected_value = terms["exchange_ratio"] * acquirer_price
    if terms["deal_type"] == "cash_and_stock":
        expected_value += terms["cash_amount"]
    assert credited_close.loc[nearest, sym] == pytest.approx(expected_value)

    day_ret = credited_close.loc[nearest, sym] / last_price - 1.0
    expected_ret = expected_value / last_price - 1.0
    assert day_ret == pytest.approx(expected_ret)


def test_stock_deal_falls_back_to_nearest_prior_acquirer_price(stock_symbol):
    """If the acquirer has no price exactly on close_date, use the nearest prior day."""
    sym, terms = stock_symbol
    acquirer = terms["acquirer_symbol"]
    closure_day = pd.Timestamp(terms["removed_date"])
    close_date  = pd.Timestamp(terms["close_date"])
    idx = pd.bdate_range(min(closure_day, close_date) - pd.Timedelta(days=25), periods=50)

    last_price = 40.0
    close = pd.DataFrame(index=idx, columns=[sym, acquirer, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[acquirer] = float("nan")
    # Acquirer only has a price several days before close_date (simulating a gap
    # on the exact close_date), and nothing after.
    lookup_idx = idx[idx <= close_date]
    acquirer_last_known_date = lookup_idx[-3]
    close.loc[idx <= acquirer_last_known_date, acquirer] = 110.0

    close[sym] = float("nan")
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]
    close.loc[idx <= last_real_date, sym] = last_price

    credited_close, _ = apply_delisting_terminal_credit(close, None)

    nearest = idx[idx <= closure_day].max()
    expected_value = terms["exchange_ratio"] * 110.0
    if terms["deal_type"] == "cash_and_stock":
        expected_value += terms["cash_amount"]
    assert credited_close.loc[nearest, sym] == pytest.approx(expected_value)


def test_position_does_not_carry_forward_after_closure(cash_symbol):
    """After the closure day, the symbol's price stays NaN -- no phantom value drift,
    no double-counted return on subsequent days."""
    sym, terms = cash_symbol
    closure_day = pd.Timestamp(terms["removed_date"])
    idx = pd.bdate_range(closure_day - pd.Timedelta(days=20), periods=40)

    close = pd.DataFrame(index=idx, columns=[sym, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[sym] = float("nan")
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]
    close.loc[idx <= last_real_date, sym] = 55.0

    credited_close, _ = apply_delisting_terminal_credit(close, None)

    nearest = idx[idx <= closure_day].max()
    after = idx[idx > nearest]
    # Every day after the credited closure day remains NaN -- the position
    # cannot leak a nonzero return into any later fold.
    assert credited_close.loc[after, sym].isna().all()

    # pct_change from the credited terminal value to NaN is NaN, and the
    # backtest engine's own .fillna(0) (ascent/backtest/engine.py) turns
    # that into a correct zero contribution afterward -- verify that
    # directly here too.
    pct = credited_close[sym].pct_change()
    assert pct.loc[after].isna().all()


def test_fold_predating_deal_is_unaffected_no_look_ahead(cash_symbol):
    """A fold with test/next-test window entirely before the real closure date
    must see exactly the same (untouched) NaN/real data as before this function
    existed -- no early leakage of the terminal value."""
    sym, terms = cash_symbol
    closure_day = pd.Timestamp(terms["removed_date"])
    idx = pd.bdate_range(closure_day - pd.Timedelta(days=60), periods=80)

    close = pd.DataFrame(index=idx, columns=[sym, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[sym] = float("nan")
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]
    # Give the symbol a real, rising price series well before closure.
    early_dates = idx[idx <= last_real_date]
    close.loc[early_dates, sym] = [50.0 + 0.1 * i for i in range(len(early_dates))]

    original_close = close.copy()
    credited_close, _ = apply_delisting_terminal_credit(close, None)

    # Every date strictly before the credited closure day is byte-identical
    # to the pre-credit data -- the deal terms are not visible before they
    # actually took effect.
    nearest = idx[idx <= closure_day].max()
    before = idx[idx < nearest]
    pd.testing.assert_series_equal(
        credited_close.loc[before, sym], original_close.loc[before, sym]
    )


def test_no_credit_when_real_data_already_covers_closure_day(cash_symbol):
    """If real market data already has a (non-NaN) price on the closure day,
    apply_delisting_terminal_credit must not override it."""
    sym, terms = cash_symbol
    closure_day = pd.Timestamp(terms["removed_date"])
    idx = pd.bdate_range(closure_day - pd.Timedelta(days=20), periods=40)

    close = pd.DataFrame(index=idx, columns=[sym, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[sym] = 55.0  # real data present on every date, including closure day

    credited_close, _ = apply_delisting_terminal_credit(close, None)

    nearest = idx[idx <= closure_day].max()
    assert credited_close.loc[nearest, sym] == pytest.approx(55.0)


def test_symbol_not_in_close_full_is_skipped_without_error():
    """A DELISTING_TERMINAL_TERMS symbol entirely absent from close_full's
    columns (no price history at all) must not raise -- just be skipped."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame(index=idx, columns=["SPY"], dtype=float)
    close["SPY"] = 100.0

    credited_close, _ = apply_delisting_terminal_credit(close, None)
    assert list(credited_close.columns) == ["SPY"]


def test_open_full_gets_matching_terminal_value(cash_symbol):
    sym, terms = cash_symbol
    closure_day = pd.Timestamp(terms["removed_date"])
    idx = pd.bdate_range(closure_day - pd.Timedelta(days=20), periods=40)

    close = pd.DataFrame(index=idx, columns=[sym, "SPY"], dtype=float)
    close["SPY"] = 100.0
    close[sym] = float("nan")
    pre_closure_dates = idx[idx <= closure_day]
    last_real_date = pre_closure_dates[-2]
    close.loc[idx <= last_real_date, sym] = 55.0

    open_ = close.copy()

    credited_close, credited_open = apply_delisting_terminal_credit(close, open_)

    nearest = idx[idx <= closure_day].max()
    assert credited_open.loc[nearest, sym] == pytest.approx(terms["cash_amount"])
    assert credited_open.loc[nearest, sym] == credited_close.loc[nearest, sym]
