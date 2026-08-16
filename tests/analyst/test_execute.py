"""Tests for the ticker-substitution postcondition and repair-history tracking.

Covers the bug documented in analyst/HANDOFF.md ("The known hole -- fix this
first"): pointing a LOAD task at a nonexistent ticker did not exhaust and
surface a clean failure. The debugger silently rewrote the ticker to a
different, valid one on repair, and shape validation passed because the
shape was perfect -- it just wasn't data for the ticker that was asked for.
"""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from analyst import toolkit
from analyst.execute import IdentityError, ValidationError, execute_node, validate_output
from analyst.types import AnalysisPlan, NodeState, OutputSchema, Task, TaskCategory

PRICE_SCHEMA = OutputSchema(
    kind="dataframe",
    columns={"Close": "float"},
    index_kind="market_trading_day",
    row_semantics="one row per trading session",
)


def _load_task(expected_ticker: str, max_repair_attempts: int = 1) -> Task:
    return Task(
        task_id="load_asset",
        title="Load asset data",
        category=TaskCategory.LOAD,
        description="load prices",
        output_name="asset_prices",
        output_schema=PRICE_SCHEMA,
        expected_ticker=expected_ticker,
        max_repair_attempts=max_repair_attempts,
    )


def _plan(task: Task) -> AnalysisPlan:
    return AnalysisPlan(question="q", tasks=(task,), params={})


def _priced_df(ticker: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2022-01-01", periods=3),
    )
    df.attrs["ticker"] = ticker
    return df


# ---------------------------------------------------------------------------
# validate_output: the postcondition itself
# ---------------------------------------------------------------------------


def test_validate_output_accepts_matching_ticker():
    task = _load_task("CL=F")
    validate_output(task, _priced_df("CL=F"))  # must not raise


def test_validate_output_rejects_substituted_ticker():
    task = _load_task("NOTAREALTICKER123")
    with pytest.raises(IdentityError, match="NOTAREALTICKER123"):
        validate_output(task, _priced_df("CL=F"))


def test_validate_output_rejects_missing_ticker_metadata():
    """A DataFrame that carries no ticker stamp at all (e.g. hand-built by a
    generated task that bypassed toolkit) must not pass just because the
    shape is right."""
    task = _load_task("CL=F")
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})  # no attrs["ticker"]
    with pytest.raises(IdentityError):
        validate_output(task, df)


def test_identity_error_is_a_validation_error():
    assert issubclass(IdentityError, ValidationError)


# ---------------------------------------------------------------------------
# toolkit.load_prices: raises loudly on a genuinely nonexistent ticker
# ---------------------------------------------------------------------------


def test_load_prices_raises_on_empty_result(monkeypatch):
    def fake_download(ticker, start, end, progress, auto_adjust):
        return pd.DataFrame()  # yfinance's shape for "no such ticker"

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(toolkit, "_CACHE", toolkit._CACHE.parent / "test_cache_nonexistent")

    with pytest.raises(ValueError, match="NOTAREALTICKER123"):
        toolkit.load_prices("NOTAREALTICKER123", "2022-01-01", "2022-02-01")


def test_load_prices_stamps_ticker_identity(monkeypatch, tmp_path):
    def fake_download(ticker, start, end, progress, auto_adjust):
        idx = pd.date_range("2022-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], [ticker]])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(toolkit, "_CACHE", tmp_path)

    df = toolkit.load_prices("CL=F", "2022-01-01", "2022-02-01")
    assert df.attrs["ticker"] == "CL=F"

    # cache hit path must re-stamp identity too (attrs are not persisted by parquet)
    df2 = toolkit.load_prices("CL=F", "2022-01-01", "2022-02-01")
    assert df2.attrs["ticker"] == "CL=F"


# ---------------------------------------------------------------------------
# execute_node: end-to-end reproduction of the original failure mode
# ---------------------------------------------------------------------------

_BAD_TICKER_CODE = textwrap.dedent(
    """
    from analyst import toolkit

    def run():
        return toolkit.load_prices("NOTAREALTICKER123", "2022-01-01", "2022-02-01")
    """
)

_SUBSTITUTED_TICKER_CODE = textwrap.dedent(
    """
    from analyst import toolkit

    def run():
        return toolkit.load_prices("CL=F", "2022-01-01", "2022-02-01")
    """
)


def test_execute_node_fails_loudly_instead_of_silently_substituting(monkeypatch):
    """Reproduces the exact HANDOFF anecdote: task asks for a nonexistent
    ticker, the (simulated) repair silently swaps in a different, valid
    ticker. Before the fix this reported `done`. After the fix it must
    report FAILED, and repair_history must record what actually happened.
    """

    def fake_load_prices(ticker, start, end):
        if ticker == "NOTAREALTICKER123":
            raise ValueError(f"no data returned for ticker {ticker!r} between {start} and {end}")
        return _priced_df(ticker)

    monkeypatch.setattr(toolkit, "load_prices", fake_load_prices)

    # Simulate the buggy debugger: whatever the prior error was, it "fixes"
    # the code by pointing it at a different, valid ticker instead of
    # surfacing the failure.
    monkeypatch.setattr(
        "analyst.codegen.generate_one",
        lambda task, plan, prior_error="", prior_code="": _SUBSTITUTED_TICKER_CODE,
    )

    task = _load_task("NOTAREALTICKER123", max_repair_attempts=1)
    plan = _plan(task)

    result = execute_node(task, plan, _BAD_TICKER_CODE, upstream={})

    assert result.state is NodeState.FAILED
    assert not result.ok
    # attempt 1: genuine fetch failure. attempt 2: identity postcondition
    # catches the substitution instead of accepting it.
    assert result.attempts == 2
    assert len(result.repair_history) == 2
    assert "NOTAREALTICKER123" in result.repair_history[0]
    assert "IdentityError" in result.repair_history[1] or "refusing to accept" in result.repair_history[1]
