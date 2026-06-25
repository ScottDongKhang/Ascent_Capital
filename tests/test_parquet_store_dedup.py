"""Regression tests for save_parquet cross-fetch deduplication.

Root cause (audit 2026-06-22): prices_live accumulated ~59% duplicate rows because
save_parquet deduped on the raw tz-aware `date` column. Different fetches wrote the
same calendar day with different intraday time / tz components, so (symbol, date) did
not match and `drop_duplicates(keep="last")` never collapsed them.
"""
import importlib
import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    mod = importlib.import_module("ascent.data.store.parquet")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    return mod


def test_same_calendar_day_different_intraday_timestamp_collapses(store):
    """Two fetches of the same (symbol, calendar-day) with different intraday
    timestamps must collapse to ONE row (last wins), not accumulate."""
    ny = "America/New_York"
    day1 = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": [pd.Timestamp("2024-06-03 19:00", tz=ny)],  # how the cache stores it
        "close": [194.03],
    })
    day1b = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": [pd.Timestamp("2024-06-03 16:00", tz=ny)],  # different intraday stamp
        "close": [194.35],
    })
    store.save_parquet(day1, "t_prices")
    store.save_parquet(day1b, "t_prices")

    out = store.load_parquet("t_prices")
    aapl = out[out["symbol"] == "AAPL"]
    assert len(aapl) == 1, f"expected 1 row per calendar day, got {len(aapl)}"
    assert aapl["close"].iloc[0] == 194.35  # keep="last"


def test_distinct_calendar_days_are_preserved(store):
    """Dedup must not over-collapse: genuinely different days stay separate."""
    ny = "America/New_York"
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [pd.Timestamp("2024-06-03 19:00", tz=ny),
                 pd.Timestamp("2024-06-04 19:00", tz=ny)],
        "close": [194.0, 195.0],
    })
    store.save_parquet(df, "t_prices2")
    store.save_parquet(df, "t_prices2")  # idempotent re-save
    out = store.load_parquet("t_prices2")
    assert len(out) == 2


def test_tz_naive_dates_still_dedup(store):
    """A tz-naive date column (other caches) must still dedup by calendar day."""
    a = pd.DataFrame({"symbol": ["X"], "date": [pd.Timestamp("2024-01-02")], "close": [1.0]})
    b = pd.DataFrame({"symbol": ["X"], "date": [pd.Timestamp("2024-01-02")], "close": [2.0]})
    store.save_parquet(a, "t_p3")
    store.save_parquet(b, "t_p3")
    out = store.load_parquet("t_p3")
    assert len(out) == 1 and out["close"].iloc[0] == 2.0


def test_mixed_tzaware_and_naive_across_fetches_collapses(store):
    """The production failure: an existing on-disk row is tz-AWARE and a later
    fetch writes the same calendar day tz-NAIVE (different ingest source). After
    pd.concat the `date` column is object dtype (pandas cannot mix aware+naive in
    one datetime column), so is_datetime64_any_dtype is False and the normalize
    branch is skipped — dedup never fires and the row accumulates. This is the
    3-generation blend that bloated prices_live ~3× and let the bad KLAC copy
    survive. Both rows are the same (symbol, calendar-day) → must collapse to 1."""
    ny = "America/New_York"
    aware = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-03 19:00", tz=ny)],
        "close": [240.0],
    })
    naive = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-03 00:00")],  # tz-naive, same calendar day
        "close": [241.0],
    })
    store.save_parquet(aware, "t_p4")
    store.save_parquet(naive, "t_p4")
    out = store.load_parquet("t_p4")
    klac = out[out["symbol"] == "KLAC"]
    assert len(klac) == 1, f"expected 1 row per calendar day, got {len(klac)}"
    assert klac["close"].iloc[0] == 241.0  # keep="last"
