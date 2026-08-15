"""Regression tests for save_parquet cross-fetch deduplication.

Root cause (audit 2026-06-22): prices_live accumulated ~59% duplicate rows because
save_parquet deduped on the raw tz-aware `date` column. Different fetches wrote the
same calendar day with different intraday time / tz components, so (symbol, date) did
not match and `drop_duplicates(keep="last")` never collapsed them.

Follow-up (2026-07-28): the 2026-06-24 dedup fix did not fully stop production
re-corruption. Two further defects were found and are covered below:

  Defect 1 — exact-duplicate rows survive because `save_parquet`'s dedup logic only
  runs inside the `if path.exists()` branch. On the very first write for a cache name,
  the incoming `df` is written straight through with NO dedup at all, so any exact
  duplicate rows already present in that first incoming frame (e.g. a hub blend of
  multiple providers concatenated before the first save_parquet call) are persisted
  forever and every subsequent append's dedup only compares against them as already
  "unique" rows.

  Defect 2 — a bar stamped >=17:00 NY (after the US close) is fetched with the WRONG
  nominal date: it is stamped with the date it was fetched relative to, but the price
  data actually belongs to the NEXT trading day. `_calendar_day_key` normalized the
  date but did not account for this, so the same real trading day fetched once as
  "D 19:00" and again (correctly) as "D+1 00:00" produced two different keys and never
  collided.
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
    timestamps must collapse to ONE row (last wins), not accumulate.

    NOTE (2026-07-28, Defect 2 fix): the original version of this test used a
    19:00 NY stamp as one of the two timestamps. Per the Defect 2 root-cause
    finding, a bar stamped >=17:00 NY (after the US close) actually belongs to
    the NEXT trading day, not the nominal date it's stamped with — so pairing
    a 19:00 stamp on date D with a 16:00 stamp also nominally on date D was
    encoding the OLD, now-understood-incorrect assumption that any two
    intraday stamps sharing a nominal date are the same trading day
    regardless of hour. That specific pairing is now covered correctly, and
    separately, by the evening-rollover tests below. This test is narrowed to
    two intraday stamps that are BOTH before the close cutoff, which is the
    unambiguous same-day case it was originally meant to guard.
    """
    ny = "America/New_York"
    day1 = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": [pd.Timestamp("2024-06-03 12:00", tz=ny)],
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
    fetch writes the SAME REAL TRADING DAY tz-NAIVE (different ingest source).
    After pd.concat the `date` column is object dtype (pandas cannot mix
    aware+naive in one datetime column), so is_datetime64_any_dtype is False and
    the normalize branch is skipped — dedup never fires and the row accumulates.
    This is the 3-generation blend that bloated prices_live ~3x and let the bad
    KLAC copy survive.

    NOTE (2026-07-28, Defect 2 fix): the original fixture paired an aware
    "2024-06-03 19:00" stamp with a naive "2024-06-03 00:00" stamp, both on the
    SAME nominal date. Per the Defect 2 finding, a >=17:00 NY stamp on date D
    is really date D+1's bar, so that original pairing was actually asserting
    that two DIFFERENT real trading days collapse to one row — which is the
    wrong behavior we now explicitly guard against below
    (test_distinct_real_trading_days_not_falsely_merged). This test is fixed to
    pair the aware evening stamp with a naive midnight stamp on D+1, which is
    the real-world pairing that produced the production duplicates.
    """
    ny = "America/New_York"
    aware = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-03 19:00", tz=ny)],  # really belongs to 06-04
        "close": [240.0],
    })
    naive = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-04 00:00")],  # tz-naive, same REAL trading day
        "close": [241.0],
    })
    store.save_parquet(aware, "t_p4")
    store.save_parquet(naive, "t_p4")
    out = store.load_parquet("t_p4")
    klac = out[out["symbol"] == "KLAC"]
    assert len(klac) == 1, f"expected 1 row per calendar day, got {len(klac)}"
    assert klac["close"].iloc[0] == 241.0  # keep="last"


# ---------------------------------------------------------------------------
# Defect 1 — exact-duplicate rows survive the FIRST write (no path.exists() yet)
# ---------------------------------------------------------------------------

def test_exact_duplicate_rows_in_first_write_collapse(store):
    """Reproduces Defect 1 through the real save_parquet API: a single incoming
    frame (as if a hub blend concatenated multiple providers before the very
    first save_parquet call for this cache name) already contains two
    byte-identical rows. Because save_parquet's dedup only runs inside
    `if path.exists()`, the very first write for a cache name bypasses dedup
    entirely and both copies would previously be persisted."""
    ny = "America/New_York"
    row = {
        "symbol": "SPY",
        "date": pd.Timestamp("2020-01-01 19:00", tz=ny),
        "close": 324.869995,
        "source": "yfinance_hub",
    }
    df = pd.DataFrame([row, row])  # exact duplicate, first write, path does NOT exist
    store.save_parquet(df, "t_p5")
    out = store.load_parquet("t_p5")
    assert len(out) == 1, f"expected exact duplicates in the first write to collapse, got {len(out)}"


def test_exact_duplicate_rows_across_two_writes_collapse(store):
    """Same exact-duplicate row delivered across two separate save_parquet calls
    (the append path) must also collapse to one — this path already worked
    before the fix and must keep working."""
    ny = "America/New_York"
    row = pd.DataFrame([{
        "symbol": "SPY",
        "date": pd.Timestamp("2020-01-01 19:00", tz=ny),
        "close": 324.869995,
        "source": "yfinance_hub",
    }])
    store.save_parquet(row, "t_p6")
    store.save_parquet(row.copy(), "t_p6")
    out = store.load_parquet("t_p6")
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Defect 2 — evening-stamped bars (>=17:00 NY) belong to the NEXT trading day
# ---------------------------------------------------------------------------

def test_evening_19h_stamp_collapses_with_next_day_midnight_stamp(store):
    """A bar stamped 2020-01-01 19:00 NY is really 2020-01-02's close (2020-01-01
    is a market holiday in the real corruption evidence). A later, correctly
    dated fetch stamps the same trading day as 2020-01-02 00:00 NY. These must
    collapse to ONE row."""
    ny = "America/New_York"
    evening = pd.DataFrame({
        "symbol": ["SPY"],
        "date": [pd.Timestamp("2020-01-01 19:00", tz=ny)],
        "close": [324.87],
    })
    midnight_next_day = pd.DataFrame({
        "symbol": ["SPY"],
        "date": [pd.Timestamp("2020-01-02 00:00", tz=ny)],
        "close": [324.87],
    })
    store.save_parquet(evening, "t_p7")
    store.save_parquet(midnight_next_day, "t_p7")
    out = store.load_parquet("t_p7")
    assert len(out) == 1, f"expected evening stamp to roll to next trading day, got {len(out)}"


def test_evening_20h_stamp_collapses_with_next_day_midnight_stamp(store):
    """Same as above but for the 20:00 NY variant also seen in production."""
    ny = "America/New_York"
    evening = pd.DataFrame({
        "symbol": ["SPY"],
        "date": [pd.Timestamp("2020-01-01 20:00", tz=ny)],
        "close": [324.87],
    })
    midnight_next_day = pd.DataFrame({
        "symbol": ["SPY"],
        "date": [pd.Timestamp("2020-01-02 00:00", tz=ny)],
        "close": [324.87],
    })
    store.save_parquet(evening, "t_p8")
    store.save_parquet(midnight_next_day, "t_p8")
    out = store.load_parquet("t_p8")
    assert len(out) == 1


def test_distinct_real_trading_days_not_falsely_merged(store):
    """Conservatism check: the evening-rollover rule must never merge two
    genuinely different trading days. 2020-01-01 19:00 (rolls to 2020-01-02)
    and 2020-01-03 00:00 (a separate, later real trading day) must stay
    distinct."""
    ny = "America/New_York"
    df = pd.DataFrame({
        "symbol": ["SPY", "SPY"],
        "date": [pd.Timestamp("2020-01-01 19:00", tz=ny),
                 pd.Timestamp("2020-01-03 00:00", tz=ny)],
        "close": [324.87, 326.50],
    })
    store.save_parquet(df, "t_p9")
    out = store.load_parquet("t_p9")
    assert len(out) == 2, f"distinct real trading days must not merge, got {len(out)}"


def test_evening_stamp_object_dtype_variant_rolls_forward(store):
    """The object-dtype (mixed aware/naive) branch of _calendar_day_key must
    apply the same evening-rollover rule as the pure-datetime64 branch."""
    ny = "America/New_York"
    aware_evening = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-03 19:00", tz=ny)],  # really 06-04
        "close": [240.0],
    })
    naive_next_day = pd.DataFrame({
        "symbol": ["KLAC"],
        "date": [pd.Timestamp("2024-06-04 00:00")],  # tz-naive
        "close": [241.0],
    })
    store.save_parquet(aware_evening, "t_p10")
    store.save_parquet(naive_next_day, "t_p10")
    out = store.load_parquet("t_p10")
    assert len(out) == 1, f"expected object-dtype evening rollover to collapse, got {len(out)}"


# ---------------------------------------------------------------------------
# Wide-format DatetimeIndex caches (prices_macro / prices_international /
# prices_alternatives) — save_parquet used to silently drop the index
# ---------------------------------------------------------------------------

def test_datetime_index_survives_round_trip(store):
    """A wide-format DataFrame's DatetimeIndex must not be lost across
    save/load. save_parquet converts it to a `date` column up front so the
    existing id_cols/calendar-day-dedup machinery can handle it; load_parquet
    itself stays generic and returns that `date` column (callers restore the
    index, see agents/macro_agent.py et al.)."""
    df = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]},
                       index=pd.date_range("2025-01-01", periods=2, freq="D"))
    store.save_parquet(df, "test_wide_cache")
    back = store.load_parquet("test_wide_cache")
    assert isinstance(back.index, pd.RangeIndex)
    assert "date" in back.columns
    assert list(back["date"]) == list(pd.date_range("2025-01-01", periods=2, freq="D"))
    assert back.set_index("date")["AAPL"].tolist() == [100.0, 101.0]
    assert back.set_index("date")["MSFT"].tolist() == [200.0, 201.0]


def test_rangeindex_dataframe_still_saves_without_index_column(store):
    """Regression guard: existing long-format callers must see byte-identical
    behavior — a plain RangeIndex input is untouched by the new DatetimeIndex
    conversion."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
        "close": [100.0, 200.0],
    })
    store.save_parquet(df, "test_long_cache")
    back = store.load_parquet("test_long_cache")
    assert isinstance(back.index, pd.RangeIndex)  # unchanged from today
    assert "date" in back.columns  # the column was already there, not index-derived
    assert len(back) == 2


def test_second_save_of_wide_dataframe_appends_and_dedupes_correctly(store):
    """The read-modify-write path (path.exists() branch) must preserve BOTH
    dates as a `date` column, not just a row count.

    Scope note: the row-count assertion alone does NOT isolate the
    ignore_index=True concat bug — two rows survived that path before the fix
    too (they were just dateless). What this test adds over the pre-fix
    behavior is the `date` column and its values; the dedup half of the
    contract is guarded by
    test_wide_dataframe_resave_of_same_day_replaces_not_duplicates, which goes
    2 rows -> 1 and genuinely fails pre-fix."""
    df1 = pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
    df2 = pd.DataFrame({"AAPL": [101.0]}, index=pd.date_range("2025-01-02", periods=1))
    store.save_parquet(df1, "test_wide_append")
    store.save_parquet(df2, "test_wide_append")
    back = store.load_parquet("test_wide_append")
    assert len(back) == 2, f"expected both dates to survive the append path, got {len(back)}"
    assert sorted(back["date"].tolist()) == [
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"),
    ]
    restored = back.set_index("date").sort_index()
    assert restored["AAPL"].tolist() == [100.0, 101.0]


def test_wide_dataframe_resave_of_same_day_replaces_not_duplicates(store):
    """Re-saving the same calendar day for a wide-format cache (e.g. a
    same-day re-fetch) must replace, not duplicate, keeping the dedup
    guarantee that already holds for long-format caches."""
    df1 = pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
    df2 = pd.DataFrame({"AAPL": [105.0]}, index=pd.date_range("2025-01-01", periods=1))
    store.save_parquet(df1, "test_wide_resave")
    store.save_parquet(df2, "test_wide_resave")
    back = store.load_parquet("test_wide_resave")
    assert len(back) == 1
    assert back["AAPL"].iloc[0] == 105.0  # keep="last"


# ---------------------------------------------------------------------------
# validate_cache — phantom-row (non-midnight time-of-day) integrity guard
# (2026-08-15 audit): defense-in-depth so this class of corruption cannot
# recur silently in a cache validate_cache is asked to trust.
# ---------------------------------------------------------------------------

def test_validate_cache_fails_on_non_midnight_date_rows(store):
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),
            pd.Timestamp("2024-06-03 19:00"),  # phantom-row corruption signature
        ],
        "close": [194.0, 420.0],
    })
    path = store.DATA_DIR / "t_validate_phantom.parquet"
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)  # write directly, bypassing save_parquet's own dedup

    ok, reason = store.validate_cache("t_validate_phantom", stale_days=0)
    assert ok is False
    assert "non-midnight" in reason
    assert "1" in reason  # exactly one offending row


def test_validate_cache_passes_all_midnight_date_rows(store):
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [pd.Timestamp("2024-06-03 00:00"), pd.Timestamp("2024-06-04 00:00")],
        "close": [194.0, 420.0],
    })
    path = store.DATA_DIR / "t_validate_clean.parquet"
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    ok, reason = store.validate_cache("t_validate_clean", stale_days=0)
    assert ok is True, reason


def test_validate_cache_ignores_non_datetime_date_col(store):
    """Gate confirmation: a date_col that is not datetime-like must not run
    the phantom-row check at all (it would error on .dt access otherwise)."""
    df = pd.DataFrame({
        "series_id": ["DGS10", "DGS10"],
        "date": ["not-a-real-date", "also-not-a-date"],
        "value": [4.5, 4.6],
    })
    path = store.DATA_DIR / "t_validate_nondatetime.parquet"
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    ok, reason = store.validate_cache("t_validate_nondatetime", stale_days=0)
    # Both values fail pd.to_datetime coercion -> "cache has no valid dates",
    # not the phantom-row check; confirms the gate gracefully no-ops rather
    # than erroring on a non-datetime column.
    assert ok is False
    assert reason == "cache has no valid dates"


def test_macro_series_id_cache_still_dedups(store):
    """Macro caches keyed on series_id (FRED/simulated ingest) must keep
    deduping correctly — the date-key rollover logic must not break the
    series_id id_cols path."""
    a = pd.DataFrame({
        "series_id": ["DGS10"],
        "date": [pd.Timestamp("2024-01-02")],
        "value": [4.5],
        "source": ["fred"],
    })
    b = pd.DataFrame({
        "series_id": ["DGS10"],
        "date": [pd.Timestamp("2024-01-02")],
        "value": [4.6],  # revised value, same series/day
        "source": ["fred"],
    })
    store.save_parquet(a, "t_p11")
    store.save_parquet(b, "t_p11")
    out = store.load_parquet("t_p11")
    assert len(out) == 1 and out["value"].iloc[0] == 4.6
