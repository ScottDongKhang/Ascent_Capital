"""Regression test for pivot_prices date normalization.

Root cause (2026-08-15 audit): prices_live accumulated phantom rows stamped
at a non-midnight time-of-day (e.g. 19:00 from an un-rolled hub fetch) for a
date that also had a legitimate 00:00 row for a DIFFERENT symbol. Because
pivot_table indexed on the raw `date` column, the phantom row's distinct
timestamp fragmented the index into two entries for what should have been a
single trading day, instead of colliding (or at worst overwriting) via
aggfunc="last".

pivot_prices() now groups on `_calendar_day_key` (the same rollover-aware
trading-day key `save_parquet`'s dedup and the repaired cache use) rather
than a plain `.dt.normalize()`. A bar stamped >=17:00 local is that trading
day's already-closed bar recorded late, and rolls forward onto the NEXT
calendar day -- it is NOT a same-day collision with a midnight row on its
own nominal date (see `_calendar_day_key`'s docstring / the repair script's
BCO example). So a same-day-collision test below uses an intraday, PRE-17:00
stamp (still non-midnight, still exercising the fragmentation bug) rather
than the >=17:00 stamp, and a separate test asserts the >=17:00 case
correctly rolls to the next trading day instead of colliding.
"""
import pandas as pd

from ascent.data.normalize.prices import pivot_prices


def test_same_day_non_midnight_phantom_row_does_not_fragment_index():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),  # real row
            pd.Timestamp("2024-06-03 15:00"),  # same-day phantom, different symbol
        ],
        "close": [194.0, 420.0],
    })

    pivot = pivot_prices(df, field="close")

    assert len(pivot.index) == 1, (
        f"expected a single index entry for 2024-06-03, got {len(pivot.index)}: "
        f"{list(pivot.index)}"
    )
    assert pivot.index[0] == pd.Timestamp("2024-06-03")
    # Both symbols' values are preserved on the single collapsed date.
    assert pivot.loc[pd.Timestamp("2024-06-03"), "AAPL"] == 194.0
    assert pivot.loc[pd.Timestamp("2024-06-03"), "MSFT"] == 420.0


def test_evening_stamp_rolls_to_next_trading_day_not_same_day():
    """A >=17:00-local stamp is the SAME trading day's close recorded late --
    per _calendar_day_key it belongs to the NEXT calendar day, not the day
    its own `date` value nominally names. pivot_prices must not collide it
    with a real midnight row sharing that literal nominal date."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),  # real row, day D
            pd.Timestamp("2024-06-03 19:00"),  # evening stamp -> rolls to D+1
        ],
        "close": [194.0, 420.0],
    })

    pivot = pivot_prices(df, field="close")

    assert list(pivot.index) == [pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-04")]
    assert pivot.loc[pd.Timestamp("2024-06-03"), "AAPL"] == 194.0
    assert pivot.loc[pd.Timestamp("2024-06-04"), "MSFT"] == 420.0


def test_same_symbol_same_day_collision_keeps_last():
    """If a future corruption produces a same-symbol same-day collision after
    normalization, aggfunc="last" must resolve it deterministically rather
    than raising or silently fragmenting the index."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),
            pd.Timestamp("2024-06-03 15:00"),
        ],
        "close": [194.0, 195.5],
    })

    pivot = pivot_prices(df, field="close")

    assert len(pivot.index) == 1
    assert pivot.loc[pd.Timestamp("2024-06-03"), "AAPL"] == 195.5


def test_collision_resolves_by_timestamp_not_input_order():
    """Collision resolution must be based on the actual timestamp, not on
    which row happens to appear first/last in the input frame. Construct the
    later-stamped row FIRST in the frame -- if pivot_prices resolved by row
    order alone, this would (wrongly) keep the earlier-stamped value."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [
            pd.Timestamp("2024-06-03 15:00"),  # later timestamp, listed FIRST
            pd.Timestamp("2024-06-03 00:00"),  # earlier timestamp, listed LAST
        ],
        "close": [195.5, 194.0],
    })

    pivot = pivot_prices(df, field="close")

    assert len(pivot.index) == 1
    # The LATER-stamped value (195.5) must win, even though it was listed
    # first in the input frame.
    assert pivot.loc[pd.Timestamp("2024-06-03"), "AAPL"] == 195.5
