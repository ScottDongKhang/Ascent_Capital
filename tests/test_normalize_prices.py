"""Regression test for pivot_prices date normalization.

Root cause (2026-08-15 audit): prices_live accumulated phantom rows stamped
at a non-midnight time-of-day (e.g. 19:00 from an un-rolled hub fetch) for a
date that also had a legitimate 00:00 row for a DIFFERENT symbol. Because
pivot_table indexed on the raw `date` column, the phantom row's distinct
timestamp fragmented the index into two entries for what should have been a
single trading day, instead of colliding (or at worst overwriting) via
aggfunc="last". pivot_prices() now normalizes `date` to midnight before
pivoting so any same-day collision resolves deterministically.
"""
import pandas as pd

from ascent.data.normalize.prices import pivot_prices


def test_same_day_non_midnight_phantom_row_does_not_fragment_index():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),  # real row
            pd.Timestamp("2024-06-03 19:00"),  # same-day phantom, different symbol
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


def test_same_symbol_same_day_collision_keeps_last():
    """If a future corruption produces a same-symbol same-day collision after
    normalization, aggfunc="last" must resolve it deterministically rather
    than raising or silently fragmenting the index."""
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [
            pd.Timestamp("2024-06-03 00:00"),
            pd.Timestamp("2024-06-03 19:00"),
        ],
        "close": [194.0, 195.5],
    })

    pivot = pivot_prices(df, field="close")

    assert len(pivot.index) == 1
    assert pivot.loc[pd.Timestamp("2024-06-03"), "AAPL"] == 195.5
