"""
Walk-Forward Window Generator
==============================
Produces IS/OOS date-index slices with mandatory purge and embargo gaps.

Boundary defenses
-----------------
Purge  : The final `purge_days` bars of the IS window are excluded from the
         IS slice. This removes dates where a forward-looking label (e.g.
         21-day forward return) would overlap with OOS data, preventing
         label leakage at the IS/OOS boundary.

Embargo: The first `embargo_days` bars after the purge window are excluded
         entirely (neither IS nor OOS). This breaks the serial correlation
         that remains between adjacent IS and OOS returns even after purging,
         caused by the autocorrelation structure of asset returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class SplitWindow:
    fold_id: int
    is_start: pd.Timestamp       # first bar available for training
    purge_start: pd.Timestamp    # first bar excluded from IS (purge begins)
    is_end: pd.Timestamp         # last bar of the raw IS window (before purge cut)
    purge_end: pd.Timestamp      # last bar of purge gap
    embargo_end: pd.Timestamp    # last bar of embargo gap
    oos_start: pd.Timestamp      # first OOS bar
    oos_end: pd.Timestamp        # last OOS bar

    def slice_is(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """IS dates: [is_start, purge_start). Purge tail excluded."""
        return dates[(dates >= self.is_start) & (dates < self.purge_start)]

    def slice_oos(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """OOS dates: [oos_start, oos_end]."""
        return dates[(dates >= self.oos_start) & (dates <= self.oos_end)]

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"IS [{self.is_start.date()} → {self.purge_start.date()}) "
            f"| purge [{self.purge_start.date()} → {self.purge_end.date()}] "
            f"| embargo [{self.purge_end.date()} → {self.embargo_end.date()}] "
            f"| OOS [{self.oos_start.date()} → {self.oos_end.date()}]"
        )


class WindowGenerator:
    """
    Generate walk-forward train/test splits with purge and embargo gaps.

    Parameters
    ----------
    is_days      : Number of trading days in the in-sample window.
    oos_days     : Number of trading days in the out-of-sample window.
    purge_days   : Bars removed from the IS tail to prevent label leakage.
                   Set this >= your maximum forward-return horizon.
    embargo_days : Bars removed from the OOS head to break serial correlation.
    window_type  : "rolling" (IS window slides) or "anchored" (IS always
                   starts from the first available date).
    step_days    : How many OOS bars to advance between folds.
                   Defaults to oos_days (non-overlapping OOS periods).
    """

    def __init__(
        self,
        is_days: int = 252,
        oos_days: int = 63,
        purge_days: int = 21,
        embargo_days: int = 5,
        window_type: Literal["rolling", "anchored"] = "rolling",
        step_days: int | None = None,
    ):
        if purge_days + embargo_days >= oos_days:
            raise ValueError(
                f"purge_days ({purge_days}) + embargo_days ({embargo_days}) "
                f"must be < oos_days ({oos_days})"
            )
        self.is_days = is_days
        self.oos_days = oos_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.window_type = window_type
        self.step_days = step_days if step_days is not None else oos_days

    def generate(self, dates: pd.DatetimeIndex) -> list[SplitWindow]:
        """
        Generate all valid SplitWindow objects for the given date index.
        """
        dates = dates.sort_values().drop_duplicates()
        n = len(dates)
        windows: list[SplitWindow] = []
        fold_id = 0

        is_end_idx = self.is_days - 1

        while True:
            if self.window_type == "anchored":
                is_start_idx = 0
            else:
                is_start_idx = is_end_idx - self.is_days + 1

            if is_start_idx < 0:
                is_end_idx += self.step_days
                continue

            purge_start_idx = is_end_idx - self.purge_days + 1
            purge_end_idx   = is_end_idx

            embargo_start_idx = purge_end_idx + 1
            embargo_end_idx   = embargo_start_idx + self.embargo_days - 1

            oos_start_idx = embargo_end_idx + 1
            oos_end_idx   = oos_start_idx + self.oos_days - 1

            if oos_end_idx >= n:
                break

            windows.append(SplitWindow(
                fold_id      = fold_id,
                is_start     = dates[is_start_idx],
                purge_start  = dates[purge_start_idx],
                is_end       = dates[is_end_idx],
                purge_end    = dates[purge_end_idx],
                embargo_end  = dates[embargo_end_idx],
                oos_start    = dates[oos_start_idx],
                oos_end      = dates[oos_end_idx],
            ))

            fold_id    += 1
            is_end_idx += self.step_days

        return windows
