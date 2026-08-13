"""Regression tests for ascent/risk/correlation_guard.py's price loading.

`prices_macro` is a wide-format cache. Since the save_parquet fix, its dates
live in a `date` COLUMN and `load_parquet` (deliberately generic) does not
restore the index, so `_load_combined_prices` must do it itself — and must
SORT, because `check_cross_agent_correlation` slices the trailing window
positionally (`returns.iloc[-lookback:]`). Without the restore the `date`
column would sit in the frame as a non-price column; without the sort the
"trailing 63 trading days" window would be an arbitrary 63 rows.
"""
import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    mod = importlib.import_module("ascent.data.store.parquet")
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    return mod


@pytest.fixture
def guard():
    return importlib.import_module("ascent.risk.correlation_guard")


def _wide_macro_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = rng.normal(0, 1, n).cumsum()
    return pd.DataFrame(
        {
            "PDBC": 100 + base,
            "KMLM": 50 + base * 0.5,              # near-perfectly correlated with PDBC
            "TLT": 90 + rng.normal(0, 1, n).cumsum(),
        },
        index=idx,
    )


def test_load_combined_prices_restores_date_index(store, guard):
    """A `date` column from a wide-format cache becomes the index again."""
    df = _wide_macro_frame(n=10)
    store.save_parquet(df, "prices_macro")

    out = guard._load_combined_prices(["PDBC", "KMLM"])

    assert isinstance(out.index, pd.DatetimeIndex)
    assert "date" not in out.columns
    assert list(out.columns) == ["PDBC", "KMLM"]
    assert len(out) == 10
    assert out.index[0] == df.index[0]


def test_load_combined_prices_sorts_by_date(store, guard):
    """Rows arriving out of order on disk must come back chronologically —
    the trailing-window slice downstream is positional."""
    df = _wide_macro_frame(n=20)
    shuffled = df.iloc[[5, 0, 19, 12, 3] + [i for i in range(20) if i not in (5, 0, 19, 12, 3)]]
    store.save_parquet(shuffled.reset_index().rename(columns={"index": "date"}), "prices_macro")

    out = guard._load_combined_prices(["PDBC", "KMLM", "TLT"])

    assert out.index.is_monotonic_increasing
    assert out.index[-1] == df.index[-1]
    assert out["PDBC"].iloc[-1] == pytest.approx(df["PDBC"].iloc[-1])


def test_check_cross_agent_correlation_flags_pair_from_wide_cache(store, guard):
    """End-to-end: a wide-format prices_macro yields a real violation instead
    of bailing out on insufficient/garbage data."""
    store.save_parquet(_wide_macro_frame(n=200), "prices_macro")

    violations = guard.check_cross_agent_correlation(
        {"macro": {"PDBC": 0.10}, "alternatives": {"KMLM": 0.08}}
    )

    pairs = {frozenset((a, b)) for a, b, _ in violations}
    assert frozenset(("PDBC", "KMLM")) in pairs
    assert all(abs(c) > guard.CORRELATION_CAP for _, _, c in violations)


def test_insufficient_rows_returns_empty(store, guard):
    """Fewer rows than the lookback must degrade to no violations, not raise."""
    store.save_parquet(_wide_macro_frame(n=10), "prices_macro")

    assert guard.check_cross_agent_correlation(
        {"macro": {"PDBC": 0.1}, "alternatives": {"KMLM": 0.1}}
    ) == []


def test_missing_cache_returns_empty_frame(store, guard):
    """No prices_macro (and no prices_live) on disk — empty frame, no raise."""
    assert guard._load_combined_prices(["PDBC", "KMLM"]).empty
