import pandas as pd
import pytest
from ascent.research.wf_framework.windows import WindowGenerator, SplitWindow

@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-02", periods=600)

def test_window_count(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    windows = gen.generate(dates)
    assert len(windows) >= 3, "Expected at least 3 folds for 600-day date range"

def test_no_overlap(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        assert w.oos_start > w.purge_end, "OOS must start after purge"
        assert w.purge_end >= w.is_end, "Purge must extend past IS end"

def test_is_slice_max_index(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        is_slice = w.slice_is(dates)
        oos_slice = w.slice_oos(dates)
        assert is_slice.max() < oos_slice.min(), "IS data must precede OOS data"

def test_is_slice_excludes_purge(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5)
    for w in gen.generate(dates):
        is_slice = w.slice_is(dates)
        assert is_slice.max() < w.purge_start, \
            "IS slice must not include dates in the purge window"

def test_rolling_window_advances(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5,
                          window_type="rolling")
    windows = gen.generate(dates)
    for i in range(1, len(windows)):
        assert windows[i].is_start > windows[i-1].is_start, \
            "Rolling window IS start must advance each fold"

def test_anchored_window_fixed_start(dates):
    gen = WindowGenerator(is_days=252, oos_days=63, purge_days=21, embargo_days=5,
                          window_type="anchored")
    windows = gen.generate(dates)
    for w in windows:
        assert w.is_start == windows[0].is_start, \
            "Anchored window IS start must be fixed"
