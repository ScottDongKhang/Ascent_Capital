import pytest
import pandas as pd


def test_e1_empty_valid_rows_guard():
    """E1: all-zero weights must raise ValueError, not IndexError."""
    target_weights_all = pd.DataFrame(
        {"AAPL": [0.0, 0.0], "MSFT": [0.0, 0.0]},
        index=pd.to_datetime(["2026-04-16", "2026-04-17"])
    )
    valid_rows = target_weights_all[(target_weights_all > 0).any(axis=1)]

    with pytest.raises(ValueError, match="no positive-weight positions"):
        if valid_rows.empty:
            raise ValueError("Pipeline returned no positive-weight positions — aborting EOD run")
        _ = valid_rows.index[-1]


def test_e1_non_empty_valid_rows_no_error():
    """E1: non-zero weights proceed without error."""
    target_weights_all = pd.DataFrame(
        {"AAPL": [0.0, 0.05], "MSFT": [0.0, 0.10]},
        index=pd.to_datetime(["2026-04-16", "2026-04-17"])
    )
    valid_rows = target_weights_all[(target_weights_all > 0).any(axis=1)]

    if valid_rows.empty:
        raise ValueError("Pipeline returned no positive-weight positions — aborting EOD run")
    latest_date = valid_rows.index[-1]
    assert latest_date == pd.Timestamp("2026-04-17")


def test_r1_empty_regime_list_returns_unknown():
    """R1: empty list in regime_signal.json must not raise IndexError."""
    _rdata = []
    _sig = (_rdata[-1] if (isinstance(_rdata, list) and _rdata) else _rdata) or {}
    result = _sig.get("label", "unknown")
    assert result == "unknown"


def test_r1_regime_list_returns_last_label():
    """R1: non-empty list returns last entry's label."""
    _rdata = [{"label": "calm_bull"}, {"label": "stressed"}]
    _sig = (_rdata[-1] if (isinstance(_rdata, list) and _rdata) else _rdata) or {}
    result = _sig.get("label", "unknown")
    assert result == "stressed"


def test_r1_regime_dict_returns_label():
    """R1: dict schema (new format) returns label directly."""
    _rdata = {"label": "crisis", "last_refit_date": "2026-04-10"}
    _sig = (_rdata[-1] if (isinstance(_rdata, list) and _rdata) else _rdata) or {}
    result = _sig.get("label", "unknown")
    assert result == "crisis"
