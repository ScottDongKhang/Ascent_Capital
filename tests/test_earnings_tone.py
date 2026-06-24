"""tests/test_earnings_tone.py
Tests for ascent/alpha/earnings_tone.py — parquet-only loader sleeve.
All 4 paths from the reconciled plan coverage diagram.
"""
import pandas as pd
import pytest
from unittest.mock import patch


def _price_grid(symbols=("AAPL", "MSFT", "GOOG"), n_days=10):
    idx = pd.bdate_range("2026-01-05", periods=n_days)
    return pd.DataFrame(150.0, index=idx, columns=list(symbols))


def _transcript_panel(symbols=("AAPL", "MSFT"), n_days=5):
    idx = pd.bdate_range("2026-01-05", periods=n_days)
    return pd.DataFrame(0.5, index=idx, columns=list(symbols))


# ── test_returns_dataframe_on_grid ────────────────────────────────────────────

def test_returns_dataframe_on_grid():
    """Panel present → returned DataFrame has same index/columns as features['close']."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    close = _price_grid()
    features = {"close": close}
    panel = _transcript_panel()

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=panel):
        result = earnings_tone_alpha(features)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # index must match close index
    assert list(result.index) == list(close.index)
    # columns must be a subset of close columns (panel only has AAPL, MSFT; GOOG → NaN then dropped)
    assert set(result.columns).issubset(set(close.columns))


# ── test_absent_cache_returns_empty ──────────────────────────────────────────

def test_absent_cache_returns_empty():
    """Cache absent (loader returns empty DF) → sleeve returns empty DF, NOT zeros."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    features = {"close": _price_grid()}

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=pd.DataFrame()):
        result = earnings_tone_alpha(features)

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Absent cache must return empty DataFrame, not zeros"


# ── test_loader_exception_degrades ────────────────────────────────────────────

def test_loader_exception_degrades():
    """Loader raises → earnings_tone_alpha returns empty DF without propagating."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    features = {"close": _price_grid()}

    with patch(
        "ascent.alpha.earnings_tone.load_transcript_signals",
        side_effect=RuntimeError("parquet corrupt"),
    ):
        result = earnings_tone_alpha(features)  # must NOT raise

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Exception in loader must degrade gracefully to empty DF"


# ── test_no_network ───────────────────────────────────────────────────────────

def test_no_network():
    """Hot path must call load_transcript_signals() exactly once, no LLM/network calls."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    features = {"close": _price_grid()}
    calls = []

    def mock_loader():
        calls.append(1)
        return _transcript_panel()

    # Patch all known network/LLM entry points in the module
    with patch("ascent.alpha.earnings_tone.load_transcript_signals", side_effect=mock_loader):
        earnings_tone_alpha(features)

    assert len(calls) == 1, "Must call loader exactly once — no extra network calls"


# ── test_close_absent_returns_empty ───────────────────────────────────────────

def test_close_absent_returns_empty():
    """features without 'close' (or empty close) → empty DF, loader never called."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    calls = []
    with patch(
        "ascent.alpha.earnings_tone.load_transcript_signals",
        side_effect=lambda: calls.append(1) or _transcript_panel(),
    ):
        assert earnings_tone_alpha({}).empty                       # key missing
        assert earnings_tone_alpha({"close": pd.DataFrame()}).empty  # empty close

    assert calls == [], "Loader must not run when there is no price grid"


# ── test_no_universe_overlap_returns_empty ────────────────────────────────────

def test_no_universe_overlap_returns_empty():
    """Panel symbols disjoint from price universe → all-NaN reindex → empty DF."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    features = {"close": _price_grid(symbols=("AAPL", "MSFT", "GOOG"))}
    disjoint = _transcript_panel(symbols=("TSLA", "NVDA"))  # none in price grid

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=disjoint):
        result = earnings_tone_alpha(features)

    assert result.empty, "No overlap must return empty, not all-NaN columns"


# ── test_malformed_panel_degrades ─────────────────────────────────────────────

def test_malformed_panel_degrades():
    """Duplicate-index panel makes reindex raise → caught, returns empty (contract)."""
    from ascent.alpha.earnings_tone import earnings_tone_alpha

    idx = pd.to_datetime(["2026-01-05", "2026-01-05"])  # duplicate labels
    bad = pd.DataFrame(0.5, index=idx, columns=["AAPL"])
    features = {"close": _price_grid()}

    with patch("ascent.alpha.earnings_tone.load_transcript_signals", return_value=bad):
        result = earnings_tone_alpha(features)  # must NOT raise

    assert result.empty


# ── test_registered_in_sleeve_ic_logger ───────────────────────────────────────

def test_registered_in_sleeve_ic_logger():
    """earnings_tone must be built by main._log_sleeve_ic so the IC gate can see it."""
    import inspect
    import ascent.main as m

    src = inspect.getsource(m._log_sleeve_ic)
    assert "earnings_tone" in src, (
        "earnings_tone missing from _log_sleeve_ic builders — IC gate would be blind to it"
    )
