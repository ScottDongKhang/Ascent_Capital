"""score_sleeve wires signal + forward returns + stats.py together correctly."""
import numpy as np
import pandas as pd
import pytest

from ascent.analyst.proof_audit.wf_scorer import (
    DegenerateSignalError,
    MIN_DENSE_DATES,
    N_LEGS,
    score_agent,
    score_signal_matrix,
    score_sleeve,
)


def _planted_features_and_prices(n_days=40, n_symbols=25):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng_signal = pd.DataFrame(
        [[(i + j) % 7 - 3 for j in range(n_symbols)] for i in range(n_days)],
        index=dates, columns=symbols,
    ).astype(float)
    # Prices constructed so tomorrow's return is proportional to today's signal
    # (planted positive IC) plus per-symbol, per-date noise. The noise is essential:
    # without it, tomorrow's return is an exact monotonic function of today's signal, so
    # Spearman IC is exactly 1.0 on every single date -- zero variance across the daily-IC
    # series, which drives scipy's ttest_1samp into catastrophic cancellation (a
    # RuntimeWarning). Jitter breaks the perfect rank correlation so daily IC varies while
    # staying reliably positive on average.
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=1.5, size=(n_days, n_symbols))
    prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    prices.iloc[0] = 100.0
    for i in range(1, n_days):
        signal_row = rng_signal.iloc[i - 1]
        denom = signal_row.abs().max() or 1
        planted_ret = 0.01 * (signal_row + noise[i - 1]) / denom
        prices.iloc[i] = prices.iloc[i - 1] * (1 + planted_ret)
    features = {"toy_signal": rng_signal}
    return features, prices


def test_score_sleeve_detects_planted_positive_ic(monkeypatch):
    features, prices = _planted_features_and_prices()

    def fake_signal_func(features):
        return features["toy_signal"]

    monkeypatch.setitem(
        __import__(
            "ascent.analyst.proof_audit.sleeve_signals", fromlist=["SLEEVE_SIGNAL_FUNCS"]
        ).SLEEVE_SIGNAL_FUNCS,
        "trend",
        fake_signal_func,
    )
    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date",
        lambda date, universe_df=None: list(prices.columns),
    )
    result = score_sleeve("trend", features, prices)
    assert result.ic_mean > 0
    assert result.n > 0


def test_score_sleeve_unknown_name_raises():
    with pytest.raises(KeyError):
        score_sleeve("not_a_sleeve", {}, pd.DataFrame())


def test_score_agent_detects_planted_positive_ic(monkeypatch):
    """score_agent wires AGENT_SIGNAL_FUNCS + forward returns + stats.py together correctly.

    Uses the same planted-correlation-plus-noise fixture as
    test_score_sleeve_detects_planted_positive_ic, for the same reason: without per-date noise,
    the daily Spearman IC is exactly 1.0 on every date, which is zero-variance input to scipy's
    ttest_1samp and trips a RuntimeWarning.
    """
    _, prices = _planted_features_and_prices()
    rng_signal = pd.DataFrame(
        [[(i + j) % 7 - 3 for j in range(len(prices.columns))] for i in range(len(prices))],
        index=prices.index, columns=prices.columns,
    ).astype(float)

    def fake_agent_signal(prices):
        return rng_signal

    monkeypatch.setitem(
        __import__(
            "ascent.analyst.proof_audit.agent_signals", fromlist=["AGENT_SIGNAL_FUNCS"]
        ).AGENT_SIGNAL_FUNCS,
        "alternatives_agent",
        fake_agent_signal,
    )
    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date",
        lambda date, universe_df=None: list(prices.columns),
    )
    result = score_agent("alternatives_agent", prices)
    assert result.ic_mean > 0
    assert result.n > 0


def test_score_agent_unknown_name_raises():
    with pytest.raises(KeyError):
        score_agent("not_an_agent", pd.DataFrame())


def test_score_signal_matrix_rejects_all_nan_signal():
    """An all-NaN signal matrix must raise, not silently score whatever falls out.

    This is the alternatives_agent failure mode: its composite depends on `vol_21d`, which is
    universally NaN on the real prices_live matrix, so the whole signal comes back NaN.
    """
    _, prices = _planted_features_and_prices()
    signal = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    with pytest.raises(DegenerateSignalError) as exc_info:
        score_signal_matrix(signal, prices, dates=list(prices.index[:-1]))
    assert "density" in str(exc_info.value)
    assert f"{N_LEGS * 2}" in str(exc_info.value)


def test_score_signal_matrix_rejects_too_few_dense_dates():
    """A signal with real values on only a handful of dates is degenerate too."""
    _, prices = _planted_features_and_prices()
    rng = np.random.default_rng(7)
    signal = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    dense_dates = prices.index[: MIN_DENSE_DATES - 1]
    for d in dense_dates:
        signal.loc[d] = rng.normal(size=len(prices.columns))
    with pytest.raises(DegenerateSignalError):
        score_signal_matrix(signal, prices, dates=list(prices.index[:-1]))


def test_score_signal_matrix_rejects_too_few_non_nan_symbols_per_date():
    """Dates carrying fewer than N_LEGS*2 real values do not count toward density."""
    _, prices = _planted_features_and_prices()
    rng = np.random.default_rng(11)
    signal = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    thin_symbols = list(prices.columns[: N_LEGS * 2 - 1])
    for d in prices.index:
        signal.loc[d, thin_symbols] = rng.normal(size=len(thin_symbols))
    with pytest.raises(DegenerateSignalError):
        score_signal_matrix(signal, prices, dates=list(prices.index[:-1]))


def test_score_signal_matrix_accepts_precomputed_dates(monkeypatch):
    """Passing `dates` skips the per-component point-in-time universe scan entirely."""
    features, prices = _planted_features_and_prices()

    def explode(date, universe_df=None):
        raise AssertionError("get_universe_on_date must not be called when dates are supplied")

    monkeypatch.setattr(
        "ascent.analyst.proof_audit.forward_returns.get_universe_on_date", explode
    )
    result = score_signal_matrix(
        features["toy_signal"], prices, dates=list(prices.index[:-1])
    )
    assert result.n > 0
