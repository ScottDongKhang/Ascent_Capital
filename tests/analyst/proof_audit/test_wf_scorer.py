"""score_sleeve wires signal + forward returns + stats.py together correctly."""
import pandas as pd
import pytest

from ascent.analyst.proof_audit.wf_scorer import score_sleeve


def _planted_features_and_prices(n_days=40, n_symbols=25):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng_signal = pd.DataFrame(
        [[(i + j) % 7 - 3 for j in range(n_symbols)] for i in range(n_days)],
        index=dates, columns=symbols,
    ).astype(float)
    # Prices constructed so tomorrow's return is proportional to today's signal
    # (planted positive IC) plus small noise.
    prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    prices.iloc[0] = 100.0
    for i in range(1, n_days):
        planted_ret = 0.01 * rng_signal.iloc[i - 1] / (rng_signal.iloc[i - 1].abs().max() or 1)
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
