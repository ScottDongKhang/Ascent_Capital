import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta


def _make_prices(symbols, n_days=90, seed=42):
    np.random.seed(seed)
    idx = pd.date_range(end=date.today(), periods=n_days, freq="B")
    returns = np.random.normal(0.0003, 0.012, size=(len(idx), len(symbols)))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=idx, columns=symbols)


def test_quant_context_keys():
    from ascent.monitoring.quant_context import build_quant_context
    weights = {"AAPL": 0.10, "MSFT": 0.10, "EEM": 0.08, "GLD": 0.07,
               "TLT": 0.06, "SPY": 0.05, "JPM": 0.09, "XOM": 0.05,
               "NEE": 0.06, "MRK": 0.06, "WMT": 0.05, "AMZN": 0.07,
               "NVDA": 0.08, "V": 0.06, "MA": 0.02}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)
    for key in ["portfolio_var_95", "portfolio_var_99", "factor_exposures",
                "sector_concentration", "top_correlated_pairs", "summary_text"]:
        assert key in ctx, f"quant_context missing key: {key}"


def test_quant_context_var_is_negative():
    from ascent.monitoring.quant_context import build_quant_context
    weights = {"AAPL": 0.30, "MSFT": 0.30, "AMZN": 0.20, "NVDA": 0.20}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)
    assert ctx["portfolio_var_95"] < 0
    assert ctx["portfolio_var_99"] < 0
    assert ctx["portfolio_var_99"] <= ctx["portfolio_var_95"]


def test_quant_context_factor_exposures():
    from ascent.monitoring.quant_context import build_quant_context
    weights = {"EEM": 0.12, "GLD": 0.08, "TLT": 0.10, "AAPL": 0.30, "JPM": 0.40}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)
    fe = ctx["factor_exposures"]
    assert "em_equity" in fe, "EEM must register as em_equity exposure"
    assert fe["em_equity"] > 0
    total = sum(fe.values())
    assert total <= 1.01


def test_quant_context_summary_text_has_var():
    from ascent.monitoring.quant_context import build_quant_context
    weights = {"AAPL": 0.25, "MSFT": 0.25, "EEM": 0.25, "GLD": 0.25}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)
    assert isinstance(ctx["summary_text"], str)
    assert len(ctx["summary_text"]) > 50
    assert "VaR" in ctx["summary_text"]


def test_extended_thinking_completion_signature():
    from ascent.llm.client import extended_thinking_completion
    import inspect
    sig = inspect.signature(extended_thinking_completion)
    assert "messages" in sig.parameters
    assert "thinking_budget" in sig.parameters


def test_chat_completion_accepts_use_cache():
    import inspect
    from ascent.llm.client import chat_completion
    sig = inspect.signature(chat_completion)
    assert "use_cache" in sig.parameters


def test_generate_structured_accepts_use_cache():
    import inspect
    from ascent.llm.client import generate_structured
    sig = inspect.signature(generate_structured)
    assert "use_cache" in sig.parameters
