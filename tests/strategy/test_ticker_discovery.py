import pytest
from unittest.mock import patch


MOCK_HIGH_CONVICTION_RESPONSE = '''{
  "symbol": "VMC",
  "conviction_score": 0.82,
  "catalyst_snippet": "Infrastructure spending accelerating — Vulcan Materials mentioned alongside CAT",
  "rationale": "CAT news about infrastructure contracts references VMC as key supplier"
}'''

MOCK_LOW_CONVICTION_RESPONSE = '''{
  "symbol": "XYZ",
  "conviction_score": 0.45,
  "catalyst_snippet": "Weak signal",
  "rationale": "Tenuous connection"
}'''


def test_run_discovery_returns_result_on_high_conviction():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=MOCK_HIGH_CONVICTION_RESPONSE):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Infrastructure bill boosts Vulcan Materials VMC"]},
            existing_universe=["CAT", "MRK", "NEE"],
        )
    assert result is not None
    assert result.symbol == "VMC"
    assert result.conviction_score == pytest.approx(0.82)


def test_run_discovery_returns_none_on_low_conviction():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=MOCK_LOW_CONVICTION_RESPONSE):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Some weak news"]},
            existing_universe=["CAT"],
        )
    assert result is None


def test_run_discovery_returns_none_if_symbol_already_in_universe():
    already_in = '''{
      "symbol": "CAT",
      "conviction_score": 0.90,
      "catalyst_snippet": "CAT is doing great",
      "rationale": "..."
    }'''
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=already_in):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["CAT headline"]},
            existing_universe=["CAT", "MRK"],
        )
    assert result is None


def test_run_discovery_returns_none_on_empty_news():
    from ascent.strategy.ticker_discovery import run_discovery
    result = run_discovery(news_context={}, existing_universe=["CAT"])
    assert result is None


def test_run_discovery_returns_none_on_all_empty_headlines():
    from ascent.strategy.ticker_discovery import run_discovery
    result = run_discovery(
        news_context={"CAT": [], "MRK": []},
        existing_universe=["CAT", "MRK"],
    )
    assert result is None


def test_run_discovery_returns_none_on_llm_failure():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               side_effect=RuntimeError("LLM down")):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Some news"]},
            existing_universe=["CAT"],
        )
    assert result is None


def test_run_discovery_truncates_catalyst_snippet():
    long_snippet = "x" * 500
    long_response = (
        f'{{"symbol": "VMC", "conviction_score": 0.85, '
        f'"catalyst_snippet": "{long_snippet}", "rationale": "ok"}}'
    )
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=long_response):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["big news"]},
            existing_universe=["CAT"],
        )
    assert result is not None
    assert len(result.catalyst_snippet) <= 200
