import pytest
from unittest.mock import patch, MagicMock
import json


MOCK_TWITS_RESPONSE = {
    "messages": [
        {"body": "CAT breaking out!", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "selling CAT",       "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "CAT hold",          "entities": {}},  # no sentiment tag
        {"body": "CAT looks good",    "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "CAT looks bad",     "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "CAT looking ok",    "entities": {"sentiment": {"basic": "Bullish"}}},
    ]
}


def _mock_get(url, timeout):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = MOCK_TWITS_RESPONSE
    return resp


def test_get_sentiment_returns_dict():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert "CAT" in result
    assert isinstance(result["CAT"]["bullish"], int)
    assert isinstance(result["CAT"]["bearish"], int)


def test_get_sentiment_counts_labels():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    # 3 bullish, 2 bearish in mock data
    assert result["CAT"]["bullish"] == 3
    assert result["CAT"]["bearish"] == 2
    assert result["CAT"]["n_labeled"] == 5
    assert result["CAT"]["n_total"] == 6


def test_get_sentiment_computes_ratio():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["ratio"] == pytest.approx(3 / 5, abs=0.01)


def test_get_sentiment_band_bullish():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["band"] == "bullish"  # ratio 0.60 → bullish


def test_get_sentiment_stale_when_few_labels():
    """Less than 5 labeled messages → stale=True."""
    sparse = {"messages": [
        {"body": "x", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "y", "entities": {}},
    ]}
    def _mock_sparse(url, timeout):
        r = MagicMock(); r.status_code = 200; r.json.return_value = sparse; return r
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_sparse):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["stale"] is True


def test_get_sentiment_handles_api_error():
    """Network error → returns stale entry, does not raise."""
    def _mock_err(url, timeout):
        raise ConnectionError("network down")
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_err):
        result = get_sentiment(["CAT"])
    assert "CAT" in result
    assert result["CAT"]["stale"] is True


def test_format_sentiment_block_shows_band():
    from ascent.integrations.stocktwits import format_sentiment_block
    data = {
        "CAT": {"bullish": 18, "bearish": 5, "n_labeled": 23, "n_total": 30,
                "ratio": 0.78, "band": "bullish", "stale": False},
        "MRK": {"bullish": 3,  "bearish": 14, "n_labeled": 17, "n_total": 30,
                "ratio": 0.18, "band": "strongly_bearish", "stale": False},
    }
    block = format_sentiment_block(data)
    assert "CAT" in block
    assert "bullish" in block.lower()
    assert "MRK" in block
    assert "strongly_bearish" in block.lower()


def test_format_sentiment_block_skips_stale():
    from ascent.integrations.stocktwits import format_sentiment_block
    data = {
        "CAT": {"bullish": 1, "bearish": 0, "n_labeled": 1, "n_total": 30,
                "ratio": 1.0, "band": "strongly_bullish", "stale": True},
    }
    block = format_sentiment_block(data)
    assert "CAT" not in block or "stale" in block.lower()
