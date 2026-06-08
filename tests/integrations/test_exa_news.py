import pytest
from unittest.mock import patch, MagicMock
import time


def _mock_exa_response(summaries: list[str]):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "results": [
            {"summary": {"answer": s}} for s in summaries
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_news_returns_dict():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["CAT beats Q1", "Infrastructure up"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"])
    assert "CAT" in result
    assert isinstance(result["CAT"], list)


def test_fetch_news_returns_summaries():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["CAT beats Q1", "Infrastructure up"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"], max_per_symbol=2)
    assert result["CAT"] == ["CAT beats Q1", "Infrastructure up"]


def test_fetch_news_caps_at_max_per_symbol():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["A", "B", "C"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"], max_per_symbol=2)
    assert len(result["CAT"]) <= 2


def test_fetch_news_returns_empty_list_on_failure():
    def bad_post(*a, **kw):
        raise RuntimeError("network error")
    with patch("ascent.integrations.exa_news.requests.post", side_effect=bad_post):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"])
    assert result["CAT"] == []


def test_fetch_news_skips_without_api_key():
    with patch("ascent.integrations.exa_news.os.getenv", return_value=""):
        from ascent.integrations.exa_news import fetch_news
        result = fetch_news(["CAT", "MRK"])
    assert result == {"CAT": [], "MRK": []}


def test_fetch_news_respects_delay():
    sleep_calls = []

    def record_sleep(n):
        sleep_calls.append(n)

    with patch("ascent.integrations.exa_news.time.sleep", side_effect=record_sleep):
        with patch("ascent.integrations.exa_news.requests.post",
                   return_value=_mock_exa_response(["headline"])):
            with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
                from ascent.integrations.exa_news import fetch_news
                fetch_news(["CAT", "MRK"])
    assert any(t >= 0.2 for t in sleep_calls), "Expected ≥0.2s delay between requests"
