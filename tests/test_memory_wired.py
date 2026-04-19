# tests/test_memory_wired.py
import inspect


def test_build_context_queries_memory():
    """_build_context must call query_memory when no memory_context pre-loaded."""
    import debate.agents as agents_module
    src = inspect.getsource(agents_module._build_context)
    assert "query_memory" in src, "_build_context must call query_memory"


def test_memory_context_injected_when_results_returned(monkeypatch):
    """If query_memory returns results, they must appear in the context string."""
    from unittest.mock import patch

    fake_results = [{"date": "2026-04-15", "recommendation": "reduce_size",
                     "reasoning": "EM overweight in stressed regime paid off — cut worked.",
                     "regime": "stressed", "key_risks": ["EM exposure"]}]

    with patch("memory.r2r_interface.query_memory", return_value=fake_results):
        from debate.agents import _build_context
        state = {
            "date": "2026-04-29",
            "us_regime": "stressed",
            "n_positions": 15,
            "allocation": {},
            "weights": {"EWY": 0.10, "PDBC": 0.09},
        }
        ctx = _build_context(state)
        assert "Historical memory" in ctx or "reduce_size" in ctx, \
            "Memory results must appear in context"


def test_memory_skipped_when_no_results(monkeypatch):
    """If query_memory returns empty, context must not crash."""
    from unittest.mock import patch

    with patch("memory.r2r_interface.query_memory", return_value=[]):
        from debate.agents import _build_context
        state = {
            "date": "2026-04-29",
            "us_regime": "calm_bull",
            "n_positions": 10,
            "allocation": {},
            "weights": {"AAPL": 0.08},
        }
        ctx = _build_context(state)
        assert "Date:" in ctx  # basic context still built
