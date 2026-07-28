"""Four AI PM defects found in the 2026-07-27 audit but not fixed at the time.

None changes trading logic; each silently removes information the agent or the
scoring layer was supposed to have.

A. Phase 1 has never seen the current portfolio. It read
   `data_cache/merged_weights.json`, while every writer in the repo uses
   `execution/merged_weights.json` — and even at the right path it took
   `.keys()` of the wrapper object, which would have yielded
   date/weights/agents/generated_at rather than tickers. So
   `_portfolio_symbols` was always [] and the whole CAUSAL INTELLIGENCE block
   was never injected.

B. `get_attribution_history` can never return data. It filters
   `r.get("symbol") == symbol` over logs/attribution_log.jsonl, whose 314 rows
   are portfolio-level: 0 carry a top-level `symbol`. Per-symbol data lives
   nested in `all_positions`. The tool is offered in BOTH phases and burns a
   call from a hard budget to say "no history".

C. Pattern memory has never reached a prompt. `update_pattern_memory` does a
   bare `json.loads` on a 200-token Haiku reply inside a try/except that only
   logs; one fenced or truncated response discards the learning, and
   `data_cache/ai_pm_pattern_memory.json` does not exist despite a post-mortem
   having been written.

D. A force-sealed run is indistinguishable from a real decision in
   `ai_pm_decision_log.jsonl`. `force_sealed` reaches the holdings log and the
   snapshots but not the decision log — which is the file that feeds override
   scoring and the authority ladder.
"""

import json

import pytest


# ── A. current holdings ───────────────────────────────────────────────────────

class TestLoadCurrentHoldings:
    def test_reads_the_weights_dict_not_the_wrapper(self, tmp_path):
        from agents.ai_pm_agent import _load_current_holdings
        p = tmp_path / "merged_weights.json"
        p.write_text(json.dumps({
            "date": "2026-07-27",
            "weights": {"UUP": 0.08, "TLT": 0.07},
            "agents": ["us_equities"],
            "generated_at": "2026-07-27T00:00:00",
        }))
        assert sorted(_load_current_holdings(p)) == ["TLT", "UUP"]

    def test_does_not_return_wrapper_keys(self, tmp_path):
        """The original bug would have yielded 'date', 'weights', 'agents'."""
        from agents.ai_pm_agent import _load_current_holdings
        p = tmp_path / "merged_weights.json"
        p.write_text(json.dumps({"date": "x", "weights": {"AAPL": 1.0}}))
        got = _load_current_holdings(p)
        assert "date" not in got and "weights" not in got

    def test_tolerates_a_bare_symbol_to_weight_mapping(self, tmp_path):
        from agents.ai_pm_agent import _load_current_holdings
        p = tmp_path / "mw.json"
        p.write_text(json.dumps({"AAPL": 0.5, "MSFT": 0.5}))
        assert sorted(_load_current_holdings(p)) == ["AAPL", "MSFT"]

    def test_missing_or_malformed_file_yields_empty(self, tmp_path):
        from agents.ai_pm_agent import _load_current_holdings
        assert _load_current_holdings(tmp_path / "nope.json") == []
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert _load_current_holdings(bad) == []

    def test_default_path_is_the_one_writers_use(self, monkeypatch, tmp_path):
        """Behavioural, not a source grep: put a book at execution/ and one at
        data_cache/ and check which the default actually reads. (A grep would
        trip over the docstring, which names the old path to explain the bug.)"""
        import agents.ai_pm_agent as m
        (tmp_path / "execution").mkdir()
        (tmp_path / "data_cache").mkdir()
        (tmp_path / "execution" / "merged_weights.json").write_text(
            json.dumps({"weights": {"RIGHT": 1.0}}))
        (tmp_path / "data_cache" / "merged_weights.json").write_text(
            json.dumps({"weights": {"WRONG": 1.0}}))
        monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)
        assert m._load_current_holdings() == ["RIGHT"]

    def test_reads_the_live_book_in_this_repo(self):
        """End-to-end against the real file, which the old path never found."""
        from agents.ai_pm_agent import _load_current_holdings
        got = _load_current_holdings()
        assert isinstance(got, list) and got, "current book should be non-empty"
        assert all(isinstance(s, str) and s.isupper() for s in got)


# ── B. attribution history ────────────────────────────────────────────────────

class TestAttributionHistory:
    def _log(self, tmp_path, monkeypatch):
        import agents.ai_pm_agent as m
        d = tmp_path / "logs"
        d.mkdir()
        (d / "attribution_log.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"date": "2026-07-20", "portfolio_return": 0.01, "spy_return": 0.005,
             "all_positions": [{"symbol": "UUP", "weight": 0.08, "return": -0.02,
                                "contribution": -0.0016},
                               {"symbol": "TLT", "weight": 0.07, "return": 0.01,
                                "contribution": 0.0007}]},
            {"date": "2026-07-21", "portfolio_return": -0.004, "spy_return": 0.001,
             "all_positions": [{"symbol": "UUP", "weight": 0.08, "return": 0.03,
                                "contribution": 0.0024}]},
        ]) + "\n")
        monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)
        return m

    def test_finds_a_symbol_nested_in_all_positions(self, tmp_path, monkeypatch):
        m = self._log(tmp_path, monkeypatch)
        out = m._tool_get_attribution_history({"symbol": "UUP"})
        assert "UUP" in out
        assert "No attribution history" not in out

    def test_reports_both_dates_for_that_symbol(self, tmp_path, monkeypatch):
        m = self._log(tmp_path, monkeypatch)
        out = m._tool_get_attribution_history({"symbol": "UUP"})
        assert "2026-07-20" in out and "2026-07-21" in out

    def test_unheld_symbol_still_says_so(self, tmp_path, monkeypatch):
        m = self._log(tmp_path, monkeypatch)
        assert "No attribution history" in m._tool_get_attribution_history({"symbol": "NVDA"})

    def test_missing_log_is_handled(self, tmp_path, monkeypatch):
        import agents.ai_pm_agent as m
        monkeypatch.setattr(m, "_REPO_ROOT", tmp_path)
        assert "not found" in m._tool_get_attribution_history({"symbol": "UUP"}).lower()


# ── C. pattern-memory extraction ──────────────────────────────────────────────

class TestPatternJsonParsing:
    def test_plain_json(self):
        from ascent.strategy.ai_pm_learning import _parse_pattern_json
        assert _parse_pattern_json('{"avoid": ["a"], "work": ["b"]}') == {
            "avoid": ["a"], "work": ["b"]}

    def test_markdown_fenced(self):
        """Haiku commonly wraps JSON in a fence; a bare json.loads discards it."""
        from ascent.strategy.ai_pm_learning import _parse_pattern_json
        txt = '```json\n{"avoid": ["a"], "work": []}\n```'
        assert _parse_pattern_json(txt) == {"avoid": ["a"], "work": []}

    def test_json_embedded_in_prose(self):
        from ascent.strategy.ai_pm_learning import _parse_pattern_json
        txt = 'Here are the rules:\n{"avoid": ["x"], "work": ["y"]}\nHope that helps.'
        assert _parse_pattern_json(txt) == {"avoid": ["x"], "work": ["y"]}

    def test_unparseable_yields_empty_arrays_not_an_exception(self):
        from ascent.strategy.ai_pm_learning import _parse_pattern_json
        assert _parse_pattern_json("no json here") == {"avoid": [], "work": []}
        assert _parse_pattern_json("") == {"avoid": [], "work": []}

    def test_non_list_values_are_coerced_away(self):
        from ascent.strategy.ai_pm_learning import _parse_pattern_json
        assert _parse_pattern_json('{"avoid": "a string", "work": null}') == {
            "avoid": [], "work": []}

    def test_token_budget_is_not_starved(self):
        import inspect
        from ascent.strategy.ai_pm_learning import update_pattern_memory
        src = inspect.getsource(update_pattern_memory)
        assert "max_tokens=200" not in src, "200 tokens truncates two rule arrays"


# ── D. force-seal visibility ──────────────────────────────────────────────────

class TestForceSealIsRecorded:
    def test_decision_log_records_force_sealed(self):
        import inspect
        import run_all_agents as ra
        src = inspect.getsource(ra._write_decision_log)
        assert "force_sealed" in src, (
            "a force-sealed run must not look like a real decision in the file "
            "that feeds override scoring and the authority ladder"
        )
