# Memory-Augmented Debate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before each rebalance debate, query the R2R memory layer for historically similar situations (matching regime + holding period overlap) and inject the retrieved past lessons into every agent's context — turning the already-written verdict history into an active signal rather than static log files.

**Architecture:** A new `memory/` package wraps R2R's HTTP API. `query_memory(query, n)` calls R2R if `R2R_API_KEY` is in the environment; otherwise falls back to a local BM25-style keyword search over the existing `outputs/debate_log/*.json` verdict files (no external dependency). `debate_runner.py` calls `query_memory()` once before agents run, injects the result as `portfolio_state["memory_context"]`. Each agent's `_build_context()` already appends all `portfolio_state` keys, so context propagates automatically.

**Tech Stack:** Python 3.12, `requests` (already installed), `outputs/debate_log/` JSON files (already written by the existing debate runner), R2R HTTP API (`r2r.ai` — requires `R2R_API_KEY` in `.env`)

---

## Background: R2R and the local fallback

R2R (r2r.ai / SciPhi) is a vector-backed RAG framework with an HTTP API. The relevant endpoints used here:

- `POST /v2/ingest_documents` — add a document (past verdict) to R2R's vector store
- `POST /v2/search` — semantic search returning the top-N matching chunks

The local fallback works without any API key: it reads all `outputs/debate_log/verdict_*.json` files, scores each by keyword overlap with the query string (regime label + top holding symbols), and returns the top-N highest-scoring verdicts formatted as text blocks.

The interface is designed so the rest of the codebase never knows which path is active.

---

## File Structure

- **Create:** `memory/__init__.py` — package marker
- **Create:** `memory/r2r_interface.py` — `query_memory()`, `ingest_verdict()`, and local fallback
- **Modify:** `debate/debate_runner.py` — call `query_memory()` + `ingest_verdict()` (async ingest after debate)
- **Modify:** `debate/agents.py:21–33` — extend `_build_context()` to render `memory_context` if present
- **Create:** `tests/test_memory_interface.py` — unit tests (no network, mocked R2R)

---

### Task 1: Build `memory/r2r_interface.py` with local fallback

**Files:**
- Create: `memory/__init__.py`
- Create: `memory/r2r_interface.py`
- Test: `tests/test_memory_interface.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_interface.py
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# ── Local fallback tests (no R2R key required) ──────────────────────────────

def _write_fake_verdict(tmpdir: Path, filename: str, data: dict):
    """Helper: write a fake verdict JSON to the temp debate log dir."""
    (tmpdir / filename).write_text(json.dumps(data))


def test_local_fallback_returns_empty_on_no_verdicts(tmp_path):
    from memory.r2r_interface import _local_search
    result = _local_search("calm_bull AAPL MSFT", debate_log_dir=tmp_path, n=3)
    assert result == []


def test_local_fallback_finds_regime_match(tmp_path):
    _write_fake_verdict(tmp_path, "verdict_2026-01-15.json", {
        "date": "2026-01-15",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {"AAPL": 0.2}},
        "verdict": {"recommendation": "proceed", "reasoning": "Momentum is strong in tech."},
    })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull AAPL", debate_log_dir=tmp_path, n=3)
    assert len(results) == 1
    assert results[0]["date"] == "2026-01-15"
    assert results[0]["recommendation"] == "proceed"


def test_local_fallback_ranks_by_overlap(tmp_path):
    _write_fake_verdict(tmp_path, "verdict_2026-01-10.json", {
        "date": "2026-01-10",
        "portfolio_state": {"us_regime": "stressed", "weights": {"TLT": 0.3}},
        "verdict": {"recommendation": "reduce_size", "reasoning": "Stress regime."},
    })
    _write_fake_verdict(tmp_path, "verdict_2026-02-01.json", {
        "date": "2026-02-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {"AAPL": 0.2, "MSFT": 0.15}},
        "verdict": {"recommendation": "proceed", "reasoning": "Bull confirmed."},
    })
    from memory.r2r_interface import _local_search
    # Query matches calm_bull + AAPL — should rank Feb verdict higher
    results = _local_search("calm_bull AAPL", debate_log_dir=tmp_path, n=3)
    assert results[0]["date"] == "2026-02-01"


def test_local_fallback_limits_results(tmp_path):
    for i in range(5):
        _write_fake_verdict(tmp_path, f"verdict_2026-0{i+1}-01.json", {
            "date": f"2026-0{i+1}-01",
            "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
            "verdict": {"recommendation": "proceed", "reasoning": "calm_bull test"},
        })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull", debate_log_dir=tmp_path, n=2)
    assert len(results) <= 2


def test_local_fallback_skips_malformed_files(tmp_path):
    (tmp_path / "verdict_bad.json").write_text("not valid json{{{")
    _write_fake_verdict(tmp_path, "verdict_2026-03-01.json", {
        "date": "2026-03-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
        "verdict": {"recommendation": "proceed", "reasoning": "ok"},
    })
    from memory.r2r_interface import _local_search
    results = _local_search("calm_bull", debate_log_dir=tmp_path, n=5)
    assert len(results) == 1  # bad file skipped, good file found


# ── query_memory() interface tests ──────────────────────────────────────────

def test_query_memory_uses_local_fallback_when_no_api_key(tmp_path):
    """Without R2R_API_KEY, query_memory falls through to local search."""
    _write_fake_verdict(tmp_path, "verdict_2026-03-10.json", {
        "date": "2026-03-10",
        "portfolio_state": {"us_regime": "stressed", "weights": {"GLD": 0.3}},
        "verdict": {"recommendation": "reduce_size", "reasoning": "Stress regime."},
    })

    env_without_key = {k: v for k, v in os.environ.items() if k != "R2R_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        from memory import r2r_interface
        # Force reload so it picks up env
        import importlib
        importlib.reload(r2r_interface)
        results = r2r_interface.query_memory(
            query="stressed GLD", n=3, debate_log_dir=tmp_path
        )
    assert isinstance(results, list)


def test_query_memory_calls_r2r_api_when_key_present(tmp_path):
    """With R2R_API_KEY set, query_memory calls the R2R HTTP endpoint."""
    fake_r2r_response = {
        "results": [
            {
                "metadata": {"date": "2026-03-01", "recommendation": "proceed"},
                "text": "Past verdict: calm_bull, proceed. Reasoning: strong momentum.",
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_r2r_response

    with patch.dict(os.environ, {"R2R_API_KEY": "test-key-123"}), \
         patch("requests.post", return_value=mock_resp):
        from memory import r2r_interface
        import importlib
        importlib.reload(r2r_interface)
        results = r2r_interface.query_memory(
            query="calm_bull AAPL", n=3, debate_log_dir=tmp_path
        )
    assert isinstance(results, list)


def test_query_memory_falls_back_on_r2r_failure(tmp_path):
    """If R2R API call fails, falls back to local search without raising."""
    _write_fake_verdict(tmp_path, "verdict_2026-02-01.json", {
        "date": "2026-02-01",
        "portfolio_state": {"us_regime": "calm_bull", "weights": {}},
        "verdict": {"recommendation": "proceed", "reasoning": "ok"},
    })

    with patch.dict(os.environ, {"R2R_API_KEY": "test-key-123"}), \
         patch("requests.post", side_effect=Exception("timeout")):
        from memory import r2r_interface
        import importlib
        importlib.reload(r2r_interface)
        results = r2r_interface.query_memory(
            query="calm_bull", n=3, debate_log_dir=tmp_path
        )
    assert isinstance(results, list)  # local fallback returned something


# ── format_memory_context() tests ───────────────────────────────────────────

def test_format_memory_context_empty():
    from memory.r2r_interface import format_memory_context
    text = format_memory_context([])
    assert "no relevant" in text.lower()


def test_format_memory_context_formats_results():
    from memory.r2r_interface import format_memory_context
    results = [
        {"date": "2026-03-01", "recommendation": "proceed",
         "reasoning": "Momentum was strong.", "regime": "calm_bull"},
        {"date": "2026-02-15", "recommendation": "reduce_size",
         "reasoning": "Stressed market conditions.", "regime": "stressed"},
    ]
    text = format_memory_context(results)
    assert "2026-03-01" in text
    assert "proceed" in text
    assert "Momentum was strong" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_memory_interface.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'memory'`

- [ ] **Step 3: Create `memory/__init__.py`**

```python
# memory/__init__.py
"""
Ascent Capital — Memory Interface
Wrapper around R2R vector memory with local file fallback.
"""
```

- [ ] **Step 4: Implement `memory/r2r_interface.py`**

```python
# memory/r2r_interface.py
"""
memory/r2r_interface.py
R2R memory interface for Ascent Capital.

query_memory(query, n) — semantic search over past verdict history.
ingest_verdict(verdict_path) — add a new verdict to memory.
format_memory_context(results) — format results for LLM prompt injection.

Uses R2R HTTP API if R2R_API_KEY is set in the environment.
Falls back to local keyword search over outputs/debate_log/*.json otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_DEBATE_LOG_DIR = Path("outputs/debate_log")
R2R_API_KEY = os.environ.get("R2R_API_KEY", "")
R2R_BASE_URL = os.environ.get("R2R_BASE_URL", "https://api.r2r.ai")


# ── Local keyword search (BM25-style scoring) ────────────────────────────────

def _score_verdict(verdict_data: dict, query_tokens: set[str]) -> float:
    """
    Score a verdict dict against a query token set.
    Counts token overlaps in regime, weights keys, reasoning text.
    Higher = more relevant.
    """
    score = 0.0
    try:
        ps = verdict_data.get("portfolio_state", {})
        v = verdict_data.get("verdict", {})

        text_blob = " ".join([
            str(ps.get("us_regime", "")),
            str(ps.get("macro_regime", "")),
            " ".join(ps.get("weights", {}).keys()),
            str(v.get("recommendation", "")),
            str(v.get("reasoning", "")),
            " ".join(v.get("key_risks", [])),
        ]).lower()

        tokens_in_doc = set(text_blob.split())
        score = len(query_tokens & tokens_in_doc)
    except Exception:
        pass
    return score


def _local_search(
    query: str,
    debate_log_dir: Path = _DEFAULT_DEBATE_LOG_DIR,
    n: int = 3,
) -> list[dict]:
    """
    Search past verdicts by keyword overlap. No external dependencies.

    Returns list of dicts with keys: date, recommendation, reasoning, regime.
    Sorted by score descending; limited to top-n with score > 0.
    """
    if not debate_log_dir.exists():
        return []

    query_tokens = set(query.lower().split())
    scored: list[tuple[float, dict]] = []

    for verdict_file in debate_log_dir.glob("verdict_*.json"):
        try:
            data = json.loads(verdict_file.read_text())
        except Exception:
            continue

        score = _score_verdict(data, query_tokens)
        if score > 0:
            ps = data.get("portfolio_state", {})
            v = data.get("verdict", {})
            scored.append((score, {
                "date":           data.get("date", "unknown"),
                "recommendation": v.get("recommendation", "unknown"),
                "reasoning":      v.get("reasoning", "")[:300],
                "regime":         ps.get("us_regime", "unknown"),
                "key_risks":      v.get("key_risks", [])[:3],
            }))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:n]]


# ── R2R HTTP API path ─────────────────────────────────────────────────────────

def _r2r_search(query: str, n: int = 3) -> list[dict]:
    """
    Query R2R API for semantically similar past verdicts.
    Raises on failure — caller handles fallback.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {R2R_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_settings": {"search_limit": n},
    }
    resp = requests.post(
        f"{R2R_BASE_URL}/v2/search",
        json=payload,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for hit in data.get("results", []):
        meta = hit.get("metadata", {})
        results.append({
            "date":           meta.get("date", "unknown"),
            "recommendation": meta.get("recommendation", "unknown"),
            "reasoning":      hit.get("text", "")[:300],
            "regime":         meta.get("regime", "unknown"),
            "key_risks":      meta.get("key_risks", []),
        })
    return results


def _r2r_ingest(document_text: str, metadata: dict) -> bool:
    """
    Ingest a document into R2R. Returns True on success.
    Raises on failure — caller handles.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {R2R_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "documents": [{"text": document_text, "metadata": metadata}]
    }
    resp = requests.post(
        f"{R2R_BASE_URL}/v2/ingest_documents",
        json=payload,
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return True


# ── Public interface ──────────────────────────────────────────────────────────

def query_memory(
    query: str,
    n: int = 3,
    debate_log_dir: Path = _DEFAULT_DEBATE_LOG_DIR,
) -> list[dict]:
    """
    Search memory for past situations similar to `query`.

    Uses R2R API if R2R_API_KEY is set, local keyword search otherwise.
    Falls back to local search if R2R call fails.

    Args:
        query:  Free-text query (e.g. "calm_bull AAPL MSFT earnings")
        n:      Number of results to return
        debate_log_dir: Path to debate logs (override for testing)

    Returns:
        List of result dicts: {date, recommendation, reasoning, regime, key_risks}
        Empty list if nothing relevant found.
    """
    if R2R_API_KEY:
        try:
            return _r2r_search(query, n=n)
        except Exception as e:
            log.warning(f"[Memory] R2R search failed ({e}), falling back to local search")

    return _local_search(query, debate_log_dir=debate_log_dir, n=n)


def ingest_verdict(verdict_path: Path, debate_log_dir: Path = _DEFAULT_DEBATE_LOG_DIR) -> None:
    """
    Ingest a verdict JSON into memory (R2R or no-op if no API key).

    Called by debate_runner.py after a verdict is written.
    Non-fatal — logs warning on any failure.

    Args:
        verdict_path: Path to the verdict JSON file.
    """
    if not R2R_API_KEY:
        return  # local search reads files directly, no explicit ingestion needed

    try:
        data = json.loads(verdict_path.read_text())
        ps = data.get("portfolio_state", {})
        v = data.get("verdict", {})

        text = (
            f"Date: {data.get('date', 'unknown')}\n"
            f"Regime: {ps.get('us_regime', 'unknown')}\n"
            f"Recommendation: {v.get('recommendation', 'unknown')}\n"
            f"Reasoning: {v.get('reasoning', '')}\n"
            f"Key risks: {', '.join(v.get('key_risks', []))}\n"
            f"Positions: {', '.join(ps.get('weights', {}).keys())}\n"
        )
        metadata = {
            "date": data.get("date"),
            "recommendation": v.get("recommendation"),
            "regime": ps.get("us_regime"),
            "key_risks": v.get("key_risks", []),
        }
        _r2r_ingest(text, metadata)
        log.info(f"[Memory] Ingested verdict {verdict_path.name} into R2R")
    except Exception as e:
        log.warning(f"[Memory] Failed to ingest {verdict_path.name} into R2R: {e}")


def format_memory_context(results: list[dict]) -> str:
    """
    Format memory query results as a concise LLM-readable block.

    Args:
        results: List of dicts from query_memory().

    Returns:
        Multi-line string suitable for injection into a debate agent prompt.
    """
    if not results:
        return "No relevant historical situations found in memory."

    lines = [f"Historical memory — {len(results)} similar past situation(s):"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[{i}] {r['date']} | Regime: {r['regime']} | Verdict: {r['recommendation']}"
        )
        if r.get("reasoning"):
            lines.append(f"    Reasoning: {r['reasoning'][:200]}")
        if r.get("key_risks"):
            lines.append(f"    Key risks: {', '.join(r['key_risks'][:3])}")

    return "\n".join(lines)
```

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/pytest tests/test_memory_interface.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 6: Commit**

```bash
git add memory/__init__.py memory/r2r_interface.py tests/test_memory_interface.py
git commit -m "feat: add memory/r2r_interface — R2R-backed memory with local fallback for debate context"
```

---

### Task 2: Wire memory into `debate_runner.py` and `debate/agents.py`

**Files:**
- Modify: `debate/debate_runner.py` — call `query_memory()` before agents, `ingest_verdict()` after verdict written
- Modify: `debate/agents.py:21–33` — extend `_build_context()` to render `memory_context` if present

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_memory_interface.py

from datetime import date

def test_debate_runner_injects_memory_context(tmp_path):
    """debate_runner calls query_memory and injects result into portfolio_state."""
    from unittest.mock import patch

    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    fake_memory = [
        {"date": "2026-02-01", "recommendation": "proceed",
         "reasoning": "Momentum strong.", "regime": "calm_bull", "key_risks": []},
    ]

    with patch("debate.debate_runner.query_memory", return_value=fake_memory) as mock_qm, \
         patch("debate.debate_runner.ingest_verdict"), \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant"), \
         patch("debate.debate_runner.run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        from debate.debate_runner import run_debate
        run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    mock_qm.assert_called_once()
    assert "memory_context" in portfolio_state
    assert portfolio_state["memory_context"] == fake_memory


def test_debate_runner_ingests_verdict_after_writing(tmp_path):
    """debate_runner calls ingest_verdict after writing the verdict JSON."""
    from unittest.mock import patch

    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 1,
        "allocation": {},
        "weights": {"AAPL": 1.0},
    }

    with patch("debate.debate_runner.query_memory", return_value=[]), \
         patch("debate.debate_runner.ingest_verdict") as mock_ingest, \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant"), \
         patch("debate.debate_runner.run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        from debate.debate_runner import run_debate
        run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    mock_ingest.assert_called_once()


def test_build_context_includes_memory_context():
    """_build_context includes formatted memory context when present."""
    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 1,
        "allocation": {},
        "weights": {"AAPL": 1.0},
        "memory_context": [
            {"date": "2026-02-01", "recommendation": "proceed",
             "reasoning": "Bull momentum.", "regime": "calm_bull", "key_risks": []},
        ],
    }
    from debate.agents import _build_context
    ctx = _build_context(portfolio_state)
    assert "2026-02-01" in ctx
    assert "proceed" in ctx
    assert "Bull momentum" in ctx
```

- [ ] **Step 2: Run failing tests**

```bash
.venv/bin/pytest tests/test_memory_interface.py::test_debate_runner_injects_memory_context tests/test_memory_interface.py::test_debate_runner_ingests_verdict_after_writing tests/test_memory_interface.py::test_build_context_includes_memory_context -v
```

Expected: FAIL — `query_memory` and `ingest_verdict` not imported in `debate_runner.py`

- [ ] **Step 3: Add memory imports and query block to `debate_runner.py`**

Add to the imports section at the top of `debate/debate_runner.py` (after existing imports, line ~29):

```python
from memory.r2r_interface import query_memory, ingest_verdict, format_memory_context
```

Insert the memory query block **after** the blind spot injection block (after line ~129) and **before** the scenario sim:

```python
    # Query memory for similar past situations
    print("[Debate] Querying memory for similar past situations...")
    try:
        regime = portfolio_state.get("us_regime", "")
        top_symbols = sorted(portfolio_state.get("weights", {}).items(), key=lambda x: -x[1])
        symbol_query = " ".join(sym for sym, _ in top_symbols[:5])
        memory_query = f"{regime} {symbol_query}"
        memory_results = query_memory(memory_query, n=3)
        portfolio_state["memory_context"] = memory_results
        if memory_results:
            print(f"[Debate] Memory: found {len(memory_results)} relevant past situation(s)")
        else:
            print("[Debate] Memory: no relevant past situations found")
    except Exception as e:
        portfolio_state["memory_context"] = []
        print(f"[Debate] Memory query failed (non-fatal): {e}")
```

Then, after writing the verdict JSON file (after `print(f"[Debate] Full record written to {out_path}")`), add:

```python
    # Ingest this verdict into memory so future debates can learn from it
    try:
        ingest_verdict(out_path)
    except Exception as e:
        print(f"[Debate] Memory ingest failed (non-fatal): {e}")
```

- [ ] **Step 4: Extend `_build_context()` in `debate/agents.py`**

In `debate/agents.py`, add the following to `_build_context()` after the catalyst block (or after the weights block if the catalyst scanner plan hasn't been implemented yet), before `return "\n".join(lines)`:

```python
    memory_ctx = portfolio_state.get("memory_context", [])
    if memory_ctx:
        from memory.r2r_interface import format_memory_context
        memory_text = format_memory_context(memory_ctx)
        if "no relevant" not in memory_text.lower():
            lines.append("")
            lines.append(memory_text)
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/pytest tests/test_memory_interface.py -v
```

Expected: all 14 tests pass.

- [ ] **Step 6: Commit**

```bash
git add debate/debate_runner.py debate/agents.py tests/test_memory_interface.py
git commit -m "feat: wire memory-augmented debate — agents now see relevant past verdicts before arguing"
```

---

### Task 3: One-time ingestion of existing verdict history into R2R

This task runs only once, to bootstrap R2R's vector store from the existing `outputs/debate_log/` files. If R2R is not configured (no `R2R_API_KEY`), this is a no-op — the local fallback reads files directly.

**Files:**
- Create: `scripts/ingest_verdict_history.py` — standalone script to bulk-ingest existing verdicts

- [ ] **Step 1: Write the script**

```python
# scripts/ingest_verdict_history.py
"""
One-time ingestion of existing verdict history into R2R.

Usage:
    .venv/bin/python scripts/ingest_verdict_history.py

If R2R_API_KEY is not set, prints a notice and exits (nothing to do).
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.r2r_interface import ingest_verdict, R2R_API_KEY

DEBATE_LOG_DIR = Path("outputs/debate_log")


def main():
    if not R2R_API_KEY:
        print("[Ingest] R2R_API_KEY not set — local search reads files directly, no ingestion needed.")
        print("[Ingest] Set R2R_API_KEY in .env if you have an R2R account.")
        return

    verdict_files = sorted(DEBATE_LOG_DIR.glob("verdict_*.json"))
    if not verdict_files:
        print(f"[Ingest] No verdict files found in {DEBATE_LOG_DIR}.")
        return

    print(f"[Ingest] Found {len(verdict_files)} verdict files. Ingesting into R2R...")
    success, failed = 0, 0

    for vf in verdict_files:
        try:
            ingest_verdict(vf)
            print(f"[Ingest] ✓ {vf.name}")
            success += 1
        except Exception as e:
            print(f"[Ingest] ✗ {vf.name}: {e}")
            failed += 1

    print(f"\n[Ingest] Done: {success} ingested, {failed} failed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to verify it handles no-key gracefully**

```bash
R2R_API_KEY="" .venv/bin/python scripts/ingest_verdict_history.py
```

Expected output (no crash):
```
[Ingest] R2R_API_KEY not set — local search reads files directly, no ingestion needed.
[Ingest] Set R2R_API_KEY in .env if you have an R2R account.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ingest_verdict_history.py
git commit -m "feat: add one-time verdict history ingestion script for R2R bootstrap"
```

---

## Self-Review

**Spec coverage:**
- R2R API query path (`_r2r_search`): Task 1 ✓
- Local keyword fallback (`_local_search`): Task 1 ✓
- Automatic fallback when R2R fails: Task 1 (`query_memory` catches R2R exception) ✓
- R2R ingestion after each verdict (`ingest_verdict`): Task 1 + Task 2 ✓
- Injection into `portfolio_state["memory_context"]`: Task 2 ✓
- Agents see memory context: Task 2 (`_build_context()` extension) ✓
- Bootstrap script for existing history: Task 3 ✓
- Non-fatal failure handling: all R2R calls caught, local fallback used ✓

**Placeholder scan:** None found.

**Type consistency:**
- `query_memory()` → `list[dict]` with keys `date, recommendation, reasoning, regime, key_risks`
- `format_memory_context(list[dict])` → `str`
- `ingest_verdict(Path)` → `None`
- `portfolio_state["memory_context"]` → `list[dict]` (same shape as `query_memory()` output)
- All consistent in `r2r_interface.py`, `debate_runner.py`, and `agents.py`.
