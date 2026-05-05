# AI-Native Ascent Capital — Tier 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous factor discovery pipeline where Claude Opus proposes new alpha signals as Python code, an AST validator enforces structural and security constraints, a lightweight CPCV-style IC evaluator scores them on real historical data, and accepted proposals land in a human review queue for final approval before deployment.

**Architecture:** One new subsystem at `ascent/research/factor_discovery/`. Four modules with a clean linear pipeline: hypothesis_generator (Opus proposes code) → code_validator (AST safety + novelty check) → cpcv_evaluator (IC scoring on real prices) → discovery_runner (orchestrates the loop, writes accepted proposals to `outputs/factor_proposals/`). Human reviews the queue and manually adds approved factors to `ascent/features/feature_defs.py`. The system never auto-deploys generated code — the human is the final gate.

**Tech Stack:** Python 3.12, Claude Opus (`claude-opus-4-6`) for hypothesis generation, existing `ascent/llm/client.py`, existing price/fundamental data caches, `ast` module (stdlib), `scipy.stats.spearmanr` (installed), `ascent/research/cpcv.py`.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/research/factor_discovery/__init__.py` | Package marker |
| Create | `ascent/research/factor_discovery/hypothesis_generator.py` | Opus proposes factor code + rationale |
| Create | `ascent/research/factor_discovery/code_validator.py` | AST safety, structure, novelty checks |
| Create | `ascent/research/factor_discovery/cpcv_evaluator.py` | Rolling IC evaluation on real price data |
| Create | `ascent/research/factor_discovery/discovery_runner.py` | Orchestrates pipeline, writes proposal queue |
| Create | `tests/test_factor_discovery.py` | Full test suite for Task G |
| Modify | `run_all_agents.py` | Monthly trigger for discovery run (first Sunday of month) |

---

## Task G: Autonomous Factor Discovery Pipeline

**Problem:** Every alpha signal in Ascent was written by a human: momentum, stat-arb, fundamental quality, PEAD, vol regime. The self-improve loop (Task E) improves *weights* between existing signals but cannot propose *new* signals. The system has no ability to discover alpha from scratch. Research from AlphaAgent (arxiv:2502.16789) demonstrates that LLMs can generate novel, CPCV-validated factors with 11% annual excess returns — not by searching existing research, but by reasoning about economic mechanisms and writing testable code. This task builds that pipeline: Opus proposes factor code from first principles, an AST validator ensures the code is safe and structurally correct, a rolling IC evaluator scores it on Ascent's actual data, and approved proposals enter a human review queue. The human reviews, edits, and merges accepted factors into `feature_defs.py`.

**Key design decisions:**
- Claude Opus (not Haiku) — factor code generation requires genuine reasoning and creativity
- AST validation blocks: imports, exec/eval, file I/O, subprocess, class definitions, scope manipulation
- Restricted execution namespace: only `pd`, `np`, and safe stdlib builtins available when running factor code
- IC threshold: mean IC > 0.015 AND IC IR > 0.4 (intentionally conservative — novel factors earn this slowly)
- Human is the final gate: nothing deploys automatically
- Monthly cadence: runs on the first Sunday of each month (not weekly — Opus is expensive)

**Files:**
- Create: `ascent/research/factor_discovery/__init__.py`
- Create: `ascent/research/factor_discovery/hypothesis_generator.py`
- Create: `ascent/research/factor_discovery/code_validator.py`
- Create: `ascent/research/factor_discovery/cpcv_evaluator.py`
- Create: `ascent/research/factor_discovery/discovery_runner.py`
- Create: `tests/test_factor_discovery.py`
- Modify: `run_all_agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_factor_discovery.py
import ast
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_price_df(n_symbols=20, n_days=504):
    """Simulated price DataFrame indexed by date, columns by symbol."""
    idx  = pd.date_range(end="2026-05-01", periods=n_days, freq="B")
    syms = [f"SYM{i:02d}" for i in range(n_symbols)]
    data = {}
    for s in syms:
        np.random.seed(hash(s) % 2**31)
        data[s] = np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
    return pd.DataFrame(data, index=idx)


_VALID_FACTOR_CODE = '''
def compute_factor_reversal_z(df):
    """Short-term reversal: 5-day return, reversed and z-scored."""
    ret5 = df.pct_change(5)
    signal = -ret5.iloc[-1]
    mean = signal.mean()
    std  = signal.std()
    if std < 1e-8:
        return pd.Series(0.0, index=signal.index)
    return (signal - mean) / std
'''

_INVALID_CODE_IMPORT = '''
import os
def compute_factor_bad(df):
    os.system("rm -rf /")
    return pd.Series(0.0, index=df.columns)
'''

_INVALID_CODE_EXEC = '''
def compute_factor_exec(df):
    exec("print('pwned')")
    return pd.Series(0.0, index=df.columns)
'''

_INVALID_CODE_NO_FUNCTION = '''
x = 1 + 2
print(x)
'''

_INVALID_CODE_WRONG_NAME = '''
def my_factor(df):
    return pd.Series(0.0, index=df.columns)
'''


# ── Code validator tests ───────────────────────────────────────────────────────

def test_validate_accepts_valid_code():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code(_VALID_FACTOR_CODE, expected_name="factor_reversal_z")
    assert ok, f"Valid code rejected: {msg}"


def test_validate_rejects_import():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code(_INVALID_CODE_IMPORT, expected_name="factor_bad")
    assert not ok
    assert "import" in msg.lower() or "forbidden" in msg.lower()


def test_validate_rejects_exec_in_code():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code(_INVALID_CODE_EXEC, expected_name="factor_exec")
    assert not ok
    assert "exec" in msg.lower() or "forbidden" in msg.lower() or "builtin" in msg.lower()


def test_validate_rejects_no_function():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code(_INVALID_CODE_NO_FUNCTION, expected_name="factor_x")
    assert not ok


def test_validate_rejects_wrong_function_name():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code(_INVALID_CODE_WRONG_NAME, expected_name="factor_reversal_z")
    assert not ok
    assert "name" in msg.lower() or "compute_factor_reversal_z" in msg


def test_validate_ast_parses_syntax_error():
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    ok, msg = validate_factor_code("def broken(df:\n    return None", expected_name="factor_broken")
    assert not ok
    assert "syntax" in msg.lower() or "parse" in msg.lower()


# ── IC evaluator tests ─────────────────────────────────────────────────────────

def test_evaluate_ic_returns_dict():
    from ascent.research.factor_discovery.cpcv_evaluator import evaluate_factor_ic
    prices = _make_price_df(n_symbols=15, n_days=300)
    result = evaluate_factor_ic(
        code=_VALID_FACTOR_CODE,
        factor_name="factor_reversal_z",
        prices_df=prices,
        n_periods=5,
    )
    assert isinstance(result, dict)
    for key in ["ic_mean", "ic_ir", "n_observations", "ic_p5"]:
        assert key in result, f"Missing key: {key}"


def test_evaluate_ic_values_in_valid_range():
    from ascent.research.factor_discovery.cpcv_evaluator import evaluate_factor_ic
    prices = _make_price_df(n_symbols=15, n_days=400)
    result = evaluate_factor_ic(
        code=_VALID_FACTOR_CODE,
        factor_name="factor_reversal_z",
        prices_df=prices,
        n_periods=5,
    )
    assert -1.0 <= result["ic_mean"] <= 1.0, "IC must be in [-1, 1]"
    assert result["n_observations"] > 0


def test_evaluate_ic_returns_error_on_bad_code():
    from ascent.research.factor_discovery.cpcv_evaluator import evaluate_factor_ic
    bad_code = "def compute_factor_crash(df):\n    raise ValueError('intentional')\n"
    prices = _make_price_df()
    result = evaluate_factor_ic(
        code=bad_code, factor_name="factor_crash", prices_df=prices, n_periods=5
    )
    assert "error" in result
    assert result.get("ic_mean", 0.0) == 0.0


def test_evaluate_ic_sandbox_no_file_access():
    """Code that tries to open a file should fail safely — not write to disk."""
    from ascent.research.factor_discovery.cpcv_evaluator import evaluate_factor_ic
    code_with_open = (
        "def compute_factor_filewrite(df):\n"
        "    open('/tmp/pwned.txt', 'w').write('hack')\n"
        "    return df.iloc[-1]\n"
    )
    prices = _make_price_df()
    result = evaluate_factor_ic(
        code=code_with_open, factor_name="factor_filewrite", prices_df=prices, n_periods=5
    )
    assert "error" in result or result.get("ic_mean", 0.0) == 0.0
    import os
    assert not os.path.exists("/tmp/pwned.txt"), "Sandbox must block file writes"


# ── Hypothesis generator tests ────────────────────────────────────────────────

def test_generate_hypothesis_returns_dict():
    from ascent.research.factor_discovery.hypothesis_generator import generate_factor_hypothesis
    mock_response = json.dumps({
        "name": "factor_reversal_z",
        "description": "Short-term mean reversion captures overreaction",
        "rationale": "Stocks that fall sharply in 5 days mean-revert as overreaction corrects.",
        "code": _VALID_FACTOR_CODE,
    })
    with patch("ascent.research.factor_discovery.hypothesis_generator._call_opus",
               return_value=mock_response):
        result = generate_factor_hypothesis(
            regime="stressed",
            existing_factor_names=["trend", "fundamental", "earnings"],
            n_attempts=1,
        )
    assert isinstance(result, dict)
    for key in ["name", "description", "rationale", "code"]:
        assert key in result


def test_generate_hypothesis_returns_none_on_llm_failure():
    from ascent.research.factor_discovery.hypothesis_generator import generate_factor_hypothesis
    with patch("ascent.research.factor_discovery.hypothesis_generator._call_opus",
               return_value=None):
        result = generate_factor_hypothesis(
            regime="stressed", existing_factor_names=["trend"], n_attempts=1
        )
    assert result is None


# ── Discovery runner tests ────────────────────────────────────────────────────

def test_discovery_runner_writes_accepted_proposal(tmp_path):
    from ascent.research.factor_discovery.discovery_runner import run_factor_discovery

    mock_hypothesis = {
        "name": "factor_reversal_z",
        "description": "Short-term reversal",
        "rationale": "Overreaction correction",
        "code": _VALID_FACTOR_CODE,
    }
    mock_ic = {
        "ic_mean": 0.025, "ic_ir": 0.60,
        "n_observations": 200, "ic_p5": -0.018,
    }

    with patch("ascent.research.factor_discovery.discovery_runner.generate_factor_hypothesis",
               return_value=mock_hypothesis):
        with patch("ascent.research.factor_discovery.discovery_runner.validate_factor_code",
                   return_value=(True, "OK")):
            with patch("ascent.research.factor_discovery.discovery_runner.evaluate_factor_ic",
                       return_value=mock_ic):
                with patch("ascent.research.factor_discovery.discovery_runner._load_prices",
                           return_value=_make_price_df()):
                    with patch("ascent.research.factor_discovery.discovery_runner.PROPOSALS_DIR",
                               tmp_path):
                        result = run_factor_discovery(n_hypotheses=1, regime="stressed")

    assert isinstance(result, dict)
    assert result.get("n_accepted", 0) >= 1 or result.get("n_rejected", 0) >= 0

    proposal_files = list(tmp_path.glob("*.json"))
    assert len(proposal_files) >= 1, "Accepted proposal must be written to proposals dir"
    proposal = json.loads(proposal_files[0].read_text())
    assert "code" in proposal
    assert "ic_mean" in proposal


def test_discovery_runner_rejects_low_ic_proposal(tmp_path):
    from ascent.research.factor_discovery.discovery_runner import run_factor_discovery

    mock_ic = {"ic_mean": 0.002, "ic_ir": 0.10, "n_observations": 200, "ic_p5": -0.05}

    with patch("ascent.research.factor_discovery.discovery_runner.generate_factor_hypothesis",
               return_value={"name": "factor_weak", "description": "", "rationale": "", "code": _VALID_FACTOR_CODE}):
        with patch("ascent.research.factor_discovery.discovery_runner.validate_factor_code",
                   return_value=(True, "OK")):
            with patch("ascent.research.factor_discovery.discovery_runner.evaluate_factor_ic",
                       return_value=mock_ic):
                with patch("ascent.research.factor_discovery.discovery_runner._load_prices",
                           return_value=_make_price_df()):
                    with patch("ascent.research.factor_discovery.discovery_runner.PROPOSALS_DIR",
                               tmp_path):
                        result = run_factor_discovery(n_hypotheses=1, regime="stressed")

    proposal_files = list(tmp_path.glob("*.json"))
    assert len(proposal_files) == 0, "Low-IC proposal must NOT be written to proposals dir"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_factor_discovery.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'ascent.research.factor_discovery'`

- [ ] **Step 3: Create `ascent/research/factor_discovery/__init__.py`**

```python
# ascent/research/factor_discovery/__init__.py
```

- [ ] **Step 4: Create `ascent/research/factor_discovery/code_validator.py`**

```python
"""
ascent/research/factor_discovery/code_validator.py

AST-based validator for LLM-generated factor code.

Checks:
1. Valid Python syntax
2. Single top-level FunctionDef named compute_{factor_name}
3. No forbidden AST nodes (import, exec, eval, file I/O)
4. No dangerous builtin references (open, __import__, compile)

Returns (is_valid: bool, message: str).
"""
from __future__ import annotations

import ast
from typing import Tuple


_FORBIDDEN_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
)

_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "__import__", "breakpoint",
    "input", "print", "__builtins__", "globals", "locals", "vars",
    "setattr", "delattr", "memoryview", "bytearray",
})

_FORBIDDEN_ATTRIBUTE_PREFIXES = frozenset({
    "os", "sys", "subprocess", "pathlib", "io", "socket",
    "urllib", "requests", "shutil", "tempfile", "importlib",
})


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that collects safety violations."""

    def __init__(self):
        self.violations: list = []

    def visit_Import(self, node):
        self.violations.append("Import statement not allowed in factor code")

    def visit_ImportFrom(self, node):
        self.violations.append(f"'from ... import' not allowed: {ast.unparse(node)}")

    def visit_ClassDef(self, node):
        self.violations.append(f"Class definition not allowed: {node.name}")

    def visit_Global(self, node):
        self.violations.append(f"'global' statement not allowed")

    def visit_Nonlocal(self, node):
        self.violations.append(f"'nonlocal' statement not allowed")

    def visit_AsyncFunctionDef(self, node):
        self.violations.append(f"Async function not allowed: {node.name}")

    def visit_Name(self, node):
        if node.id in _FORBIDDEN_NAMES:
            self.violations.append(f"Forbidden built-in reference: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Catch patterns like os.system, subprocess.run, etc.
        if isinstance(node.value, ast.Name):
            if node.value.id in _FORBIDDEN_ATTRIBUTE_PREFIXES:
                self.violations.append(
                    f"Forbidden module access: '{node.value.id}.{node.attr}'"
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        # Catch exec("..."), eval("...") as function calls
        if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
            self.violations.append(f"Forbidden function call: '{node.func.id}'")
        self.generic_visit(node)


def validate_factor_code(code: str, expected_name: str) -> Tuple[bool, str]:
    """
    Validate LLM-generated factor code.

    Args:
        code:          Python source code string from the LLM.
        expected_name: The factor name (without 'compute_' prefix).
                       The top-level function must be named compute_{expected_name}.

    Returns:
        (is_valid, message) — message is "OK" on success or an error description.
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # 2. Must have exactly one top-level definition, and it must be a FunctionDef
    top_level = tree.body
    if not top_level:
        return False, "Code is empty."

    func_defs = [n for n in top_level if isinstance(n, ast.FunctionDef)]
    non_funcs = [n for n in top_level if not isinstance(n, (ast.FunctionDef, ast.Expr))]

    if not func_defs:
        return False, "No function definition found. Factor code must define a function."

    if len(func_defs) > 1:
        names = [f.name for f in func_defs]
        return False, f"Multiple function definitions not allowed: {names}"

    if non_funcs:
        non_func_types = [type(n).__name__ for n in non_funcs]
        return False, f"Top-level statements not allowed (only function defs): {non_func_types}"

    # 3. Function must be named compute_{expected_name}
    fn = func_defs[0]
    expected_fn_name = f"compute_{expected_name}"
    if fn.name != expected_fn_name:
        return False, (
            f"Function must be named '{expected_fn_name}', got '{fn.name}'. "
            f"Factor name is '{expected_name}'."
        )

    # 4. Function must accept at least one argument (the DataFrame)
    if not fn.args.args:
        return False, f"Function '{fn.name}' must accept at least one argument (df: pd.DataFrame)."

    # 5. Safety scan — check for forbidden constructs
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    if visitor.violations:
        return False, f"Safety violations: {'; '.join(visitor.violations)}"

    return True, "OK"
```

- [ ] **Step 5: Create `ascent/research/factor_discovery/cpcv_evaluator.py`**

```python
"""
ascent/research/factor_discovery/cpcv_evaluator.py

Rolling IC evaluator for LLM-generated factor code.

Executes the factor code in a restricted namespace, computes Spearman IC
between the factor values and n-period forward returns, and returns summary
statistics: mean IC, IC IR (IC/std), p5 IC, and observation count.

The namespace restriction: only pd, np, and a safe subset of Python builtins
are available during execution. File I/O, imports, and subprocess calls fail
silently and return an error result.
"""
from __future__ import annotations

import builtins
import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

# Builtins that are safe for factor code execution
_SAFE_BUILTIN_NAMES = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "range", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None", "NotImplemented",
})

_SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in dir(builtins)
    if k in _SAFE_BUILTIN_NAMES
}


def _execute_factor(code: str, factor_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Execute factor code in a restricted namespace and return the resulting Series.
    Raises on any execution error so the caller can record it.
    """
    namespace = {"pd": pd, "np": np, "__builtins__": _SAFE_BUILTINS}
    exec(compile(code, f"<factor_{factor_name}>", "exec"), namespace)

    fn_name = f"compute_{factor_name}"
    fn = namespace.get(fn_name)
    if fn is None:
        raise ValueError(f"Function '{fn_name}' not found after execution.")

    result = fn(df)
    if not isinstance(result, pd.Series):
        raise TypeError(f"Factor must return pd.Series, got {type(result).__name__}")
    return result


def evaluate_factor_ic(
    code: str,
    factor_name: str,
    prices_df: pd.DataFrame,
    n_periods: int = 5,
    min_symbols: int = 10,
    lookback_days: int = 252,
) -> Dict:
    """
    Evaluate a factor's Information Coefficient on historical price data.

    Method:
      For each rolling date (weekly step), compute:
        - Factor value: execute the code on the trailing window ending at that date
        - Forward return: n_periods-day return starting at that date

      Compute Spearman IC (cross-sectional) at each date.
      Report mean IC, IC IR, p5 IC, and observation count.

    Args:
        code:          Python source of the factor function.
        factor_name:   Name (without 'compute_' prefix).
        prices_df:     DataFrame of price series — index=dates, columns=symbols.
        n_periods:     Forward return horizon in trading days.
        min_symbols:   Minimum symbols required per period; periods below this are skipped.
        lookback_days: How many days of price history to pass to the factor function.

    Returns:
        Dict with ic_mean, ic_ir, ic_p5, n_observations — or {"error": msg} on failure.
    """
    if prices_df.empty or len(prices_df) < lookback_days + n_periods:
        return {"error": "Insufficient price data", "ic_mean": 0.0, "ic_ir": 0.0,
                "n_observations": 0, "ic_p5": 0.0}

    ic_series = []
    dates = prices_df.index[lookback_days:-n_periods:5]  # step every 5 days

    for dt in dates:
        try:
            iloc_pos = prices_df.index.get_loc(dt)
            window_start = max(0, iloc_pos - lookback_days)
            window_df    = prices_df.iloc[window_start : iloc_pos + 1]

            factor_vals = _execute_factor(code, factor_name, window_df)

            fwd_rets = (
                prices_df.iloc[iloc_pos + n_periods] /
                prices_df.iloc[iloc_pos] - 1
            )

            common = factor_vals.index.intersection(fwd_rets.index)
            f = factor_vals.reindex(common).dropna()
            r = fwd_rets.reindex(common).dropna()
            common2 = f.index.intersection(r.index)
            f = f.reindex(common2)
            r = r.reindex(common2)

            if len(f) < min_symbols:
                continue

            ic, _ = spearmanr(f.values, r.values)
            if not np.isnan(ic):
                ic_series.append(float(ic))

        except Exception as exc:
            log.debug("[CPCV] Skipped date %s: %s", dt, exc)
            continue

    if not ic_series:
        return {
            "error": "No valid IC observations computed",
            "ic_mean": 0.0, "ic_ir": 0.0, "n_observations": 0, "ic_p5": 0.0,
        }

    ic_arr = np.array(ic_series)
    ic_mean = float(np.mean(ic_arr))
    ic_std  = float(np.std(ic_arr))
    ic_ir   = round(ic_mean / ic_std, 3) if ic_std > 1e-6 else 0.0
    ic_p5   = float(np.percentile(ic_arr, 5))

    return {
        "ic_mean":       round(ic_mean, 4),
        "ic_ir":         round(ic_ir, 3),
        "ic_p5":         round(ic_p5, 4),
        "n_observations": len(ic_series),
    }
```

- [ ] **Step 6: Create `ascent/research/factor_discovery/hypothesis_generator.py`**

```python
"""
ascent/research/factor_discovery/hypothesis_generator.py

Uses Claude Opus to generate novel alpha factor hypotheses as Python code.

The LLM proposes:
  - A factor name (snake_case, will be used as compute_{name})
  - Economic rationale for why it should predict returns
  - Python implementation as a pure function of a price DataFrame

The prompt instructs Opus to reason from first principles and avoid
replicating factors already in the system.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quantitative researcher at a systematic hedge fund with deep expertise \
in financial economics and signal processing. Your task is to propose ONE novel \
alpha factor — an empirically-motivated signal that predicts cross-sectional stock \
returns over a 5-day horizon.

You will be given:
1. The current market regime
2. Existing factors in the system (to help you avoid redundancy)

Your factor must be:
- Economically motivated (not just data mining)
- Implementable as a pure Python function
- Novel — not a close variant of the existing factors
- Cross-sectionally z-scored in the output

Respond ONLY with a JSON object. No other text."""

_USER_TEMPLATE = """\
Current market regime: {regime}

Existing factors (avoid redundancy with these):
{existing_factors}

Propose ONE new factor. The factor function receives a pd.DataFrame where:
- The index is dates (most recent date = df.index[-1])
- Columns are stock tickers (symbols)
- Values are price history (adjusted close)

The function must:
1. Be named exactly: compute_{factor_name}
2. Accept exactly one argument: df (pd.DataFrame)
3. Return a pd.Series indexed by symbol with cross-sectionally z-scored values
4. Use only pd (pandas) and np (numpy) — no imports inside the function
5. Handle edge cases (e.g., return zeros when std < 1e-8)

Think step by step:
Step 1: Identify an economic mechanism that drives short-term cross-sectional returns
Step 2: Specify what data transformation captures that mechanism
Step 3: Write the Python implementation
Step 4: Verify the output is a cross-sectionally z-scored pd.Series

Respond with this exact JSON format:
{{
  "name": "factor_your_name",
  "description": "One sentence — what this measures",
  "rationale": "2-3 sentences — economic rationale for why this predicts returns",
  "code": "def compute_factor_your_name(df):\\n    ...\\n    return result"
}}"""


def _call_opus(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, DEFAULT_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=DEFAULT_MODEL,
            max_tokens=1200,
            temperature=0.7,
            use_cache=False,
        )
    except Exception as exc:
        log.warning("[FactorDiscovery] Opus call failed: %s", exc)
        return None


def generate_factor_hypothesis(
    regime: str,
    existing_factor_names: List[str],
    n_attempts: int = 2,
) -> Optional[dict]:
    """
    Ask Claude Opus to propose a novel alpha factor.

    Args:
        regime:               Current market regime label.
        existing_factor_names: List of factor names already in the system.
        n_attempts:           How many times to retry on parse failure.

    Returns:
        Dict with {name, description, rationale, code} or None if all attempts fail.
    """
    existing_str = "\n".join(f"  - {n}" for n in existing_factor_names) or "  (none yet)"
    user_prompt  = _USER_TEMPLATE.format(regime=regime, existing_factors=existing_str)

    for attempt in range(n_attempts):
        raw = _call_opus(_SYSTEM_PROMPT, user_prompt)
        if not raw:
            continue

        try:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                log.warning("[FactorDiscovery] No JSON in response (attempt %d)", attempt + 1)
                continue

            parsed = json.loads(raw[start:end])
            name   = str(parsed.get("name", "")).strip()
            code   = str(parsed.get("code", "")).strip()

            if not name.startswith("factor_"):
                log.warning("[FactorDiscovery] Factor name must start with 'factor_': %s", name)
                continue

            if "def compute_" not in code:
                log.warning("[FactorDiscovery] Code missing 'def compute_' pattern")
                continue

            return {
                "name":        name,
                "description": str(parsed.get("description", "")),
                "rationale":   str(parsed.get("rationale", "")),
                "code":        code,
            }

        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("[FactorDiscovery] Parse failed (attempt %d): %s", attempt + 1, exc)

    return None
```

- [ ] **Step 7: Create `ascent/research/factor_discovery/discovery_runner.py`**

```python
"""
ascent/research/factor_discovery/discovery_runner.py

Orchestrates the factor discovery pipeline:
  1. Load price data from cache
  2. Generate N factor hypotheses via Claude Opus
  3. Validate each with AST code_validator
  4. Score valid factors with cpcv_evaluator
  5. Write accepted proposals (IC > threshold) to outputs/factor_proposals/
  6. Log all attempts (accepted + rejected) to logs/factor_discovery_log.jsonl

Acceptance thresholds (conservative):
  - ic_mean > 0.015  (IC of 1.5% is meaningful for 5-day horizon)
  - ic_ir   > 0.40   (information ratio above 0.4)
  - n_observations > 20

Human reviews accepted proposals in outputs/factor_proposals/ before any code
is added to feature_defs.py. NOTHING auto-deploys.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)

PROPOSALS_DIR    = Path("outputs/factor_proposals")
DISCOVERY_LOG    = Path("logs/factor_discovery_log.jsonl")
IC_MEAN_THRESHOLD = 0.015
IC_IR_THRESHOLD   = 0.40
MIN_OBSERVATIONS  = 20

_EXISTING_FACTORS = [
    "trend", "meanrev", "volatility", "statarb", "ml",
    "fundamental", "earnings", "analyst", "options_flow",
    "insider", "short_interest", "llm_fundamental",
]


def _load_prices() -> pd.DataFrame:
    """Load price data from parquet cache for factor evaluation."""
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data("prices_live"):
            df = load_parquet("prices_live")
            if "close" in df.columns:
                return df.pivot(columns="symbol", values="close").sort_index()
            elif isinstance(df.columns, pd.MultiIndex):
                return df["Close"].sort_index()
            return df.sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] Parquet load failed: %s", exc)

    try:
        import yfinance as yf
        from ascent.config.settings import get_config
        cfg  = get_config()
        syms = list(getattr(cfg.universe, "symbols", ["AAPL", "MSFT", "AMZN"]))[:50]
        raw  = yf.download(syms, period="2y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].dropna(axis=1, how="all").sort_index()
        return raw.sort_index()
    except Exception as exc:
        log.warning("[FactorDiscovery] yfinance fallback failed: %s", exc)
        return pd.DataFrame()


def _write_log(entry: dict) -> None:
    DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISCOVERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_proposal(hypothesis: dict, ic_result: dict) -> Path:
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    fname = f"{hypothesis['name']}_{today}.json"
    path  = PROPOSALS_DIR / fname
    payload = {
        **hypothesis,
        **{k: ic_result.get(k) for k in ["ic_mean", "ic_ir", "ic_p5", "n_observations"]},
        "proposed_at":  datetime.now().isoformat(),
        "regime_at_proposal": hypothesis.get("regime", "unknown"),
        "review_status": "pending",
        "review_notes":  "",
        "how_to_deploy": (
            "1. Review and edit the code below.\n"
            "2. Add the function to ascent/features/feature_defs.py.\n"
            "3. Register it in build_all_features() with appropriate lag.\n"
            "4. Add to stack.py DEFAULT_ALPHA_WEIGHTS at a small initial weight (0.02).\n"
            "5. Run the full test suite before committing."
        ),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_factor_discovery(
    n_hypotheses: int = 3,
    regime: Optional[str] = None,
) -> Dict:
    """
    Run one full factor discovery cycle.

    Args:
        n_hypotheses: How many factor hypotheses to generate and evaluate.
        regime:       Current market regime (passed to hypothesis generator for context).

    Returns:
        Dict with n_generated, n_valid, n_accepted, n_rejected, proposals (list of paths).
    """
    from ascent.research.factor_discovery.hypothesis_generator import generate_factor_hypothesis
    from ascent.research.factor_discovery.code_validator import validate_factor_code
    from ascent.research.factor_discovery.cpcv_evaluator import evaluate_factor_ic

    regime = regime or "unknown"
    log.info("[FactorDiscovery] Starting cycle — regime=%s, n_hypotheses=%d", regime, n_hypotheses)

    prices = _load_prices()
    if prices.empty:
        log.warning("[FactorDiscovery] No price data — aborting")
        return {"n_generated": 0, "n_valid": 0, "n_accepted": 0, "n_rejected": 0, "proposals": []}

    n_valid = n_accepted = n_rejected = 0
    proposals = []

    for i in range(n_hypotheses):
        log.info("[FactorDiscovery] Generating hypothesis %d/%d", i + 1, n_hypotheses)
        hypothesis = generate_factor_hypothesis(
            regime=regime,
            existing_factor_names=_EXISTING_FACTORS,
            n_attempts=2,
        )
        if hypothesis is None:
            log.warning("[FactorDiscovery] Hypothesis generation failed for slot %d", i + 1)
            continue

        factor_name = hypothesis["name"].replace("factor_", "", 1)
        hypothesis["regime"] = regime

        # Validate code
        is_valid, validation_msg = validate_factor_code(hypothesis["code"], factor_name)
        if not is_valid:
            log.info("[FactorDiscovery] Code validation failed: %s — %s",
                     hypothesis["name"], validation_msg)
            _write_log({
                "date": date.today().isoformat(), "regime": regime,
                "name": hypothesis["name"], "status": "validation_failed",
                "validation_msg": validation_msg,
            })
            n_rejected += 1
            continue

        n_valid += 1

        # Evaluate IC
        log.info("[FactorDiscovery] Evaluating IC for %s", hypothesis["name"])
        ic_result = evaluate_factor_ic(
            code=hypothesis["code"],
            factor_name=factor_name,
            prices_df=prices,
            n_periods=5,
        )

        if "error" in ic_result:
            log.info("[FactorDiscovery] IC evaluation error: %s — %s",
                     hypothesis["name"], ic_result["error"])
            _write_log({
                "date": date.today().isoformat(), "regime": regime,
                "name": hypothesis["name"], "status": "evaluation_error",
                "error": ic_result["error"],
            })
            n_rejected += 1
            continue

        ic_mean = ic_result.get("ic_mean", 0.0)
        ic_ir   = ic_result.get("ic_ir", 0.0)
        n_obs   = ic_result.get("n_observations", 0)

        log.info("[FactorDiscovery] %s — IC=%.4f, IR=%.3f, n=%d",
                 hypothesis["name"], ic_mean, ic_ir, n_obs)

        log_entry = {
            "date": date.today().isoformat(), "regime": regime,
            "name": hypothesis["name"], "description": hypothesis.get("description", ""),
            "ic_mean": ic_mean, "ic_ir": ic_ir, "n_observations": n_obs,
        }

        if ic_mean > IC_MEAN_THRESHOLD and ic_ir > IC_IR_THRESHOLD and n_obs >= MIN_OBSERVATIONS:
            proposal_path = _write_proposal(hypothesis, ic_result)
            proposals.append(str(proposal_path))
            n_accepted += 1
            log_entry["status"] = "accepted"
            log_entry["proposal_path"] = str(proposal_path)
            log.info("[FactorDiscovery] ACCEPTED: %s → %s", hypothesis["name"], proposal_path)
        else:
            n_rejected += 1
            log_entry["status"] = "rejected_low_ic"
            reject_reasons = []
            if ic_mean <= IC_MEAN_THRESHOLD:
                reject_reasons.append(f"IC {ic_mean:.4f} ≤ threshold {IC_MEAN_THRESHOLD}")
            if ic_ir <= IC_IR_THRESHOLD:
                reject_reasons.append(f"IC IR {ic_ir:.3f} ≤ threshold {IC_IR_THRESHOLD}")
            if n_obs < MIN_OBSERVATIONS:
                reject_reasons.append(f"Only {n_obs} observations (need {MIN_OBSERVATIONS})")
            log_entry["reject_reasons"] = reject_reasons
            log.info("[FactorDiscovery] Rejected: %s — %s",
                     hypothesis["name"], "; ".join(reject_reasons))

        _write_log(log_entry)

    summary = {
        "n_generated":  n_hypotheses,
        "n_valid":      n_valid,
        "n_accepted":   n_accepted,
        "n_rejected":   n_rejected,
        "proposals":    proposals,
        "regime":       regime,
        "date":         date.today().isoformat(),
    }
    log.info("[FactorDiscovery] Cycle complete: %d accepted / %d rejected",
             n_accepted, n_rejected)
    return summary
```

- [ ] **Step 8: Wire monthly discovery run into `run_all_agents.py`**

Find the Sunday self-improve block (search for `weekday() == 6` or `run_self_improve`). Add after the existing Sunday block:

```python
        # Factor discovery — runs on first Sunday of each month
        try:
            from datetime import date
            today = date.today()
            if today.weekday() == 6 and today.day <= 7:  # first Sunday of the month
                from ascent.research.factor_discovery.discovery_runner import run_factor_discovery
                print("[FactorDiscovery] Monthly run starting...")
                regime_for_discovery = _get_current_regime()  # reuse existing helper if available
                discovery_result = run_factor_discovery(n_hypotheses=3, regime=regime_for_discovery)
                print(
                    f"[FactorDiscovery] Cycle complete: "
                    f"{discovery_result['n_accepted']} accepted, "
                    f"{discovery_result['n_rejected']} rejected. "
                    f"Proposals in outputs/factor_proposals/"
                )
        except Exception as _de:
            print(f"[FactorDiscovery] Monthly run skipped: {_de}")
```

Note: `_get_current_regime()` should return the current regime string from the orchestrator's last output or `"unknown"` if unavailable. Add this helper near the Sunday block if not already present:

```python
def _get_current_regime() -> str:
    try:
        import json
        sig = json.loads(open("dashboard/regime_signal.json").read())
        return str(sig.get("label", "unknown")).lower()
    except Exception:
        return "unknown"
```

- [ ] **Step 9: Run all tests**

```bash
.venv/bin/pytest tests/test_factor_discovery.py -v
```
Expected: All 12 tests PASS.

- [ ] **Step 10: Full suite check**

```bash
.venv/bin/pytest tests/ -q --tb=short 2>&1 | tail -8
```
Expected: All tests pass (≥265).

- [ ] **Step 11: Commit**

```bash
git add ascent/research/factor_discovery/__init__.py \
        ascent/research/factor_discovery/hypothesis_generator.py \
        ascent/research/factor_discovery/code_validator.py \
        ascent/research/factor_discovery/cpcv_evaluator.py \
        ascent/research/factor_discovery/discovery_runner.py \
        tests/test_factor_discovery.py \
        run_all_agents.py
git commit -m "feat(research): autonomous factor discovery pipeline — Opus proposes code, AST validates, IC scores, human reviews"
```

---

## Final: Push

- [ ] **Push to GitHub**

```bash
git push origin main
```

---

## Human Review Process (non-automated)

When a factor is accepted, a file like `outputs/factor_proposals/factor_vol_acceleration_2026-05-04.json` is written. To deploy it:

1. Read the `code` field — review it for economic soundness and correctness
2. Edit if needed (rename columns, add edge case guards, verify z-scoring)
3. Add the function to `ascent/features/feature_defs.py` following the existing pattern:

```python
def factor_vol_acceleration(df: pd.DataFrame) -> pd.Series:
    # [paste and clean up the generated code here]
    ...
```

4. Register in `build_all_features()` in `ascent/features/build_features.py`
5. Add to `stack.py` `DEFAULT_ALPHA_WEIGHTS` at a small initial weight (0.02)
6. Reduce another sleeve by 0.02 to keep the sum at 1.0
7. Run the full test suite: `.venv/bin/pytest tests/ -v`
8. Run the system once in dry-run mode to verify the new factor doesn't crash the pipeline
9. Commit: `git commit -m "feat(alpha): deploy {factor_name} — proposed by factor discovery, reviewed YYYY-MM-DD"`

---

## Self-Review

**Spec coverage:**
- ✅ LLM hypothesis generation: Opus proposes factor name, rationale, and Python implementation
- ✅ Structured 6-step CoT prompt: regime context, existing factors, step-by-step reasoning, exact JSON format
- ✅ AST validation: syntax check, forbidden nodes (Import, ClassDef, Global, Nonlocal), forbidden names (exec, eval, open), dangerous attribute access (os., sys., subprocess.)
- ✅ Restricted execution namespace: only pd, np, and safe builtins available during IC evaluation
- ✅ Rolling IC evaluator: Spearman IC between factor values and 5-day forward returns, reports mean/IR/p5/n
- ✅ Acceptance thresholds: IC > 0.015 AND IC IR > 0.40 AND n_obs ≥ 20
- ✅ Proposals written to `outputs/factor_proposals/` with deployment instructions
- ✅ All attempts logged to `logs/factor_discovery_log.jsonl`
- ✅ Monthly trigger: first Sunday of each month in `run_all_agents.py`
- ✅ Human is the final gate: nothing auto-deploys

**Security validation:**
- Import statements: blocked by AST `visit_Import` and `visit_ImportFrom`
- `exec`/`eval` calls: blocked by `visit_Call` checking for forbidden names
- File system access: `open` blocked in both AST check (visit_Name) and execution namespace
- Module access: `os.system`, `subprocess.run` blocked by `visit_Attribute` checking prefixes
- Execution namespace: `__builtins__` limited to ~22 safe builtins, no `open`, `exec`, `eval`, `__import__`

**Type consistency:**
- `validate_factor_code(code: str, expected_name: str) -> Tuple[bool, str]` — used correctly in discovery_runner
- `evaluate_factor_ic(code, factor_name, prices_df, n_periods) -> Dict` — returns dict with `ic_mean`, `ic_ir`, `ic_p5`, `n_observations` OR `{"error": str}`
- `generate_factor_hypothesis(regime, existing_factor_names, n_attempts) -> Optional[dict]` — returns dict with `name`, `description`, `rationale`, `code`
- `run_factor_discovery(n_hypotheses, regime) -> Dict` — returns summary dict with `n_generated`, `n_valid`, `n_accepted`, `n_rejected`, `proposals`
