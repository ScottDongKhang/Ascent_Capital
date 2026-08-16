"""Tests that a self-healed node keeps a record of what it healed from.

Before the fix, `NodeResult.error` was overwritten to "" on a successful
repair, so a run that took 2 attempts left no trace that anything had gone
wrong on attempt 1. `repair_history` accumulates each failed attempt's
error instead of discarding it.
"""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from analyst.execute import execute_node
from analyst.types import AnalysisPlan, NodeState, OutputSchema, Task, TaskCategory

SCHEMA = OutputSchema(kind="dataframe", columns={"x": "int"}, row_semantics="one row")

_BROKEN_CODE = "def run():\n    raise RuntimeError('boom')\n"
_FIXED_CODE = textwrap.dedent(
    """
    import pandas as pd

    def run():
        return pd.DataFrame({"x": [1]})
    """
)


def _task() -> Task:
    return Task(
        task_id="t1",
        title="t1",
        category=TaskCategory.LOAD,
        description="d",
        output_name="out",
        output_schema=SCHEMA,
        max_repair_attempts=1,
    )


def test_successful_repair_preserves_history(monkeypatch):
    task = _task()
    plan = AnalysisPlan(question="q", tasks=(task,), params={})

    monkeypatch.setattr(
        "analyst.codegen.generate_one",
        lambda task, plan, prior_error="", prior_code="": _FIXED_CODE,
    )

    result = execute_node(task, plan, _BROKEN_CODE, upstream={})

    assert result.state is NodeState.DONE
    assert result.ok
    assert result.attempts == 2
    # the final `error` reflects the winning attempt: empty
    assert result.error == ""
    # but the failure that got repaired is not lost
    assert len(result.repair_history) == 1
    assert "RuntimeError" in result.repair_history[0]
    assert "boom" in result.repair_history[0]


def test_first_attempt_success_has_empty_history(monkeypatch):
    task = _task()
    plan = AnalysisPlan(question="q", tasks=(task,), params={})

    result = execute_node(task, plan, _FIXED_CODE, upstream={})

    assert result.state is NodeState.DONE
    assert result.attempts == 1
    assert result.repair_history == []
