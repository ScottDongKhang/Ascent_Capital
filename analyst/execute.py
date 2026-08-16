"""Run the generated code as a DAG, validate each output, repair on failure.

The framework executes the code. The model never invokes its own work, and it
cannot skip validation because it is never asked to run it. That inversion is
the containment mechanism: a bad generation fails a check here rather than
quietly producing a plausible wrong number downstream.
"""
from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display, and safe to build figures off the main thread

import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import codegen  # noqa: E402
from .plans import layers  # noqa: E402
from .types import AnalysisPlan, NodeResult, NodeState, Task  # noqa: E402

log = logging.getLogger(__name__)


class ValidationError(Exception):
    """The code ran but produced something the task did not promise."""


class IdentityError(ValidationError):
    """The code ran, and the shape was right, but it is not data for the
    ticker the task actually asked for. This is the failure mode shape
    checking cannot see: a repaired loader that silently substitutes a
    different, valid ticker produces perfectly-shaped output."""


def validate_output(task: Task, value: Any) -> None:
    """Shape check, plus an identity postcondition for LOAD tasks that name
    a specific ticker.

    Shape checking alone is deliberately shallow -- deeper semantic checking
    is a later phase, and pretending to it now would give false confidence.
    But shape-only checking has one proven hole: a task pointed at a bad
    ticker can get "repaired" by an LLM that just swaps in a different,
    valid ticker instead of surfacing the failure. That produces a
    DataFrame of the exact right shape, so it passes every check below
    unless identity is checked too. `toolkit.load_prices` stamps
    `df.attrs["ticker"]` with whatever ticker it actually fetched; here we
    cross-check that against what the task's plan said to fetch.
    """
    s = task.output_schema

    if task.expected_ticker is not None and s.kind == "dataframe":
        actual = None
        if isinstance(value, pd.DataFrame):
            actual = value.attrs.get("ticker")
        if actual != task.expected_ticker:
            raise IdentityError(
                f"task {task.task_id!r} was supposed to load ticker "
                f"{task.expected_ticker!r}, but the returned data is stamped "
                f"{actual!r} -- refusing to accept a silently substituted ticker"
            )

    if s.kind == "figure":
        if not isinstance(value, Figure):
            raise ValidationError(
                f"expected a matplotlib Figure, got {type(value).__name__}"
            )
        if not value.get_axes():
            raise ValidationError("figure has no axes -- nothing was plotted")
        return

    if not isinstance(value, pd.DataFrame):
        raise ValidationError(f"expected a pandas DataFrame, got {type(value).__name__}")
    if value.empty:
        raise ValidationError("DataFrame is empty")

    missing = [c for c in s.columns if c not in value.columns]
    if missing:
        raise ValidationError(
            f"missing declared columns {missing}; got {list(value.columns)}"
        )


def _run_code(code: str, args: list[Any]) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    if "run" not in namespace:
        raise ValidationError("generated module defines no function named `run`")
    return namespace["run"](*args)


def execute_node(
    task: Task,
    plan: AnalysisPlan,
    code: str,
    upstream: dict[str, Any],
) -> NodeResult:
    """Execute one task, repairing up to `task.max_repair_attempts` times."""
    result = NodeResult(task_id=task.task_id, code=code)
    args = [upstream[name] for name in codegen.signature(task, plan)]

    for attempt in range(task.max_repair_attempts + 1):
        result.attempts = attempt + 1
        try:
            value = _run_code(result.code, args)
            validate_output(task, value)
        except Exception:
            err = traceback.format_exc(limit=6)
            log.warning(
                "%s failed on attempt %d/%d",
                task.task_id,
                result.attempts,
                task.max_repair_attempts + 1,
            )
            result.error = err
            result.repair_history.append(err)
            if attempt == task.max_repair_attempts:
                result.state = NodeState.FAILED
                return result
            log.info("repairing %s", task.task_id)
            result.code = codegen.generate_one(
                task, plan, prior_error=err, prior_code=result.code
            )
            continue

        result.value = value
        result.state = NodeState.DONE
        # `error` reflects only the latest attempt (empty = the winning one
        # raised nothing), but `repair_history` keeps every prior failure so
        # a self-healed run still shows what it healed from.
        result.error = ""
        return result

    return result  # unreachable; kept so every path returns a NodeResult


def run_plan(
    plan: AnalysisPlan, code: dict[str, str], max_workers: int = 4
) -> dict[str, NodeResult]:
    """Execute the DAG layer by layer, in parallel within each layer."""
    results: dict[str, NodeResult] = {}
    values: dict[str, Any] = {}

    for layer in layers(plan):
        runnable, skipped = [], []
        for tid in layer:
            task = plan.by_id(tid)
            dead = [d for d in task.depends_on if not results[d].ok]
            (skipped if dead else runnable).append((tid, dead))

        for tid, dead in skipped:
            log.warning("skipping %s: upstream failed %s", tid, dead)
            results[tid] = NodeResult(
                task_id=tid,
                state=NodeState.SKIPPED,
                error=f"upstream failed: {', '.join(dead)}",
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                tid: pool.submit(
                    execute_node, plan.by_id(tid), plan, code[tid], dict(values)
                )
                for tid, _ in runnable
            }
            for tid, fut in futures.items():
                res = fut.result()
                results[tid] = res
                if res.ok:
                    values[plan.by_id(tid).output_name] = res.value

    return results
