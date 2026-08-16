"""Core types for the research compiler.

A plan is a DAG of tasks. Each task compiles, via one LLM call, to a Python
function named `run`. The plan is fully specified before any code exists, which
is what lets every task be generated in parallel: a VISUALIZE task already knows
the schema of a dataframe whose code has not been written yet.

This module is deliberately behaviour-free. It holds the contract, nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class TaskCategory(str, Enum):
    """What a task is allowed to do.

    The categories are enforced in `plans.validate_plan`, not decorative:
    LOAD may perform IO and has no upstreams; TRANSFORM is pure and must have
    upstreams; VISUALIZE renders and registers no series.
    """

    LOAD = "load"
    TRANSFORM = "transform"
    VISUALIZE = "visualize"


@dataclass(frozen=True)
class OutputSchema:
    """The declared shape of a task's output, checked after execution."""

    kind: Literal["dataframe", "figure"]
    columns: dict[str, str] = field(default_factory=dict)
    index_kind: str = "none"  # market_trading_day | calendar_day | none
    row_semantics: str = ""


@dataclass(frozen=True)
class Task:
    """One node of the plan. Compiles to exactly one Python function."""

    task_id: str
    title: str
    category: TaskCategory
    description: str
    output_name: str
    output_schema: OutputSchema
    depends_on: tuple[str, ...] = ()
    max_repair_attempts: int = 2
    expected_ticker: str | None = None
    """For LOAD tasks that fetch a single named ticker. When set, execution
    checks that the DataFrame actually returned carries this ticker (via
    `toolkit.load_prices`'s `attrs["ticker"]`), not just the right shape.
    Closes the silent-substitution hole: a repaired loader that quietly
    swaps in a different, valid ticker produces perfectly shaped output,
    so shape checking alone cannot catch it."""


@dataclass(frozen=True)
class AnalysisPlan:
    """A question plus the task DAG that answers it."""

    question: str
    tasks: tuple[Task, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def by_id(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(task_id)


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Execution outcome for one task. Mutable: the runner fills it in."""

    task_id: str
    state: NodeState = NodeState.PENDING
    value: Any = None
    code: str = ""
    attempts: int = 0
    error: str = ""
    repair_history: list[str] = field(default_factory=list)
    """Error from every failed attempt, oldest first, preserved even after a
    later attempt succeeds. `error` alone is cleared on success (it reflects
    only the most recent attempt's status), so this is the only record that a
    self-healed run took N tries and what broke on the earlier ones."""

    @property
    def ok(self) -> bool:
        return self.state is NodeState.DONE
