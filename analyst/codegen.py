"""One LLM call per task, all tasks generated in parallel.

The parallelism is not a trick -- it falls out of the plan being fully specified
before any code exists. `asset_moves_table` knows the exact columns of
`asset_moves` without having seen the code that builds it, so it can be
generated at the same time. That is the whole reason the plan phase is worth
paying for.
"""
from __future__ import annotations

import ast
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from .llm import complete
from .types import AnalysisPlan, Task

log = logging.getLogger(__name__)

SYSTEM = """You write one small, self-contained Python function per task, for a \
deterministic research pipeline.

Rules, all of them hard:
- Emit exactly one ```python code block. No prose outside it.
- The block defines a single function named `run`, plus imports. Nothing else \
executes at module level.
- `run` must take exactly the parameters named in the signature you are given, \
in that order.
- Return exactly the declared output. Do not print, do not write files, do not \
call plt.show().
- You may import: pandas as pd, numpy as np, matplotlib, and \
`from analyst import toolkit`.
- Use the toolkit helpers you are told to use. They exist so that every task \
reaches data the same way.
- No network access except through toolkit helpers. No file IO. No imports from \
any package beginning with `ascent`.
- Match the declared output schema exactly: the column names given are the \
column names expected, spelled that way."""

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def signature(task: Task, plan: AnalysisPlan) -> list[str]:
    """Parameter names for `run`: the output_name of each upstream, in order."""
    return [plan.by_id(dep).output_name for dep in task.depends_on]


def _schema_block(task: Task) -> str:
    s = task.output_schema
    if s.kind == "figure":
        return "Return a matplotlib Figure object."
    cols = ", ".join(f"{k} ({v})" for k, v in s.columns.items())
    return (
        f"Return a pandas DataFrame.\n"
        f"Required columns: {cols}\n"
        f"Index: {s.index_kind}\n"
        f"Rows: {s.row_semantics}"
    )


def build_prompt(task: Task, plan: AnalysisPlan, prior_error: str = "", prior_code: str = "") -> str:
    params = signature(task, plan)
    sig = f"def run({', '.join(params)}):" if params else "def run():"

    upstream = ""
    if task.depends_on:
        lines = []
        for dep in task.depends_on:
            d = plan.by_id(dep)
            lines.append(f"- `{d.output_name}`: {_schema_block(d)}".replace("\n", " "))
        upstream = "\n\nYou are given these inputs:\n" + "\n".join(lines)

    prompt = (
        f"Research question: {plan.question}\n\n"
        f"Task: {task.title}\n"
        f"Category: {task.category.value}\n\n"
        f"What to compute:\n{task.description}\n"
        f"{upstream}\n\n"
        f"Required signature:\n    {sig}\n\n"
        f"Output contract:\n{_schema_block(task)}\n"
    )

    if prior_error:
        prompt += (
            f"\n---\n\nYour previous attempt failed. Here is the code you wrote:\n\n"
            f"```python\n{prior_code}\n```\n\n"
            f"And here is what went wrong:\n\n```\n{prior_error}\n```\n\n"
            f"Return a corrected version of the whole function. Fix the actual "
            f"cause; do not paper over it with a try/except that returns empty "
            f"or fabricated data."
        )

    return prompt


def extract_code(text: str) -> str:
    m = _FENCE.search(text)
    code = m.group(1) if m else text
    code = code.strip()
    ast.parse(code)  # fail here, loudly, rather than at exec time
    return code


def generate_one(task: Task, plan: AnalysisPlan, prior_error: str = "", prior_code: str = "") -> str:
    log.info("codegen: %s", task.task_id)
    text = complete(SYSTEM, build_prompt(task, plan, prior_error, prior_code))
    return extract_code(text)


def generate_all(plan: AnalysisPlan, max_workers: int = 8) -> dict[str, str]:
    """Generate every task's code concurrently. Returns task_id -> source."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {t.task_id: pool.submit(generate_one, t, plan) for t in plan.tasks}
        return {tid: f.result() for tid, f in futures.items()}
