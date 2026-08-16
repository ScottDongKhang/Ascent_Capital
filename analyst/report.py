"""Write the run out to disk: figures, tables, the generated code, and an index.

The generated code is saved alongside the outputs on purpose. An analysis you
cannot read the source of is not reproducible, and the whole point of compiling
the plan rather than chatting with an agent is that the artifact survives the
session.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from .types import AnalysisPlan, NodeResult, NodeState

log = logging.getLogger(__name__)


def _markdown_table(df: pd.DataFrame) -> str:
    """Render without pulling in `tabulate` just to draw pipes."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "| " + " | ".join("-" * w for w in widths) + " |"]
    out += ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join(out)


def write_report(
    plan: AnalysisPlan, results: dict[str, NodeResult], outdir: Path
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    code_dir = outdir / "code"
    code_dir.mkdir(exist_ok=True)

    lines = [f"# {plan.question}", ""]
    lines.append(f"*Event date: {plan.params['event_date']} "
                 f"({plan.params['event_name']})*")
    lines.append("")

    for task in plan.tasks:
        res = results[task.task_id]
        if res.code:
            (code_dir / f"{task.task_id}.py").write_text(res.code)

        if res.state is not NodeState.DONE:
            lines += [
                f"## {task.title} — {res.state.value.upper()}",
                "",
                "```",
                res.error.strip() or "(no detail)",
                "```",
                "",
            ]
            continue

        repaired_note = ""
        if res.attempts > 1 and res.repair_history:
            repaired_note = (
                f"\n*Self-healed after {res.attempts} attempts. "
                f"Earlier failures: "
                + "; ".join(e.strip().splitlines()[-1] for e in res.repair_history)
                + "*\n"
            )

        if isinstance(res.value, Figure):
            png = outdir / f"{task.task_id}.png"
            res.value.savefig(png, dpi=140, bbox_inches="tight")
            lines += [f"## {task.title}", "", f"![{task.title}]({png.name})", repaired_note]
        elif isinstance(res.value, pd.DataFrame):
            csv = outdir / f"{task.task_id}.csv"
            res.value.to_csv(csv, index=False)
            if task.task_id.endswith("table"):
                lines += [
                    f"## {task.title}",
                    "",
                    _markdown_table(res.value),
                    "",
                    f"*[{csv.name}]({csv.name})*",
                    repaired_note,
                    "",
                ]

    summary = {
        "question": plan.question,
        "params": plan.params,
        "tasks": {
            tid: {
                "state": r.state.value,
                "attempts": r.attempts,
                "error": r.error.strip()[-800:] if r.error else "",
                "repair_history": [e.strip()[-800:] for e in r.repair_history],
            }
            for tid, r in results.items()
        },
    }
    (outdir / "run.json").write_text(json.dumps(summary, indent=2, default=str))

    path = outdir / "report.md"
    path.write_text("\n".join(lines))
    return path
