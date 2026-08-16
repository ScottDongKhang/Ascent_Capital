"""End-to-end entrypoint: plan -> codegen -> execute -> report.

    .venv/bin/python -m analyst.cli --event ukraine_2022
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import codegen, events, plans, report
from .execute import run_plan
from .types import NodeState

ASSETS = {
    "oil": ("CL=F", "WTI crude"),
    "brent": ("BZ=F", "Brent crude"),
    "dxy": ("DX-Y.NYB", "US dollar index"),
    "eurusd": ("EURUSD=X", "EUR/USD"),
    "gold": ("GC=F", "Gold"),
}

OUT = Path(__file__).resolve().parent.parent / "outputs" / "analyst"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event", default="ukraine_2022", choices=sorted(events.EVENTS))
    p.add_argument("--asset-a", default="oil", choices=sorted(ASSETS))
    p.add_argument("--asset-b", default="dxy", choices=sorted(ASSETS))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    event = events.resolve(args.event)
    plan = plans.build_event_move_plan(event, ASSETS[args.asset_a], ASSETS[args.asset_b])

    print(f"\nQuestion: {plan.question}")
    print(f"Plan:     {len(plan.tasks)} tasks, {len(plans.layers(plan))} layers")
    for i, layer in enumerate(plans.layers(plan), 1):
        print(f"  layer {i}: {', '.join(layer)}")

    t0 = time.time()
    print(f"\nGenerating {len(plan.tasks)} code blocks in parallel...")
    code = codegen.generate_all(plan)
    print(f"  done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("\nExecuting DAG...")
    results = run_plan(plan, code)
    print(f"  done in {time.time() - t1:.1f}s")

    outdir = OUT / f"{event.key}_{args.asset_a}_{args.asset_b}"
    path = report.write_report(plan, results, outdir)

    print("\nResults:")
    failed = []
    for task in plan.tasks:
        r = results[task.task_id]
        mark = {"done": "ok", "failed": "FAILED", "skipped": "skipped"}[r.state.value]
        note = f" ({r.attempts} attempts)" if r.attempts > 1 else ""
        print(f"  {mark:8s} {task.task_id}{note}")
        if r.state is not NodeState.DONE:
            failed.append((task, r))

    print(f"\nReport: {path}")

    if failed:
        print("\n" + "=" * 72)
        print("FAILURES -- surfacing rather than retrying further")
        print("=" * 72)
        for task, r in failed:
            print(f"\n--- {task.task_id} ({r.state.value}, {r.attempts} attempts) ---")
            print(r.error.strip() or "(no detail)")
            if r.code:
                print(f"\nlast code written to: {outdir / 'code' / (task.task_id + '.py')}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
