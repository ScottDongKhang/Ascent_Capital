"""The one hardcoded plan, plus the structural rules it must satisfy.

There is no general planner yet. This module stands in for one, and it exists
separately from `types.py` so that when a planner LLM does arrive it has an
obvious seam to plug into: it must emit an `AnalysisPlan` that survives
`validate_plan`. The validator is the part worth keeping.
"""
from __future__ import annotations

from .events import Event
from .types import AnalysisPlan, OutputSchema, Task, TaskCategory

HORIZONS = (1, 5, 20, 60)


def build_event_move_plan(
    event: Event,
    asset_a: tuple[str, str],
    asset_b: tuple[str, str],
    lookback_days: int = 180,
    lookforward_days: int = 120,
) -> AnalysisPlan:
    """The 5-task DAG from the architecture diagrams.

        load_asset_a ─┬─► calc_asset_moves ──► asset_moves_table
        load_asset_b ─┘
        load_asset_a ────► asset_a_chart

    `asset_a` and `asset_b` are (ticker, label) pairs.
    """
    a_ticker, a_label = asset_a
    b_ticker, b_label = asset_b

    window = _window(event.date, lookback_days, lookforward_days)
    params = {
        "event_key": event.key,
        "event_name": event.name,
        "event_date": event.date,
        "asset_a_ticker": a_ticker,
        "asset_a_label": a_label,
        "asset_b_ticker": b_ticker,
        "asset_b_label": b_label,
        "start": window[0],
        "end": window[1],
        "horizons": list(HORIZONS),
    }

    price_schema = OutputSchema(
        kind="dataframe",
        columns={"Close": "float"},
        index_kind="market_trading_day",
        row_semantics="one row per trading session in the window",
    )

    tasks = (
        Task(
            task_id="load_asset_a",
            title=f"Load {a_label} data",
            category=TaskCategory.LOAD,
            description=(
                f"Load daily prices for {a_label} (yfinance ticker {a_ticker!r}) "
                f"from {window[0]} to {window[1]} using "
                f"toolkit.load_prices(ticker, start, end). Return the full "
                f"OHLCV DataFrame unchanged; it already has a DatetimeIndex and "
                f"flat columns including 'Close'."
            ),
            output_name="asset_a_prices",
            output_schema=price_schema,
            expected_ticker=a_ticker,
        ),
        Task(
            task_id="load_asset_b",
            title=f"Load {b_label} data",
            category=TaskCategory.LOAD,
            description=(
                f"Load daily prices for {b_label} (yfinance ticker {b_ticker!r}) "
                f"from {window[0]} to {window[1]} using "
                f"toolkit.load_prices(ticker, start, end). Return the full "
                f"OHLCV DataFrame unchanged."
            ),
            output_name="asset_b_prices",
            output_schema=price_schema,
            expected_ticker=b_ticker,
        ),
        Task(
            task_id="calc_asset_moves",
            title="Calculate asset price moves",
            category=TaskCategory.TRANSFORM,
            description=(
                f"For each asset, compute the percent change in the 'Close' "
                f"column measured from the last close at or before the event "
                f"date {event.date!r}, at each horizon in {list(HORIZONS)} "
                f"trading sessions after that anchor. Use "
                f"toolkit.pct_move(series, anchor, horizon_days), passing "
                f"pandas.Timestamp({event.date!r}) as the anchor. Return a "
                f"DataFrame with exactly the columns 'asset', 'horizon_days' "
                f"and 'pct_move', one row per (asset, horizon) pair, with "
                f"asset taking the values {a_label!r} and {b_label!r}. Use a "
                f"default RangeIndex."
            ),
            depends_on=("load_asset_a", "load_asset_b"),
            output_name="asset_moves",
            output_schema=OutputSchema(
                kind="dataframe",
                columns={"asset": "str", "horizon_days": "int", "pct_move": "float"},
                index_kind="none",
                row_semantics="one row per (asset, horizon) pair",
            ),
        ),
        Task(
            task_id="asset_a_chart",
            title=f"{a_label} price chart",
            category=TaskCategory.VISUALIZE,
            description=(
                f"Plot the 'Close' column of asset_a_prices over the full "
                f"window as a matplotlib line chart. Draw a vertical dashed "
                f"line at {event.date!r} and label it with the event name "
                f"{event.name!r}. Title the chart, label both axes including "
                f"the units, and add a legend. Create the figure with "
                f"fig, ax = plt.subplots(figsize=(11, 5)) and return fig. "
                f"Do not call plt.show()."
            ),
            depends_on=("load_asset_a",),
            output_name="asset_a_figure",
            output_schema=OutputSchema(kind="figure", row_semantics="one line chart"),
        ),
        Task(
            task_id="asset_moves_table",
            title="Asset price moves table",
            category=TaskCategory.VISUALIZE,
            description=(
                "Reshape asset_moves into a presentation table: one row per "
                "asset, one column per horizon, named like '+1d', '+5d', "
                "'+20d', '+60d', values being the percent move rounded to 2 "
                "decimals. The asset name must be a column called 'asset', not "
                "the index. Use a default RangeIndex."
            ),
            depends_on=("calc_asset_moves",),
            output_name="asset_moves_table",
            output_schema=OutputSchema(
                kind="dataframe",
                columns={"asset": "str"},
                index_kind="none",
                row_semantics="one row per asset",
            ),
        ),
    )

    plan = AnalysisPlan(
        question=(
            f"How did {a_label} and {b_label} move during {event.name} "
            f"({event.date})?"
        ),
        tasks=tasks,
        params=params,
    )
    validate_plan(plan)
    return plan


def validate_plan(plan: AnalysisPlan) -> None:
    """Structural rules. Raises ValueError on the first violation.

    This is the deterministic half of the compiler doing its job. A malformed
    plan must never reach codegen, because every downstream failure would then
    be diagnosed as a model problem when it is really a specification problem.
    """
    ids = [t.task_id for t in plan.tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task_id in plan: {ids}")

    names = [t.output_name for t in plan.tasks]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate output_name in plan: {names}")

    known = set(ids)
    for t in plan.tasks:
        for dep in t.depends_on:
            if dep not in known:
                raise ValueError(f"{t.task_id} depends on unknown task {dep!r}")
            if dep == t.task_id:
                raise ValueError(f"{t.task_id} depends on itself")

        if t.category is TaskCategory.LOAD and t.depends_on:
            raise ValueError(f"LOAD task {t.task_id} must have no upstreams")
        if t.category is TaskCategory.TRANSFORM and not t.depends_on:
            raise ValueError(f"TRANSFORM task {t.task_id} must have upstreams")

        if t.output_schema.kind == "dataframe" and not t.output_schema.columns:
            raise ValueError(f"{t.task_id} declares a dataframe with no columns")

    layers(plan)  # raises on a cycle


def layers(plan: AnalysisPlan) -> list[list[str]]:
    """Topological layers. Tasks within a layer are independent."""
    remaining = {t.task_id: set(t.depends_on) for t in plan.tasks}
    out: list[list[str]] = []
    done: set[str] = set()

    while remaining:
        ready = sorted(tid for tid, deps in remaining.items() if deps <= done)
        if not ready:
            raise ValueError(f"cycle in plan among tasks {sorted(remaining)}")
        out.append(ready)
        done.update(ready)
        for tid in ready:
            del remaining[tid]

    return out


def _window(event_date: str, back: int, forward: int) -> tuple[str, str]:
    import datetime as dt

    d = dt.date.fromisoformat(event_date)
    return (
        (d - dt.timedelta(days=back)).isoformat(),
        (d + dt.timedelta(days=forward)).isoformat(),
    )
