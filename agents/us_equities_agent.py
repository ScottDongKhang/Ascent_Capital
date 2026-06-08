"""
agents/us_equities_agent.py
US Equities specialist agent — wrapper around the existing ascent pipeline.

NOT a new pipeline. Calls run_pipeline() exactly as eod_runner.py does,
but wraps the output in AgentOutput format for the orchestrator.

Usage:
    python3 -m agents.us_equities_agent --dry-run
"""

import sys
from datetime import date
import pandas as pd

from ascent.config.settings import get_config
from ascent.config.types import AgentOutput
from ascent.main import run_pipeline

AGENT_ID = "us_equities"


def run_us_equities_agent(
    dry_run: bool = False,
    as_of_date: date = None,
    extra_symbols: list[str] | None = None,
) -> AgentOutput:
    """
    Run the full US equities pipeline and wrap output as AgentOutput.
    """
    as_of_date = as_of_date or date.today()

    print(f"\n{'='*60}")
    print(f"[USEquities] Running US equities agent | {as_of_date}")
    print(f"{'='*60}")

    try:
        (
            result,
            regime_engine,
            spy_wide,
            univ_wide,
            vix_series,
            target_weights_all,
            price_df,
            macro_df,
            price_cache_name,
            _alpha_breakdown,
        ) = run_pipeline(live=True, extra_symbols=extra_symbols)

        # Extract latest non-zero target weights
        if target_weights_all is not None and not target_weights_all.empty:
            valid_rows = target_weights_all[(target_weights_all > 0).any(axis=1)]
            if not valid_rows.empty:
                latest_row = valid_rows.iloc[-1]
                nonzero    = latest_row[latest_row > 0].dropna()
                target_weights = {sym: round(float(w), 6) for sym, w in nonzero.items()}
            else:
                target_weights = {}
        else:
            target_weights = {}

        # Extract regime signal
        regime_signal = None
        try:
            if regime_engine is not None:
                # Try get_signal with the latest date
                latest_ts = pd.Timestamp(target_weights_all.index[-1]) if (
                    target_weights_all is not None and not target_weights_all.empty
                ) else pd.Timestamp.now()

                sig = regime_engine.get_signal(latest_ts)
                if sig is not None:
                    # Use .value to get clean string ("stressed") not "RegimeLabel.STRESSED"
                    regime_signal = sig.label.value if hasattr(sig.label, "value") else str(sig.label)
        except Exception:
            pass

        # Build signal quality summary for top holdings — feeds AI PM two-way loop
        _signal_quality: dict = {}
        try:
            _conv  = _alpha_breakdown.get("convergence", {})
            _pslv  = _alpha_breakdown.get("primary_sleeve", {})
            _per_s = _alpha_breakdown.get("per_sleeve", {})
            for sym in list(target_weights.keys())[:20]:
                _signal_quality[sym] = {
                    "convergence":     round(_conv.get(sym, 0.5), 3),
                    "primary_sleeve":  _pslv.get(sym, "trend"),
                    "sleeve_scores":   {
                        sl: round(float(scores.get(sym, 0.0)), 3)
                        for sl, scores in _per_s.items()
                        if sym in scores
                    },
                }
        except Exception:
            pass

        output = AgentOutput(
            agent_id=AGENT_ID,
            as_of_date=as_of_date,
            target_weights=target_weights,
            regime_signal=regime_signal,
            skill_score=None,  # skill tracker fills this
            metadata={
                "n_positions":    len(target_weights),
                "pipeline_mode":  "live",
                "signal_quality": _signal_quality,   # per-stock sleeve breakdown for AI PM
            },
        )

    except Exception as e:
        print(f"[USEquities] Pipeline failed: {e}")
        output = AgentOutput(
            agent_id=AGENT_ID,
            as_of_date=as_of_date,
            target_weights={},
            metadata={"error": str(e)},
        )

    print(f"[USEquities] {output.summary()}")
    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    output  = run_us_equities_agent(dry_run=dry_run)
