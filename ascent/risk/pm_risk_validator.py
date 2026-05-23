# ascent/risk/pm_risk_validator.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]  # ascent/risk/ → repo root

MAX_POSITION = 0.15
MAX_SECTOR = 0.40
MIN_POSITIONS = 5
DISTRESSED_THRESHOLD = -0.65


def validate(
    portfolio: Dict[str, float],
    allow_shorts: bool = False,
) -> Tuple[bool, List[str]]:
    """Pre-blend hard-limit check. Returns (ok, violations). Never raises.
    allow_shorts: if True, negative weights are permitted (130/30 mode).
    """
    if not portfolio:
        return False, ["Empty portfolio"]

    violations: List[str] = []

    if not allow_shorts:
        for sym, w in portfolio.items():
            if w < 0:
                violations.append(f"Negative weight for {sym}: {w:.4f}")

    if violations:
        return False, violations

    total = sum(portfolio.values())
    if total <= 0:
        return False, ["Portfolio weights sum to zero or negative"]
    weights = {sym: w / total for sym, w in portfolio.items()}

    for sym, w in weights.items():
        if w > MAX_POSITION:
            violations.append(f"{sym} weight {w:.1%} exceeds max {MAX_POSITION:.0%}")

    if len(weights) < MIN_POSITIONS:
        violations.append(f"Only {len(weights)} positions (min {MIN_POSITIONS})")

    sector_weights = _compute_sector_weights(weights)
    # Only check sector limits if we have real sector mappings (not all "unknown")
    if sector_weights and sector_weights.get("unknown", 0.0) < 1.0:
        for sector, sw in sector_weights.items():
            if sw > MAX_SECTOR and sector != "unknown":
                violations.append(f"Sector {sector} at {sw:.1%} exceeds max {MAX_SECTOR:.0%}")

    distressed = _get_distressed_names(list(weights.keys()))
    for sym in distressed:
        if sym in weights:
            violations.append(f"{sym} is distressed (mom_252d < {DISTRESSED_THRESHOLD:.0%})")

    return len(violations) == 0, violations


def _compute_sector_weights(weights: Dict[str, float]) -> Dict[str, float]:
    sector_map = _load_sector_map()
    result: Dict[str, float] = {}
    for sym, w in weights.items():
        sector = sector_map.get(sym, "unknown")
        result[sector] = result.get(sector, 0.0) + w
    return result


def _load_sector_map() -> Dict[str, str]:
    try:
        import pandas as pd
        p = _REPO_ROOT / "data_cache" / "profiles.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" in df.columns and "sector" in df.columns:
                return dict(zip(df["symbol"], df["sector"]))
    except Exception as exc:
        log.warning("[PMValidator] Could not load sector map: %s", exc)
    return {}


def _get_distressed_names(symbols: List[str]) -> List[str]:
    try:
        import pandas as pd
        p = _REPO_ROOT / "data_cache" / "features_cache.parquet"
        if not p.exists():
            return []
        df = pd.read_parquet(p)
        if "symbol" not in df.columns or "mom_252d" not in df.columns:
            return []
        latest = df.sort_values("date").groupby("symbol").last().reset_index()
        distressed = latest[latest["mom_252d"] < DISTRESSED_THRESHOLD]["symbol"].tolist()
        return [s for s in distressed if s in symbols]
    except Exception as exc:
        log.warning("[PMValidator] Could not check distressed names: %s", exc)
        return []
