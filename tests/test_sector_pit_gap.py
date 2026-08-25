# tests/test_sector_pit_gap.py
"""
Tests for scripts/measure_sector_pit_gap.py's reason-string classification.

This is the one piece of nontrivial logic in that script: which REMOVED_STOCKS
reason strings count as a "business-identity change" (acquisition / merger /
spin-off) versus a routine index reshuffle or a failure/receivership. A
regression here (e.g. a reason phrasing the regex stops matching) would
silently under-count the measured PIT sector gap without raising an error.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_sector_pit_gap.py"
_spec = importlib.util.spec_from_file_location("measure_sector_pit_gap", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["measure_sector_pit_gap"] = _mod
_spec.loader.exec_module(_mod)

classify_reason = _mod.classify_reason
is_business_identity_change = _mod.is_business_identity_change


@pytest.mark.parametrize(
    "reason,expected_flagged",
    [
        ("Acquired by Aetna (AET)", True),
        ("HNZ acquired by consortium", True),
        ("Zoetis spun off by Pfizer", True),
        ("Kraft merger with Heinz.", True),
        ("WarnerMedia and Discovery merge to create Warner Bros. Discovery.", True),
        ("Arconic separated into 2 companies.", True),
        ("BMC taken private by consortium", True),
        ("Market capitalization changes", False),
        ("Market Capitalization Changes.", False),
        ("FDIC placed Silicon Valley Bank into receivership.", False),
        ("CMCSK shares no longer listed", False),
    ],
)
def test_classify_reason_flags_business_identity_changes(reason, expected_flagged):
    bucket = classify_reason(reason)
    assert is_business_identity_change(bucket) == expected_flagged


def test_market_cap_reason_bucketed_as_index_reshuffle():
    bucket = classify_reason("Market capitalization change.")
    assert "index reshuffle" in bucket


def test_receivership_bucketed_separately_from_business_identity_change():
    bucket = classify_reason("FDIC placed First Republic Bank into receivership.")
    assert "receivership" in bucket
    assert not is_business_identity_change(bucket)


def test_all_removed_stocks_reasons_classify_without_error():
    """Every real reason string in REMOVED_STOCKS must classify to some bucket
    without raising, and the flagged count must be a strict subset."""
    from ascent.data.universe import REMOVED_STOCKS

    n_flagged = 0
    for _symbol, _sector, _removed_date, reason in REMOVED_STOCKS:
        bucket = classify_reason(reason)
        assert bucket  # never empty
        if is_business_identity_change(bucket):
            n_flagged += 1

    assert 0 < n_flagged < len(REMOVED_STOCKS)
