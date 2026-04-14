"""
scripts/ingest_verdict_history.py
One-time ingestion of existing verdict history into R2R.

Usage:
    .venv/bin/python scripts/ingest_verdict_history.py

If R2R_API_KEY is not set, prints a notice and exits (nothing to do —
the local fallback in r2r_interface.py reads files directly).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.r2r_interface import ingest_verdict, R2R_API_KEY

DEBATE_LOG_DIR = Path("outputs/debate_log")


def main():
    if not R2R_API_KEY:
        print("[Ingest] R2R_API_KEY not set — local search reads files directly, no ingestion needed.")
        print("[Ingest] Set R2R_API_KEY in .env if you have an R2R account.")
        return

    verdict_files = sorted(DEBATE_LOG_DIR.glob("verdict_*.json"))
    if not verdict_files:
        print(f"[Ingest] No verdict files found in {DEBATE_LOG_DIR}.")
        return

    print(f"[Ingest] Found {len(verdict_files)} verdict files. Ingesting into R2R...")
    success, failed = 0, 0

    for vf in verdict_files:
        try:
            ingest_verdict(vf)
            print(f"[Ingest] + {vf.name}")
            success += 1
        except Exception as e:
            print(f"[Ingest] x {vf.name}: {e}")
            failed += 1

    print(f"\n[Ingest] Done: {success} ingested, {failed} failed.")


if __name__ == "__main__":
    main()
