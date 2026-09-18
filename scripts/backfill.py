"""Backfill the processed dataset from the NYISO public archive.

    python scripts/backfill.py                      # 2025-06-01 .. yesterday, all tables
    python scripts/backfill.py --start 2025-06-01 --end 2026-08-31 --tables load_slots isolf

Downloads go to data/cache (monthly zips for past months, daily files for the last
few days), normalized frames to data/processed/*.pkl. Both are gitignored: NYISO data
is fetched, never redistributed (data/README.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.log import configure_logging  # noqa: E402
from app.nyiso import ArchiveClient  # noqa: E402
from src import dataset  # noqa: E402
from src.config import WARMUP_START  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=date.fromisoformat, default=WARMUP_START)
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None, help="default: yesterday (ET)"
    )
    parser.add_argument("--tables", nargs="+", choices=dataset.NAMES, default=list(dataset.NAMES))
    args = parser.parse_args(argv)

    configure_logging()
    logging.getLogger(__name__).info(
        "backfill %s .. %s: %s", args.start, args.end or "yesterday", args.tables
    )
    frames = dataset.build(
        ArchiveClient(), start=args.start, end=args.end, names=tuple(args.tables)
    )
    report = dataset.coverage_report(frames)
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
