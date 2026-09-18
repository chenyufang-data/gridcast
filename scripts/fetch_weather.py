"""Fetch day-ahead-issued temperature forecasts for every zone into data/weather.csv.

    python scripts/fetch_weather.py                  # 2025-06-01 .. today + 7
    python scripts/fetch_weather.py --start 2025-06-01 --end 2026-09-16

Source: Open-Meteo previous-runs API (CC BY 4.0). See app/weather.py for the leakage
rule (previous_day2 = issued before the D-1 05:00 ET cutoff for every hour of D).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import weather  # noqa: E402
from app.log import configure_logging  # noqa: E402
from src.config import WARMUP_START  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=date.fromisoformat, default=WARMUP_START)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    ok = weather.refresh(args.start, args.end)
    if ok:
        df = weather.load_weather()
        assert df is not None
        print(
            f"{len(df)} zone-days, {df['date'].min().date()} .. {df['date'].max().date()}, "
            f"zones {df['zone'].nunique()}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
