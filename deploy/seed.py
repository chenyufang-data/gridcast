"""Seed a deployment: backfill the store from the NYISO archive, weather, forecasts, scores.

Runs inside the backend container (the code and the data volume are there):

    docker compose -f docker-compose.prod.yml exec backend python deploy/seed.py
    docker compose -f docker-compose.prod.yml exec backend python deploy/seed.py --months 15 --forecast-days 30

Steps: (1) ingest every table from the retention boundary (or ``--months`` back) to
yesterday, monthly zips for past months and daily files for the rest; (2) fetch the
weather history for the same range plus a week ahead; (3) forecast the trailing
``--forecast-days`` days with the served model (the ONNX TFT when the bundle is on the
volume, else the trees) so the cards and the accuracy views are populated on day one;
(4) score them and forecast tomorrow. Re-running is safe: every step upserts.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service, weather  # noqa: E402
from app.log import configure_logging  # noqa: E402
from app.serving import registry  # noqa: E402

log = logging.getLogger("seed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--months", type=int, default=service.RETENTION_MONTHS)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--forecast-days", type=int, default=14)
    parser.add_argument("--model", choices=service.MODELS, default="auto")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()

    today = service.TODAY()
    yesterday = today - timedelta(days=1)
    start = args.start or service.retention_boundary(today, args.months)
    t0 = time.perf_counter()

    if not args.skip_ingest:
        print(f"1/4 ingest {start} .. {yesterday} (+ tomorrow's isolf / DA prices)")
        summary = service.ingest_days(start, yesterday)
        for ft, s in summary.items():
            print(
                f"    {ft:14s} {s['days']:4d} days {s['rows']:9d} rows, {len(s['missing'])} missing"
            )
        service.ingest_days(today, today + timedelta(days=1), ("isolf", "damlbmp_zone"))
        service.ingest_days(today, today, ("pal",))
    if not args.skip_weather:
        print(f"2/4 weather {start} .. {today + timedelta(days=7)}")
        ok = weather.update(start, today + timedelta(days=service.WEATHER_DAYS_AHEAD))
        print(f"    {'ok' if ok else 'FAILED (models run without weather)'}")

    status = registry.reload()
    print(f"3/4 forecasts: TFT {status['tft'].get('version') or status['tft'].get('error')}")
    first = today - timedelta(days=args.forecast_days)
    for k in range((today + timedelta(days=1) - first).days + 1):
        target = first + timedelta(days=k)
        results = service.forecast_all(target, args.model)
        ok = sum(r["status"] == "ok" for r in results)
        models = {r.get("model", "").split(":")[0] for r in results if r["status"] == "ok"}
        print(f"    {target}: {ok}/{len(results)} zones ({', '.join(sorted(models))})")

    print("4/4 scoring")
    scored = service.score_pending()
    print(
        f"    {len(scored['forecasts'])} forecasts, {len(scored['schedules'])} schedules, "
        f"{len(scored['alerts'])} alerts"
    )
    for card in service.zone_cards():
        week = card["last_7d"]
        mape = week["mape_hour"]
        iso = week["isolf_mape_hour"]
        print(
            f"    {card['zone']:7s} {card['history_days']:4d} d history, "
            f"7-day MAPE {mape:.2f}% (ISO {iso:.2f}%)"
            if mape is not None and iso is not None
            else f"    {card['zone']:7s} {card['history_days']:4d} d history, not scored yet"
        )
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
