"""In-process scheduler: the daily jobs of the service, on ET wall-clock times.

Jobs (times configurable through the environment, all in ``America/New_York``):

=================  =========  ==========================================================
``forecast_all``   04:30 ET   refresh weather, fetch today's partial load, forecast
                              tomorrow for every zone (TFT when valid, trees otherwise)
``ingest_score``   06:30 ET   catch up every table to yesterday, score yesterday's
                              forecasts and schedules in MAPE and dollars, refresh alerts
``isolf_refresh``  08:30 ET   fetch tomorrow's ISO forecast and DA prices (posted after
                              the 05:00 close) for the overlay and the pre-close benchmark
``retention``      07:00 ET   on the 1st of the month: prune data past the retention window
=================  =========  ==========================================================

A job is *due* when its latest scheduled occurrence at or before now is later than its
last recorded run, so a job missed while the VM was down runs at the next start (the
catch-up semantics the data layer expects). One daemon thread runs the loop; jobs never
raise out of it, and every run lands in the ``jobs`` table for ``/jobs`` and ``/health``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from typing import Any

from app import db, service
from src.config import MARKET_TZ

log = logging.getLogger(__name__)

ENABLED = (os.environ.get("SCHEDULER_ENABLED") or "1") not in ("0", "false", "no")
TICK_SECONDS = float(os.environ.get("SCHEDULER_TICK_SECONDS") or 60)


def _et(name: str, default: str) -> dtime:
    raw = os.environ.get(name) or default
    hh, mm = raw.split(":")
    return dtime(int(hh), int(mm))


@dataclass(frozen=True)
class Job:
    name: str
    at: dtime  # local ET wall-clock time
    run: Callable[[], Any]
    monthly: bool = False  # only on the 1st of the month


def run_forecast_all() -> dict[str, Any]:
    today = service.TODAY()
    weather_ok = service.refresh_weather(today)
    partial = service.ingest_days(today, today, ("pal",))["pal"]
    results = service.forecast_all(today + timedelta(days=1), "auto")
    return {
        "target_date": (today + timedelta(days=1)).isoformat(),
        "weather_refreshed": weather_ok,
        "partial_load_rows": partial["rows"],
        "zones": results,
    }


def run_ingest_score() -> dict[str, Any]:
    ingested = service.catch_up()
    scored = service.score_pending()
    return {
        "ingested": {k: v for k, v in ingested.items() if not k.endswith(":retry")},
        "scored_forecasts": len(scored["forecasts"]),
        "scored_schedules": len(scored["schedules"]),
        "alerts": scored["alerts"],
    }


def run_isolf_refresh() -> dict[str, Any]:
    today = service.TODAY()
    tomorrow = today + timedelta(days=1)
    return service.ingest_days(today, tomorrow, ("isolf", "damlbmp_zone"))


def run_retention() -> dict[str, Any]:
    return service.prune()


JOBS: tuple[Job, ...] = (
    Job("forecast_all", _et("SCHEDULE_FORECAST_ET", "04:30"), run_forecast_all),
    Job("ingest_score", _et("SCHEDULE_INGEST_ET", "06:30"), run_ingest_score),
    Job("isolf_refresh", _et("SCHEDULE_ISOLF_ET", "08:30"), run_isolf_refresh),
    Job("retention", _et("SCHEDULE_RETENTION_ET", "07:00"), run_retention, monthly=True),
)
JOB_BY_NAME = {j.name: j for j in JOBS}


def last_occurrence(job: Job, now_utc: datetime) -> datetime:
    """The most recent scheduled time of `job` at or before `now_utc` (UTC, tz-aware)."""
    local = now_utc.astimezone(MARKET_TZ)
    day: date = local.date()
    for _ in range(0, 40):
        if not job.monthly or day.day == 1:
            candidate = datetime.combine(day, job.at, tzinfo=MARKET_TZ)
            if candidate <= local:
                return candidate.astimezone(UTC)
        day -= timedelta(days=1)
    raise RuntimeError("no scheduled occurrence found")  # pragma: no cover


def is_due(job: Job, now_utc: datetime, last_run: str | None) -> bool:
    due_at = last_occurrence(job, now_utc)
    if last_run is None:
        return True
    last = datetime.strptime(last_run, db.TS_FORMAT).replace(tzinfo=UTC)
    return last < due_at


def run_job(name: str) -> dict[str, Any]:
    """Run one job now (scheduler tick or the admin endpoint); records the outcome."""
    job = JOB_BY_NAME[name]
    t0 = time.perf_counter()
    log.info("job %s: start", name)
    try:
        detail = job.run()
    except Exception as exc:
        log.exception("job %s failed", name)
        service.record_job(name, "error", {"error": str(exc)})
        return {"name": name, "status": "error", "error": str(exc)}
    elapsed = round(time.perf_counter() - t0, 1)
    service.record_job(name, "ok", {"seconds": elapsed, "result": detail})
    log.info("job %s: done in %.1fs", name, elapsed)
    return {"name": name, "status": "ok", "seconds": elapsed, "result": detail}


def _last_runs() -> dict[str, str | None]:
    return {j["name"]: j["last_run_at"] for j in service.job_status()}


def due_jobs(now_utc: datetime | None = None) -> list[str]:
    now_utc = now_utc or datetime.now(UTC)
    last = _last_runs()
    return [j.name for j in JOBS if is_due(j, now_utc, last.get(j.name))]


class Scheduler:
    def __init__(self, tick: float = TICK_SECONDS) -> None:
        self.tick = tick
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="gridcast-scheduler", daemon=True)
        self._thread.start()
        log.info(
            "scheduler started: %s",
            ", ".join(f"{j.name}@{j.at:%H:%M}ET{' monthly' if j.monthly else ''}" for j in JOBS),
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for name in due_jobs():
                    run_job(name)
            except Exception:
                log.exception("scheduler tick failed")
            self._stop.wait(self.tick)


scheduler = Scheduler()
