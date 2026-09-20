"""FastAPI entry point for the NYISO day-ahead forecasting service.

Run locally:   uvicorn app.main:app --reload
Docs UI:       http://127.0.0.1:8000/docs

Access model (docs/plan.md §5): every GET is public; POST / PATCH / DELETE require the
``X-Admin-Token`` header (``ADMIN_TOKEN`` in the environment) and pass a per-IP token
bucket. Data: NYISO public MIS archive, fetched at runtime and never redistributed;
responses carry derived aggregates and model outputs only.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import __version__, scheduler, service
from app.log import configure_logging
from app.serving import registry
from src.config import DEMO_DEFAULT_ZONE, MARKET_TZ, NYISO_ARCHIVE_BASE

configure_logging()
log = logging.getLogger(__name__)

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN") or ""
WRITE_METHODS = {"POST", "PATCH", "PUT", "DELETE"}
RATE_PER_MIN = float(os.environ.get("WRITE_RATE_PER_MIN") or 30)
RATE_BURST = int(os.environ.get("WRITE_RATE_BURST") or 10)
DATA_CREDIT = (
    f"Data: NYISO public MIS archive ({NYISO_ARCHIVE_BASE}/), fetched at runtime and not "
    "redistributed; weather: Open-Meteo (CC BY 4.0). Not affiliated with or endorsed by NYISO."
)


class TokenBucket:
    """Per-key token bucket: `rate` tokens per minute, `burst` capacity."""

    def __init__(self, rate_per_min: float, burst: int) -> None:
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1.0, now)
            return True


limiter = TokenBucket(RATE_PER_MIN, RATE_BURST)


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not ADMIN_TOKEN:
        log.warning("ADMIN_TOKEN is not set: every write endpoint will answer 503")
    status = registry.reload()
    log.info("models: %s", status["tft"].get("version") or status["tft"].get("error"))
    if scheduler.ENABLED:
        scheduler.scheduler.start()
    else:
        log.info("scheduler disabled (SCHEDULER_ENABLED=0)")
    yield
    scheduler.scheduler.stop()


app = FastAPI(
    title="gridcast",
    description=(
        "Day-ahead zonal load forecasts for the NYISO market: 15-min forecasts with "
        "P10-P90 bands, a cost-aware DAM bid, and settlement in dollars against "
        f"real-time prices. {DATA_CREDIT}"
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def guard_writes(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.method in WRITE_METHODS:
        if not ADMIN_TOKEN:
            return JSONResponse({"detail": "writes disabled: ADMIN_TOKEN not set"}, 503)
        supplied = request.headers.get("x-admin-token") or ""
        if not secrets.compare_digest(supplied, ADMIN_TOKEN):
            return JSONResponse({"detail": "missing or invalid X-Admin-Token"}, 401)
        if not limiter.allow(client_key(request)):
            return JSONResponse({"detail": "too many write requests; slow down"}, 429)
    return await call_next(request)


# ─── schemas ──────────────────────────────────────────────────────────────────────────
class ScheduleCreate(BaseModel):
    target_date: date
    hourly_mw: list[float] = Field(
        description="one MW bid per hour of the target day, chronological (24; 23/25 on DST days)"
    )
    note: str | None = None
    adjustments: dict[str, Any] | None = None


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(400, f"bad date {value!r}: use YYYY-MM-DD") from exc


def _range(start: str | None, end: str | None, days: int = 7) -> tuple[date, date]:
    e = _date(end) if end else service.TODAY() - timedelta(days=1)
    s = _date(start) if start else e - timedelta(days=days - 1)
    if s > e:
        raise HTTPException(400, "start must not be after end")
    if (e - s).days > 400:
        raise HTTPException(400, "range too long (max 400 days)")
    return s, e


# ─── health, models, jobs ─────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    now = datetime.now(MARKET_TZ)
    return {
        "status": "ok",
        "version": __version__,
        "now_et": now.isoformat(),
        "next_target": service.default_target().isoformat(),
        "models": registry.status(service.default_target()),
        "data": service.data_status(),
        "jobs": service.job_status(),
        "scheduler": scheduler.ENABLED,
        "data_credit": DATA_CREDIT,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    """Which model serves next: the ONNX TFT bundle (with its age) or the LightGBM trees."""
    return registry.status(service.default_target())


@app.post("/admin/models/reload")
def reload_models() -> dict[str, Any]:
    """Re-read the TFT bundle from the data volume (after a monthly upload)."""
    return registry.reload(force=True)


@app.get("/jobs")
def jobs() -> dict[str, Any]:
    return {
        "enabled": scheduler.ENABLED,
        "jobs": [
            {"name": j.name, "at_et": j.at.strftime("%H:%M"), "monthly": j.monthly}
            for j in scheduler.JOBS
        ],
        "runs": service.job_status(),
        "due": scheduler.due_jobs(),
    }


@app.post("/admin/jobs/{name}/run")
def run_job(name: str) -> dict[str, Any]:
    """Run a scheduler job now: forecast_all, ingest_score, isolf_refresh or retention."""
    if name not in scheduler.JOB_BY_NAME:
        raise HTTPException(404, f"unknown job {name!r}")
    return scheduler.run_job(name)


@app.post("/admin/ingest")
def admin_ingest(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    types: str | None = Query(None, description="comma-separated file types (default: all)"),
) -> dict[str, Any]:
    """Fetch and store a date range from the NYISO archive (synchronous; seeding, repairs)."""
    s, e = _date(start), _date(end)
    file_types = tuple(t.strip() for t in types.split(",")) if types else None
    try:
        return service.ingest_days(s, e, file_types or service.FILE_TYPES)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/admin/score")
def admin_score(zone: str | None = None) -> dict[str, Any]:
    """Score every forecast and schedule that actuals now cover (the 06:30 job's second half)."""
    try:
        return service.score_pending(service.resolve_zone(zone) if zone else None)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/admin/weather/refresh")
def admin_weather() -> dict[str, Any]:
    return {"refreshed": service.refresh_weather()}


# ─── zones ────────────────────────────────────────────────────────────────────────────
@app.get("/zones")
def zones() -> list[dict[str, Any]]:
    """Zone cards: history span, latest forecast, 7-day MAPE and imbalance $, alerts."""
    return service.zone_cards()


@app.get("/zones/default")
def default_zone() -> dict[str, str]:
    return {"zone": DEMO_DEFAULT_ZONE}


@app.get("/zones/{zone}")
def zone_card(zone: str) -> dict[str, Any]:
    name = _zone(zone)
    return next(c for c in service.zone_cards() if c["zone"] == name)


def _zone(zone: str) -> str:
    try:
        return service.resolve_zone(zone)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/zones/{zone}/forecasts")
def create_forecast(
    zone: str,
    target_date: str | None = Query(None, description="YYYY-MM-DD (default: tomorrow ET)"),
    model: str = Query("auto", pattern="^(auto|tft|lgbm)$"),
) -> dict[str, Any]:
    """Forecast a day as of its D-1 05:00 ET cutoff and store it.

    ``auto`` serves the ONNX TFT when the bundle is valid and falls back to the trees;
    ``lgbm`` retrains the trees on demand (the demo's retrain button).
    """
    name = _zone(zone)
    target = _date(target_date) if target_date else None
    try:
        return service.generate_forecast(name, target, model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/zones/{zone}/forecasts")
def list_forecasts(zone: str, limit: int = Query(400, ge=1, le=2000)) -> list[dict[str, Any]]:
    return service.list_forecasts(_zone(zone), limit)


@app.get("/zones/{zone}/forecasts/{target_date}")
def get_forecast(zone: str, target_date: str) -> dict[str, Any]:
    """Latest stored forecast for a day with the ISO overlay and actuals where they exist."""
    try:
        return service.get_forecast(_zone(zone), _date(target_date))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/zones/{zone}/scores")
def scores(zone: str, limit: int = Query(400, ge=1, le=2000)) -> list[dict[str, Any]]:
    """Persisted accuracy history: MAPE, band coverage and dollars per scored day."""
    return service.list_scores(_zone(zone), limit)


@app.get("/zones/{zone}/compare")
def compare(
    zone: str,
    start: str | None = Query(None, description="YYYY-MM-DD"),
    end: str | None = Query(None, description="YYYY-MM-DD"),
    granularity: str = Query("hour", pattern="^(slot|hour|day)$"),
) -> dict[str, Any]:
    """Stored forecasts vs actuals vs NYISO's pre-close forecast, with prices and $ per bucket."""
    s, e = _range(start, end)
    return service.compare(_zone(zone), s, e, granularity)


@app.get("/zones/{zone}/load")
def load(
    zone: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    granularity: str = Query("hour", pattern="^(slot|hour|day)$"),
) -> dict[str, Any]:
    s, e = _range(start, end)
    return service.load_view(_zone(zone), s, e, granularity)


@app.get("/zones/{zone}/prices")
def prices(
    zone: str, start: str | None = Query(None), end: str | None = Query(None)
) -> dict[str, Any]:
    """Hourly DA and RT LBMP with the spread that settles forecast errors."""
    s, e = _range(start, end)
    try:
        return service.prices_view(_zone(zone), s, e)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/zones/{zone}/isolf")
def isolf(zone: str, target_date: str | None = Query(None)) -> dict[str, Any]:
    """NYISO's own hourly forecast for a day: pre-close (file named D-1) and post-close (D)."""
    t = _date(target_date) if target_date else service.default_target()
    return service.isolf_view(_zone(zone), t)


@app.get("/zones/{zone}/alpha")
def alpha(zone: str, target_date: str | None = Query(None)) -> dict[str, Any]:
    """The newsvendor bid quantile α and the trailing spread costs behind it."""
    t = _date(target_date) if target_date else service.default_target()
    conn = service.db.connect()
    try:
        return service.alpha_stats(conn, _zone(zone), t)
    finally:
        conn.close()


# ─── DAM schedules ────────────────────────────────────────────────────────────────────
@app.post("/zones/{zone}/schedules")
def create_schedule(zone: str, body: ScheduleCreate) -> dict[str, Any]:
    """Store a DAM schedule (hourly MW bids) pinned to the latest forecast of that day."""
    try:
        return service.create_schedule(
            _zone(zone), body.target_date, body.hourly_mw, body.note, body.adjustments
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/zones/{zone}/schedules")
def list_schedules(zone: str, limit: int = Query(200, ge=1, le=2000)) -> list[dict[str, Any]]:
    return service.list_schedules(_zone(zone), limit)


@app.get("/zones/{zone}/schedules/{target_date}")
def get_schedule(zone: str, target_date: str) -> dict[str, Any]:
    try:
        return service.get_schedule(_zone(zone), _date(target_date))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/alerts")
def alerts(days: int = Query(14, ge=1, le=400)) -> list[dict[str, Any]]:
    return service.list_alerts(days)
