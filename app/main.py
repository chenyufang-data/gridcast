"""FastAPI entry point for the NYISO day-ahead forecasting service.

Run locally:   uvicorn app.main:app --reload
Docs UI:       http://127.0.0.1:8000/docs

Phase 0 exposes the health endpoint only. Phase 3 ports the service surface (zones,
forecasts, DAM schedules, scores, prices, isolf) and adds the X-Admin-Token middleware
on every write endpoint.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app import __version__
from app.log import configure_logging

configure_logging()
log = logging.getLogger(__name__)

app = FastAPI(
    title="gridcast",
    description=(
        "Day-ahead zonal load forecasts for the NYISO market: 15-min forecasts with "
        "P10-P90 bands, a cost-aware DAM bid, and settlement in dollars against "
        "real-time prices. Data: NYISO public MIS archive, fetched at runtime."
    ),
    version=__version__,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
