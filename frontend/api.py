"""HTTP client of the backend API: the only way the frontend touches data.

The frontend imports nothing from ``app/``; every number it shows comes from these
calls. Writes carry the admin token (``ADMIN_TOKEN``), which lives with the frontend
process, never in the browser. Tests swap the client with :func:`set_client`.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:  # secrets live in .env locally; in docker they arrive as real env vars
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


class BackendDown(Exception):
    """The backend did not answer at all."""


class ApiError(Exception):
    """The backend answered with an error status; ``detail`` is its message."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _iso(d: date | str | None) -> str | None:
    if d is None:
        return None
    return d if isinstance(d, str) else d.isoformat()


class Api:
    def __init__(self, base_url: str = API_BASE, token: str = ADMIN_TOKEN, timeout: float = 60):
        self.base_url = base_url
        self.token = token
        self.timeout = timeout

    # --- transport --------------------------------------------------------------------
    def request(self, method: str, path: str, timeout: float | None = None, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}) or {})
        if method != "GET" and self.token:
            headers["X-Admin-Token"] = self.token
        try:
            resp = requests.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=timeout or self.timeout,
                **kwargs,
            )
        except requests.ConnectionError as exc:
            raise BackendDown(str(exc)) from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ApiError(resp.status_code, str(detail))
        return resp.json()

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, params: dict[str, Any] | None = None, json: Any = None,
             timeout: float | None = None) -> Any:  # fmt: skip
        return self.request(
            "POST",
            path,
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=json,
            timeout=timeout,
        )

    # --- typed calls ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self.get("/health")

    def models(self) -> dict[str, Any]:
        return self.get("/models")

    def zones(self) -> list[dict[str, Any]]:
        return self.get("/zones")

    def zone(self, zone: str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}")

    def forecasts(self, zone: str, limit: int = 400) -> list[dict[str, Any]]:
        return self.get(f"/zones/{zone}/forecasts", limit=limit)

    def forecast(self, zone: str, target: date | str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}/forecasts/{_iso(target)}")

    def create_forecast(
        self, zone: str, target: date | str | None = None, model: str = "auto"
    ) -> dict[str, Any]:
        return self.post(
            f"/zones/{zone}/forecasts",
            params={"target_date": _iso(target), "model": model},
            timeout=600,
        )

    def scores(self, zone: str, limit: int = 400) -> list[dict[str, Any]]:
        return self.get(f"/zones/{zone}/scores", limit=limit)

    def compare(
        self, zone: str, start: date | str, end: date | str, granularity: str = "hour"
    ) -> dict[str, Any]:
        return self.get(
            f"/zones/{zone}/compare", start=_iso(start), end=_iso(end), granularity=granularity
        )

    def load(
        self, zone: str, start: date | str, end: date | str, granularity: str = "hour"
    ) -> dict[str, Any]:
        return self.get(
            f"/zones/{zone}/load", start=_iso(start), end=_iso(end), granularity=granularity
        )

    def prices(self, zone: str, start: date | str, end: date | str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}/prices", start=_iso(start), end=_iso(end))

    def isolf(self, zone: str, target: date | str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}/isolf", target_date=_iso(target))

    def alpha(self, zone: str, target: date | str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}/alpha", target_date=_iso(target))

    def schedules(self, zone: str) -> list[dict[str, Any]]:
        return self.get(f"/zones/{zone}/schedules")

    def schedule(self, zone: str, target: date | str) -> dict[str, Any]:
        return self.get(f"/zones/{zone}/schedules/{_iso(target)}")

    def create_schedule(
        self,
        zone: str,
        target: date | str,
        hourly_mw: list[float],
        note: str | None = None,
        adjustments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.post(
            f"/zones/{zone}/schedules",
            json={
                "target_date": _iso(target),
                "hourly_mw": hourly_mw,
                "note": note,
                "adjustments": adjustments,
            },
        )

    def alerts(self, days: int = 14) -> list[dict[str, Any]]:
        return self.get("/alerts", days=days)


_client: Api = Api()


def client() -> Api:
    return _client


def set_client(api: Api) -> None:
    """Replace the shared client (tests inject a fake backend)."""
    global _client
    _client = api
