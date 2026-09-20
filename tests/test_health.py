"""/health: liveness plus the operator's view (served model, data coverage, job runs)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.main import app


def test_health_reports_version_models_and_data() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["models"]["tft"]["loaded"] is False  # tests never see a real bundle
    assert body["models"]["lgbm"]["identity"].startswith("lgbm:")
    assert set(body["data"]) >= {"load_slots", "rt_slots", "da_hourly", "isolf", "forecasts"}
    assert body["scheduler"] is False and "NYISO" in body["data_credit"]
