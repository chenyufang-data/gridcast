from fastapi.testclient import TestClient

from app import __version__
from app.main import app


def test_health_reports_version() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}
