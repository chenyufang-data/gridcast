"""/health: liveness plus the operator's view (served model, data coverage, job runs)."""

from __future__ import annotations

from pathlib import Path

import pytest
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


# ---------------------------------------------------------------------------- probes
def test_livez_and_readyz_on_a_fresh_store() -> None:
    with TestClient(app) as client:  # the lifespan creates the schema
        assert client.get("/livez").json() == {"status": "ok"}
        ready = client.get("/readyz")
    assert ready.status_code == 200, ready.text
    body = ready.json()
    assert body["ready"] is True and body["checks"] == {"database": "ok", "model": "ok"}
    assert body["pinned_version"] is None


def test_readyz_is_not_blocked_by_a_writer() -> None:
    import sqlite3
    import time

    from app import db

    with TestClient(app) as client:
        writer = sqlite3.connect(db.DB_PATH, timeout=1)
        try:
            writer.execute("BEGIN IMMEDIATE")  # holds the write lock, as a long ingest does
            writer.execute(
                "INSERT INTO jobs (name, last_run_at, last_status) VALUES ('t', 'x', 'ok')"
            )
            t0 = time.perf_counter()
            ready = client.get("/readyz")
            elapsed = time.perf_counter() - t0
        finally:
            writer.rollback()
            writer.close()
    assert ready.status_code == 200 and elapsed < 1.0


def test_readyz_reports_a_missing_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app import db

    with TestClient(app) as client:
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "gone.db")
        ready = client.get("/readyz")
    assert ready.status_code == 503
    assert "does not exist" in ready.json()["checks"]["database"]
    assert not (tmp_path / "gone.db").exists()  # the probe never creates the file


def test_readyz_requires_the_tft_when_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.serving import registry

    monkeypatch.setattr(registry, "require_tft", True)
    with TestClient(app) as client:
        ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["checks"]["model"].startswith("TFT required but not served")


def test_registry_rejects_a_bundle_that_is_not_the_pinned_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from app import serving

    for name in ("tft.onnx", "tft.json"):
        (tmp_path / name).write_text("x")
    fake = SimpleNamespace(
        version="tft-onnx:aaaaaaaaaaaa:2026-09-19", meta={}, zones=["N.Y.C."], zone_id={"N.Y.C.": 0}
    )
    monkeypatch.setattr(serving.OnnxTFT, "load", classmethod(lambda cls, path: fake))
    reg = serving.ModelRegistry(bundle_dir=tmp_path)
    reg.require_tft = True
    reg.expected_version = "tft-onnx:aaaaaaaaaaaa:2026-09-19"
    reg.reload(force=True)
    assert reg.tft is fake and reg.ready() == (True, None)
    assert reg.status()["tft"]["pinned_version"] == reg.expected_version

    reg.expected_version = "tft-onnx:bbbbbbbbbbbb:2026-10-15"  # a different model was deployed
    reg.reload(force=True)
    assert reg.tft is None and "is not the pinned" in (reg.error or "")
    ok, why = reg.ready()
    assert ok is False and "is not the pinned" in (why or "")
    model, reason = reg.tft_for("N.Y.C.", serving.date(2026, 10, 1))
    assert model is None and "is not the pinned" in (reason or "")  # the trees serve instead


# ---------------------------------------------------------------------------- job runs
def test_job_endpoint_answers_409_while_running_and_500_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import time as dtime

    from app import scheduler

    token = {"X-Admin-Token": "test-admin-token"}
    with TestClient(app) as client:
        lock = scheduler._RUNNING["retention"]
        assert lock.acquire(blocking=False)
        try:
            busy = client.post("/admin/jobs/retention/run", headers=token)
        finally:
            lock.release()
        assert busy.status_code == 409 and "already running" in busy.json()["detail"]

        def boom() -> None:
            raise RuntimeError("archive unreachable")

        monkeypatch.setitem(
            scheduler.JOB_BY_NAME, "retention", scheduler.Job("retention", dtime(7, 0), boom, True)
        )
        failed = client.post("/admin/jobs/retention/run", headers=token)
        assert failed.status_code == 500
        assert (
            failed.json()["status"] == "error" and "archive unreachable" in failed.json()["error"]
        )
        assert client.post("/admin/jobs/nope/run", headers=token).status_code == 404
