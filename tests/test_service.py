"""End-to-end service tests through the FastAPI TestClient on a throwaway store.

The archive client reads a synthetic NYISO archive (tests/synthetic.py) and "today" is
pinned, so the flows below run the real code path: fetch -> normalize -> SQLite ->
train (LightGBM, and the ONNX TFT when torch is installed) -> forecast -> DAM schedule
-> score in MAPE and dollars -> alerts -> cards. Tests build on each other in
definition order; the tree fits make this module take a minute or two.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db, nyiso, service
from app.nyiso import ArchiveClient
from app.serving import registry
from models import cutoff_for
from src.config import NYCA, NYISO_ARCHIVE_BASE, ZONES
from tests.synthetic import SyntheticNYISO

TODAY = date(2025, 9, 12)
FIRST = date(2025, 8, 1)
TARGET = date(2025, 9, 10)  # fully covered by actuals -> scorable right away
TOKEN = {"X-Admin-Token": "test-admin-token"}


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("archive")
    # August comes from the monthly zip, September from daily files, like the real archive
    SyntheticNYISO(FIRST, TODAY, seed=21).write_archive(root, monthly_zips=True)
    return root


@pytest.fixture(scope="module", autouse=True)
def wired(archive: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the service at the synthetic archive, a pinned today and an empty bundle dir."""

    def fetch(url: str) -> bytes | None:
        path = archive / url.removeprefix(NYISO_ARCHIVE_BASE + "/")
        return path.read_bytes() if path.exists() else None

    cache = tmp_path_factory.mktemp("cache")
    old = (service.CLIENT, service.TODAY, registry.bundle_dir, registry.max_age_days)
    service.CLIENT = ArchiveClient(cache_dir=cache, fetch=fetch, today=lambda: TODAY)
    service.TODAY = lambda: TODAY
    registry.bundle_dir = tmp_path_factory.mktemp("no_bundle")
    registry.reload(force=True)
    yield
    service.CLIENT, service.TODAY, registry.bundle_dir, registry.max_age_days = old
    registry.reload(force=True)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------- ingest
def test_ingest_populates_every_table(client: TestClient) -> None:
    r = client.post(
        "/admin/ingest",
        params={"start": FIRST.isoformat(), "end": (TODAY - timedelta(days=1)).isoformat()},
        headers=TOKEN,
    )
    assert r.status_code == 200, r.text
    summary = r.json()
    assert set(summary) == set(service.FILE_TABLES)
    assert summary["pal"]["days"] == 42 and summary["pal"]["missing"] == []
    assert summary["pal"]["rows"] == 42 * 96 * (len(ZONES) + 1)  # NYCA added
    assert summary["realtime_zone"]["rows"] == 42 * 96 * len(ZONES)

    health = client.get("/health").json()
    assert health["data"]["load_slots"]["last_ingested_day"] == "2025-09-11"
    assert health["models"]["tft"]["loaded"] is False
    assert health["next_target"] == "2025-09-13"

    # today's partial file is stored but not logged as complete
    partial = service.ingest_days(TODAY, TODAY, ("pal",))["pal"]
    assert partial["rows"] > 0
    conn = db.connect()
    try:
        assert service.last_ingested(conn, "pal") == TODAY - timedelta(days=1)
        n = conn.execute("SELECT COUNT(*) AS n FROM load_slots WHERE zone='N.Y.C.'").fetchone()
        assert n["n"] == 43 * 96
        # the store holds exactly what the normalizer + resampler produce for the day
        # (normalizer vs synthetic truth is test_nyiso's job)
        synth = SyntheticNYISO(FIRST, TODAY, seed=21)
        expected = nyiso.resample_slots(nyiso.normalize_pal(synth.pal_csv(TARGET)), "load_mw")
        expected = expected[expected["zone"] == "N.Y.C."].set_index("ts_utc")["load_mw"]
        got = db.read_frame(
            conn,
            "SELECT ts_utc, load_mw FROM load_slots WHERE zone='N.Y.C.' AND ts_utc BETWEEN ? AND ?",
            (db.ts_text(expected.index.min()), db.ts_text(expected.index.max())),
        ).set_index("ts_utc")["load_mw"]
        assert len(got) == 96 and np.allclose(got, expected.reindex(got.index))
    finally:
        conn.close()


def test_catch_up_is_incremental(client: TestClient) -> None:
    out = service.catch_up()
    assert "pal" not in out  # already at yesterday
    assert out["isolf"]["days"] >= 1  # today's file (named for today) is fetched fresh
    cards = client.get("/zones").json()
    assert [c["zone"] for c in cards] == [*ZONES, NYCA]
    assert all(c["history_days"] == 43 for c in cards)


# ---------------------------------------------------------------------------- access
def test_writes_need_the_admin_token(client: TestClient) -> None:
    assert client.post("/zones/nyc/forecasts").status_code == 401
    assert client.post("/zones/nyc/forecasts", headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.get("/zones").status_code == 200
    assert client.get("/zones/nowhere").status_code == 404
    assert client.get("/zones/default").json() == {"zone": "N.Y.C."}


def test_zone_names_resolve_loosely() -> None:
    assert service.resolve_zone("nyc") == "N.Y.C."
    assert service.resolve_zone("hud_vl") == "HUD VL"
    assert service.resolve_zone("Mhk Vl") == "MHK VL"
    assert service.resolve_zone("nyca") == NYCA
    with pytest.raises(LookupError):
        service.resolve_zone("PJM")


# ---------------------------------------------------------------------------- forecasts
@pytest.mark.slow
def test_tree_forecast_is_stored_and_idempotent(client: TestClient) -> None:
    r = client.post(
        "/zones/nyc/forecasts",
        params={"target_date": TARGET.isoformat(), "model": "lgbm"},
        headers=TOKEN,
    )
    assert r.status_code == 200, r.text
    fc = r.json()
    assert fc["model"].startswith("lgbm:nyiso-fv") and fc["new"] is True
    assert fc["cutoff_utc"] == cutoff_for(TARGET).isoformat()
    assert fc["history"][1] < fc["cutoff_utc"]  # the leakage guard held
    assert fc["band_scale"] == {"p10": 1.0, "p90": 1.0}  # no scored history yet
    assert fc["weather_features"] is False
    vals = pd.DataFrame(fc["values"])
    assert len(vals) == 96
    assert (vals["p10"] <= vals["predicted"]).all() and (vals["predicted"] <= vals["p90"]).all()
    truth = SyntheticNYISO(TARGET, TARGET, seed=21).load_5min()
    actual = truth[truth["zone"] == "N.Y.C."]["load_mw"].to_numpy().reshape(96, 3).mean(axis=1)
    assert np.mean(np.abs(vals["predicted"] - actual) / actual) < 0.05  # synthetic is easy

    again = client.post(
        "/zones/nyc/forecasts",
        params={"target_date": TARGET.isoformat(), "model": "lgbm"},
        headers=TOKEN,
    ).json()
    assert again["model_version"] == fc["model_version"] and again["new"] is False

    # auto without a bundle falls back to the trees and says why
    auto = client.post(
        "/zones/nyc/forecasts", params={"target_date": TARGET.isoformat()}, headers=TOKEN
    ).json()
    assert auto["model"].startswith("lgbm:") and "bundle missing" in auto["fallback_reason"]
    assert (
        client.post(
            "/zones/nyc/forecasts",
            params={"target_date": TARGET.isoformat(), "model": "tft"},
            headers=TOKEN,
        ).status_code
        == 400
    )

    stored = client.get(f"/zones/nyc/forecasts/{TARGET}").json()
    assert stored["model_version"] == fc["model_version"] and len(stored["values"]) == 96
    assert (
        stored["values"][0]["actual"] is not None and stored["values"][0]["isolf_pre"] is not None
    )
    listed = client.get("/zones/nyc/forecasts").json()
    assert listed[0]["target_date"] == TARGET.isoformat()


@pytest.mark.slow
def test_nyca_forecast_has_no_prices(client: TestClient) -> None:
    fc = client.post(
        "/zones/nyca/forecasts",
        params={"target_date": TARGET.isoformat(), "model": "lgbm"},
        headers=TOKEN,
    ).json()
    assert fc["alpha"] == 0.5 and len(fc["values"]) == 96
    assert client.get("/zones/nyca/prices").status_code == 400


# ---------------------------------------------------------------------------- scoring
@pytest.mark.slow
def test_scoring_in_mape_and_dollars(client: TestClient) -> None:
    r = client.post("/admin/score", headers=TOKEN)
    assert r.status_code == 200, r.text
    out = r.json()
    nyc = next(s for s in out["forecasts"] if s["zone"] == "N.Y.C.")
    assert nyc["coverage"] == 1.0 and nyc["mape_hour"] is not None and nyc["mape_slot"] is not None
    assert nyc["imbalance_usd"] is not None and nyc["da_cost_usd"] > 0
    assert nyc["isolf_mape_hour"] is not None and nyc["isolf_imbalance_usd"] is not None
    assert 0 <= nyc["band_coverage"] <= 100
    nyca = next(s for s in out["forecasts"] if s["zone"] == NYCA)
    assert nyca["imbalance_usd"] is None and nyca["mape_hour"] is not None

    hist = client.get("/zones/nyc/scores").json()
    assert hist[0]["target_date"] == TARGET.isoformat() and hist[0]["mape_hour"] == nyc["mape_hour"]

    # the on-the-fly compare view agrees with the persisted score
    cmp_ = client.get(
        "/zones/nyc/compare",
        params={"start": TARGET.isoformat(), "end": TARGET.isoformat(), "granularity": "hour"},
    ).json()
    assert len(cmp_["points"]) == 24 and cmp_["summary"]["days"] == 1
    assert abs(cmp_["summary"]["mape_hour"] - nyc["mape_hour"]) < 1e-3
    assert abs(cmp_["summary"]["imbalance_usd"] - nyc["imbalance_usd"]) < 0.05
    assert abs(cmp_["summary"]["isolf_mape_hour"] - nyc["isolf_mape_hour"]) < 1e-3
    point = cmp_["points"][12]
    assert point["actual"] and point["p_da"] is not None and point["imbalance_usd"] is not None
    daily = client.get(
        "/zones/nyc/compare",
        params={"start": TARGET.isoformat(), "end": TARGET.isoformat(), "granularity": "day"},
    ).json()
    assert len(daily["points"]) == 1
    assert abs(daily["points"][0]["imbalance_usd"] - nyc["imbalance_usd"]) < 0.05

    # re-scoring an already complete day is a no-op
    assert client.post("/admin/score", headers=TOKEN).json()["forecasts"] == []

    card = client.get("/zones/nyc").json()
    assert card["last_7d"]["days"] == 1
    assert abs(card["last_7d"]["mape_hour"] - nyc["mape_hour"]) < 0.01  # card rounds to 2 dp
    assert card["latest_forecast"]["target_date"] == TARGET.isoformat()


def test_views(client: TestClient) -> None:
    load = client.get(
        "/zones/nyc/load", params={"start": "2025-09-09", "end": "2025-09-10", "granularity": "day"}
    ).json()
    assert len(load["points"]) == 2
    prices = client.get(
        "/zones/nyc/prices", params={"start": "2025-09-10", "end": "2025-09-10"}
    ).json()
    assert len(prices["points"]) == 24 and prices["points"][0]["spread"] is not None
    iso = client.get("/zones/nyc/isolf", params={"target_date": TARGET.isoformat()}).json()
    assert len(iso["points"]) == 24
    assert iso["points"][0]["isolf_pre"] is not None and iso["points"][0]["isolf_post"] is not None
    alpha = client.get("/zones/nyc/alpha", params={"target_date": TARGET.isoformat()}).json()
    assert 0 < alpha["alpha"] < 1 and alpha["slots"] > 0
    assert (
        client.get(
            "/zones/nyc/compare", params={"start": "2025-09-11", "end": "2025-09-10"}
        ).status_code
        == 400
    )


# ---------------------------------------------------------------------------- schedules
@pytest.mark.slow
def test_schedule_flow(client: TestClient) -> None:
    fc = client.get(f"/zones/nyc/forecasts/{TARGET}").json()
    vals = pd.DataFrame(fc["values"])
    vals["hour"] = pd.to_datetime(vals["ts_utc"]).dt.floor("h")
    hourly = vals.groupby("hour", sort=True)["predicted"].mean()
    bids = (hourly * 1.05).tolist()
    bids[0] = float(hourly.iloc[0])  # hour 0 left at the model
    r = client.post(
        "/zones/nyc/schedules",
        json={
            "target_date": TARGET.isoformat(),
            "hourly_mw": bids,
            "note": "test",
            "adjustments": {"fills": ["+5%"], "manual_hours": [0]},
        },
        headers=TOKEN,
    )
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["model_version"] == fc["model_version"]
    assert abs(s["total_bid_mwh"] - sum(bids)) < 1e-2  # MW per hour == MWh per hour
    assert s["usd_at_risk"] is not None and s["usd_at_risk"] > 0

    g = client.get(f"/zones/nyc/schedules/{TARGET}").json()
    gv = pd.DataFrame(g["values"])
    assert len(gv) == 96 and len(g["hourly"]) == 24
    h3 = gv[gv["slot"] // 4 == 3]
    assert np.allclose(h3["bid_mw"], bids[3])  # flat MW within the hour
    assert np.allclose(gv[gv["slot"] // 4 == 0]["bid_mw"], bids[0])
    assert g["adjustments"]["manual_hours"] == [0] and g["score"] is None

    scored = client.post("/admin/score", headers=TOKEN).json()
    assert len(scored["schedules"]) == 1 and scored["schedules"][0]["mape_hour"] is not None
    g = client.get(f"/zones/nyc/schedules/{TARGET}").json()
    assert g["score"]["coverage"] == 1.0 and g["forecast_score"]["mape_hour"] is not None
    assert client.get("/zones/nyc/schedules").json()[0]["schedule_id"] == s["schedule_id"]

    # guards
    bad = client.post(
        "/zones/nyc/schedules",
        json={"target_date": TARGET.isoformat(), "hourly_mw": bids[:12]},
        headers=TOKEN,
    )
    assert bad.status_code == 400 and "24" in bad.json()["detail"]
    neg = bids.copy()
    neg[5] = -1.0
    assert (
        client.post(
            "/zones/nyc/schedules",
            json={"target_date": TARGET.isoformat(), "hourly_mw": neg},
            headers=TOKEN,
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/zones/nyc/schedules",
            json={"target_date": "2030-01-01", "hourly_mw": bids},
            headers=TOKEN,
        ).status_code
        == 404
    )
    assert client.get("/zones/nyc/schedules/2030-01-01").status_code == 404


# ---------------------------------------------------------------------------- calibration + alerts
@pytest.mark.slow
def test_conformal_band_and_alerts_after_history(client: TestClient) -> None:
    for k in range(8):
        target = date(2025, 9, 2) + timedelta(days=k)
        fc = service.generate_forecast("WEST", target, "lgbm")
        assert len(fc["values"]) == 96
    fresh = service.generate_forecast("WEST", date(2025, 9, 11), "lgbm")
    s10, s90 = fresh["band_scale"]["p10"], fresh["band_scale"]["p90"]
    assert (s10, s90) != (1.0, 1.0) and 0.5 <= s10 <= 2.0 and 0.5 <= s90 <= 2.0
    vals = pd.DataFrame(fresh["values"])
    assert (vals["p10"] <= vals["predicted"]).all() and (vals["predicted"] <= vals["p90"]).all()

    out = client.post("/admin/score", headers=TOKEN).json()
    west = [s for s in out["forecasts"] if s["zone"] == "WEST"]
    assert len(west) == 9
    # the synthetic series is easy: no MAPE alert; the $ alert needs 10 scored days
    assert out["alerts"] == [] and client.get("/alerts").json() == []
    conn = db.connect()
    try:
        forced = dict(west[-1], mape_hour=12.5)
        alerts = service._refresh_alerts(conn, forced)
        conn.commit()
    finally:
        conn.close()
    assert [a["kind"] for a in alerts] == ["mape"]
    assert client.get("/alerts").json()[0]["zone"] == "WEST"
    assert client.get("/zones/west").json()["alert"] is True


# ---------------------------------------------------------------------------- served TFT
@pytest.mark.slow
def test_served_tft_and_fallbacks(client: TestClient, tmp_path: Path) -> None:
    T = pytest.importorskip("models.tft")
    pytest.importorskip("onnxruntime")
    conn = db.connect()
    try:
        zones: dict[str, Any] = {}
        for z in ("N.Y.C.", "WEST"):
            hist = service.load_history(conn, z, cutoff_for(TODAY + timedelta(days=1)))
            zones[z] = T.prepare_zone(z, hist, None, TODAY + timedelta(days=1))
    finally:
        conn.close()
    c0 = cutoff_for(TARGET)
    data = {z: (zd, T.series_asof(zd, c0)[0]) for z, zd in zones.items()}
    f = T.TFTForecaster(
        list(zones),
        enc_days=2,
        hidden=8,
        heads=2,
        epochs=2,
        batch_size=16,
        val_days=3,
        device="cpu",
    )
    f.fit(data, [date(2025, 8, 5) + timedelta(days=k) for k in range(34)])
    bundle = tmp_path / "tft"
    f.export(bundle, fit_cutoff=c0.isoformat(), window_days=34)

    registry.bundle_dir = bundle
    status = client.post("/admin/models/reload", headers=TOKEN).json()
    assert status["tft"]["loaded"] is True and status["tft"]["zones"] == ["N.Y.C.", "WEST"]
    assert client.get("/models").json()["tft"]["stale"] is False

    fc = client.post(
        "/zones/nyc/forecasts", params={"target_date": TARGET.isoformat()}, headers=TOKEN
    ).json()
    assert fc["model"].startswith("tft-onnx:") and fc["fallback_reason"] is None
    vals = pd.DataFrame(fc["values"])
    assert len(vals) == 96 and (vals["p10"] <= vals["predicted"]).all()
    latest = client.get(f"/zones/nyc/forecasts/{TARGET}").json()
    assert latest["model"] == fc["model"]  # the newest version is served

    # a zone outside the bundle and a stale bundle both fall back to the trees
    capitl = client.post(
        "/zones/capitl/forecasts", params={"target_date": TARGET.isoformat()}, headers=TOKEN
    ).json()
    assert capitl["model"].startswith("lgbm:") and "not in the bundle" in capitl["fallback_reason"]
    registry.max_age_days = 0
    try:
        stale = client.post(
            "/zones/west/forecasts", params={"target_date": TODAY.isoformat()}, headers=TOKEN
        ).json()
        assert stale["model"].startswith("lgbm:") and "days old" in stale["fallback_reason"]
        assert client.get("/models").json()["tft"]["stale"] is True
    finally:
        registry.max_age_days = 60

    # a corrupted bundle is rejected on reload, the service keeps working
    (bundle / "tft.onnx").write_bytes((bundle / "tft.onnx").read_bytes() + b"\0")
    status = client.post("/admin/models/reload", headers=TOKEN).json()
    assert status["tft"]["loaded"] is False and "rejected" in status["tft"]["error"]


# ---------------------------------------------------------------------------- jobs + retention
def test_jobs_and_retention(client: TestClient) -> None:
    jobs = client.get("/jobs").json()
    assert {j["name"] for j in jobs["jobs"]} == {
        "forecast_all", "ingest_score", "isolf_refresh", "retention"
    }  # fmt: skip
    assert client.post("/admin/jobs/nope/run", headers=TOKEN).status_code == 404
    run = client.post("/admin/jobs/isolf_refresh/run", headers=TOKEN).json()
    assert run["status"] == "ok" and "isolf" in run["result"]
    assert any(r["name"] == "isolf_refresh" and r["last_status"] == "ok" for r in jobs_now(client))

    assert service.retention_boundary(date(2026, 9, 19), 24) == date(2024, 9, 1)
    assert service.retention_boundary(date(2026, 1, 15), 14) == date(2024, 11, 1)
    with pytest.raises(ValueError):
        service.prune(months=6, today=date(2026, 9, 19))
    conn = db.connect()
    try:
        before = conn.execute("SELECT COUNT(*) AS n FROM load_slots").fetchone()["n"]
        forecasts = conn.execute("SELECT COUNT(*) AS n FROM forecasts").fetchone()["n"]
    finally:
        conn.close()
    out = service.prune(months=14, today=date(2026, 11, 1))  # boundary 2025-09-01: August goes
    assert out["boundary"] == "2025-09-01" and out["load_slots"] == 31 * 96 * (len(ZONES) + 1)
    conn = db.connect()
    try:
        assert (
            conn.execute("SELECT COUNT(*) AS n FROM load_slots").fetchone()["n"]
            == before - out["load_slots"]
        )
        assert conn.execute("SELECT COUNT(*) AS n FROM forecasts").fetchone()["n"] == forecasts
        assert (
            conn.execute("SELECT MIN(ts_utc) AS lo FROM load_slots").fetchone()["lo"]
            == "2025-09-01 04:00:00"
        )
    finally:
        conn.close()
    assert out["cache_files"] > 0
    assert not any(
        p.name.startswith("202508")
        for p in service.get_client().cache_dir.rglob("*")
        if p.is_file()
    )


def jobs_now(client: TestClient) -> list[dict[str, Any]]:
    return client.get("/jobs").json()["runs"]
