"""Core service logic: ingest the NYISO archive, forecast, settle, score, alert, prune.

Every forecast is produced by the same code as the offline backtest (``models`` for the
features and estimators, ``src.settlement`` for α and dollars) as of the bid cutoff
D-1 05:00 ET, and stored immutably with a ``model_version`` so "forecast vs actual"
always shows what the model said at the time. The served model is the ONNX TFT when the
bundle is valid (:mod:`app.serving`), the trees otherwise; the row records which one.

Module-level ``CLIENT`` / ``TODAY`` are the seams tests use: an archive client backed by
the synthetic archive, and a fixed "today".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app import db, weather
from app.nyiso import ArchiveClient, NotAvailable, add_nyca, normalize, resample_slots, today_et
from app.serving import (
    LGBM_IDENTITY,
    MODEL_THREADS,
    TREE_DECAY_FLOOR,
    TREE_HALF_LIFE,
    TREE_QUANTILES,
    TREE_WINDOW_DAYS,
    registry,
)
from models import SLOT, cutoff_for, forecast_day, local_midnight_utc
from models.tft_data import prepare_zone, series_asof
from src.config import FILE_TYPES, MARKET_TZ, NYCA
from src.metrics import mape
from src.settlement import MWH_PER_MW_SLOT, estimate_alpha, settle, slot_prices

log = logging.getLogger(__name__)

# --- settings -------------------------------------------------------------------------
HISTORY_DAYS = TREE_WINDOW_DAYS + 35  # window + the longest lag (21 d) + margin
ALPHA_WINDOW_DAYS = 30
CONFORMAL_WINDOW_DAYS = 30  # trailing scored days ending D-2, as in quantile_calibration.py
CONFORMAL_MIN_DAYS = 7
CONFORMAL_CLIP = (0.5, 2.0)
SCORE_WINDOW_DAYS = 14  # re-score partially covered days this far back
SCORE_MIN_COVERAGE = 0.5  # partial days below this are stored but never alert
SCORE_ALERT_MAPE = 10.0  # hourly MAPE (%) that raises an alert
IMBALANCE_ALERT_MIN_DAYS = 10  # trailing scored days needed for the $ alert
IMBALANCE_ALERT_QUANTILE = 0.9
CATCHUP_MAX_DAYS = 45  # a scheduler tick never backfills further than this (seed does)
DAILY_FILE_DAYS = 11  # daily files live about this long: retry missing days within it
RETENTION_MIN_MONTHS = 14  # the trees' window + lags must always fit
RETENTION_MONTHS = max(int(os.environ.get("RETENTION_MONTHS") or 24), RETENTION_MIN_MONTHS)
WEATHER_REFRESH = (os.environ.get("WEATHER_REFRESH") or "1") not in ("0", "false", "no")
WEATHER_DAYS_BACK = 10
WEATHER_DAYS_AHEAD = 7

FILE_TABLES = {
    "pal": "load_slots",
    "realtime_zone": "rt_slots",
    "damlbmp_zone": "da_hourly",
    "rtlbmp_zone": "rt_hourly",
    "isolf": "isolf",
}
MODELS = ("auto", "tft", "lgbm")

CLIENT: ArchiveClient | None = None
TODAY: Callable[[], date] = today_et
_train_lock = threading.Lock()


def get_client() -> ArchiveClient:
    return CLIENT if CLIENT is not None else ArchiveClient(today=TODAY)


# --- small helpers ----------------------------------------------------------------------
def _num(x: Any, digits: int | None = None) -> float | None:
    """JSON-safe float: None for NaN/None."""
    if x is None:
        return None
    v = float(x)
    if not np.isfinite(v):
        return None
    return round(v, digits) if digits is not None else v


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _key(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def resolve_zone(name: str) -> str:
    """Canonical zone name from any spelling (``nyc``, ``N.Y.C.``, ``hud vl``, ``HUD_VL``)."""
    conn = db.connect()
    try:
        names = db.zone_names(conn)
    finally:
        conn.close()
    wanted = _key(name)
    for z in names:
        if _key(z) == wanted:
            return z
    raise LookupError(f"unknown zone {name!r}; known: {', '.join(names)}")


def default_target() -> date:
    """Tomorrow in ET: the day whose bid is open (or just closed) today."""
    return TODAY() + timedelta(days=1)


def _local_label(ts: pd.Timestamp) -> str:
    return ts.tz_convert(MARKET_TZ).strftime("%Y-%m-%d %H:%M")


def _local_day(stored: str) -> date:
    """Local ET day of a stored UTC text timestamp."""
    return pd.Timestamp(stored, tz="UTC").tz_convert(MARKET_TZ).date()


# ─── ingestion ────────────────────────────────────────────────────────────────────────
def _clip_day(slots: pd.DataFrame, day: date) -> pd.DataFrame:
    """Keep the slots of the local day only: an off-schedule stamp near midnight spills a
    sliver into the next day's first slot, which that day's own file covers in full."""
    lo, hi = local_midnight_utc(day), local_midnight_utc(day + timedelta(days=1))
    return slots[(slots["ts_utc"] >= lo) & (slots["ts_utc"] < hi)]


def _store_day(conn: sqlite3.Connection, file_type: str, frame: pd.DataFrame, day: date) -> int:
    if file_type == "pal":
        slots = _clip_day(add_nyca(resample_slots(frame, "load_mw"), "load_mw"), day)
        return db.upsert_frame(conn, "load_slots", slots[["zone", "ts_utc", "load_mw", "coverage"]])
    if file_type == "realtime_zone":
        slots = _clip_day(resample_slots(frame, "p_rt"), day)
        return db.upsert_frame(conn, "rt_slots", slots[["zone", "ts_utc", "p_rt", "coverage"]])
    if file_type == "damlbmp_zone":
        return db.upsert_frame(conn, "da_hourly", frame[["zone", "ts_utc", "p_da"]])
    if file_type == "rtlbmp_zone":
        return db.upsert_frame(conn, "rt_hourly", frame[["zone", "ts_utc", "p_rt_hourly"]])
    if file_type == "isolf":
        return db.upsert_frame(conn, "isolf", frame[["issued", "zone", "ts_utc", "isolf_mw"]])
    raise ValueError(f"unknown file type {file_type!r}")


def ingest_days(
    start: date,
    end: date,
    file_types: Sequence[str] = FILE_TYPES,
    client: ArchiveClient | None = None,
) -> dict[str, Any]:
    """Fetch, normalize and upsert every day of `start`..`end` for the given file types.

    Complete days (before today, ET) are recorded in ``ingest_log`` so the next catch-up
    starts after them; today's partial files and tomorrow's ``isolf`` / ``damlbmp`` are
    stored but not logged, so they are fetched again when final. Never raises on a
    missing or failing day: the summary lists them and the next run retries.
    """
    client = client or get_client()
    today = client.today()
    summary: dict[str, Any] = {}
    conn = db.connect()
    try:
        for ft in file_types:
            if ft not in FILE_TABLES:
                raise ValueError(f"unknown file type {ft!r}")
            s: dict[str, Any] = {"days": 0, "rows": 0, "missing": [], "errors": []}
            for day in _days(start, end):
                try:
                    text = client.day_csv(ft, day)
                except NotAvailable:
                    s["missing"].append(day.isoformat())
                    if day < today:
                        db.upsert(
                            conn,
                            "ingest_log",
                            ("file_type", "day", "status", "rows", "fetched_at"),
                            [(ft, day.isoformat(), "missing", 0, db.now_text())],
                        )
                        conn.commit()
                    continue
                except Exception as exc:  # network / archive trouble: retry next run
                    log.error("%s %s: fetch failed: %s", ft, day, exc)
                    s["errors"].append(f"{day.isoformat()}: {exc}")
                    continue
                try:
                    n = _store_day(conn, ft, normalize(ft, text, day), day)
                except Exception as exc:
                    log.exception("%s %s: normalize/store failed", ft, day)
                    s["errors"].append(f"{day.isoformat()}: {exc}")
                    conn.rollback()
                    continue
                if day < today:
                    db.upsert(
                        conn,
                        "ingest_log",
                        ("file_type", "day", "status", "rows", "fetched_at"),
                        [(ft, day.isoformat(), "ok", n, db.now_text())],
                    )
                conn.commit()
                s["days"] += 1
                s["rows"] += n
            summary[ft] = s
            log.info(
                "ingest %s %s..%s: %d days, %d rows, %d missing, %d errors",
                ft,
                start,
                end,
                s["days"],
                s["rows"],
                len(s["missing"]),
                len(s["errors"]),
            )
    finally:
        conn.close()
    return summary


def last_ingested(conn: sqlite3.Connection, file_type: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(day) AS d FROM ingest_log WHERE file_type=? AND status='ok'", (file_type,)
    ).fetchone()
    return date.fromisoformat(row["d"]) if row and row["d"] else None


def catch_up(client: ArchiveClient | None = None) -> dict[str, Any]:
    """Bring every table up to yesterday (ET); retry recently missing days; try tomorrow's isolf.

    A fresh store starts ``CATCHUP_MAX_DAYS`` back; ``deploy/seed.py`` does the long
    backfill. Days the archive still lacked are retried while their daily file can
    still appear, then left to the monthly zip on the next pass.
    """
    client = client or get_client()
    today = client.today()
    yesterday = today - timedelta(days=1)
    out: dict[str, Any] = {}
    conn = db.connect()
    try:
        plan: dict[str, tuple[date, date]] = {}
        retry: dict[str, list[date]] = {}
        for ft in FILE_TYPES:
            last = last_ingested(conn, ft)
            start = (
                max(last + timedelta(days=1), today - timedelta(days=CATCHUP_MAX_DAYS))
                if last
                else today - timedelta(days=CATCHUP_MAX_DAYS)
            )
            end = today + timedelta(days=1) if ft in ("isolf", "damlbmp_zone") else yesterday
            plan[ft] = (start, end)
            rows = conn.execute(
                "SELECT day FROM ingest_log WHERE file_type=? AND status='missing' AND day>=?",
                (ft, (today - timedelta(days=DAILY_FILE_DAYS)).isoformat()),
            ).fetchall()
            retry[ft] = [date.fromisoformat(r["day"]) for r in rows]
    finally:
        conn.close()
    for ft, (start, end) in plan.items():
        if start <= end:
            out[ft] = ingest_days(start, end, (ft,), client)[ft]
        for day in retry[ft]:
            if day < start:
                out.setdefault(f"{ft}:retry", []).append(ingest_days(day, day, (ft,), client)[ft])
    return out


# ─── weather ─────────────────────────────────────────────────────────────────────────
_weather_cache: dict[str, Any] = {"sig": None, "daily": None, "hourly": None}


def _weather_signature() -> str:
    parts = []
    for p in (weather.WEATHER_PATH, weather.WEATHER_HOURLY_PATH):
        parts.append(f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() else "none")
    return "|".join(parts)


def weather_frames() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Both weather stores, re-read when the files change; ``sig`` identifies the version."""
    sig = _weather_signature()
    if sig != _weather_cache["sig"]:
        _weather_cache["daily"] = weather.load_weather()
        _weather_cache["hourly"] = weather.load_weather_hourly()
        _weather_cache["sig"] = sig
    return _weather_cache["daily"], _weather_cache["hourly"], sig


def weather_features(zone: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """(daily features, hourly features incl. the extra variables, signature) for `zone`."""
    daily, hourly, sig = weather_frames()
    d = weather.features_for(daily, zone)
    h = weather.hourly_features_for(hourly, zone, extra=True)
    stamp = "none"
    if h is not None and not h.empty:
        stamp = f"{len(h)}:{h['hour_utc'].max():%Y-%m-%dT%H}"
    return d, h, stamp


def refresh_weather(today: date | None = None) -> bool:
    """Incremental Open-Meteo update (trailing days + a week ahead); off with WEATHER_REFRESH=0."""
    if not WEATHER_REFRESH:
        return False
    today = today or TODAY()
    return weather.update(
        today - timedelta(days=WEATHER_DAYS_BACK), today + timedelta(days=WEATHER_DAYS_AHEAD)
    )


# ─── data access ─────────────────────────────────────────────────────────────────────
def load_history(
    conn: sqlite3.Connection, zone: str, cutoff: pd.Timestamp, days: int = HISTORY_DAYS
) -> pd.DataFrame:
    """Slots of `zone` ending at or before `cutoff`, at most `days` back: the model input."""
    start = cutoff - pd.Timedelta(days=days)
    hist = db.read_frame(
        conn,
        "SELECT ts_utc, load_mw, coverage FROM load_slots WHERE zone=? AND ts_utc>=? AND ts_utc<? "
        "ORDER BY ts_utc",
        (zone, db.ts_text(start), db.ts_text(cutoff)),
    )
    return hist[hist["ts_utc"] + SLOT <= cutoff].reset_index(drop=True)


def slot_prices_for(
    conn: sqlite3.Connection, zone: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """``ts_utc, zone, p_rt, p_da, spread`` for `zone` in [start, end] (empty for NYCA)."""
    if zone == NYCA:
        return pd.DataFrame(columns=["ts_utc", "zone", "p_rt", "p_da", "spread"])
    rt = db.read_frame(
        conn,
        "SELECT ts_utc, zone, p_rt FROM rt_slots WHERE zone=? AND ts_utc BETWEEN ? AND ?",
        (zone, db.ts_text(start), db.ts_text(end)),
    )
    da = db.read_frame(
        conn,
        "SELECT ts_utc, zone, p_da FROM da_hourly WHERE zone=? AND ts_utc BETWEEN ? AND ?",
        (zone, db.ts_text(start.floor("h")), db.ts_text(end)),
    )
    if rt.empty or da.empty:
        return pd.DataFrame(columns=["ts_utc", "zone", "p_rt", "p_da", "spread"])
    return slot_prices(rt, da)


def alpha_stats(conn: sqlite3.Connection, zone: str, target: date) -> dict[str, Any]:
    """Newsvendor α and its ingredients for `zone` as of the cutoff of `target`."""
    cutoff = cutoff_for(target)
    start = cutoff - pd.Timedelta(days=ALPHA_WINDOW_DAYS)
    prices = slot_prices_for(conn, zone, start, cutoff)
    p = prices[prices["ts_utc"] + SLOT <= cutoff] if not prices.empty else prices
    spread = p["spread"].dropna().to_numpy() if not p.empty else np.array([])
    c_under = float(np.maximum(spread, 0).mean()) if len(spread) else None
    c_over = float(np.maximum(-spread, 0).mean()) if len(spread) else None
    alpha = estimate_alpha(prices, zone, target, ALPHA_WINDOW_DAYS) if not p.empty else 0.5
    return {
        "zone": zone,
        "target_date": target.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "window_days": ALPHA_WINDOW_DAYS,
        "slots": int(len(spread)),
        "alpha": round(float(alpha), 4),
        "c_under_usd_per_mwh": _num(c_under, 3),
        "c_over_usd_per_mwh": _num(c_over, 3),
        "mean_abs_spread_usd_per_mwh": _num(np.abs(spread).mean() if len(spread) else None, 3),
    }


def alpha_for(conn: sqlite3.Connection, zone: str, target: date) -> float:
    if zone == NYCA:
        return 0.5  # no zonal price for the statewide total
    return float(alpha_stats(conn, zone, target)["alpha"])


# ─── forecasting ─────────────────────────────────────────────────────────────────────
def conformal_scales(
    conn: sqlite3.Connection, zone: str, family: str, target: date
) -> tuple[float, float]:
    """Split-conformal band scales from the trailing scored days of the same model family.

    For day D: the 0.1-quantile of actual/P10 and the 0.9-quantile of actual/P90 over the
    latest forecasts of the ``CONFORMAL_WINDOW_DAYS`` days ending D-2 (1.0 with fewer
    than ``CONFORMAL_MIN_DAYS`` days), clipped to ``CONFORMAL_CLIP``.
    """
    lo = (target - timedelta(days=CONFORMAL_WINDOW_DAYS + 1)).isoformat()
    hi = (target - timedelta(days=2)).isoformat()
    like = f"{family}%"
    rows = db.read_frame(
        conn,
        """
        SELECT f.target_date, v.ts_utc, v.p10_raw, v.p90_raw, l.load_mw
        FROM forecasts f
        JOIN forecast_values v ON v.forecast_id = f.id
        JOIN load_slots l ON l.zone = f.zone AND l.ts_utc = v.ts_utc
        WHERE f.zone = ? AND f.model LIKE ? AND f.target_date BETWEEN ? AND ?
          AND f.id = (
            SELECT f2.id FROM forecasts f2
            WHERE f2.zone = f.zone AND f2.target_date = f.target_date AND f2.model LIKE ?
            ORDER BY f2.created_at DESC, f2.id DESC LIMIT 1
          )
        """,
        (zone, like, lo, hi, like),
    )
    if rows.empty or rows["target_date"].nunique() < CONFORMAL_MIN_DAYS:
        return 1.0, 1.0
    r10 = (rows["load_mw"] / rows["p10_raw"]).replace([np.inf, -np.inf], np.nan).dropna()
    r90 = (rows["load_mw"] / rows["p90_raw"]).replace([np.inf, -np.inf], np.nan).dropna()
    if r10.empty or r90.empty:
        return 1.0, 1.0
    s10 = float(np.clip(np.quantile(r10, 0.1), *CONFORMAL_CLIP))
    s90 = float(np.clip(np.quantile(r90, 0.9), *CONFORMAL_CLIP))
    return s10, s90


def _tft_rows(
    tft: Any,
    zone: str,
    history: pd.DataFrame,
    hourly_feats: pd.DataFrame | None,
    target: date,
    cutoff: pd.Timestamp,
    alpha: float,
) -> pd.DataFrame:
    zd = prepare_zone(zone, history, hourly_feats, target)
    y_asof, _ = series_asof(zd, cutoff)
    return tft.forecast(zd, y_asof, target, alpha)


def _tree_rows(
    history: pd.DataFrame,
    target: date,
    daily_feats: pd.DataFrame | None,
    hourly_feats: pd.DataFrame | None,
    alpha: float,
) -> pd.DataFrame:
    return forecast_day(
        history,
        target,
        weather=daily_feats,
        weather_hourly=hourly_feats,
        quantiles=TREE_QUANTILES,
        alpha=alpha,
        window_days=TREE_WINDOW_DAYS,
        model_overrides={
            "n_jobs": MODEL_THREADS,
            "decay_half_life": TREE_HALF_LIFE,
            "decay_floor": TREE_DECAY_FLOOR,
        },
    )


def generate_forecast(zone: str, target: date | None = None, model: str = "auto") -> dict[str, Any]:
    """Forecast `target` for `zone` as of its cutoff and store it (idempotent per version).

    `model`: ``auto`` (the TFT bundle when valid, else the trees), ``tft`` (fail instead
    of falling back) or ``lgbm`` (always the trees, the demo's retrain button).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    zone = resolve_zone(zone)
    target = target or default_target()
    cutoff = cutoff_for(target)
    with _train_lock:
        conn = db.connect()
        try:
            history = load_history(conn, zone, cutoff)
            if history.empty:
                raise ValueError(f"no load data for {zone} before the cutoff {cutoff}")
            alpha = alpha_for(conn, zone, target)
            daily_w, hourly_w, weather_sig = weather_features(zone)

            rows: pd.DataFrame | None = None
            identity = LGBM_IDENTITY
            fallback_reason: str | None = None
            if model in ("auto", "tft"):
                tft, why = registry.tft_for(zone, target)
                if tft is not None:
                    try:
                        rows = _tft_rows(tft, zone, history, hourly_w, target, cutoff, alpha)
                        identity = tft.version
                    except ValueError as exc:
                        why = f"TFT failed: {exc}"
                if rows is None:
                    if model == "tft":
                        raise ValueError(f"TFT unavailable for {zone} {target}: {why}")
                    fallback_reason = why
                    log.warning("%s %s: %s; using the trees", zone, target, why)
            if rows is None:
                rows = _tree_rows(history, target, daily_w, hourly_w, alpha)
            family = "tft" if identity.startswith("tft") else "lgbm"
            s10, s90 = conformal_scales(conn, zone, family, target)
            rows = rows.copy()
            rows["p10_raw"], rows["p90_raw"] = rows["p10"], rows["p90"]
            rows["p10"] = np.minimum(rows["p10_raw"] * s10, rows["pred"])
            rows["p90"] = np.maximum(rows["p90_raw"] * s90, rows["pred"])

            h0, h1 = history["ts_utc"].min(), history["ts_utc"].max()
            key = "|".join(
                [
                    zone,
                    target.isoformat(),
                    identity,
                    db.ts_text(h0),
                    db.ts_text(h1),
                    str(len(history)),
                    f"{history['load_mw'].sum():.3f}",
                    f"alpha={alpha:.4f}",
                    f"band={s10:.4f},{s90:.4f}",
                    f"w={weather_sig}",
                    f"trees={TREE_WINDOW_DAYS},{TREE_HALF_LIFE},{TREE_DECAY_FLOOR}",
                ]
            )
            version = hashlib.sha1(key.encode()).hexdigest()[:12]
            existing = conn.execute(
                "SELECT id, created_at, requested FROM forecasts "
                "WHERE zone=? AND target_date=? AND model_version=?",
                (zone, target.isoformat(), version),
            ).fetchone()
            if existing:
                fid, created_at, is_new = existing["id"], existing["created_at"], False
                if model == "auto" and existing["requested"] != "auto":
                    # the served model reproduced a manual version: it is the day's forecast
                    conn.execute("UPDATE forecasts SET requested='auto' WHERE id=?", (fid,))
                    conn.commit()
            else:
                created_at, is_new = db.now_text(), True
                cur = conn.execute(
                    "INSERT INTO forecasts (zone, target_date, model, model_version, requested, "
                    "cutoff_utc, created_at, alpha, band_scale_p10, band_scale_p90, "
                    "history_start, history_end, weather) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        zone,
                        target.isoformat(),
                        identity,
                        version,
                        model,
                        db.ts_text(cutoff),
                        created_at,
                        float(alpha),
                        s10,
                        s90,
                        db.ts_text(h0),
                        db.ts_text(h1),
                        int(hourly_w is not None),
                    ),
                )
                fid = cur.lastrowid
                conn.executemany(
                    "INSERT INTO forecast_values (forecast_id, slot, ts_utc, predicted, p10, p90, "
                    "p10_raw, p90_raw, p_alpha) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            fid,
                            int(r.slot),
                            db.ts_text(r.ts_utc),
                            float(r.pred),
                            float(r.p10),
                            float(r.p90),
                            float(r.p10_raw),
                            float(r.p90_raw),
                            float(r.p_alpha),
                        )
                        for r in rows.itertuples()
                    ],
                )
                conn.commit()
            primary = primary_forecast(conn, zone, target)
            is_primary = primary is not None and primary["id"] == fid
            log.info(
                "forecast %s %s: %s v%s (%s, %s), alpha %.3f, band x%.3f/%.3f",
                zone,
                target,
                identity,
                version,
                "new" if is_new else "existing",
                "primary" if is_primary else "overlay",
                alpha,
                s10,
                s90,
            )
        finally:
            conn.close()
    return {
        "forecast_id": fid,
        "zone": zone,
        "target_date": target.isoformat(),
        "model": identity,
        "model_version": version,
        "requested": model,
        "primary": is_primary,
        "created_at": created_at,
        "new": is_new,
        "cutoff_utc": cutoff.isoformat(),
        "alpha": round(float(alpha), 4),
        "band_scale": {"p10": round(s10, 4), "p90": round(s90, 4)},
        "history": [h0.isoformat(), h1.isoformat()],
        "weather_features": hourly_w is not None,
        "fallback_reason": fallback_reason,
        "values": _value_rows(rows),
    }


def _value_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "slot": int(r.slot),
            "ts_utc": r.ts_utc.isoformat(),
            "local": _local_label(r.ts_utc),
            "predicted": round(float(r.pred), 3),
            "p10": round(float(r.p10), 3),
            "p90": round(float(r.p90), 3),
            "p_alpha": round(float(r.p_alpha), 3),
        }
        for r in rows.itertuples()
    ]


def forecast_all(target: date | None = None, model: str = "auto") -> list[dict[str, Any]]:
    """One forecast per zone; per-zone failures are reported, never raised."""
    conn = db.connect()
    try:
        zones = db.zone_names(conn)
    finally:
        conn.close()
    out = []
    for zone in zones:
        try:
            fc = generate_forecast(zone, target, model)
            out.append(
                {
                    "zone": zone,
                    "status": "ok",
                    "target_date": fc["target_date"],
                    "model": fc["model"],
                    "model_version": fc["model_version"],
                    "new": fc["new"],
                    "fallback_reason": fc["fallback_reason"],
                }
            )
        except (ValueError, LookupError) as exc:
            log.warning("forecast %s skipped: %s", zone, exc)
            out.append({"zone": zone, "status": "skipped", "reason": str(exc)})
        except Exception as exc:
            log.exception("forecast %s failed", zone)
            out.append({"zone": zone, "status": "error", "reason": str(exc)})
    return out


# ─── retrieval ────────────────────────────────────────────────────────────────────────
# A day can hold several versions (the served model's forecast, a live retrain, a refit
# bundle). The *primary* one counts for the cards, the compare view, scores and schedules:
# the newest version the served model produced (``requested = 'auto'``), else the newest
# of all. Every other version is an overlay: kept, scored, shown on request.
_PRIMARY_ORDER = "ORDER BY (f2.requested = 'auto') DESC, f2.created_at DESC, f2.id DESC LIMIT 1"
_PRIMARY = (
    "f.id = (SELECT f2.id FROM forecasts f2 WHERE f2.zone = f.zone "
    f"AND f2.target_date = f.target_date {_PRIMARY_ORDER})"
)


def primary_forecast(conn: sqlite3.Connection, zone: str, target: date) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT * FROM forecasts f2 WHERE zone=? AND target_date=? {_PRIMARY_ORDER}",
        (zone, target.isoformat()),
    ).fetchone()


def list_forecasts(zone: str, limit: int = 400) -> list[dict[str, Any]]:
    """The primary version per target date (newest day first) with its score, if any."""
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        rows = conn.execute(
            f"""
            SELECT f.id AS forecast_id, f.target_date, f.model, f.model_version, f.requested,
                   f.created_at, f.alpha, s.coverage, s.mape_hour, s.imbalance_usd,
                   s.isolf_mape_hour,
                   (SELECT COUNT(*) FROM forecasts f3 WHERE f3.zone = f.zone
                    AND f3.target_date = f.target_date) AS n_versions
            FROM forecasts f LEFT JOIN forecast_scores s ON s.forecast_id = f.id
            WHERE f.zone = ? AND {_PRIMARY}
            ORDER BY f.target_date DESC LIMIT ?
            """,
            (zone, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def forecast_versions(conn: sqlite3.Connection, zone: str, target: date) -> list[dict[str, Any]]:
    """Every stored version of a day, oldest first, with its score summary and primary flag."""
    rows = conn.execute(
        f"""
        SELECT f.id AS forecast_id, f.model, f.model_version, f.requested, f.created_at,
               f.alpha, ({_PRIMARY}) AS is_primary, s.coverage, s.mape_hour, s.band_coverage,
               s.imbalance_usd, s.imbalance_alpha_usd, s.isolf_mape_hour, s.isolf_imbalance_usd
        FROM forecasts f LEFT JOIN forecast_scores s ON s.forecast_id = f.id
        WHERE f.zone = ? AND f.target_date = ? ORDER BY f.created_at, f.id
        """,
        (zone, target.isoformat()),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["primary"] = bool(d.pop("is_primary"))
        out.append(d)
    return out


def _forecast_values(conn: sqlite3.Connection, forecast_id: int) -> pd.DataFrame:
    return db.read_frame(
        conn,
        "SELECT slot, ts_utc, predicted, p10, p90, p10_raw, p90_raw, p_alpha "
        "FROM forecast_values WHERE forecast_id=? ORDER BY slot",
        (forecast_id,),
    )


def _isolf_for_day(
    conn: sqlite3.Connection, zone: str, target: date, lag_days: int
) -> pd.DataFrame:
    """Hourly ISO forecast rows of `target` from the file named `target - lag_days`."""
    issued = (target - timedelta(days=lag_days)).isoformat()
    lo, hi = local_midnight_utc(target), local_midnight_utc(target + timedelta(days=1))
    return db.read_frame(
        conn,
        "SELECT ts_utc, isolf_mw FROM isolf WHERE zone=? AND issued=? AND ts_utc>=? AND ts_utc<? "
        "ORDER BY ts_utc",
        (zone, issued, db.ts_text(lo), db.ts_text(hi)),
    )


def _to_slots(hourly: pd.DataFrame, value: str, slots: pd.Series) -> np.ndarray:
    """Repeat an hourly column onto slot timestamps (NaN where the hour is absent)."""
    if hourly.empty:
        return np.full(len(slots), np.nan)
    table = hourly.set_index("ts_utc")[value]
    table = table[~table.index.duplicated(keep="last")]
    return table.reindex(slots.dt.floor("h")).to_numpy(dtype=float)


def get_forecast(zone: str, target: date, version: str | None = None) -> dict[str, Any]:
    """A day's primary forecast (or the version asked for) with the ISO overlay, the
    actuals where they exist, and the list of every version stored for that day."""
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        if version:
            f = conn.execute(
                "SELECT * FROM forecasts WHERE zone=? AND target_date=? AND model_version=?",
                (zone, target.isoformat(), version),
            ).fetchone()
            if f is None:
                raise LookupError(f"no version {version} stored for {zone} on {target}")
        else:
            f = primary_forecast(conn, zone, target)
            if f is None:
                raise LookupError(f"no forecast stored for {zone} on {target}")
        versions = forecast_versions(conn, zone, target)
        vals = _forecast_values(conn, f["id"])
        vals = vals.rename(columns={"predicted": "pred"})
        lo, hi = vals["ts_utc"].min(), vals["ts_utc"].max()
        act = db.read_frame(
            conn,
            "SELECT ts_utc, load_mw FROM load_slots WHERE zone=? AND ts_utc BETWEEN ? AND ?",
            (zone, db.ts_text(lo), db.ts_text(hi)),
        )
        vals = vals.merge(act, on="ts_utc", how="left")
        vals["isolf_pre"] = _to_slots(
            _isolf_for_day(conn, zone, target, 1), "isolf_mw", vals["ts_utc"]
        )
        vals["isolf_post"] = _to_slots(
            _isolf_for_day(conn, zone, target, 0), "isolf_mw", vals["ts_utc"]
        )
        score = conn.execute(
            "SELECT * FROM forecast_scores WHERE forecast_id=?", (f["id"],)
        ).fetchone()
        out = dict(f)
        out["forecast_id"] = out.pop("id")
        out["primary"] = any(v["primary"] and v["forecast_id"] == f["id"] for v in versions)
        out["versions"] = versions
        out["score"] = dict(score) if score else None
        out["values"] = [
            {
                "slot": int(r.slot),
                "ts_utc": r.ts_utc.isoformat(),
                "local": _local_label(r.ts_utc),
                "predicted": round(float(r.pred), 3),
                "p10": round(float(r.p10), 3),
                "p90": round(float(r.p90), 3),
                "p_alpha": round(float(r.p_alpha), 3),
                "actual": _num(r.load_mw, 3),
                "isolf_pre": _num(r.isolf_pre, 1),
                "isolf_post": _num(r.isolf_post, 1),
            }
            for r in vals.itertuples()
        ]
        return out
    finally:
        conn.close()


# ─── scoring, settlement and alerts ──────────────────────────────────────────────────
def _hourly_bid(values: pd.Series, ts: pd.Series) -> np.ndarray:
    """The DAM product is one MW per hour: each slot takes its hour's mean forecast."""
    return values.groupby(ts.dt.floor("h")).transform("mean").to_numpy(dtype=float)


def _settle_bid(
    scored: pd.DataFrame, zone: str, bid: np.ndarray, prices: pd.DataFrame
) -> tuple[float | None, float | None]:
    """(imbalance $, DA cost $) of a slot-level bid against the actuals in `scored`."""
    if prices.empty:
        return None, None
    actual = scored[["ts_utc", "load_mw"]].copy()
    actual["zone"] = zone
    settled = settle(actual, bid, prices).dropna(subset=["imbalance_usd"])
    if settled.empty:
        return None, None
    return float(settled["imbalance_usd"].sum()), float(settled["da_cost_usd"].sum())


def _complete_hours(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Hourly means over UTC hours where every slot has an actual (DST-safe)."""
    d = frame.copy()
    d["hour_utc"] = d["ts_utc"].dt.floor("h")
    ok = d.groupby("hour_utc")["load_mw"].transform(lambda v: v.notna().all())
    d = d[ok]
    return d.groupby("hour_utc")[list(cols)].mean().reset_index()


def score_forecast(conn: sqlite3.Connection, forecast_id: int) -> dict[str, Any] | None:
    """Score one stored forecast against actuals, prices and the ISO's pre-close forecast.

    Hourly metrics use complete UTC hours; dollars settle the hourly-mean bid per 15-min
    slot at (RT - DA) (``src.settlement``); NYCA has no zonal price, so no dollars.
    Returns None (and stores nothing) while no actual exists.
    """
    f = conn.execute("SELECT * FROM forecasts WHERE id=?", (forecast_id,)).fetchone()
    if f is None:
        raise LookupError(f"forecast {forecast_id} not found")
    zone, target = f["zone"], date.fromisoformat(f["target_date"])
    vals = _forecast_values(conn, forecast_id)
    lo, hi = vals["ts_utc"].min(), vals["ts_utc"].max()
    act = db.read_frame(
        conn,
        "SELECT ts_utc, load_mw FROM load_slots WHERE zone=? AND ts_utc BETWEEN ? AND ?",
        (zone, db.ts_text(lo), db.ts_text(hi)),
    )
    m = vals.merge(act, on="ts_utc", how="left")
    coverage = float(m["load_mw"].notna().mean())
    if coverage == 0.0:
        return None
    m["isolf_pre"] = _to_slots(_isolf_for_day(conn, zone, target, 1), "isolf_mw", m["ts_utc"])
    scored = m.dropna(subset=["load_mw"]).reset_index(drop=True)
    hours = _complete_hours(m, ["predicted", "p_alpha", "isolf_pre", "load_mw"])
    band = float(
        ((scored["p10"] <= scored["load_mw"]) & (scored["load_mw"] <= scored["p90"])).mean()
    )

    prices = slot_prices_for(conn, zone, lo, hi)
    imb, da_cost = _settle_bid(
        scored, zone, _hourly_bid(scored["predicted"], scored["ts_utc"]), prices
    )
    imb_alpha, _ = _settle_bid(
        scored, zone, _hourly_bid(scored["p_alpha"], scored["ts_utc"]), prices
    )
    iso_rows = scored.dropna(subset=["isolf_pre"])
    iso_imb, _ = _settle_bid(iso_rows, zone, iso_rows["isolf_pre"].to_numpy(dtype=float), prices)
    iso_hours = hours.dropna(subset=["isolf_pre"])

    score = {
        "forecast_id": forecast_id,
        "zone": zone,
        "target_date": f["target_date"],
        "model": f["model"],
        "coverage": round(coverage, 4),
        "mape_slot": _num(mape(scored["load_mw"], scored["predicted"]), 3),
        "mape_hour": _num(mape(hours["load_mw"], hours["predicted"]), 3) if len(hours) else None,
        "mape_hour_alpha": _num(mape(hours["load_mw"], hours["p_alpha"]), 3)
        if len(hours)
        else None,
        "band_coverage": round(band * 100, 2),
        "imbalance_usd": _num(imb, 2),
        "imbalance_alpha_usd": _num(imb_alpha, 2),
        "da_cost_usd": _num(da_cost, 2),
        "isolf_mape_hour": (
            _num(mape(iso_hours["load_mw"], iso_hours["isolf_pre"]), 3) if len(iso_hours) else None
        ),
        "isolf_imbalance_usd": _num(iso_imb, 2),
    }
    conn.execute(
        "INSERT OR REPLACE INTO forecast_scores (forecast_id, scored_at, coverage, mape_slot, "
        "mape_hour, mape_hour_alpha, band_coverage, imbalance_usd, imbalance_alpha_usd, "
        "da_cost_usd, isolf_mape_hour, isolf_imbalance_usd) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            forecast_id,
            db.now_text(),
            score["coverage"],
            score["mape_slot"],
            score["mape_hour"],
            score["mape_hour_alpha"],
            score["band_coverage"],
            score["imbalance_usd"],
            score["imbalance_alpha_usd"],
            score["da_cost_usd"],
            score["isolf_mape_hour"],
            score["isolf_imbalance_usd"],
        ),
    )
    return score


def score_schedule(conn: sqlite3.Connection, schedule_id: int) -> dict[str, Any] | None:
    s = conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
    if s is None:
        raise LookupError(f"schedule {schedule_id} not found")
    zone = s["zone"]
    vals = db.read_frame(
        conn,
        "SELECT slot, ts_utc, bid_mw FROM schedule_values WHERE schedule_id=? ORDER BY slot",
        (schedule_id,),
    )
    lo, hi = vals["ts_utc"].min(), vals["ts_utc"].max()
    act = db.read_frame(
        conn,
        "SELECT ts_utc, load_mw FROM load_slots WHERE zone=? AND ts_utc BETWEEN ? AND ?",
        (zone, db.ts_text(lo), db.ts_text(hi)),
    )
    m = vals.merge(act, on="ts_utc", how="left")
    coverage = float(m["load_mw"].notna().mean())
    if coverage == 0.0:
        return None
    scored = m.dropna(subset=["load_mw"]).reset_index(drop=True)
    hours = _complete_hours(m, ["bid_mw", "load_mw"])
    prices = slot_prices_for(conn, zone, lo, hi)
    imb, da_cost = _settle_bid(scored, zone, scored["bid_mw"].to_numpy(dtype=float), prices)
    score = {
        "schedule_id": schedule_id,
        "zone": zone,
        "target_date": s["target_date"],
        "coverage": round(coverage, 4),
        "mape_hour": _num(mape(hours["load_mw"], hours["bid_mw"]), 3) if len(hours) else None,
        "imbalance_usd": _num(imb, 2),
        "da_cost_usd": _num(da_cost, 2),
    }
    conn.execute(
        "INSERT OR REPLACE INTO schedule_scores (schedule_id, scored_at, coverage, mape_hour, "
        "imbalance_usd, da_cost_usd) VALUES (?, ?, ?, ?, ?, ?)",
        (
            schedule_id,
            db.now_text(),
            score["coverage"],
            score["mape_hour"],
            score["imbalance_usd"],
            score["da_cost_usd"],
        ),
    )
    return score


def _refresh_alerts(conn: sqlite3.Connection, score: dict[str, Any]) -> list[dict[str, Any]]:
    """Accuracy and imbalance alerts for one scored day (idempotent per zone/day/kind)."""
    if score["coverage"] < SCORE_MIN_COVERAGE:
        return []
    zone, day = score["zone"], score["target_date"]
    alerts = []
    if score["mape_hour"] is not None and score["mape_hour"] > SCORE_ALERT_MAPE:
        alerts.append(
            ("mape", f"hourly MAPE {score['mape_hour']:.1f}% exceeds {SCORE_ALERT_MAPE:.0f}%")
        )
    imb = score["imbalance_usd"]
    if imb is not None:
        hist = conn.execute(
            f"""
            SELECT s.imbalance_usd FROM forecast_scores s JOIN forecasts f ON f.id = s.forecast_id
            WHERE f.zone = ? AND f.target_date < ? AND f.target_date >= ? AND {_PRIMARY}
              AND s.imbalance_usd IS NOT NULL AND s.coverage >= ?
            """,
            (
                zone,
                day,
                (date.fromisoformat(day) - timedelta(days=CONFORMAL_WINDOW_DAYS)).isoformat(),
                SCORE_MIN_COVERAGE,
            ),
        ).fetchall()
        past = np.array([r["imbalance_usd"] for r in hist], dtype=float)
        if len(past) >= IMBALANCE_ALERT_MIN_DAYS:
            p90 = float(np.quantile(past, IMBALANCE_ALERT_QUANTILE))
            if imb > p90:
                alerts.append(
                    (
                        "imbalance",
                        f"imbalance cost ${imb:,.0f} above the trailing-"
                        f"{CONFORMAL_WINDOW_DAYS}-day P90 (${p90:,.0f})",
                    )
                )
    out = []
    for kind, message in alerts:
        conn.execute(
            "INSERT OR IGNORE INTO alerts (zone, target_date, kind, message, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (zone, day, kind, message, db.now_text()),
        )
        out.append({"zone": zone, "target_date": day, "kind": kind, "message": message})
    return out


def score_pending(zone: str | None = None, today: date | None = None) -> dict[str, Any]:
    """Score every forecast version and schedule that actuals now cover better than before.

    Overlays are scored like the primary so a live retrain gets its own number the next
    morning; only the primary version raises alerts.
    """
    today = today or TODAY()
    lo = (today - timedelta(days=SCORE_WINDOW_DAYS)).isoformat()
    hi = (today - timedelta(days=1)).isoformat()
    zone_sql, params = ("AND f.zone = ?", [zone]) if zone else ("", [])
    conn = db.connect()
    try:
        rows = conn.execute(
            f"""
            SELECT f.id, s.coverage AS prev, ({_PRIMARY}) AS is_primary FROM forecasts f
            LEFT JOIN forecast_scores s ON s.forecast_id = f.id
            WHERE f.target_date BETWEEN ? AND ? {zone_sql}
              AND (s.forecast_id IS NULL OR s.coverage < 1.0)
            ORDER BY f.zone, f.target_date, f.id
            """,
            [lo, hi, *params],
        ).fetchall()
        scores, alerts = [], []
        for r in rows:
            score = score_forecast(conn, r["id"])
            if score is not None and score["coverage"] != r["prev"]:
                score["primary"] = bool(r["is_primary"])
                scores.append(score)
                if score["primary"]:
                    alerts.extend(_refresh_alerts(conn, score))
        srows = conn.execute(
            f"""
            SELECT d.id, ss.coverage AS prev FROM schedules d
            LEFT JOIN schedule_scores ss ON ss.schedule_id = d.id
            WHERE d.target_date BETWEEN ? AND ? {zone_sql.replace("f.", "d.")}
              AND (ss.schedule_id IS NULL OR ss.coverage < 1.0)
            """,
            [lo, hi, *params],
        ).fetchall()
        sched = [s for r in srows if (s := score_schedule(conn, r["id"])) is not None]
        conn.commit()
    finally:
        conn.close()
    log.info("scored %d forecasts, %d schedules, %d alerts", len(scores), len(sched), len(alerts))
    return {"forecasts": scores, "schedules": sched, "alerts": alerts}


def list_scores(zone: str, limit: int = 400) -> list[dict[str, Any]]:
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        rows = conn.execute(
            f"""
            SELECT f.target_date, f.model, f.model_version, s.*
            FROM forecast_scores s JOIN forecasts f ON f.id = s.forecast_id
            WHERE f.zone = ? AND {_PRIMARY} ORDER BY f.target_date DESC LIMIT ?
            """,
            (zone, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_alerts(days: int = 14, today: date | None = None) -> list[dict[str, Any]]:
    today = today or TODAY()
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT zone, target_date, kind, message, created_at FROM alerts "
            "WHERE target_date >= ? ORDER BY target_date DESC, zone",
            ((today - timedelta(days=days)).isoformat(),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── zone cards and views ─────────────────────────────────────────────────────────────
def zone_cards(today: date | None = None) -> list[dict[str, Any]]:
    """One card per zone: history span, latest forecast, 7-day accuracy and dollars, alerts."""
    today = today or TODAY()
    week = (today - timedelta(days=7)).isoformat()
    conn = db.connect()
    try:
        cards = []
        for zone in db.zone_names(conn):
            span = conn.execute(
                "SELECT MIN(ts_utc) AS first_ts, MAX(ts_utc) AS last_ts, COUNT(*) AS n "
                "FROM load_slots WHERE zone=?",
                (zone,),
            ).fetchone()
            latest = conn.execute(
                "SELECT target_date, model, model_version, created_at FROM forecasts "
                "WHERE zone=? ORDER BY target_date DESC, (requested = 'auto') DESC, "
                "created_at DESC, id DESC LIMIT 1",
                (zone,),
            ).fetchone()
            recent = conn.execute(
                f"""
                SELECT COUNT(*) AS days, AVG(s.mape_hour) AS mape_hour,
                       SUM(s.imbalance_usd) AS imbalance_usd, AVG(s.isolf_mape_hour) AS isolf_mape,
                       SUM(s.isolf_imbalance_usd) AS isolf_imbalance_usd,
                       AVG(s.band_coverage) AS band_coverage
                FROM forecast_scores s JOIN forecasts f ON f.id = s.forecast_id
                WHERE f.zone = ? AND f.target_date >= ? AND s.coverage >= ? AND {_PRIMARY}
                """,
                (zone, week, SCORE_MIN_COVERAGE),
            ).fetchone()
            alerts = conn.execute(
                "SELECT target_date, kind, message FROM alerts WHERE zone=? AND target_date>=? "
                "ORDER BY target_date DESC",
                (zone, week),
            ).fetchall()
            cards.append(
                {
                    "zone": zone,
                    "is_total": zone == NYCA,
                    "first_ts": span["first_ts"],
                    "last_ts": span["last_ts"],
                    "history_days": (
                        (_local_day(span["last_ts"]) - _local_day(span["first_ts"])).days + 1
                        if span["n"]
                        else 0
                    ),
                    "latest_forecast": dict(latest) if latest else None,
                    "last_7d": {
                        "days": recent["days"],
                        "mape_hour": _num(recent["mape_hour"], 2),
                        "imbalance_usd": _num(recent["imbalance_usd"], 0),
                        "isolf_mape_hour": _num(recent["isolf_mape"], 2),
                        "isolf_imbalance_usd": _num(recent["isolf_imbalance_usd"], 0),
                        "band_coverage": _num(recent["band_coverage"], 1),
                    },
                    "alerts": [dict(a) for a in alerts],
                    "alert": bool(alerts),
                }
            )
        return cards
    finally:
        conn.close()


def _latest_values_between(
    conn: sqlite3.Connection, zone: str, start: date, end: date
) -> pd.DataFrame:
    return db.read_frame(
        conn,
        f"""
        SELECT f.target_date, f.model, v.ts_utc, v.predicted, v.p10, v.p90, v.p_alpha
        FROM forecasts f JOIN forecast_values v ON v.forecast_id = f.id
        WHERE f.zone = ? AND f.target_date BETWEEN ? AND ? AND {_PRIMARY}
        ORDER BY v.ts_utc
        """,
        (zone, start.isoformat(), end.isoformat()),
    )


def _isolf_pre_between(conn: sqlite3.Connection, zone: str, start: date, end: date) -> pd.DataFrame:
    """Hourly ``ts_utc, isolf_pre`` for every day in the range from the file named D-1."""
    parts = [_isolf_for_day(conn, zone, d, 1) for d in _days(start, end)]
    frames = [p for p in parts if not p.empty]
    if not frames:
        return pd.DataFrame({"ts_utc": pd.Series(dtype="datetime64[ns, UTC]"), "isolf_mw": []})
    return pd.concat(frames, ignore_index=True)


def _bucket(frame: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate slot rows: MW columns by mean, dollar columns by sum."""
    d = frame.copy()
    if granularity == "slot":
        d["bucket"] = d["ts_utc"]
    elif granularity == "hour":
        d["bucket"] = d["ts_utc"].dt.floor("h")
    elif granularity == "day":
        d["bucket"] = d["ts_utc"].dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_convert("UTC")
    else:
        raise ValueError("granularity must be slot, hour or day")
    usd = [c for c in d.columns if c.endswith("_usd")]
    mw = [c for c in d.columns if c not in ("ts_utc", "bucket", *usd) and d[c].dtype != object]
    agg = {**{c: "mean" for c in mw}, **{c: "sum" for c in usd}}
    g = d.groupby("bucket").agg(agg)
    # a bucket with any missing actual is incomplete: no error, no dollars
    incomplete = d.groupby("bucket")["load_mw"].apply(lambda v: v.isna().any())
    g.loc[incomplete, "load_mw"] = np.nan
    for c in usd:
        g.loc[incomplete, c] = np.nan
    return g.reset_index()


def compare(zone: str, start: date, end: date, granularity: str = "hour") -> dict[str, Any]:
    """Stored forecasts vs actuals vs the ISO's pre-close forecast, with prices and dollars."""
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        preds = _latest_values_between(conn, zone, start, end)
        if preds.empty:
            return {
                "zone": zone,
                "granularity": granularity,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "points": [],
                "summary": None,
                "message": "no stored forecasts in this range",
            }
        lo, hi = preds["ts_utc"].min(), preds["ts_utc"].max()
        act = db.read_frame(
            conn,
            "SELECT ts_utc, load_mw FROM load_slots WHERE zone=? AND ts_utc BETWEEN ? AND ?",
            (zone, db.ts_text(lo), db.ts_text(hi)),
        )
        m = preds.merge(act, on="ts_utc", how="left")
        m["isolf_pre"] = _to_slots(
            _isolf_pre_between(conn, zone, start, end), "isolf_mw", m["ts_utc"]
        )
        prices = slot_prices_for(conn, zone, lo, hi)
        if not prices.empty:
            m = m.merge(prices[["ts_utc", "p_rt", "p_da", "spread"]], on="ts_utc", how="left")
        else:
            m["p_rt"] = m["p_da"] = m["spread"] = np.nan
        bid = _hourly_bid(m["predicted"], m["ts_utc"])
        bid_alpha = _hourly_bid(m["p_alpha"], m["ts_utc"])
        dev = (m["load_mw"] - bid) * MWH_PER_MW_SLOT
        m["imbalance_usd"] = dev * m["spread"]
        m["imbalance_alpha_usd"] = (m["load_mw"] - bid_alpha) * MWH_PER_MW_SLOT * m["spread"]
        m["isolf_imbalance_usd"] = (m["load_mw"] - m["isolf_pre"]) * MWH_PER_MW_SLOT * m["spread"]
        m["da_cost_usd"] = bid * MWH_PER_MW_SLOT * m["p_da"]
        g = _bucket(m.drop(columns=["target_date", "model"]), granularity)

        points = []
        for r in g.itertuples():
            actual = _num(r.load_mw)
            pred = float(r.predicted)
            points.append(
                {
                    "ts_utc": r.bucket.isoformat(),
                    "local": _local_label(r.bucket),
                    "predicted": round(pred, 3),
                    "p10": round(float(r.p10), 3),
                    "p90": round(float(r.p90), 3),
                    "p_alpha": round(float(r.p_alpha), 3),
                    "actual": _num(actual, 3),
                    "isolf_pre": _num(r.isolf_pre, 1),
                    "ape": round(abs(pred - actual) / actual * 100, 3) if actual else None,
                    "p_da": _num(r.p_da, 3),
                    "p_rt": _num(r.p_rt, 3),
                    "spread": _num(r.spread, 3),
                    "imbalance_usd": _num(r.imbalance_usd, 2),
                    "imbalance_alpha_usd": _num(r.imbalance_alpha_usd, 2),
                    "isolf_imbalance_usd": _num(r.isolf_imbalance_usd, 2),
                }
            )
        hours = _complete_hours(m, ["predicted", "p_alpha", "isolf_pre", "load_mw"])
        iso_h = hours.dropna(subset=["isolf_pre"])
        settled = m.dropna(subset=["load_mw", "spread"])
        summary = {
            "days": int(m["target_date"].nunique()),
            "mape_hour": _num(mape(hours["load_mw"], hours["predicted"]), 3)
            if len(hours)
            else None,
            "mape_hour_alpha": (
                _num(mape(hours["load_mw"], hours["p_alpha"]), 3) if len(hours) else None
            ),
            "isolf_mape_hour": _num(mape(iso_h["load_mw"], iso_h["isolf_pre"]), 3)
            if len(iso_h)
            else None,
            "imbalance_usd": _num(settled["imbalance_usd"].sum(), 2) if len(settled) else None,
            "imbalance_alpha_usd": (
                _num(settled["imbalance_alpha_usd"].sum(), 2) if len(settled) else None
            ),
            "isolf_imbalance_usd": (
                _num(settled["isolf_imbalance_usd"].sum(), 2)
                if settled["isolf_imbalance_usd"].notna().any()
                else None
            ),
            "models": sorted(m["model"].unique().tolist()),
        }
        return {
            "zone": zone,
            "granularity": granularity,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "points": points,
            "summary": summary,
        }
    finally:
        conn.close()


def load_view(zone: str, start: date, end: date, granularity: str = "hour") -> dict[str, Any]:
    """Actual load for plotting at slot / hour / day resolution (local ET buckets)."""
    zone = resolve_zone(zone)
    lo, hi = local_midnight_utc(start), local_midnight_utc(end + timedelta(days=1))
    conn = db.connect()
    try:
        act = db.read_frame(
            conn,
            "SELECT ts_utc, load_mw FROM load_slots WHERE zone=? AND ts_utc>=? AND ts_utc<? "
            "ORDER BY ts_utc",
            (zone, db.ts_text(lo), db.ts_text(hi)),
        )
    finally:
        conn.close()
    if act.empty:
        return {"zone": zone, "granularity": granularity, "points": []}
    g = _bucket(act, granularity)
    return {
        "zone": zone,
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "points": [
            {
                "ts_utc": r.bucket.isoformat(),
                "local": _local_label(r.bucket),
                "actual": _num(r.load_mw, 3),
            }
            for r in g.itertuples()
            if _num(r.load_mw) is not None
        ],
    }


def prices_view(zone: str, start: date, end: date) -> dict[str, Any]:
    """Hourly DA and RT prices with the spread (the settlement driver) for `zone`."""
    zone = resolve_zone(zone)
    if zone == NYCA:
        raise ValueError("NYCA has no zonal price; pick a zone")
    lo, hi = local_midnight_utc(start), local_midnight_utc(end + timedelta(days=1))
    conn = db.connect()
    try:
        da = db.read_frame(
            conn,
            "SELECT ts_utc, p_da FROM da_hourly WHERE zone=? AND ts_utc>=? AND ts_utc<? "
            "ORDER BY ts_utc",
            (zone, db.ts_text(lo), db.ts_text(hi)),
        )
        rt = db.read_frame(
            conn,
            "SELECT ts_utc, p_rt FROM rt_slots WHERE zone=? AND ts_utc>=? AND ts_utc<?",
            (zone, db.ts_text(lo), db.ts_text(hi)),
        )
    finally:
        conn.close()
    if da.empty and rt.empty:
        return {"zone": zone, "points": []}
    if not rt.empty:
        rt["ts_utc"] = rt["ts_utc"].dt.floor("h")
        rt = rt.groupby("ts_utc", as_index=False)["p_rt"].mean()
    m = da.merge(rt, on="ts_utc", how="outer").sort_values("ts_utc")
    m["spread"] = m["p_rt"] - m["p_da"]
    return {
        "zone": zone,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "points": [
            {
                "ts_utc": r.ts_utc.isoformat(),
                "local": _local_label(r.ts_utc),
                "p_da": _num(r.p_da, 3),
                "p_rt": _num(r.p_rt, 3),
                "spread": _num(r.spread, 3),
            }
            for r in m.itertuples()
        ],
    }


def isolf_view(zone: str, target: date) -> dict[str, Any]:
    """NYISO's hourly forecast of `target`: pre-close (file named D-1) and post-close (D)."""
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        pre = _isolf_for_day(conn, zone, target, 1).rename(columns={"isolf_mw": "isolf_pre"})
        post = _isolf_for_day(conn, zone, target, 0).rename(columns={"isolf_mw": "isolf_post"})
    finally:
        conn.close()
    m = pre.merge(post, on="ts_utc", how="outer").sort_values("ts_utc")
    return {
        "zone": zone,
        "target_date": target.isoformat(),
        "points": [
            {
                "ts_utc": r.ts_utc.isoformat(),
                "local": _local_label(r.ts_utc),
                "isolf_pre": _num(r.isolf_pre, 1),
                "isolf_post": _num(r.isolf_post, 1),
            }
            for r in m.itertuples()
        ],
    }


# ─── DAM schedules ────────────────────────────────────────────────────────────────────
def _hour_positions(ts: pd.Series) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Chronological hour index of each slot (24 / 23 / 25 hours on a local day)."""
    hours = ts.dt.floor("h")
    order = sorted(hours.unique())
    pos = {h: i for i, h in enumerate(order)}
    return hours.map(pos).to_numpy(dtype=int), order


def create_schedule(
    zone: str,
    target: date,
    hourly_mw: Sequence[float],
    note: str | None = None,
    adjustments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a DAM schedule (one MW bid per hour of `target`) on top of the latest forecast.

    The bid is flat within each hour, as the market product is; the pinned forecast stays
    immutable so the schedule and the model can both be scored once actuals arrive.
    """
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        f = primary_forecast(conn, zone, target)
        if f is None:
            raise LookupError(f"no forecast stored for {zone} on {target}; generate one first")
        vals = _forecast_values(conn, f["id"])
        pos, hours = _hour_positions(vals["ts_utc"])
        bids = np.asarray(list(hourly_mw), dtype=float)
        if bids.shape != (len(hours),) or not np.isfinite(bids).all() or (bids < 0).any():
            raise ValueError(
                f"hourly_mw must be {len(hours)} finite non-negative MW values "
                f"(one per hour of {target}, chronological)"
            )
        vals["bid_mw"] = bids[pos]
        stats = alpha_stats(conn, zone, target)
        created_at = db.now_text()
        cur = conn.execute(
            "INSERT INTO schedules (zone, forecast_id, target_date, created_at, note, "
            "adjustments_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                zone,
                f["id"],
                target.isoformat(),
                created_at,
                (str(note).strip() or None) if note else None,
                json.dumps(adjustments) if adjustments else None,
            ),
        )
        sid = cur.lastrowid
        conn.executemany(
            "INSERT INTO schedule_values (schedule_id, slot, ts_utc, bid_mw) VALUES (?, ?, ?, ?)",
            [(sid, int(r.slot), db.ts_text(r.ts_utc), float(r.bid_mw)) for r in vals.itertuples()],
        )
        conn.commit()
        return {
            "schedule_id": sid,
            "zone": zone,
            "target_date": target.isoformat(),
            "forecast_id": f["id"],
            "model": f["model"],
            "model_version": f["model_version"],
            "created_at": created_at,
            "note": note,
            **_schedule_totals(vals, stats),
        }
    finally:
        conn.close()


def _schedule_totals(vals: pd.DataFrame, stats: dict[str, Any]) -> dict[str, Any]:
    energy = float(vals["bid_mw"].sum() * MWH_PER_MW_SLOT)
    forecast = float(vals["predicted"].sum() * MWH_PER_MW_SLOT)
    dev_mwh = float((vals["bid_mw"] - vals["predicted"]).abs().sum() * MWH_PER_MW_SLOT)
    spread = stats.get("mean_abs_spread_usd_per_mwh")
    return {
        "total_bid_mwh": round(energy, 3),
        "total_forecast_mwh": round(forecast, 3),
        "delta_pct": round((energy / forecast - 1) * 100, 3) if forecast else None,
        "deviation_from_model_mwh": round(dev_mwh, 3),
        # guardrail: the deviation from the model priced at the trailing mean |RT - DA|
        "usd_at_risk": round(dev_mwh * spread, 2) if spread is not None else None,
        "alpha": stats["alpha"],
    }


def _schedule_frame(conn: sqlite3.Connection, schedule_id: int, forecast_id: int) -> pd.DataFrame:
    return db.read_frame(
        conn,
        "SELECT sv.slot, sv.ts_utc, sv.bid_mw, fv.predicted, fv.p10, fv.p90, fv.p_alpha "
        "FROM schedule_values sv JOIN forecast_values fv "
        "ON fv.forecast_id = ? AND fv.slot = sv.slot WHERE sv.schedule_id = ? ORDER BY sv.slot",
        (forecast_id, schedule_id),
    )


def get_schedule(zone: str, target: date) -> dict[str, Any]:
    """Latest schedule for a day: slot bids with the pinned forecast, hourly rollup, score."""
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        s = conn.execute(
            "SELECT d.*, f.model, f.model_version FROM schedules d JOIN forecasts f "
            "ON f.id = d.forecast_id WHERE d.zone=? AND d.target_date=? "
            "ORDER BY d.created_at DESC, d.id DESC LIMIT 1",
            (zone, target.isoformat()),
        ).fetchone()
        if s is None:
            raise LookupError(f"no schedule stored for {zone} on {target}")
        vals = _schedule_frame(conn, s["id"], s["forecast_id"])
        stats = alpha_stats(conn, zone, target)
        score = conn.execute(
            "SELECT * FROM schedule_scores WHERE schedule_id=?", (s["id"],)
        ).fetchone()
        fscore = conn.execute(
            "SELECT mape_hour, imbalance_usd, imbalance_alpha_usd FROM forecast_scores "
            "WHERE forecast_id=?",
            (s["forecast_id"],),
        ).fetchone()
        pos, hours = _hour_positions(vals["ts_utc"])
        vals["hour_pos"] = pos
        hourly = vals.groupby("hour_pos").agg(
            ts_utc=("ts_utc", "min"), bid_mw=("bid_mw", "mean"), predicted=("predicted", "mean")
        )
        return {
            "schedule_id": s["id"],
            "zone": zone,
            "target_date": target.isoformat(),
            "forecast_id": s["forecast_id"],
            "model": s["model"],
            "model_version": s["model_version"],
            "created_at": s["created_at"],
            "note": s["note"],
            "adjustments": json.loads(s["adjustments_json"]) if s["adjustments_json"] else None,
            **_schedule_totals(vals, stats),
            "values": [
                {
                    "slot": int(r.slot),
                    "ts_utc": r.ts_utc.isoformat(),
                    "local": _local_label(r.ts_utc),
                    "bid_mw": round(float(r.bid_mw), 3),
                    "predicted": round(float(r.predicted), 3),
                    "p10": round(float(r.p10), 3),
                    "p90": round(float(r.p90), 3),
                    "p_alpha": round(float(r.p_alpha), 3),
                }
                for r in vals.itertuples()
            ],
            "hourly": [
                {
                    "hour": int(h),
                    "local": _local_label(r.ts_utc),
                    "bid_mw": round(float(r.bid_mw), 3),
                    "predicted": round(float(r.predicted), 3),
                }
                for h, r in hourly.iterrows()
            ],
            "score": dict(score) if score else None,
            "forecast_score": dict(fscore) if fscore else None,
        }
    finally:
        conn.close()


def list_schedules(zone: str, limit: int = 200) -> list[dict[str, Any]]:
    zone = resolve_zone(zone)
    conn = db.connect()
    try:
        rows = conn.execute(
            """
            SELECT d.id AS schedule_id, d.target_date, d.created_at, d.note, d.forecast_id,
                   f.model, f.model_version, ss.coverage, ss.mape_hour, ss.imbalance_usd,
                   fs.mape_hour AS model_mape_hour, fs.imbalance_usd AS model_imbalance_usd
            FROM schedules d
            JOIN forecasts f ON f.id = d.forecast_id
            LEFT JOIN schedule_scores ss ON ss.schedule_id = d.id
            LEFT JOIN forecast_scores fs ON fs.forecast_id = d.forecast_id
            WHERE d.zone = ? ORDER BY d.target_date DESC, d.created_at DESC, d.id DESC LIMIT ?
            """,
            (zone, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── retention ────────────────────────────────────────────────────────────────────────
def retention_boundary(today: date, months: int) -> date:
    """First day of the month `months` before `today`'s month: whole months are kept."""
    y, m = today.year, today.month - months
    while m <= 0:
        y, m = y - 1, m + 12
    return date(y, m, 1)


def prune(months: int | None = None, today: date | None = None) -> dict[str, Any]:
    """Drop market data, cache files and weather older than the retention window.

    Forecasts, scores, schedules and alerts are never pruned: they are the track record.
    """
    months = months or RETENTION_MONTHS
    if months < RETENTION_MIN_MONTHS:
        raise ValueError(f"retention must be at least {RETENTION_MIN_MONTHS} months")
    today = today or TODAY()
    boundary = retention_boundary(today, months)
    ts = db.ts_text(local_midnight_utc(boundary))
    out: dict[str, Any] = {"boundary": boundary.isoformat()}
    conn = db.connect()
    try:
        for table in ("load_slots", "rt_slots", "da_hourly", "rt_hourly"):
            out[table] = conn.execute(f"DELETE FROM {table} WHERE ts_utc < ?", (ts,)).rowcount
        out["isolf"] = conn.execute(
            "DELETE FROM isolf WHERE issued < ?", (boundary.isoformat(),)
        ).rowcount
        out["ingest_log"] = conn.execute(
            "DELETE FROM ingest_log WHERE day < ?", (boundary.isoformat(),)
        ).rowcount
        conn.commit()
        conn.execute("PRAGMA incremental_vacuum")
    finally:
        conn.close()
    out["cache_files"] = _prune_cache(get_client().cache_dir, boundary)
    out["weather_rows"] = weather.trim(boundary)
    log.info("retention: pruned before %s: %s", boundary, out)
    return out


def _prune_cache(cache_dir: Path, boundary: date) -> int:
    """Delete cached daily files and monthly zips dated before `boundary`."""
    n = 0
    if not cache_dir.exists():
        return 0
    for path in cache_dir.rglob("*"):
        if not path.is_file() or not path.name[:8].isdigit():
            continue
        y, m, d = int(path.name[:4]), int(path.name[4:6]), int(path.name[6:8])
        try:
            stamp = date(y, m, d)
        except ValueError:
            continue
        if stamp < boundary:
            path.unlink(missing_ok=True)
            n += 1
    return n


# ─── health ───────────────────────────────────────────────────────────────────────────
def data_status() -> dict[str, Any]:
    """Coverage per table for /health: first and last stamps and the last ingested day."""
    conn = db.connect()
    try:
        out: dict[str, Any] = {}
        for ft, table in FILE_TABLES.items():
            col = "issued" if table == "isolf" else "ts_utc"
            span = conn.execute(
                f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM {table}"
            ).fetchone()
            last = last_ingested(conn, ft)
            out[table] = {
                "first": span["lo"],
                "last": span["hi"],
                "last_ingested_day": last.isoformat() if last else None,
            }
        out["forecasts"] = conn.execute("SELECT COUNT(*) AS n FROM forecasts").fetchone()["n"]
        out["scored"] = conn.execute("SELECT COUNT(*) AS n FROM forecast_scores").fetchone()["n"]
        return out
    finally:
        conn.close()


def record_job(name: str, status: str, detail: Any = None) -> None:
    conn = db.connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (name, last_run_at, last_status, detail_json) "
            "VALUES (?, ?, ?, ?)",
            (name, db.now_text(), status, json.dumps(detail, default=str) if detail else None),
        )
        conn.commit()
    finally:
        conn.close()


def job_status() -> list[dict[str, Any]]:
    conn = db.connect()
    try:
        rows = conn.execute("SELECT * FROM jobs ORDER BY name").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            raw = d.pop("detail_json")
            d["detail"] = json.loads(raw) if raw else None
            out.append(d)
        return out
    finally:
        conn.close()


def zone_list() -> list[str]:
    conn = db.connect()
    try:
        return db.zone_names(conn)
    finally:
        conn.close()
