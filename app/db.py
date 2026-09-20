"""SQLite storage for the NYISO forecasting service.

Tables
------
zones                       the 11 NYISO zones + NYCA, seeded from ``src.config``
load_slots                  15-min actual load per zone (slot start, UTC), NYCA included
rt_slots                    15-min real-time LBMP per zone (time-weighted mean of the RTD prices)
da_hourly / rt_hourly       hourly day-ahead LBMP / hourly integrated real-time LBMP
isolf                       NYISO's own hourly load forecast, keyed by the file's issue day
ingest_log                  one row per (file type, day) fetched: catch-up and health
forecasts / forecast_values immutable forecasts versioned by ``model_version``; every value
                            row carries the raw and the conformally scaled P10-P90 band
                            and the α-bid
schedules / schedule_values DAM schedules (one MW bid per hour) pinned to one forecast
forecast_scores             MAPE, band coverage and dollars once actuals and prices exist
schedule_scores             the same for a schedule
alerts                      accuracy / imbalance alerts per (zone, day, kind)
jobs                        scheduler bookkeeping: last run, status, detail

Timestamps are text ``YYYY-MM-DD HH:MM:SS`` in UTC (sorts chronologically); dates are
``YYYY-MM-DD`` local ET days. ``APP_DB_PATH`` is read once at import, so tests set it
before importing the app.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import NYCA, ZONE_PTID, ZONES

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("APP_DB_PATH") or PROJECT_ROOT / "data" / "app.db")
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    name        TEXT PRIMARY KEY,
    ptid        INTEGER,
    is_total    INTEGER NOT NULL DEFAULT 0,
    sort_order  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS load_slots (
    zone        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    load_mw     REAL NOT NULL,
    coverage    REAL NOT NULL,
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE IF NOT EXISTS rt_slots (
    zone        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    p_rt        REAL NOT NULL,
    coverage    REAL NOT NULL,
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE IF NOT EXISTS da_hourly (
    zone        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    p_da        REAL NOT NULL,
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE IF NOT EXISTS rt_hourly (
    zone        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    p_rt_hourly REAL NOT NULL,
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE IF NOT EXISTS isolf (
    issued      TEXT NOT NULL,      -- the day the file is named for (posted the morning before)
    zone        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,      -- hour start
    isolf_mw    REAL NOT NULL,
    PRIMARY KEY (issued, zone, ts_utc)
);
CREATE TABLE IF NOT EXISTS ingest_log (
    file_type   TEXT NOT NULL,
    day         TEXT NOT NULL,
    status      TEXT NOT NULL,      -- ok | missing
    rows        INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (file_type, day)
);
CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    zone            TEXT NOT NULL REFERENCES zones(name),
    target_date     TEXT NOT NULL,
    model           TEXT NOT NULL,  -- 'tft-onnx:<sha12>:<fit cutoff>' or 'lgbm:<feature version>'
    model_version   TEXT NOT NULL,  -- hash of the data window, model identity and settings
    cutoff_utc      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    alpha           REAL NOT NULL,
    band_scale_p10  REAL NOT NULL DEFAULT 1.0,
    band_scale_p90  REAL NOT NULL DEFAULT 1.0,
    history_start   TEXT,
    history_end     TEXT,
    weather         INTEGER NOT NULL DEFAULT 0,
    UNIQUE (zone, target_date, model_version)
);
CREATE TABLE IF NOT EXISTS forecast_values (
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    slot        INTEGER NOT NULL,
    ts_utc      TEXT NOT NULL,
    predicted   REAL NOT NULL,
    p10         REAL NOT NULL,
    p90         REAL NOT NULL,
    p10_raw     REAL NOT NULL,
    p90_raw     REAL NOT NULL,
    p_alpha     REAL NOT NULL,
    PRIMARY KEY (forecast_id, slot)
);
CREATE TABLE IF NOT EXISTS schedules (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    zone             TEXT NOT NULL REFERENCES zones(name),
    forecast_id      INTEGER NOT NULL REFERENCES forecasts(id),
    target_date      TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    note             TEXT,
    adjustments_json TEXT
);
CREATE TABLE IF NOT EXISTS schedule_values (
    schedule_id INTEGER NOT NULL REFERENCES schedules(id),
    slot        INTEGER NOT NULL,
    ts_utc      TEXT NOT NULL,
    bid_mw      REAL NOT NULL,
    PRIMARY KEY (schedule_id, slot)
);
CREATE TABLE IF NOT EXISTS forecast_scores (
    forecast_id         INTEGER PRIMARY KEY REFERENCES forecasts(id),
    scored_at           TEXT NOT NULL,
    coverage            REAL NOT NULL,      -- fraction of the day's slots with actuals
    mape_slot           REAL,
    mape_hour           REAL,               -- median forecast, complete UTC hours
    mape_hour_alpha     REAL,               -- the α-bid
    band_coverage       REAL,               -- fraction of slots inside the scaled P10-P90
    imbalance_usd       REAL,               -- median bid settled at RT - DA
    imbalance_alpha_usd REAL,
    da_cost_usd         REAL,
    isolf_mape_hour     REAL,               -- NYISO's pre-close forecast, same day
    isolf_imbalance_usd REAL
);
CREATE TABLE IF NOT EXISTS schedule_scores (
    schedule_id     INTEGER PRIMARY KEY REFERENCES schedules(id),
    scored_at       TEXT NOT NULL,
    coverage        REAL NOT NULL,
    mape_hour       REAL,
    imbalance_usd   REAL,
    da_cost_usd     REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    zone        TEXT NOT NULL,
    target_date TEXT NOT NULL,
    kind        TEXT NOT NULL,      -- mape | imbalance
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (zone, target_date, kind)
);
CREATE TABLE IF NOT EXISTS jobs (
    name         TEXT PRIMARY KEY,
    last_run_at  TEXT,
    last_status  TEXT,
    detail_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecasts_zone_target ON forecasts(zone, target_date);
CREATE INDEX IF NOT EXISTS idx_schedules_zone_target ON schedules(zone, target_date);
CREATE INDEX IF NOT EXISTS idx_alerts_zone_target ON alerts(zone, target_date);
"""


def now_text() -> str:
    return datetime.now(UTC).strftime(TS_FORMAT)


def ts_text(ts: pd.Timestamp | datetime) -> str:
    """UTC text form of a tz-aware timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}; the store holds UTC only")
    return t.tz_convert("UTC").strftime(TS_FORMAT)


def ts_column(values: pd.Series) -> pd.Series:
    """Text column (as stored) -> tz-aware UTC timestamps."""
    return pd.to_datetime(values, format=TS_FORMAT, utc=True)


def day_text(day: date | pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialise) the database; every call gets its own connection."""
    db_path = Path(path) if path is not None else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not db_path.exists()
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    if fresh:
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")  # space comes back after pruning
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _seed_zones(conn)
    return conn


def _seed_zones(conn: sqlite3.Connection) -> None:
    rows: list[tuple[str, int | None, int, int]] = [
        (z, ZONE_PTID[z], 0, i) for i, z in enumerate(ZONES)
    ]
    rows.append((NYCA, None, 1, len(ZONES)))
    conn.executemany(
        "INSERT OR IGNORE INTO zones (name, ptid, is_total, sort_order) VALUES (?, ?, ?, ?)", rows
    )
    conn.commit()


def zone_names(conn: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM zones ORDER BY sort_order")]


def upsert(
    conn: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]
) -> int:
    """``INSERT OR REPLACE`` many rows; returns how many were written."""
    rows = list(rows)
    if not rows:
        return 0
    marks = ",".join("?" * len(columns))
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({marks})", rows)
    return len(rows)


def upsert_frame(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    """Upsert a canonical frame; ``ts_utc`` / ``issued`` columns are converted to text."""
    if frame.empty:
        return 0
    f = frame.copy()
    if "ts_utc" in f.columns:
        f["ts_utc"] = f["ts_utc"].dt.tz_convert("UTC").dt.strftime(TS_FORMAT)
    if "issued" in f.columns:
        f["issued"] = pd.to_datetime(f["issued"]).dt.strftime("%Y-%m-%d")
    cols = list(f.columns)
    return upsert(conn, table, cols, f.itertuples(index=False, name=None))


def read_frame(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
    ts_cols: Sequence[str] = ("ts_utc",),
) -> pd.DataFrame:
    """``pd.read_sql_query`` with the stored text timestamps parsed back to UTC."""
    df = pd.read_sql_query(sql, conn, params=list(params))
    for c in ts_cols:
        if c in df.columns:
            df[c] = ts_column(df[c])
    return df
