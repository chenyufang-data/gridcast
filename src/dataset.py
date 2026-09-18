"""Processed dataset for offline modeling: fetch, normalize, resample, cache as pickles.

Everything lands under ``data/processed/`` (gitignored, derived from NYISO data that is
fetched at runtime; see data/README.md):

==============  =================================================  ===================
name            columns                                            grain
==============  =================================================  ===================
``load_slots``  ``ts_utc, zone, load_mw, coverage``                15-min, incl. NYCA
``rt_slots``    ``ts_utc, zone, p_rt, coverage``                   15-min RT LBMP
``da_hourly``   ``ts_utc, zone, p_da``                             hourly DA LBMP
``rt_hourly``   ``ts_utc, zone, p_rt_hourly``                      hourly integrated RT
``isolf``       ``issued, ts_utc, zone, isolf_mw``                 hourly, incl. NYCA
==============  =================================================  ===================

``ts_utc`` is always the interval start. Zones carry no NYCA price (prices are zonal).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.nyiso import PROJECT_ROOT, ArchiveClient, add_nyca, resample_slots
from src.config import BACKTEST_END, WARMUP_START

log = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NAMES = ("load_slots", "rt_slots", "da_hourly", "rt_hourly", "isolf")


def _path(name: str, out_dir: Path | None = None) -> Path:
    return (out_dir or PROCESSED_DIR) / f"{name}.pkl"


def build(
    client: ArchiveClient,
    start: date = WARMUP_START,
    end: date | None = None,
    out_dir: Path | None = None,
    names: tuple[str, ...] = NAMES,
) -> dict[str, pd.DataFrame]:
    """Fetch `start`..`end` for the requested tables, normalize, resample and save.

    `end` defaults to yesterday (ET), the last complete day. Returns the frames; missing
    days are logged, never invented.
    """
    end = end or (client.today() - timedelta(days=1))
    out: dict[str, pd.DataFrame] = {}
    if "load_slots" in names:
        load, missing = client.frame("pal", start, end)
        _report("pal", missing)
        out["load_slots"] = add_nyca(resample_slots(load, "load_mw"), "load_mw")
    if "rt_slots" in names:
        rt, missing = client.frame("realtime_zone", start, end)
        _report("realtime_zone", missing)
        out["rt_slots"] = resample_slots(rt, "p_rt")
    if "da_hourly" in names:
        out["da_hourly"], missing = client.frame("damlbmp_zone", start, end)
        _report("damlbmp_zone", missing)
    if "rt_hourly" in names:
        out["rt_hourly"], missing = client.frame("rtlbmp_zone", start, end)
        _report("rtlbmp_zone", missing)
    if "isolf" in names:
        # the file named D is posted on D-1 after the close; fetch one day past `end`
        # so the post-close reference for `end` exists as well
        out["isolf"], missing = client.frame("isolf", start, end + timedelta(days=1))
        _report("isolf", missing)
    for name, frame in out.items():
        path = _path(name, out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
        log.info("saved %s: %d rows -> %s", name, len(frame), path)
    return out


def _report(file_type: str, missing: list[date]) -> None:
    if missing:
        log.warning(
            "%s: %d day(s) missing: %s ... %s", file_type, len(missing), missing[0], missing[-1]
        )


def load(name: str, out_dir: Path | None = None) -> pd.DataFrame:
    path = _path(name, out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/backfill.py` first")
    return pd.read_pickle(path)


def load_all(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {name: load(name, out_dir) for name in NAMES}


def coverage_report(
    frames: dict[str, pd.DataFrame], start: date = WARMUP_START, end: date = BACKTEST_END
) -> pd.DataFrame:
    """Days per table with any data, and slots below full coverage, for a quick sanity check."""
    rows = []
    for name, frame in frames.items():
        if frame.empty:
            rows.append(
                {
                    "table": name,
                    "rows": 0,
                    "days": 0,
                    "first": None,
                    "last": None,
                    "partial_slots": 0,
                }
            )
            continue
        local_day = frame["ts_utc"].dt.tz_convert("America/New_York").dt.date
        partial = int((frame["coverage"] < 0.999).sum()) if "coverage" in frame.columns else 0
        rows.append(
            {
                "table": name,
                "rows": len(frame),
                "days": local_day.nunique(),
                "first": local_day.min(),
                "last": local_day.max(),
                "partial_slots": partial,
            }
        )
    return pd.DataFrame(rows)
