"""NYISO public archive client: fetch, cache, normalize, resample.

Sources, terms and the verified file layouts are documented in data/README.md. Policy:
fetch-not-redistribute. The cache under NYISO_CACHE_DIR mirrors the archive path
(``<cache>/<dir>/<file>``) and is never committed.

Archive behaviour this module relies on (verified 2026-09-17):

- A day's file is immutable once the day is over. Daily files
  (``YYYYMMDD<type>.csv``) exist for roughly the last 11 days only.
- The monthly zip (``YYYYMM01<type>_csv.zip``, flat daily files inside) is final on the
  1st of the next month and, for the current month, rebuilt every morning (~05:00 ET)
  with every day so far, including the partial current day.
- Files for a future day (tomorrow's ``isolf``/``damlbmp``) appear during the previous
  day and may still change, so days >= today (ET) are fetched fresh and never cached.

Canonical frames (all timestamps tz-aware UTC):

=================  ==============================================================
``normalize_pal``        ``ts_utc`` (interval start), ``ts_end_utc``, ``zone``, ``load_mw``
``normalize_realtime``   ``ts_utc`` (interval start), ``ts_end_utc``, ``zone``, ``p_rt``
``normalize_damlbmp``    ``ts_utc`` (hour start), ``zone``, ``p_da``
``normalize_rtlbmp``     ``ts_utc`` (hour start), ``zone``, ``p_rt_hourly``
``normalize_isolf``      ``issued``, ``ts_utc`` (hour start), ``zone``, ``isolf_mw`` (+ NYCA)
``resample_slots``       ``ts_utc`` (slot start), ``zone``, value, ``coverage``
=================  ==============================================================
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    ARCHIVE_DIRS,
    ISOLF_COLUMNS,
    MARKET_TZ,
    NYCA,
    RT_INTERVAL_MIN,
    ZONE_PTID,
    ZONES,
    archive_url,
    daily_filename,
    monthly_zip_name,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("NYISO_CACHE_DIR") or PROJECT_ROOT / "data" / "cache")

PTID_ZONE: dict[int, str] = {ptid: zone for zone, ptid in ZONE_PTID.items()}
_TZ_OFFSET_HOURS = {"EDT": 4, "EST": 5}  # hours behind UTC
_NOMINAL = pd.Timedelta(minutes=RT_INTERVAL_MIN)
_PRICE_COL = "LBMP ($/MWHr)"
_NS = 10**9


class NotAvailable(Exception):
    """The archive has no such file (HTTP 404, or a day missing from the monthly zip)."""


# ---------------------------------------------------------------------------- fetching
def today_et() -> date:
    return datetime.now(MARKET_TZ).date()


def http_fetch(
    url: str, timeout: float = 60.0, retries: int = 3, backoff: float = 2.0
) -> bytes | None:
    """GET `url`; None on 404; retries connection errors, timeouts and 5xx with backoff."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            log.warning("fetch %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
        else:
            if resp.status_code == 404:
                return None
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp.content
            last_error = requests.HTTPError(f"{resp.status_code} for {url}")
            log.warning(
                "fetch %s returned %d (attempt %d/%d)", url, resp.status_code, attempt, retries
            )
        if attempt < retries:
            time.sleep(backoff**attempt)
    assert last_error is not None
    raise last_error


class ArchiveClient:
    """Cached access to the NYISO MIS archive, one day of one file type at a time.

    `fetch(url)` returns bytes or None (404); the default is :func:`http_fetch`. Tests
    inject a function that reads a synthetic archive from disk. `today` is injectable
    for the same reason: it decides which months are final and which days are cached.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        fetch: Callable[[str], bytes | None] = http_fetch,
        today: Callable[[], date] = today_et,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.fetch = fetch
        self.today = today

    # --- cache paths ---------------------------------------------------------------
    def cache_path(self, file_type: str, name: str) -> Path:
        return self.cache_dir / ARCHIVE_DIRS[file_type] / name

    def _download(self, file_type: str, name: str) -> bytes | None:
        url = archive_url(file_type, name)
        log.info("fetching %s", url)
        return self.fetch(url)

    def _download_to_cache(
        self, file_type: str, name: str, cache_as: str | None = None
    ) -> Path | None:
        data = self._download(file_type, name)
        if data is None:
            return None
        path = self.cache_path(file_type, cache_as or name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    # --- month zips ------------------------------------------------------------------
    @staticmethod
    def _month_start(day: date) -> date:
        return day.replace(day=1)

    @staticmethod
    def _next_month(month_start: date) -> date:
        return (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)

    def _month_is_final(self, month_start: date) -> bool:
        """The archive rewrites the final zip early on the 1st; trust it from the 2nd on."""
        return self.today() >= self._next_month(month_start) + timedelta(days=1)

    @staticmethod
    def _read_from_zip(path: Path, member: str) -> str | None:
        try:
            with zipfile.ZipFile(path) as zf:
                if member not in zf.namelist():
                    return None
                return zf.read(member).decode("utf-8")
        except zipfile.BadZipFile:
            log.warning("corrupt zip %s removed from cache", path)
            path.unlink(missing_ok=True)
            return None

    def _from_month_zip(self, file_type: str, day: date) -> str | None:
        """Serve `day` from the month zip: cached final zip, else cached/refreshed partial."""
        month_start = self._month_start(day)
        zip_name = monthly_zip_name(file_type, month_start)
        member = daily_filename(file_type, day)
        final_path = self.cache_path(file_type, zip_name)
        partial_path = self.cache_path(file_type, zip_name + ".partial")

        if self._month_is_final(month_start):
            if not final_path.exists():
                if self._download_to_cache(file_type, zip_name) is None:
                    return None
                partial_path.unlink(missing_ok=True)
            return self._read_from_zip(final_path, member)

        if partial_path.exists():
            text = self._read_from_zip(partial_path, member)
            if text is not None:
                return text
        # the partial zip is rebuilt every morning; refresh it once and look again
        if self._download_to_cache(file_type, zip_name, cache_as=partial_path.name) is None:
            return None
        return self._read_from_zip(partial_path, member)

    # --- public API ----------------------------------------------------------------
    def day_csv(self, file_type: str, day: date) -> str:
        """Raw CSV text of one day's file. Raises :class:`NotAvailable` if the archive lacks it."""
        if file_type not in ARCHIVE_DIRS:
            raise ValueError(f"unknown NYISO file type {file_type!r}")
        name = daily_filename(file_type, day)

        if day >= self.today():  # still changing: fetch fresh, never cache
            data = self._download(file_type, name)
            if data is None:
                raise NotAvailable(f"{name} not published yet")
            return data.decode("utf-8")

        daily_path = self.cache_path(file_type, name)
        if daily_path.exists():
            return daily_path.read_text(encoding="utf-8")

        if self._month_is_final(self._month_start(day)):
            text = self._from_month_zip(file_type, day)
        else:
            partial_path = self.cache_path(
                file_type, monthly_zip_name(file_type, day.replace(day=1)) + ".partial"
            )
            text = self._read_from_zip(partial_path, name) if partial_path.exists() else None
            if text is None:
                path = self._download_to_cache(file_type, name)
                if path is not None:
                    return path.read_text(encoding="utf-8")
                text = self._from_month_zip(file_type, day)
        if text is None:
            raise NotAvailable(f"{name} is not in the archive")
        return text

    def frame(self, file_type: str, start: date, end: date) -> tuple[pd.DataFrame, list[date]]:
        """Normalized frame for `start`..`end` (inclusive) plus the list of days not available."""
        parts: list[pd.DataFrame] = []
        missing: list[date] = []
        day = start
        while day <= end:
            try:
                text = self.day_csv(file_type, day)
            except NotAvailable:
                log.warning("%s: no data for %s", file_type, day)
                missing.append(day)
            else:
                parts.append(normalize(file_type, text, day))
            day += timedelta(days=1)
        if not parts:
            return normalize(file_type, "", start).iloc[0:0], missing
        return pd.concat(parts, ignore_index=True), missing


# ---------------------------------------------------------------------------- parsing helpers
def _read(text: str, required: Iterable[str]) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame(columns=list(required))
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"NYISO file lacks columns {missing}; has {list(df.columns)}")
    return df


def _zones_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 11 load zones (by PTID) and name them canonically; drops external proxies."""
    out = df[df["PTID"].isin(PTID_ZONE)].copy()
    out["zone"] = out["PTID"].map(PTID_ZONE)
    return out


def _localize_by_order(naive: pd.Series, group: pd.Series | None) -> pd.Series:
    """Local ET stamps without offsets -> UTC, resolving the fall-back hour by row order.

    Rows are chronological within a file, so on the fall-back day the local clock runs
    00:00 .. 01:55 (EDT), then 01:00 .. (EST). The first non-increasing step per group
    marks the switch: ambiguous stamps before it are daylight time, after it standard.
    """
    prev = naive.groupby(group).shift() if group is not None else naive.shift()
    reset = naive <= prev
    after = reset.groupby(group).cummax() if group is not None else reset.cummax()
    ambiguous = (~after).to_numpy()
    local = naive.dt.tz_localize(MARKET_TZ, ambiguous=ambiguous, nonexistent="raise")
    return local.dt.tz_convert("UTC")


def _ns(ts: pd.Series) -> np.ndarray:
    """tz-aware timestamps -> int64 nanoseconds since the epoch."""
    return (
        ts.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .to_numpy()
        .astype("datetime64[ns]")
        .astype("int64")
    )


# ---------------------------------------------------------------------------- normalizers
def normalize_pal(text: str) -> pd.DataFrame:
    """5-min actual load -> ``ts_utc, ts_end_utc, zone, load_mw`` (interval start/end)."""
    raw = _read(text, ["Time Stamp", "Time Zone", "Name", "PTID", "Load"])
    df = _zones_only(raw)
    offset = df["Time Zone"].map(_TZ_OFFSET_HOURS)
    if offset.isna().any():
        raise ValueError(
            f"unknown time zone labels: {sorted(df.loc[offset.isna(), 'Time Zone'].unique())}"
        )
    naive = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    out = pd.DataFrame(
        {
            "ts_utc": (naive + pd.to_timedelta(offset.astype(int), unit="h")).dt.tz_localize("UTC"),
            "zone": df["zone"].to_numpy(),
            "load_mw": df["Load"].astype(float).to_numpy(),
        }
    )
    out = out.drop_duplicates(["ts_utc", "zone"], keep="last").sort_values(["zone", "ts_utc"])
    nxt = out.groupby("zone")["ts_utc"].shift(-1)
    cap = out["ts_utc"] + _NOMINAL
    out["ts_end_utc"] = nxt.where(nxt < cap, cap)
    return (
        out[["ts_utc", "ts_end_utc", "zone", "load_mw"]]
        .sort_values(["ts_utc", "zone"])
        .reset_index(drop=True)
    )


def normalize_realtime(text: str) -> pd.DataFrame:
    """5-min RT LBMP -> ``ts_utc, ts_end_utc, zone, p_rt``.

    The file stamps the END of each RTD interval; the start is the previous stamp of
    the same zone, capped at one nominal interval so gaps show up as missing coverage.
    """
    raw = _read(text, ["Time Stamp", "Name", "PTID", _PRICE_COL])
    df = _zones_only(raw).reset_index(drop=True)
    naive = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    out = pd.DataFrame(
        {
            "ts_end_utc": _localize_by_order(naive, df["zone"]),
            "zone": df["zone"].to_numpy(),
            "p_rt": df[_PRICE_COL].astype(float).to_numpy(),
        }
    )
    out = out.drop_duplicates(["ts_end_utc", "zone"], keep="last").sort_values(
        ["zone", "ts_end_utc"]
    )
    prev = out.groupby("zone")["ts_end_utc"].shift(1)
    cap = out["ts_end_utc"] - _NOMINAL
    out["ts_utc"] = prev.where(prev > cap, cap)
    return (
        out[["ts_utc", "ts_end_utc", "zone", "p_rt"]]
        .sort_values(["ts_utc", "zone"])
        .reset_index(drop=True)
    )


def _normalize_hourly_lbmp(text: str, value_name: str) -> pd.DataFrame:
    raw = _read(text, ["Time Stamp", "Name", "PTID", _PRICE_COL])
    df = _zones_only(raw).reset_index(drop=True)
    naive = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
    out = pd.DataFrame(
        {
            "ts_utc": _localize_by_order(naive, df["zone"]),
            "zone": df["zone"].to_numpy(),
            value_name: df[_PRICE_COL].astype(float).to_numpy(),
        }
    )
    return (
        out.drop_duplicates(["ts_utc", "zone"], keep="last")
        .sort_values(["ts_utc", "zone"])
        .reset_index(drop=True)
    )


def normalize_damlbmp(text: str) -> pd.DataFrame:
    """Hourly day-ahead LBMP -> ``ts_utc (hour start), zone, p_da``."""
    return _normalize_hourly_lbmp(text, "p_da")


def normalize_rtlbmp(text: str) -> pd.DataFrame:
    """Hourly integrated real-time LBMP -> ``ts_utc (hour start), zone, p_rt_hourly``."""
    return _normalize_hourly_lbmp(text, "p_rt_hourly")


def normalize_isolf(text: str, issued: date) -> pd.DataFrame:
    """ISO load forecast file (wide, 6 days) -> ``issued, ts_utc, zone, isolf_mw`` incl. NYCA.

    `issued` is the day the file is named for (its first forecast day); the file is
    posted on the morning of the day before, after the DAM closes (data/README.md).
    """
    raw = _read(text, ["Time Stamp", *ISOLF_COLUMNS])
    naive = pd.to_datetime(raw["Time Stamp"], format="%m/%d/%Y %H:%M")
    ts_utc = _localize_by_order(naive, None)
    wide = raw[list(ISOLF_COLUMNS)].rename(columns=ISOLF_COLUMNS).astype(float)
    wide.insert(0, "ts_utc", ts_utc)
    long = wide.melt(id_vars="ts_utc", var_name="zone", value_name="isolf_mw")
    long.insert(0, "issued", pd.Timestamp(issued).date())
    return (
        long.drop_duplicates(["ts_utc", "zone"], keep="last")
        .sort_values(["ts_utc", "zone"])
        .reset_index(drop=True)[["issued", "ts_utc", "zone", "isolf_mw"]]
    )


def normalize(file_type: str, text: str, day: date) -> pd.DataFrame:
    if file_type == "pal":
        return normalize_pal(text)
    if file_type == "realtime_zone":
        return normalize_realtime(text)
    if file_type == "damlbmp_zone":
        return normalize_damlbmp(text)
    if file_type == "rtlbmp_zone":
        return normalize_rtlbmp(text)
    if file_type == "isolf":
        return normalize_isolf(text, issued=day)
    raise ValueError(f"unknown NYISO file type {file_type!r}")


# ---------------------------------------------------------------------------- resampling
def resample_slots(df: pd.DataFrame, value_col: str, minutes: int = 15) -> pd.DataFrame:
    """Time-weighted mean of interval observations on a fixed UTC grid.

    Input rows are observations valid on ``[ts_utc, ts_end_utc)``; they may be
    off-schedule, overlapping slot boundaries, or leave gaps. Output: ``ts_utc`` (slot
    start), ``zone``, `value_col`, ``coverage`` (fraction of the slot with data; 1.0 on
    a clean day, less where stamps are missing). On the regular 5-min grid this equals
    the plain mean of the three values in each 15-min slot.
    """
    if df.empty:
        return pd.DataFrame(
            {
                "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "zone": [],
                value_col: [],
                "coverage": [],
            }
        )
    step = minutes * 60 * _NS
    s, e = _ns(df["ts_utc"]), _ns(df["ts_end_utc"])
    v = df[value_col].to_numpy(dtype=float)
    z = df["zone"].to_numpy(dtype=object)
    keep = e > s
    s, e, v, z = s[keep], e[keep], v[keep], z[keep]
    slots, zones, vals, weights = [], [], [], []
    while len(s):
        boundary = (s // step + 1) * step
        cut = np.minimum(e, boundary)
        slots.append(s // step * step)
        zones.append(z)
        vals.append(v)
        weights.append((cut - s).astype(float))
        more = e > boundary
        s, e, v, z = boundary[more], e[more], v[more], z[more]
    w = np.concatenate(weights)
    pieces = pd.DataFrame(
        {
            "slot": np.concatenate(slots),
            "zone": np.concatenate(zones),
            "wv": np.concatenate(vals) * w,
            "w": w,
        }
    )
    g = pieces.groupby(["slot", "zone"], sort=True)[["wv", "w"]].sum().reset_index()
    return pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(g["slot"], unit="ns", utc=True),
            "zone": g["zone"],
            value_col: g["wv"] / g["w"],
            "coverage": g["w"] / step,
        }
    )


def add_nyca(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Append the statewide total (zone ``NYCA``) where all 11 zones are present.

    Sums `value_col` per timestamp; a ``coverage`` column, if present, becomes the
    minimum over zones. Rows already labelled NYCA are replaced.
    """
    base = df[df["zone"] != NYCA]
    wide = base.pivot(index="ts_utc", columns="zone", values=value_col)
    have = [z for z in ZONES if z in wide.columns]
    complete = (
        wide[have].notna().all(axis=1)
        if len(have) == len(ZONES)
        else pd.Series(False, index=wide.index)
    )
    total = pd.DataFrame(
        {
            "ts_utc": wide.index[complete],
            "zone": NYCA,
            value_col: wide.loc[complete, have].sum(axis=1).to_numpy(),
        }
    )
    if "coverage" in base.columns:
        cov = base.pivot(index="ts_utc", columns="zone", values="coverage")
        total["coverage"] = cov.loc[complete, have].min(axis=1).to_numpy()
    extra = [c for c in base.columns if c not in total.columns]
    for col in extra:  # e.g. `issued`: constant per timestamp, take the first
        total[col] = base.groupby("ts_utc")[col].first().reindex(total["ts_utc"]).to_numpy()
    return (
        pd.concat([base, total[base.columns]], ignore_index=True)
        .sort_values(["ts_utc", "zone"])
        .reset_index(drop=True)
    )
