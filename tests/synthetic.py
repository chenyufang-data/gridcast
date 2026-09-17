"""Deterministic synthetic NYISO-shaped fixtures (no NYISO bytes anywhere in the repo).

Generates CSV text in the exact layouts observed in the public MIS archive (verified
2026-09-17 on the files of 2026-09-15; see data/README.md) so the normalizers can be
tested offline, including the DST and irregular-interval gotchas. The *truth* frames
(clean series in UTC) are what a correct normalizer must recover from the files.

Layout facts reproduced here
----------------------------
pal             ``"Time Stamp","Time Zone","Name","PTID","Load"``; strings quoted,
                numbers bare; ``MM/DD/YYYY HH:MM:SS`` local ET, tz ``EDT``/``EST``;
                the day file holds stamps 00:00:00 .. 23:55:00 (interval start); the 11
                zones per stamp in alphabetical order.
realtime_zone   ``"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses
                ($/MWHr)","Marginal Cost Congestion ($/MWHr)"``; header and strings
                quoted; the day file holds 00:05:00 .. next day 00:00:00 (interval END);
                15 names (11 zones + H Q, NPX, O H, PJM), no tz column.
damlbmp_zone    same columns, nothing quoted; ``MM/DD/YYYY HH:MM``; 24 stamps (23/25 on
                DST days; fall-back lists 01:00 twice, first = EDT); 15 names.
rtlbmp_zone     quoted like realtime; hourly ``MM/DD/YYYY HH:MM``, 00:00 .. 23:00.
isolf           ``"Time Stamp","Capitl",...,"West","NYISO"``; header and stamp quoted,
                integer MW; the file issued on day X covers X 00:00 .. X+5 23:00 hourly
                (144 rows); ``NYISO`` equals the sum of the rounded zone values.
RTD irregularity
                pal and realtime_zone share the same off-schedule stamps (e.g.
                04:04:18) on top of the 5-min grid: 296 stamps on 2026-09-15, not 288.

Unverified (⚠️): how isolf and rtlbmp list the fall-back hour (here: like damlbmp,
01:00 twice) and the internal layout of monthly zips (here: flat daily files).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARCHIVE_DIRS,
    EXTERNAL_PTID,
    ISOLF_COLUMNS,
    MARKET_TZ,
    PRICE_NAMES,
    ZONE_PTID,
    ZONES,
    daily_filename,
    monthly_zip_name,
)

ISOLF_HORIZON_DAYS = 6

# typical zone size in MW: the load curve is scale * (0.75 .. 1.05) over the day
ZONE_SCALE_MW: dict[str, float] = {
    "CAPITL": 1350, "CENTRL": 1660, "DUNWOD": 660, "GENESE": 1080, "HUD VL": 1080,
    "LONGIL": 2380, "MHK VL": 790, "MILLWD": 250, "N.Y.C.": 6100, "NORTH": 630,
    "WEST": 1730,
}  # fmt: skip
PRICE_PTID: dict[str, int] = {**ZONE_PTID, **EXTERNAL_PTID}
PRICE_HEADER = (
    '"Time Stamp","Name","PTID","LBMP ($/MWHr)","Marginal Cost Losses ($/MWHr)",'
    '"Marginal Cost Congestion ($/MWHr)"'
)
PAL_HEADER = '"Time Stamp","Time Zone","Name","PTID","Load"'
ISOLF_HEADER = '"Time Stamp",' + ",".join(f'"{c}"' for c in ISOLF_COLUMNS)
_FIVE_MIN = pd.Timedelta(minutes=5)


def _local_midnight_utc(day: date) -> pd.Timestamp:
    """00:00 local ET of `day` as a UTC timestamp (never ambiguous: DST shifts at 02:00)."""
    return pd.Timestamp(datetime(day.year, day.month, day.day), tz=MARKET_TZ).tz_convert("UTC")


def _stamp(ts_utc: pd.Timestamp, fmt: str) -> str:
    return ts_utc.tz_convert(MARKET_TZ).strftime(fmt)


def _tz_abbrev(ts_utc: pd.Timestamp) -> str:
    return str(ts_utc.tz_convert(MARKET_TZ).tzname())


def _repeat_utc(idx: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    """Each UTC stamp repeated k times (one row per zone/name), still tz-aware UTC."""
    return pd.DatetimeIndex(np.repeat(idx.tz_localize(None).to_numpy(), k), tz="UTC")


@dataclass(frozen=True)
class SyntheticNYISO:
    """Synthetic archive for the local-ET days `start` .. `end` (inclusive).

    `irregular=True` adds the off-schedule RTD stamps seen in the real files; the truth
    frames stay on the clean grid. Values depend only on (seed, absolute timestamp), so
    two generators with overlapping ranges and the same seed agree on every shared day.
    """

    start: date
    end: date
    seed: int = 0
    irregular: bool = True

    # ------------------------------------------------------------------ master grids
    def _by_day(
        self,
        stream: int,
        index: pd.DatetimeIndex,
        draw: Callable[[np.random.Generator, int], np.ndarray],
    ) -> np.ndarray:
        """Random draws seeded per (seed, stream, local day), so they do not depend on range."""
        local_days = index.tz_convert(MARKET_TZ).normalize()
        blocks = []
        for day_ts in local_days.unique():  # chronological: the index is sorted
            n = int((local_days == day_ts).sum())
            rng = np.random.default_rng([self.seed, stream, day_ts.date().toordinal()])
            blocks.append(draw(rng, n))
        return np.concatenate(blocks)

    @cached_property
    def _grid_utc(self) -> pd.DatetimeIndex:
        """5-min interval starts covering start .. end + isolf horizon (for isolf truth)."""
        lo = _local_midnight_utc(self.start)
        hi = _local_midnight_utc(self.end + timedelta(days=ISOLF_HORIZON_DAYS + 1))
        return pd.date_range(lo, hi, freq="5min", inclusive="left")

    @cached_property
    def _hours_utc(self) -> pd.DatetimeIndex:
        return self._grid_utc[::12]

    @cached_property
    def _load(self) -> np.ndarray:
        """(n_stamps, 11) MW on the 5-min grid: daily/weekly/seasonal shape + 1% noise."""
        local = self._grid_utc.tz_convert(MARKET_TZ)
        h = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
        doy = local.dayofyear.to_numpy()
        wd = local.weekday.to_numpy()
        daily = 0.5 * (1 - np.cos(2 * np.pi * (h - 4) / 24))  # trough 04:00, peak 16:00
        season = 0.10 * np.cos(2 * np.pi * (doy - 200) / 365.25)  # summer peak
        weekend = np.where(wd >= 5, -0.06, 0.0)
        base = 0.75 + 0.30 * daily + season + weekend
        noise = self._by_day(
            1, self._grid_utc, lambda rng, n: rng.normal(0.0, 0.01, (n, len(ZONES)))
        )
        scale = np.array([ZONE_SCALE_MW[z] for z in ZONES])
        return np.round(scale[None, :] * (base[:, None] + noise), 4)

    @cached_property
    def _da(self) -> np.ndarray:
        """(n_hours, 15) day-ahead LBMP, losses, congestion stacked as (n, 15, 3)."""
        local = self._hours_utc.tz_convert(MARKET_TZ)
        h = local.hour.to_numpy().astype(float)
        doy = local.dayofyear.to_numpy()
        daily = 0.5 * (1 - np.cos(2 * np.pi * (h - 4) / 24))
        season = 0.15 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        m = len(PRICE_NAMES)
        offset = np.random.default_rng([self.seed, 2]).normal(0.0, 4.0, size=m)  # per-name level

        def draw(rng: np.random.Generator, n: int) -> np.ndarray:
            noise = rng.normal(0.0, 2.0, (n, m))
            losses = rng.normal(0.5, 0.6, (n, m))
            congestion = np.where(rng.random((n, m)) < 0.2, -np.abs(rng.normal(0, 3, (n, m))), 0.0)
            return np.stack([noise, losses, congestion], axis=-1)

        rnd = self._by_day(2, self._hours_utc, draw)
        lbmp = 30 + 25 * daily[:, None] * (1 + season[:, None]) + offset[None, :] + rnd[:, :, 0]
        return np.round(np.stack([lbmp, rnd[:, :, 1], rnd[:, :, 2]], axis=-1), 2)

    @cached_property
    def _rt(self) -> np.ndarray:
        """(n_stamps, 15, 3) real-time prices for the interval ENDING at grid stamp + 5 min."""
        ends = self._grid_utc + _FIVE_MIN
        hour_pos = np.searchsorted(self._hours_utc.to_numpy(), self._grid_utc.to_numpy(), "right")
        hour_pos = hour_pos - 1
        m = len(PRICE_NAMES)
        da = self._da[hour_pos]

        def draw(rng: np.random.Generator, n: int) -> np.ndarray:
            spikes = np.where(rng.random((n, m)) < 0.03, rng.exponential(80.0, (n, m)), 0.0)
            return np.stack(
                [rng.normal(0.0, 6.0, (n, m)), rng.normal(0.0, 0.3, (n, m)), spikes], -1
            )

        rnd = self._by_day(3, ends - _FIVE_MIN, draw)  # keyed by the interval's start day
        spikes = rnd[:, :, 2]
        lbmp = da[:, :, 0] + rnd[:, :, 0] + spikes
        losses = da[:, :, 1] + rnd[:, :, 1]
        congestion = da[:, :, 2] + np.where(spikes > 0, -spikes * 0.5, 0.0)
        return np.round(np.stack([lbmp, losses, congestion], axis=-1), 2)

    # ------------------------------------------------------------------ truth frames
    def load_5min(self) -> pd.DataFrame:
        """Clean 5-min load, long format: ts_utc (interval start), zone, load_mw."""
        idx = self._grid_utc[self._grid_utc < _local_midnight_utc(self.end + timedelta(days=1))]
        n = len(idx)
        return pd.DataFrame(
            {
                "ts_utc": _repeat_utc(idx, len(ZONES)),
                "zone": np.tile(np.array(ZONES, dtype=object), n),
                "load_mw": self._load[:n].ravel(),
            }
        )

    def rt_prices_5min(self) -> pd.DataFrame:
        """Clean 5-min RT prices, long: ts_utc (interval END), name, lbmp, losses, congestion."""
        hi = _local_midnight_utc(self.end + timedelta(days=1))
        idx = self._grid_utc[self._grid_utc < hi]
        n, m = len(idx), len(PRICE_NAMES)
        vals = self._rt[:n].reshape(n * m, 3)
        return pd.DataFrame(
            {
                "ts_utc": _repeat_utc(idx + _FIVE_MIN, m),
                "name": np.tile(np.array(PRICE_NAMES, dtype=object), n),
                "lbmp": vals[:, 0],
                "losses": vals[:, 1],
                "congestion": vals[:, 2],
            }
        )

    def da_prices_hourly(self) -> pd.DataFrame:
        """Clean hourly DA prices, long: ts_utc (hour start), name, lbmp, losses, congestion."""
        hi = _local_midnight_utc(self.end + timedelta(days=1))
        idx = self._hours_utc[self._hours_utc < hi]
        n, m = len(idx), len(PRICE_NAMES)
        vals = self._da[:n].reshape(n * m, 3)
        return pd.DataFrame(
            {
                "ts_utc": _repeat_utc(idx, m),
                "name": np.tile(np.array(PRICE_NAMES, dtype=object), n),
                "lbmp": vals[:, 0],
                "losses": vals[:, 1],
                "congestion": vals[:, 2],
            }
        )

    def isolf_truth(self, issue_day: date) -> pd.DataFrame:
        """The ISO forecast issued on `issue_day`, long: ts_utc (hour start), zone, isolf_mw.

        Includes the NYCA row (sum of the rounded zones). Forecast = hourly mean of the
        truth load times a per-issue, per-zone bias (2%) plus hourly noise (1%).
        """
        wide = self._isolf_wide(issue_day)
        rows = wide.melt(id_vars="ts_utc", var_name="zone", value_name="isolf_mw")
        rows["zone"] = rows["zone"].map(ISOLF_COLUMNS)
        return rows.sort_values(["ts_utc", "zone"], ignore_index=True)

    # ------------------------------------------------------------------ day slices
    def _day_bounds(self, day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        return _local_midnight_utc(day), _local_midnight_utc(day + timedelta(days=1))

    def offgrid_stamps(self, day: date) -> pd.DatetimeIndex:
        """Off-schedule RTD stamps (UTC) inside `day`; shared by pal and realtime_zone."""
        lo, hi = self._day_bounds(day)
        if not self.irregular:
            return pd.DatetimeIndex([], tz="UTC")
        rng = np.random.default_rng([self.seed, 4, day.toordinal()])
        k = int(rng.integers(0, 10))
        span = int((hi - lo).total_seconds())
        secs = np.unique(rng.integers(1, span, size=k))
        secs = secs[secs % 300 != 0]
        return pd.DatetimeIndex([lo + pd.Timedelta(seconds=int(s)) for s in secs], tz="UTC")

    def _interp(self, values: np.ndarray, at: pd.DatetimeIndex) -> np.ndarray:
        """Linear interpolation of a (n_stamps, ...) grid array at arbitrary UTC times."""
        x = self._grid_utc.asi8.astype(float)
        xq = at.asi8.astype(float)
        flat = values.reshape(len(values), -1)
        out = np.column_stack([np.interp(xq, x, flat[:, j]) for j in range(flat.shape[1])])
        return out.reshape((len(at), *values.shape[1:]))

    def _pal_rows(self, day: date) -> list[tuple[pd.Timestamp, np.ndarray]]:
        lo, hi = self._day_bounds(day)
        mask = (self._grid_utc >= lo) & (self._grid_utc < hi)
        grid = [(ts, row) for ts, row in zip(self._grid_utc[mask], self._load[mask], strict=True)]
        off = self.offgrid_stamps(day)
        if len(off):
            rng = np.random.default_rng([self.seed, 5, day.toordinal()])
            vals = self._interp(self._load, off) * (
                1 + rng.normal(0, 0.002, (len(off), len(ZONES)))
            )
            grid += [(ts, np.round(row, 4)) for ts, row in zip(off, vals, strict=True)]
        return sorted(grid, key=lambda item: item[0])

    def pal_csv(self, day: date) -> str:
        lines = [PAL_HEADER]
        for ts, row in self._pal_rows(day):
            stamp, tz = _stamp(ts, "%m/%d/%Y %H:%M:%S"), _tz_abbrev(ts)
            for zone, mw in zip(ZONES, row, strict=True):
                lines.append(f'"{stamp}","{tz}","{zone}",{ZONE_PTID[zone]},{mw:.4f}')
        return "\n".join(lines) + "\n"

    def _rt_rows(self, day: date) -> list[tuple[pd.Timestamp, np.ndarray]]:
        lo, hi = self._day_bounds(day)
        ends = self._grid_utc + _FIVE_MIN
        mask = (ends > lo) & (ends <= hi)
        grid = [(ts, row) for ts, row in zip(ends[mask], self._rt[mask], strict=True)]
        off = self.offgrid_stamps(day)
        if len(off):
            # the interval ending at an off-grid stamp: price of the grid interval it falls in
            vals = self._interp(self._rt, off - _FIVE_MIN)
            grid += [(ts, np.round(row, 2)) for ts, row in zip(off, vals, strict=True)]
        return sorted(grid, key=lambda item: item[0])

    @staticmethod
    def _price_line(stamp: str, name: str, v: np.ndarray, quoted: bool) -> str:
        nums = f"{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}"
        if quoted:
            return f'"{stamp}","{name}",{PRICE_PTID[name]},{nums}'
        return f"{stamp},{name},{PRICE_PTID[name]},{nums}"

    def realtime_zone_csv(self, day: date) -> str:
        lines = [PRICE_HEADER]
        for ts, block in self._rt_rows(day):
            stamp = _stamp(ts, "%m/%d/%Y %H:%M:%S")
            for name, v in zip(PRICE_NAMES, block, strict=True):
                lines.append(self._price_line(stamp, name, v, quoted=True))
        return "\n".join(lines) + "\n"

    def _hour_blocks(self, day: date, values: np.ndarray) -> list[tuple[pd.Timestamp, np.ndarray]]:
        lo, hi = self._day_bounds(day)
        mask = (self._hours_utc >= lo) & (self._hours_utc < hi)
        return list(zip(self._hours_utc[mask], values[mask], strict=True))

    def damlbmp_zone_csv(self, day: date) -> str:
        lines = [PRICE_HEADER.replace('"', "")]
        for ts, block in self._hour_blocks(day, self._da):
            stamp = _stamp(ts, "%m/%d/%Y %H:%M")
            for name, v in zip(PRICE_NAMES, block, strict=True):
                lines.append(self._price_line(stamp, name, v, quoted=False))
        return "\n".join(lines) + "\n"

    @cached_property
    def _rt_hourly(self) -> np.ndarray:
        """Hourly integrated RT price = mean of the 12 grid intervals ending in the hour."""
        n_hours = len(self._hours_utc)
        return np.round(
            self._rt[: n_hours * 12].reshape(n_hours, 12, len(PRICE_NAMES), 3).mean(axis=1), 2
        )

    def rtlbmp_zone_csv(self, day: date) -> str:
        lines = [PRICE_HEADER]
        for ts, block in self._hour_blocks(day, self._rt_hourly):
            stamp = _stamp(ts, "%m/%d/%Y %H:%M")
            for name, v in zip(PRICE_NAMES, block, strict=True):
                lines.append(self._price_line(stamp, name, v, quoted=True))
        return "\n".join(lines) + "\n"

    def _isolf_wide(self, issue_day: date) -> pd.DataFrame:
        lo = _local_midnight_utc(issue_day)
        hi = _local_midnight_utc(issue_day + timedelta(days=ISOLF_HORIZON_DAYS))
        mask = (self._hours_utc >= lo) & (self._hours_utc < hi)
        hours = self._hours_utc[mask]
        n_hours = len(self._hours_utc)
        hourly_load = self._load[: n_hours * 12].reshape(n_hours, 12, len(ZONES)).mean(axis=1)[mask]
        rng = np.random.default_rng([self.seed, 6, issue_day.toordinal()])
        bias = rng.normal(0.0, 0.02, size=len(ZONES))
        noise = rng.normal(0.0, 0.01, size=hourly_load.shape)
        zones = np.rint(hourly_load * (1 + bias[None, :] + noise)).astype(int)
        wide = pd.DataFrame(zones, columns=[c for c in ISOLF_COLUMNS if c != "NYISO"])
        wide["NYISO"] = zones.sum(axis=1)
        wide.insert(0, "ts_utc", hours)
        return wide

    def isolf_csv(self, issue_day: date) -> str:
        wide = self._isolf_wide(issue_day)
        lines = [ISOLF_HEADER]
        for ts, *vals in wide.itertuples(index=False):
            stamp = _stamp(ts, "%m/%d/%Y %H:%M")
            lines.append(f'"{stamp}",' + ",".join(str(int(v)) for v in vals))
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ archive on disk
    def csv_for(self, file_type: str, day: date) -> str:
        writers = {
            "pal": self.pal_csv,
            "damlbmp_zone": self.damlbmp_zone_csv,
            "realtime_zone": self.realtime_zone_csv,
            "rtlbmp_zone": self.rtlbmp_zone_csv,
            "isolf": self.isolf_csv,
        }
        return writers[file_type](day)

    def days(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]

    def write_archive(
        self,
        root: Path,
        file_types: Iterable[str] = (
            "pal",
            "damlbmp_zone",
            "realtime_zone",
            "rtlbmp_zone",
            "isolf",
        ),
        monthly_zips: bool = False,
    ) -> list[Path]:
        """Write daily files as `<root>/<archive dir>/<YYYYMMDD><type>.csv`.

        With `monthly_zips`, every calendar month fully inside start..end also gets its
        `<YYYYMM>01<type>_csv.zip` holding the month's daily files (flat, ⚠️ unverified).
        """

        written: list[Path] = []
        for file_type in file_types:
            folder = root / ARCHIVE_DIRS[file_type]
            folder.mkdir(parents=True, exist_ok=True)
            by_month: dict[date, list[tuple[str, str]]] = {}
            for day in self.days():
                name, text = daily_filename(file_type, day), self.csv_for(file_type, day)
                path = folder / name
                path.write_text(text, encoding="utf-8", newline="")
                written.append(path)
                by_month.setdefault(day.replace(day=1), []).append((name, text))
            if not monthly_zips:
                continue
            for month_start, members in by_month.items():
                nxt = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
                if month_start < self.start or nxt - timedelta(days=1) > self.end:
                    continue  # partial month: the real archive would serve daily files
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, text in members:
                        zf.writestr(name, text)
                path = folder / monthly_zip_name(file_type, month_start)
                path.write_bytes(buf.getvalue())
                written.append(path)
        return written
