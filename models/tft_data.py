"""Data side of the Temporal Fusion Transformer, without torch.

Everything the served model needs besides the network itself: the per-zone grid with the
known inputs (:func:`prepare_zone`), the leakage-safe load series (:func:`series_asof`),
the encoder / decoder windows (:func:`make_sample`, :func:`sample_arrays`), the quantile
post-processing (:func:`quantile_frame`, :func:`to_result_rows`) and the training-set
helpers (:func:`aggregate_zone`, :func:`block_bootstrap`, :func:`periods`). The backend
imports this module and :mod:`models.tft_onnx`; :mod:`models.tft` (torch) re-exports it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from models.features import (
    SLOT,
    cutoff_for,
    day_grid,
    is_holiday,
    local_fields,
    local_midnight_utc,
    regularize,
    repair,
    training_targets,
)
from src.config import SLOTS_PER_HOUR

SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR
DEC_LEN = 19 * SLOTS_PER_HOUR + SLOTS_PER_DAY + SLOTS_PER_HOUR  # 05:00 D-1 .. end of a 25-h D
QUANTILES = (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9)
WEATHER_COLS = ["temp_h", "app_h", "dew_h", "rh_h", "cloud_h", "wind_h", "rad_h", "temp_h_prev3"]
CALENDAR_COLS = ["tod_sin", "tod_cos", "is_holiday", "holiday_tomorrow", "rel_pos"]

ONNX_INPUTS = ("x_enc", "dow_enc", "x_dec", "dow_dec", "zone")
ONNX_FORMAT = "gridcast-tft-onnx-1"


# ---------------------------------------------------------------------------- data
@dataclass
class ZoneData:
    """One zone on a complete 15-min UTC grid that extends to the end of the last target."""

    zone: str
    index: pd.DatetimeIndex
    raw: np.ndarray  # coverage-masked load (NaN = missing), NaN beyond the data
    known: np.ndarray  # [T, n_known] float32: calendar + weather (rel_pos filled per sample)
    dow: np.ndarray  # [T] int64
    dates: np.ndarray  # [T] datetime64[D] local date of each slot
    tod: np.ndarray  # [T] int64 wall-clock quarter hour
    scale: float = 1.0
    known_cols: list[str] = field(default_factory=list)


def prepare_zone(
    zone: str, zone_slots: pd.DataFrame, hourly_feats: pd.DataFrame | None, last_target: date
) -> ZoneData:
    """Grid one zone's slots, attach the known inputs, extend the grid to `last_target`."""
    series = regularize(zone_slots)
    end = local_midnight_utc(last_target + timedelta(days=1))
    index = pd.date_range(series.index.min(), max(series.index.max(), end - SLOT), freq=SLOT)
    raw = series.reindex(index).to_numpy(dtype=float)
    fields = local_fields(pd.Series(index, index=index))
    tod = fields["tod"].to_numpy()
    k = pd.DataFrame(index=index)
    k["tod_sin"] = np.sin(2 * np.pi * tod / SLOTS_PER_DAY)
    k["tod_cos"] = np.cos(2 * np.pi * tod / SLOTS_PER_DAY)
    k["is_holiday"] = is_holiday(fields["date"]).to_numpy()
    k["holiday_tomorrow"] = is_holiday(fields["date"] + pd.Timedelta(days=1)).to_numpy()
    k["rel_pos"] = 0.0
    cols = list(CALENDAR_COLS)
    if hourly_feats is not None and not hourly_feats.empty:
        w = hourly_feats.set_index("hour_utc")
        present = [c for c in WEATHER_COLS if c in w.columns]
        wk = w[present].reindex(index.floor("h"))
        wk = wk.ffill(limit=8 * SLOTS_PER_HOUR).bfill(limit=8 * SLOTS_PER_HOUR)
        wk = wk.fillna(wk.mean())
        for c in present:
            k[c] = wk[c].to_numpy()
        cols += present
    return ZoneData(
        zone=zone,
        index=index,
        raw=raw,
        known=k[cols].to_numpy(dtype=np.float32),
        dow=fields["date"].dt.dayofweek.to_numpy(),
        dates=fields["date"].to_numpy().astype("datetime64[D]"),
        tod=tod.astype(np.int64),
        known_cols=cols,
    )


def aggregate_zone(name: str, members: list[ZoneData], weights: dict[str, float]) -> ZoneData:
    """A synthetic zone: the sum of `members` (loads add exactly), weather = load-weighted mean.

    Used as extra training series for the global model; never forecast itself.
    """
    first = members[0]
    w = np.array([weights.get(m.zone, 1.0) for m in members], dtype=float)
    w = w / w.sum()
    raw = np.zeros(len(first.index))
    known = first.known.copy()
    weather_j = [j for j, c in enumerate(first.known_cols) if c in WEATHER_COLS]
    known[:, weather_j] = 0.0
    for wi, m in zip(w, members, strict=True):
        pos = m.index.get_indexer(first.index)
        raw = raw + np.where(pos >= 0, m.raw[np.clip(pos, 0, None)], np.nan)
        known[:, weather_j] += wi * np.where(
            pos[:, None] >= 0, m.known[np.clip(pos, 0, None)][:, weather_j], np.nan
        )
    return ZoneData(
        zone=name,
        index=first.index,
        raw=raw,
        known=known.astype(np.float32),
        dow=first.dow,
        dates=first.dates,
        tod=first.tod,
        known_cols=list(first.known_cols),
    )


def block_bootstrap(
    zd: ZoneData, y: np.ndarray, rng: np.random.Generator, block: int = SLOTS_PER_DAY
) -> np.ndarray:
    """A bootstrap replica of `y`: trend + (weekday, tod) profile + block-resampled residual.

    Trend = centred 7-day mean; profile = mean of the detrended load per (weekday, tod);
    the residual is resampled in moving blocks of one day (Bergmeir et al., 2016). The
    NaN tail beyond the last observation is kept, so the replica obeys the same cutoff.
    """
    valid = np.flatnonzero(~np.isnan(y))
    if len(valid) == 0:
        return y.copy()
    n = int(valid.max()) + 1
    s = pd.Series(y[:n])
    trend = s.rolling(7 * SLOTS_PER_DAY, center=True, min_periods=SLOTS_PER_DAY).mean()
    trend = trend.ffill().bfill()
    detrended = s - trend
    key = zd.dow[:n] * SLOTS_PER_DAY + zd.tod[:n]
    profile = detrended.groupby(key).transform("mean")
    resid = np.nan_to_num((detrended - profile).to_numpy())
    n_blocks = -(-n // block)
    starts = rng.integers(0, max(n - block, 0) + 1, n_blocks)
    boot = np.concatenate([resid[st : st + block] for st in starts])[:n]
    out = np.full(len(y), np.nan)
    out[:n] = np.maximum((trend + profile).to_numpy() + boot, 1.0)
    out[:n][np.isnan(y[:n])] = np.nan
    return out


def series_asof(
    zd: ZoneData, cutoff: pd.Timestamp, tod_means: dict[int, float] | None = None
) -> tuple[np.ndarray, dict[int, float]]:
    """Repaired load on the zone grid, using only slots that end at or before `cutoff`.

    Returns ``(y, tod_means)`` with `y` NaN from the cutoff on, so a window that reads
    past the cutoff is visibly wrong instead of silently leaky.
    """
    n = int(zd.index.searchsorted(cutoff))  # slots strictly before the cutoff (start < cutoff)
    s = pd.Series(zd.raw[:n], index=zd.index[:n])
    repaired, means = repair(s, tod_means)
    y = np.full(len(zd.index), np.nan)
    y[:n] = repaired.to_numpy()
    return y, means


@dataclass
class Sample:
    zone_id: int
    enc: slice
    dec: slice
    cutoff_pos: int


def make_sample(zd: ZoneData, zone_id: int, target: date, enc_len: int) -> Sample | None:
    cutoff = cutoff_for(target)
    p0 = int(zd.index.searchsorted(cutoff))
    p1 = int(zd.index.searchsorted(local_midnight_utc(target + timedelta(days=1))))
    if p0 - enc_len < 0 or p1 - p0 > DEC_LEN or p1 > len(zd.index):
        return None
    return Sample(zone_id, slice(p0 - enc_len, p0), slice(p0, p1), p0)


def sample_arrays(
    zd: ZoneData, y: np.ndarray, s: Sample, enc_len: int, wmean: np.ndarray, wstd: np.ndarray
) -> dict[str, np.ndarray]:
    """One sample as padded arrays (decoder padded to DEC_LEN, ``mask`` marks real slots)."""
    known = (zd.known - wmean) / wstd
    rel_col = zd.known_cols.index("rel_pos")
    kenc = known[s.enc].copy()
    kenc[:, rel_col] = np.arange(-enc_len, 0) / SLOTS_PER_DAY
    n_dec = s.dec.stop - s.dec.start
    kdec = np.zeros((DEC_LEN, known.shape[1]), dtype=np.float32)
    kdec[:n_dec] = known[s.dec]
    kdec[:n_dec, rel_col] = np.arange(n_dec) / SLOTS_PER_DAY
    yenc = np.nan_to_num(y[s.enc] / zd.scale, nan=1.0).astype(np.float32)
    ydec = np.zeros(DEC_LEN, dtype=np.float32)
    mask = np.zeros(DEC_LEN, dtype=np.float32)
    yd = y[s.dec] / zd.scale
    ok = ~np.isnan(yd)
    ydec[:n_dec][ok] = yd[ok]
    mask[:n_dec][ok] = 1.0
    dow_dec = np.zeros(DEC_LEN, dtype=np.int64)
    dow_dec[:n_dec] = zd.dow[s.dec]
    return {
        "x_enc": np.concatenate([yenc[:, None], kenc], axis=1).astype(np.float32),
        "dow_enc": zd.dow[s.enc].astype(np.int64),
        "x_dec": kdec,
        "dow_dec": dow_dec,
        "zone": np.asarray(s.zone_id, dtype=np.int64),
        "y": ydec,
        "mask": mask,
    }


def check_sample(
    zd: ZoneData, zone_id: int, y_asof: np.ndarray, target: date, enc_len: int
) -> Sample:
    """The forecast window of `target`, or a ValueError when history is short or leaks."""
    s = make_sample(zd, zone_id, target, enc_len)
    if s is None:
        raise ValueError(f"not enough history for {zd.zone} {target}")
    if not np.isnan(y_asof[s.dec]).all():
        raise ValueError("y_asof reaches past the cutoff")
    return s


def quantile_frame(
    zd: ZoneData, s: Sample, out: np.ndarray, target: date, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    """The slots of `target` with sorted ``q<pct>`` columns from the decoder output (MW)."""
    n_dec = s.dec.stop - s.dec.start
    out = np.sort(out[:n_dec], axis=1)  # monotone quantiles
    pick = zd.dates[s.dec] == np.datetime64(target)
    grid = day_grid(target)
    if pick.sum() != len(grid):
        raise ValueError(
            f"{zd.zone} {target}: {pick.sum()} decoder slots for a {len(grid)}-slot day"
        )
    qcols = {f"q{int(round(q * 100))}": out[pick, i] for i, q in enumerate(quantiles)}
    return pd.concat([grid, pd.DataFrame(qcols)], axis=1)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile_at(frame: pd.DataFrame, quantiles: tuple[float, ...], level: float) -> np.ndarray:
    """Linear interpolation of the ``q<pct>`` columns at `level` (clipped to the grid)."""
    cols = [f"q{int(round(q * 100))}" for q in quantiles]
    vals = frame[cols].to_numpy()
    return np.array([np.interp(level, quantiles, row) for row in vals])


def to_result_rows(frame: pd.DataFrame, quantiles: tuple[float, ...], alpha: float) -> pd.DataFrame:
    """``pred, p10, p90, p_alpha, alpha`` from the quantile columns; band kept around the median."""
    out = frame[["ts_utc", "date", "slot", "tod"]].copy()
    out["pred"] = quantile_at(frame, quantiles, 0.5)
    out["p10"] = np.minimum(quantile_at(frame, quantiles, 0.1), out["pred"])
    out["p90"] = np.maximum(quantile_at(frame, quantiles, 0.9), out["pred"])
    out["p_alpha"] = (
        out["pred"] if abs(alpha - 0.5) < 1e-9 else quantile_at(frame, quantiles, alpha)
    )
    out["alpha"] = float(alpha)
    return out


def periods(targets: list[date], refit_days: int) -> list[list[date]]:
    """Consecutive blocks of `refit_days` targets; each block is served by one fit."""
    return [targets[i : i + refit_days] for i in range(0, len(targets), refit_days)]


def fit_targets(first_target: date, window_days: int, available_dates: pd.Index) -> list[date]:
    """Training days for the fit that serves `first_target`: ``[D0-1-window, D0-2]`` with data."""
    return [d.date() for d in training_targets(first_target, window_days, available_dates)]


def zone_dates(zd: ZoneData) -> pd.Index:
    have = ~np.isnan(zd.raw)
    return pd.Index(pd.to_datetime(np.unique(zd.dates[have])))
