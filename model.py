"""Modeling core: slot grid, repair, features as of the bid cutoff, LightGBM wrapper.

Shared by ``src/`` (offline backtest) and ``app/`` (service), like the source repo.
Everything is relative to a target day ``D`` (local ET) and its bid cutoff
``D-1 05:00 ET`` (``src.config.BID_CUTOFF_TIME``):

- ``history`` is the 15-min slot frame of one zone (``ts_utc, load_mw[, coverage]``)
  and must end at or before the cutoff: :func:`forecast_day` raises
  :class:`LeakageError` otherwise, so no caller can train on the future by accident.
- Lag features align on the wall-clock quarter hour (``tod`` = 0..95), so DST days line
  up with human schedules; the output grid is chronological (92 / 96 / 100 slots).
- The partial day ``D-1`` (00:00-04:45, 20 slots) feeds "morning" level features; every
  training target ``T <= D-2`` is featurized the same way, as of its own cutoff.
- Weather is optional: a day-level frame (``date, temp_mean, temp_min, temp_max,
  temp_dev``) built from day-ahead-issued forecasts by ``app.weather``; without it the
  feature set degrades gracefully.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import BID_CUTOFF_TIME, MARKET_TZ, SLOTS_PER_HOUR

log = logging.getLogger(__name__)

SLOT = pd.Timedelta(minutes=60 // SLOTS_PER_HOUR)
SLOTS_PER_WEEK = 7 * 24 * SLOTS_PER_HOUR
MORNING_TODS = 20  # 00:00 .. 04:45 of D-1 are complete at the 05:00 cutoff
MIN_COVERAGE = 0.5  # a slot with less data than this is treated as missing
MIN_TRAIN_DAYS = 7
LAGS = (2, 3, 7, 14, 21)
RECENT_LAGS = ["lag_2d", "lag_3d"]
WEEK_LAGS = ["lag_7d", "lag_14d", "lag_21d"]
ALL_LAGS = [f"lag_{k}d" for k in LAGS]
WEATHER_FEATURES = ["temp_mean", "temp_min", "temp_max", "temp_dev"]
WEATHER_HOURLY_FEATURES = ["temp_h"]
AGE_COL = "_age_days"  # days before the newest training day; only for decay weights
FEATURE_VERSION = "nyiso-fv1"  # bump when features or hyper-parameters change

# --- repair thresholds (port of the source; NYISO data rarely triggers them) --------
ABNORMAL_ZERO_EPS = 1e-6
ABNORMAL_LOW_RATIO = 0.30
ABNORMAL_HIGH_RATIO = 3.0
WEEK_NEIGHBOR_OFFSETS = (7, -7, 14, -14)

DECAY_HALF_LIFE_DAYS = 32


class LeakageError(ValueError):
    """History reaches past the bid cutoff."""


# ---------------------------------------------------------------------------- calendar
def cutoff_for(target: date) -> pd.Timestamp:
    """The DAM close for `target`: D-1 05:00 ET, as a UTC timestamp."""
    d1 = datetime.combine(target - timedelta(days=1), BID_CUTOFF_TIME)
    return pd.Timestamp(d1, tz=MARKET_TZ).tz_convert("UTC")


def local_midnight_utc(day: date) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(MARKET_TZ).tz_convert("UTC")


def day_grid(target: date) -> pd.DataFrame:
    """Chronological 15-min grid of one local day: ``ts_utc, date, slot, tod``."""
    idx = pd.date_range(
        local_midnight_utc(target),
        local_midnight_utc(target + timedelta(days=1)),
        freq=SLOT,
        inclusive="left",
    )
    out = pd.DataFrame({"ts_utc": idx})
    out[["date", "tod"]] = local_fields(out["ts_utc"])
    out["slot"] = np.arange(len(out))
    return out[["ts_utc", "date", "slot", "tod"]]


def local_fields(ts_utc: pd.Series) -> pd.DataFrame:
    """``date`` (naive local midnight) and ``tod`` (wall-clock quarter hour) per stamp."""
    local = ts_utc.dt.tz_convert(MARKET_TZ)
    return pd.DataFrame(
        {
            "date": local.dt.normalize().dt.tz_localize(None),
            "tod": (
                local.dt.hour * SLOTS_PER_HOUR + local.dt.minute // (60 // SLOTS_PER_HOUR)
            ).astype(int),
        },
        index=ts_utc.index,
    )


@lru_cache(maxsize=8)
def holiday_dates(years: tuple[int, ...]) -> frozenset[date]:
    """US federal holidays (the NY-only observances do not move load)."""
    return frozenset(holidays.country_holidays("US", years=list(years)).keys())


def is_holiday(dates: pd.Series) -> pd.Series:
    years = tuple(sorted(dates.dt.year.unique()))
    hol = holiday_dates(years)
    return dates.dt.date.map(lambda d: d in hol).astype(int)


# ---------------------------------------------------------------------------- grid + repair
def regularize(history: pd.DataFrame) -> pd.Series:
    """One zone's slots -> a complete 15-min UTC grid (NaN where missing or thin)."""
    h = history.sort_values("ts_utc")
    values = h["load_mw"].to_numpy(dtype=float)
    if "coverage" in h.columns:
        values = np.where(h["coverage"].to_numpy() < MIN_COVERAGE, np.nan, values)
    s = pd.Series(values, index=pd.DatetimeIndex(h["ts_utc"]).tz_convert("UTC"))
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index.min().floor(SLOT), s.index.max().floor(SLOT), freq=SLOT)
    return s.reindex(full)


def repair(
    series: pd.Series, tod_means: dict[int, float] | None = None
) -> tuple[pd.Series, dict[int, float]]:
    """Treat missing / non-positive / wildly off-level slots as absent and refill them.

    Refill = mean of the same slot ±7 / ±14 days (weekly cycle), then the per-tod mean
    (computed here when `tod_means` is None, reused otherwise), then the global mean.
    Holidays are exempt from the level test: a real holiday trough is not a fault.
    """
    s = series.copy()
    fields = local_fields(pd.Series(s.index, index=s.index))
    tod = fields["tod"]
    positive = s.where(s > ABNORMAL_ZERO_EPS)
    tod_median = positive.groupby(tod.to_numpy()).transform("median")
    hol = is_holiday(fields["date"]).astype(bool)
    abnormal = s.isna() | (s <= ABNORMAL_ZERO_EPS)
    abnormal |= (
        s.notna()
        & tod_median.notna()
        & ~hol
        & ((s < ABNORMAL_LOW_RATIO * tod_median) | (s > ABNORMAL_HIGH_RATIO * tod_median))
    )
    clean = s.mask(abnormal)
    neighbors = pd.concat(
        [clean.shift(off * SLOTS_PER_WEEK // 7) for off in WEEK_NEIGHBOR_OFFSETS], axis=1
    )
    repaired = clean.copy()
    repaired[abnormal] = neighbors.mean(axis=1)[abnormal]
    if tod_means is None:
        means = clean.groupby(tod.to_numpy()).mean()
        tod_means = means.fillna(clean.mean()).to_dict()
    still = repaired.isna()
    if still.any():
        repaired[still] = tod[still].map(tod_means)
        repaired = repaired.fillna(clean.mean())
    n_fixed = int(abnormal.sum())
    if n_fixed:
        log.debug("repair: %d of %d slots refilled", n_fixed, len(s))
    return repaired, tod_means


# ---------------------------------------------------------------------------- features
def _lookup(table: pd.Series, dates: pd.Series, tods: pd.Series) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([dates.to_numpy(), tods.to_numpy()])
    return table.reindex(idx).to_numpy(dtype=float)


def build_features(
    repaired: pd.Series,
    train_dates: list[pd.Timestamp],
    predict_target: date | None,
    weather: pd.DataFrame | None = None,
    weather_hourly: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Feature rows for every slot of the training days and of the prediction day.

    Returns ``(rows, feature_columns)``; `rows` carries ``ts_utc, date, slot, tod, y``
    (``y`` NaN on the prediction day), the features, and ``_age_days``.
    """
    tbl = pd.DataFrame({"ts_utc": repaired.index, "y": repaired.to_numpy()})
    tbl[["date", "tod"]] = local_fields(tbl["ts_utc"])
    tbl = tbl.sort_values("ts_utc").reset_index(drop=True)
    tbl["slot"] = tbl.groupby("date").cumcount()

    by_tod = tbl.pivot_table(index="date", columns="tod", values="y", aggfunc="mean")
    long = by_tod.stack()  # (date, tod) -> value
    level = tbl.groupby("date")["y"].mean()  # daily mean MW (DST-day safe)
    morning = by_tod.loc[:, [t for t in range(MORNING_TODS) if t in by_tod.columns]].mean(axis=1)
    morning_last = by_tod[MORNING_TODS - 1] if MORNING_TODS - 1 in by_tod.columns else morning

    parts = [tbl[tbl["date"].isin(train_dates)][["ts_utc", "date", "slot", "tod", "y"]]]
    if predict_target is not None:
        grid = day_grid(predict_target)
        grid["y"] = np.nan
        parts.append(grid[["ts_utc", "date", "slot", "tod", "y"]])
    rows = pd.concat(parts, ignore_index=True)

    for k in LAGS:
        d = rows["date"] - pd.Timedelta(days=k)
        rows[f"lag_{k}d"] = _lookup(long, d, rows["tod"])
        rows[f"level_{k}d"] = level.reindex(d).to_numpy()
    d1 = rows["date"] - pd.Timedelta(days=1)
    rows["d1_morning"] = morning.reindex(d1).to_numpy()
    rows["d1_last"] = morning_last.reindex(d1).to_numpy()
    rows["d2_morning"] = morning.reindex(rows["date"] - pd.Timedelta(days=2)).to_numpy()
    rows["d8_morning"] = morning.reindex(rows["date"] - pd.Timedelta(days=8)).to_numpy()
    rows["d1_vs_d2"] = rows["d1_morning"] / rows["d2_morning"]
    rows["d1_vs_d8"] = rows["d1_morning"] / rows["d8_morning"]
    rows["lag_2d_adj"] = rows["lag_2d"] * rows["d1_vs_d2"]
    rows["lag_7d_adj"] = rows["lag_7d"] * rows["d1_vs_d8"]

    rows["lag_recent_mean"] = rows[RECENT_LAGS].mean(axis=1)
    rows["lag_week_mean"] = rows[WEEK_LAGS].mean(axis=1)
    rows["lag_week_median"] = rows[WEEK_LAGS].median(axis=1)
    rows["lag_all_mean"] = rows[ALL_LAGS].mean(axis=1)
    rows["lag_all_min"] = rows[ALL_LAGS].min(axis=1)
    rows["lag_all_max"] = rows[ALL_LAGS].max(axis=1)
    rows["lag_all_std"] = rows[ALL_LAGS].std(axis=1)
    rows["trend_2_7"] = rows["lag_2d"] - rows["lag_7d"]
    rows["trend_7_14"] = rows["lag_7d"] - rows["lag_14d"]
    rows["ratio_2_7"] = rows["lag_2d"] / rows["lag_7d"]
    rows["ratio_7_14"] = rows["lag_7d"] / rows["lag_14d"]
    rows["level_week_mean"] = rows[["level_7d", "level_14d", "level_21d"]].mean(axis=1)
    for k in (2, 7, 14):
        rows[f"shape_{k}d"] = rows[f"lag_{k}d"] / rows[f"level_{k}d"]

    rows["dayofweek"] = rows["date"].dt.dayofweek
    rows["is_holiday"] = is_holiday(rows["date"])
    rows["holiday_tomorrow"] = is_holiday(rows["date"] + pd.Timedelta(days=1))

    weather_cols: list[str] = []
    if weather is not None and not weather.empty:
        w = weather[["date", *WEATHER_FEATURES]].drop_duplicates("date")
        rows = rows.merge(w, on="date", how="left")
        weather_cols = list(WEATHER_FEATURES)
    if weather_hourly is not None and not weather_hourly.empty:
        rows["hour_utc"] = rows["ts_utc"].dt.floor("h")
        wh = weather_hourly[["hour_utc", *WEATHER_HOURLY_FEATURES]].drop_duplicates("hour_utc")
        rows = rows.merge(wh, on="hour_utc", how="left").drop(columns="hour_utc")
        weather_cols += list(WEATHER_HOURLY_FEATURES)

    rows = rows.replace([np.inf, -np.inf], np.nan)
    newest = max(train_dates) if train_dates else rows["date"].max()
    rows[AGE_COL] = (newest - rows["date"]).dt.days.clip(lower=0).astype(float)
    return rows, feature_columns(weather_cols)


def feature_columns(weather_cols: list[str] | tuple[str, ...] = ()) -> list[str]:
    return [
        "tod", "dayofweek", "is_holiday", "holiday_tomorrow",
        *ALL_LAGS,
        "level_2d", "level_7d", "level_14d", "level_21d", "level_week_mean",
        "d1_morning", "d1_last", "d1_vs_d2", "d1_vs_d8", "lag_2d_adj", "lag_7d_adj",
        "lag_recent_mean", "lag_week_mean", "lag_week_median",
        "lag_all_mean", "lag_all_min", "lag_all_max", "lag_all_std",
        "trend_2_7", "trend_7_14", "ratio_2_7", "ratio_7_14",
        "shape_2d", "shape_7d", "shape_14d",
        *weather_cols,
        AGE_COL,
    ]  # fmt: skip


# ---------------------------------------------------------------------------- model
class DecayWeightedLGBMRegressor:
    """LightGBM with time-decay sample weights ``0.5 ** (age_days / half_life)``.

    Softer than a hard window: data just outside the window fades instead of vanishing,
    recent behaviour dominates, older weekly structure is kept.
    """

    def __init__(self, decay_half_life: float = DECAY_HALF_LIFE_DAYS, **lgbm_params: Any) -> None:
        self.decay_half_life = decay_half_life
        self.lgbm_params = lgbm_params
        self.model = LGBMRegressor(**lgbm_params)

    @staticmethod
    def _pop_age(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        if AGE_COL in X.columns:
            return X.drop(columns=[AGE_COL]), X[AGE_COL].to_numpy(dtype=float)
        return X, None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> DecayWeightedLGBMRegressor:
        X, age = self._pop_age(X)
        weight = (
            0.5 ** (age / self.decay_half_life)
            if age is not None and self.decay_half_life
            else None
        )
        self.model.fit(X, y, sample_weight=weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X, _ = self._pop_age(X)
        return np.asarray(self.model.predict(X), dtype=float)


def get_model(**overrides: Any) -> DecayWeightedLGBMRegressor:
    """Default median configuration; overrides derive variants (quantile objective + alpha)."""
    params: dict[str, Any] = dict(
        decay_half_life=DECAY_HALF_LIFE_DAYS,
        objective="mae",
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=30,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=0,
        verbose=-1,
    )
    params.update(overrides)
    return DecayWeightedLGBMRegressor(**params)


# ---------------------------------------------------------------------------- one forecast
def _ratio_denominator(rows: pd.DataFrame) -> pd.Series:
    """Level the ratio target is expressed against: same-slot mean of the last three weeks."""
    return rows["lag_week_mean"].fillna(rows["lag_all_mean"])


def training_targets(target: date, window_days: int, available: pd.Index) -> list[pd.Timestamp]:
    """Training days: ``[D-1-window, D-2]`` (complete before the cutoff) that have data."""
    last = pd.Timestamp(target) - pd.Timedelta(days=2)
    first = pd.Timestamp(target) - pd.Timedelta(days=1 + window_days)
    return [d for d in available if first <= d <= last]


def forecast_day(
    history: pd.DataFrame,
    target: date,
    *,
    weather: pd.DataFrame | None = None,
    weather_hourly: pd.DataFrame | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.9),
    alpha: float | None = None,
    window_days: int = 120,
    model_overrides: dict[str, Any] | None = None,
    target_mode: str = "mw",
) -> pd.DataFrame:
    """Train on `history` (one zone, slots ending at or before the cutoff) and forecast `target`.

    Returns the chronological grid of `target` with ``pred`` (median), one column per
    requested quantile (``p10``, ``p90``, ...) and, when `alpha` is given, ``p_alpha``
    (the α-quantile bid, or the median when α is 0.5).

    `target_mode`: ``"ratio"`` fits ``y / lag_week_mean`` (the same slot's mean over the
    three previous weeks) and rescales the predictions, which lets trees follow level
    shifts they never saw in the window; ``"mw"`` fits the load directly.
    """
    cutoff = cutoff_for(target)
    if history.empty:
        raise ValueError("empty history")
    last_end = history["ts_utc"].max() + SLOT
    if last_end > cutoff:
        raise LeakageError(f"history ends {last_end}, after the cutoff {cutoff} for {target}")

    series = regularize(history)
    repaired, _ = repair(series)
    fields = local_fields(pd.Series(repaired.index, index=repaired.index))
    targets = training_targets(target, window_days, pd.Index(fields["date"].unique()))
    if len(targets) < MIN_TRAIN_DAYS:
        raise ValueError(
            f"only {len(targets)} training days before {target}; need {MIN_TRAIN_DAYS}"
        )

    rows, cols = build_features(repaired, targets, target, weather, weather_hourly)
    train = rows[rows["y"].notna() & rows["lag_7d"].notna()]
    pred_rows = (
        rows[rows["date"] == pd.Timestamp(target)].sort_values("slot").reset_index(drop=True)
    )
    if train.empty or pred_rows.empty:
        raise ValueError("no usable training rows")
    if target_mode == "ratio":
        denom_train = _ratio_denominator(train)
        denom_pred = _ratio_denominator(pred_rows)
        keep = denom_train.notna() & (denom_train > 0)
        train, denom_train = train[keep], denom_train[keep]
        y = train["y"] / denom_train
        scale = denom_pred.fillna(denom_pred.mean()).to_numpy()
    elif target_mode == "mw":
        y = train["y"]
        scale = np.ones(len(pred_rows))
    else:
        raise ValueError(f"unknown target_mode {target_mode!r}")
    X = train[cols]
    Xp = pred_rows[cols]
    overrides = dict(model_overrides or {})

    out = pred_rows[["ts_utc", "date", "slot", "tod"]].copy()
    out["pred"] = get_model(**overrides).fit(X, y).predict(Xp) * scale
    fitted: dict[float, np.ndarray] = {}
    for q in quantiles:
        fitted[q] = (
            get_model(objective="quantile", alpha=q, **overrides).fit(X, y).predict(Xp) * scale
        )
        out[f"p{int(round(q * 100))}"] = fitted[q]
    if alpha is not None:
        if abs(alpha - 0.5) < 1e-9:
            out["p_alpha"] = out["pred"].to_numpy()
        elif alpha in fitted:
            out["p_alpha"] = fitted[alpha]
        else:
            out["p_alpha"] = (
                get_model(objective="quantile", alpha=float(alpha), **overrides)
                .fit(X, y)
                .predict(Xp)
                * scale
            )
        out["alpha"] = float(alpha)
    # quantile crossing: keep the band ordered around the median
    if {"p10", "p90"} <= set(out.columns):
        lo = np.minimum(out["p10"], out["pred"])
        hi = np.maximum(out["p90"], out["pred"])
        out["p10"], out["p90"] = lo, hi
    return out
