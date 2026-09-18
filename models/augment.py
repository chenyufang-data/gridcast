"""Training-set augmentation for the tabular estimators.

Every kind (``KINDS``) acts on the training rows only, after the cutoff split, and returns
``(X, y, sample_weight)``; the originals come first with weight 1, augmented copies after
them with the weight passed in (multiplied into the decay weights by the wrapper).

- ``swap``: Jahrer's swap noise as plain augmentation: every feature cell of a copy is
  replaced with probability ``p`` by the same feature of a donor row, by default a row of
  the same quarter hour so a night-time lag never lands in a noon row.
- ``wnoise``: weather-forecast-error injection: the forecast-temperature columns of a
  copy are shifted by one offset per training day plus per-slot jitter for the hourly
  columns, with the spread of the 2-day-lead forecast updates (about 1.5 C each). The
  model learns that the forecast it is given is uncertain, which is the dominant error
  source of the day-ahead problem.
- ``extreme``: no new rows; rows of days in the hot or cold tail of the window, or with a
  large temperature deviation, get a higher weight (the heat-wave misses of the error
  analysis).
- ``cmixup``: C-Mixup for regression: every row of a copy is interpolated with a row of
  the same quarter hour whose target is among the ``k`` nearest, features and target
  alike, with ``lambda ~ Beta(alpha, alpha)``.

Calendar columns and the decay-age column are never modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from models.features import AGE_COL

CALENDAR_COLUMNS = ("tod", "dayofweek", "is_holiday", "holiday_tomorrow", "is_weekend", AGE_COL)
TEMPERATURE_DAILY = ("temp_mean", "temp_min", "temp_max", "temp_dev")
TEMPERATURE_HOURLY = ("temp_h", "app_h", "dew_h", "temp_h_prev3")
KINDS = ("swap", "wnoise", "extreme", "cmixup")

Augmented = tuple[pd.DataFrame, np.ndarray, np.ndarray]


def _groups(X: pd.DataFrame, within: str | None) -> list[np.ndarray]:
    if within is not None and within in X.columns:
        values = X[within].to_numpy()
        return [np.flatnonzero(values == g) for g in np.unique(values)]
    return [np.arange(len(X))]


def _day_ids(X: pd.DataFrame) -> np.ndarray:
    """Rows of one training day share the decay age; use it as the day key."""
    if AGE_COL in X.columns:
        return np.unique(X[AGE_COL].to_numpy(), return_inverse=True)[1]
    return np.zeros(len(X), dtype=int)


def _stack(
    X: pd.DataFrame, copies: list[np.ndarray], ys: list[np.ndarray], weight: float
) -> Augmented:
    n = len(X)
    out = pd.DataFrame(np.vstack([X.to_numpy(dtype=float), *copies]), columns=list(X.columns))
    w = np.concatenate([np.ones(n), np.full(n * len(copies), weight)])
    return out, np.concatenate([np.asarray(ys[0], dtype=float), *ys[1:]]), w


def swap_noise(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    p: float = 0.1,
    copies: int = 1,
    weight: float = 0.5,
    within: str | None = "tod",
    exclude: tuple[str, ...] = CALENDAR_COLUMNS,
    rng: np.random.Generator | None = None,
) -> Augmented:
    """Append `copies` swap-noised copies of ``(X, y)``; `within` names the donor group column."""
    yv = np.asarray(y, dtype=float)
    if copies < 1 or p <= 0:
        return X, yv, np.ones(len(X))
    rng = rng or np.random.default_rng(0)
    cols = list(X.columns)
    swap_idx = np.array([i for i, c in enumerate(cols) if c not in exclude])
    base = X.to_numpy(dtype=float)
    n = len(base)
    groups = _groups(X, within)
    outs, ys = [], [yv]
    for _ in range(copies):
        donor = np.arange(n)
        for idx in groups:
            donor[idx] = rng.choice(idx, size=len(idx), replace=True)
        mask = rng.random((n, len(swap_idx))) < p
        copy = base.copy()
        copy[:, swap_idx] = np.where(mask, base[donor][:, swap_idx], copy[:, swap_idx])
        outs.append(copy)
        ys.append(yv)
    return _stack(X, outs, ys, weight)


def weather_noise(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    day_sigma: float = 1.55,
    slot_sigma: float = 1.58,
    copies: int = 1,
    weight: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Augmented:
    """Copies whose forecast-temperature columns carry a per-day offset (+ hourly jitter)."""
    yv = np.asarray(y, dtype=float)
    cols = list(X.columns)
    daily = [cols.index(c) for c in TEMPERATURE_DAILY if c in cols]
    hourly = [cols.index(c) for c in TEMPERATURE_HOURLY if c in cols]
    if copies < 1 or not (daily or hourly):
        return X, yv, np.ones(len(X))
    rng = rng or np.random.default_rng(0)
    base = X.to_numpy(dtype=float)
    n = len(base)
    day = _day_ids(X)
    outs, ys = [], [yv]
    for _ in range(copies):
        eps_day = rng.normal(0.0, day_sigma, day.max() + 1)[day]
        copy = base.copy()
        for j in daily:
            copy[:, j] += eps_day
        for j in hourly:
            copy[:, j] += eps_day + rng.normal(0.0, slot_sigma, n)
        outs.append(copy)
        ys.append(yv)
    return _stack(X, outs, ys, weight)


def extreme_weight(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    quantile: float = 0.9,
    factor: float = 3.0,
    rng: np.random.Generator | None = None,
) -> Augmented:
    """Weight `factor` on rows of hot / cold-tail days or days with a large temperature swing."""
    yv = np.asarray(y, dtype=float)
    w = np.ones(len(X))
    if "temp_mean" in X.columns:
        tm = X["temp_mean"].to_numpy(dtype=float)
        hi, lo = np.nanquantile(tm, quantile), np.nanquantile(tm, 1 - quantile)
        extreme = (tm >= hi) | (tm <= lo)
        if "temp_dev" in X.columns:
            dev = np.abs(X["temp_dev"].to_numpy(dtype=float))
            extreme |= dev >= np.nanquantile(dev, quantile)
        w[extreme] = factor
    return X, yv, w


def cmixup(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    alpha: float = 2.0,
    k: int = 5,
    copies: int = 1,
    weight: float = 0.5,
    within: str | None = "tod",
    exclude: tuple[str, ...] = CALENDAR_COLUMNS,
    rng: np.random.Generator | None = None,
) -> Augmented:
    """Copies interpolated with a same-group partner among the `k` nearest targets."""
    yv = np.asarray(y, dtype=float)
    if copies < 1:
        return X, yv, np.ones(len(X))
    rng = rng or np.random.default_rng(0)
    cols = list(X.columns)
    mix_idx = np.array([i for i, c in enumerate(cols) if c not in exclude])
    base = X.to_numpy(dtype=float)
    n = len(base)
    groups = _groups(X, within)
    outs, ys = [], [yv]
    for _ in range(copies):
        partner = np.arange(n)
        for idx in groups:
            if len(idx) < 2:
                continue
            order = idx[np.argsort(yv[idx])]
            m = len(order)
            pos = np.arange(m)
            step = rng.integers(1, k + 1, m) * rng.choice([-1, 1], m)
            j = np.clip(pos + step, 0, m - 1)
            same = j == pos
            j[same] = (pos[same] + 1) % m
            partner[order] = order[j]
        lam = rng.beta(alpha, alpha, n)
        copy = base.copy()
        copy[:, mix_idx] = (
            lam[:, None] * base[:, mix_idx] + (1 - lam[:, None]) * base[partner][:, mix_idx]
        )
        outs.append(copy)
        ys.append(lam * yv + (1 - lam) * yv[partner])
    return _stack(X, outs, ys, weight)


BUILDERS: dict[str, Callable[..., Augmented]] = {
    "swap": swap_noise,
    "wnoise": weather_noise,
    "extreme": extreme_weight,
    "cmixup": cmixup,
}


def apply(
    kind: str | None,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Dispatch by name; ``None`` returns the inputs unchanged (weight ``None``)."""
    if kind is None:
        return X, np.asarray(y, dtype=float), None
    if kind not in BUILDERS:
        raise ValueError(f"unknown augmentation {kind!r}; known: {KINDS}")
    return BUILDERS[kind](X, y, rng=np.random.default_rng(seed), **(params or {}))
