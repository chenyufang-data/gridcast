"""models.augment: weather-noise injection, extreme-day weights, C-Mixup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models as M
from models import augment


def _table(n_days: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(4)
    n = n_days * 96
    X = pd.DataFrame(
        {
            "tod": np.tile(np.arange(96), n_days),
            "dayofweek": np.repeat(np.arange(n_days) % 7, 96),
            "lag_2d": rng.normal(1000, 100, n),
            "temp_mean": np.repeat(rng.normal(15, 8, n_days), 96),
            "temp_dev": np.repeat(rng.normal(0, 3, n_days), 96),
            "temp_h": rng.normal(15, 8, n),
            "rh_h": rng.uniform(30, 90, n),
            M.AGE_COL: np.repeat(np.arange(n_days), 96).astype(float),
        }
    )
    return X, X["lag_2d"].to_numpy() * 1.02


def test_weather_noise_shifts_temperatures_per_day_only() -> None:
    X, y = _table()
    Xa, ya, w = augment.weather_noise(
        X, y, day_sigma=2.0, slot_sigma=0.5, copies=1, weight=0.3, rng=np.random.default_rng(1)
    )
    n = len(X)
    assert len(Xa) == 2 * n and np.allclose(ya[n:], y) and (w[n:] == 0.3).all()
    copy = Xa.iloc[n:]
    for col in ("tod", "dayofweek", "lag_2d", "rh_h", M.AGE_COL):
        assert np.array_equal(copy[col].to_numpy(), X[col].to_numpy()), col
    shift = copy["temp_mean"].to_numpy() - X["temp_mean"].to_numpy()
    # one offset per day, identical for temp_mean and temp_dev
    per_day = pd.Series(shift).groupby(X[M.AGE_COL].to_numpy()).nunique()
    assert (per_day == 1).all() and len(per_day) == 10
    assert np.allclose(shift, copy["temp_dev"].to_numpy() - X["temp_dev"].to_numpy())
    assert 0.5 < np.std(pd.Series(shift).groupby(X[M.AGE_COL].to_numpy()).first()) < 5.0
    # hourly temperature carries the day offset plus its own jitter
    jitter = (copy["temp_h"].to_numpy() - X["temp_h"].to_numpy()) - shift
    assert 0.3 < jitter.std() < 0.8 and abs(jitter.mean()) < 0.1


def test_extreme_weight_marks_tails() -> None:
    X, y = _table(20)
    same, ys, w = augment.extreme_weight(X, y, quantile=0.9, factor=3.0)
    assert same is X and np.allclose(ys, y) and set(np.unique(w)) <= {1.0, 3.0}
    tm = X["temp_mean"].to_numpy()
    hot = tm >= np.quantile(tm, 0.9)
    cold = tm <= np.quantile(tm, 0.1)
    assert (w[hot] == 3.0).all() and (w[cold] == 3.0).all()
    mild = (
        ~hot & ~cold & (np.abs(X["temp_dev"].to_numpy()) < np.quantile(np.abs(X["temp_dev"]), 0.9))
    )
    assert (w[mild] == 1.0).all()
    no_weather = X.drop(columns=["temp_mean", "temp_dev"])
    assert (augment.extreme_weight(no_weather, y)[2] == 1.0).all()


def test_cmixup_interpolates_within_tod() -> None:
    X, y = _table()
    Xa, ya, w = augment.cmixup(
        X, y, alpha=2.0, k=3, copies=2, weight=0.5, rng=np.random.default_rng(2)
    )
    n = len(X)
    assert len(Xa) == 3 * n and (w[n:] == 0.5).all() and np.allclose(ya[:n], y)
    copy, y_mix = Xa.iloc[n : 2 * n], ya[n : 2 * n]
    for col in ("tod", "dayofweek", M.AGE_COL):
        assert np.array_equal(copy[col].to_numpy(), X[col].to_numpy()), col
    # every mixed target lies between the row's target and one of the k nearest same-tod targets
    for tod in (5, 50):
        rows = np.flatnonzero(X["tod"].to_numpy() == tod)
        ys_tod = np.sort(y[rows])
        for r in rows:
            lo, hi = min(y[r], y_mix[r]), max(y[r], y_mix[r])
            rank = np.searchsorted(ys_tod, y[r])
            near = ys_tod[max(rank - 3, 0) : rank + 4]
            assert lo - 1e-9 <= y_mix[r] <= hi + 1e-9
            assert ((near >= lo - 1e-9) & (near <= hi + 1e-9)).any()
    assert not np.allclose(copy["lag_2d"].to_numpy(), X["lag_2d"].to_numpy())


def test_apply_dispatches_every_kind() -> None:
    X, y = _table(4)
    for kind in augment.KINDS:
        Xa, ya, w = augment.apply(kind, X, y, seed=3)
        assert w is not None and len(Xa) == len(ya) == len(w) >= len(X)
    with pytest.raises(ValueError, match="unknown augmentation"):
        augment.apply("timewarp", X, y)
