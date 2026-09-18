"""models.augment (swap noise), the XGBoost estimator, and the history_start restriction."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

import models as M
from models import augment
from src import backtest as BT
from tests.synthetic import SyntheticNYISO
from tests.test_model import zone_slots


def _table(n: int = 960) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "tod": np.tile(np.arange(96), n // 96),
            "dayofweek": rng.integers(0, 7, n),
            "lag_2d": rng.normal(1000, 100, n),
            "temp_h": rng.normal(15, 8, n),
            M.AGE_COL: np.repeat(np.arange(n // 96), 96).astype(float),
        }
    )
    return X, X["lag_2d"].to_numpy() * 1.01


def test_swap_noise_keeps_originals_and_swaps_within_tod() -> None:
    X, y = _table()
    Xa, ya, w = augment.swap_noise(X, y, p=0.3, copies=2, weight=0.4, rng=np.random.default_rng(1))
    n = len(X)
    assert len(Xa) == 3 * n and len(ya) == 3 * n and len(w) == 3 * n
    assert np.allclose(Xa.iloc[:n].to_numpy(), X.to_numpy()) and np.allclose(ya[:n], y)
    assert (w[:n] == 1).all() and (w[n:] == 0.4).all() and np.allclose(ya[n:], np.tile(y, 2))
    copy = Xa.iloc[n : 2 * n]
    # calendar and age columns are never touched
    for col in ("tod", "dayofweek", M.AGE_COL):
        assert np.array_equal(copy[col].to_numpy(), X[col].to_numpy())
    changed = ~np.isclose(copy["lag_2d"].to_numpy(), X["lag_2d"].to_numpy())
    assert 0.2 < changed.mean() < 0.4  # about p of the cells
    # every swapped value comes from a row with the same tod
    for tod in (0, 48, 95):
        rows = copy[(copy["tod"] == tod) & changed]
        pool = set(X.loc[X["tod"] == tod, "lag_2d"].round(9))
        assert set(rows["lag_2d"].round(9)) <= pool


def test_swap_noise_anywhere_and_dispatch() -> None:
    X, y = _table()
    Xa, _, _ = augment.swap_noise(X, y, p=0.5, within=None, rng=np.random.default_rng(2))
    copy = Xa.iloc[len(X) :]
    changed = ~np.isclose(copy["temp_h"].to_numpy(), X["temp_h"].to_numpy())
    assert 0.4 < changed.mean() < 0.6
    same, ys, w = augment.apply(None, X, y)
    assert same is X and w is None and np.allclose(ys, y)
    off, _, w0 = augment.swap_noise(X, y, p=0.0)
    assert off is X and (w0 == 1).all()
    with pytest.raises(ValueError, match="unknown augmentation"):
        augment.apply("mixup", X, y)


def test_get_model_names_and_decay_weights() -> None:
    with pytest.raises(ValueError, match="unknown estimator"):
        M.get_model("catboost")
    m = M.get_model("lgbm", quantile=0.9, decay_half_life=10, n_estimators=5)
    X, _ = _table()
    w = m.weights(X, sample_weight=np.full(len(X), 0.5))
    assert w is not None and w[0] == 0.5 and w[-1] == pytest.approx(0.5 * 0.5 ** (9 / 10))
    assert isinstance(m.model, LGBMRegressor) and m.model.get_params()["objective"] == "quantile"


@pytest.mark.slow
def test_xgb_estimator_forecasts_like_lgbm() -> None:
    pytest.importorskip("xgboost")
    synth = SyntheticNYISO(date(2025, 8, 1), date(2025, 9, 20), seed=5, irregular=False)
    slots = zone_slots(synth)
    target = date(2025, 9, 18)
    history = slots[slots["ts_utc"] + M.SLOT <= M.cutoff_for(target)]
    fast = {"n_estimators": 120, "n_jobs": 2}
    out = M.forecast_day(history, target, alpha=0.4, model_overrides=fast, estimator="xgb")
    actual = slots.merge(out[["ts_utc"]], on="ts_utc")["load_mw"].to_numpy()
    assert len(out) == 96 and np.mean(np.abs(actual - out["pred"]) / actual) < 0.04
    assert (out["p10"] <= out["pred"]).all() and (out["pred"] <= out["p90"]).all()
    assert (out["p_alpha"] <= out["pred"] + 1e-6).mean() > 0.7  # α < 0.5 bids short
    aug = M.forecast_day(
        history,
        target,
        model_overrides=fast,
        augment_kind="swap",
        augment_params={"p": 0.1, "copies": 1},
    )
    assert len(aug) == 96 and np.mean(np.abs(actual - aug["pred"]) / actual) < 0.04


def test_restrict_history() -> None:
    ts = pd.date_range("2025-09-30 00:00", periods=8, freq="6h", tz="UTC")
    df = pd.DataFrame({"ts_utc": ts, "zone": "WEST", "load_mw": 1.0})
    assert BT.restrict_history(df, None) is df
    kept = BT.restrict_history(df, date(2025, 10, 1))
    # local midnight EDT = 04:00 UTC; the 00:00 UTC row of 10-01 is still 09-30 in ET
    assert kept["ts_utc"].min() == pd.Timestamp("2025-10-01 06:00", tz="UTC")
    assert len(kept) == 3
