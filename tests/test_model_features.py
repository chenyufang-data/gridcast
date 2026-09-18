"""Day-type lag anchors, decay floor, and extra hourly weather columns in model.py."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import model as M
from app import weather as W
from tests.synthetic import SyntheticNYISO
from tests.test_model import zone_slots


def _series() -> pd.Series:
    synth = SyntheticNYISO(date(2025, 9, 1), date(2025, 10, 20), seed=21, irregular=False)
    return M.regularize(zone_slots(synth))


def test_daytype_anchors_pick_the_same_day_type() -> None:
    series = _series()
    # 2025-10-04 is a Saturday in a holiday-free week: the nearest same-type day
    # at or before D-2 is Sunday 09-28 (k=6); Columbus Day would make 10-13 weekend-type
    rows, cols = M.build_features(
        series, [pd.Timestamp("2025-09-29")], date(2025, 10, 4), daytype=True
    )
    assert {
        "is_weekend",
        "holiday_yesterday",
        "lag_same_type",
        "lag_same_type2",
        "level_same_type",
        "shape_same_type",
    } <= set(cols)
    sat = rows[rows["date"] == pd.Timestamp("2025-10-04")].sort_values("slot")
    assert (sat["is_weekend"] == 1).all()
    expected = series.reindex(sat["ts_utc"] - pd.Timedelta(days=6)).to_numpy()
    assert np.allclose(sat["lag_same_type"].to_numpy(), expected)
    # the second-nearest same-type day is Saturday 09-27 (k=7): lag_same_type2 is the mean
    expected2 = series.reindex(sat["ts_utc"] - pd.Timedelta(days=7)).to_numpy()
    assert np.allclose(sat["lag_same_type2"].to_numpy(), (expected + expected2) / 2)
    # a Monday training row anchors on Friday (k=3), not on the weekend
    mon = rows[rows["date"] == pd.Timestamp("2025-09-29")].sort_values("slot")
    assert (mon["is_weekend"] == 0).all()
    assert np.allclose(
        mon["lag_same_type"].to_numpy(),
        series.reindex(mon["ts_utc"] - pd.Timedelta(days=3)).to_numpy(),
    )
    assert np.allclose(mon["shape_same_type"] * mon["level_same_type"], mon["lag_same_type"])


def test_holiday_counts_as_weekend() -> None:
    series = M.regularize(
        zone_slots(SyntheticNYISO(date(2025, 10, 20), date(2025, 12, 5), seed=22, irregular=False))
    )
    # Thanksgiving 2025-11-27 (Thursday) is weekend-type: the nearest same-type day
    # at or before D-2 is Sunday 11-23 (k=4)
    rows, _ = M.build_features(
        series, [pd.Timestamp("2025-11-20")], date(2025, 11, 27), daytype=True
    )
    thu = rows[rows["date"] == pd.Timestamp("2025-11-27")].sort_values("slot")
    assert (thu["is_weekend"] == 1).all() and (thu["is_holiday"] == 1).all()
    assert np.allclose(
        thu["lag_same_type"].to_numpy(),
        series.reindex(thu["ts_utc"] - pd.Timedelta(days=4)).to_numpy(),
    )
    fri = M.build_features(series, [pd.Timestamp("2025-11-20")], date(2025, 11, 28), daytype=True)[
        0
    ]
    fri = fri[fri["date"] == pd.Timestamp("2025-11-28")]
    assert (fri["holiday_yesterday"] == 1).all() and (fri["is_weekend"] == 0).all()


def test_decay_floor_keeps_old_rows_weighted() -> None:
    X = pd.DataFrame({"a": np.linspace(0, 1, 50), M.AGE_COL: np.linspace(0, 400, 50)})
    y = pd.Series(np.linspace(0, 1, 50))
    captured: dict[str, np.ndarray] = {}

    class Fake:
        def fit(
            self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
        ) -> None:
            assert sample_weight is not None
            captured["w"] = sample_weight

    m = M.DecayWeightedLGBMRegressor(decay_half_life=32, decay_floor=0.2)
    m.model = Fake()  # type: ignore[assignment]
    m.fit(X, y)
    w = captured["w"]
    assert w[0] == 1.0 and w[-1] == 0.2 and (w >= 0.2).all()
    plain = M.DecayWeightedLGBMRegressor(decay_half_life=32)
    plain.model = Fake()  # type: ignore[assignment]
    plain.fit(X, y)
    assert captured["w"][-1] < 1e-3


def test_extra_hourly_weather_columns_become_features() -> None:
    ts = pd.date_range("2025-10-10", periods=24 * 12, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "ts_utc": ts,
            "zone": "N.Y.C.",
            "tfc1": 10.0,
            "tfc2": np.arange(len(ts)) % 24 + 5.0,
            "x2_app": 1.0,
            "x2_dew": 2.0,
            "x2_rh": 50.0,
            "x2_cloud": 30.0,
            "x2_wind": 9.0,
            "x2_rad": 100.0,
        }  # fmt: skip
    )
    feats = W.hourly_features_for(hourly, "N.Y.C.", extra=True)
    assert feats is not None
    assert list(feats.columns) == [
        "hour_utc",
        "temp_h",
        "app_h",
        "dew_h",
        "rh_h",
        "cloud_h",
        "wind_h",
        "rad_h",
        "temp_h_prev3",
    ]
    assert np.isnan(feats["temp_h_prev3"].iloc[0]) and feats["temp_h_prev3"].iloc[3] == np.mean(
        feats["temp_h"].iloc[0:3]
    )
    series = _series()
    rows, cols = M.build_features(
        series, [pd.Timestamp("2025-10-13")], date(2025, 10, 15), weather_hourly=feats
    )
    assert {"temp_h", "app_h", "rad_h", "temp_h_prev3"} <= set(cols)
    day = rows[rows["date"] == pd.Timestamp("2025-10-15")]
    assert day["app_h"].eq(1.0).all() and day["temp_h"].notna().all()
