"""app.weather: offline parts only (feature semantics, statewide weighting, graceful absence)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app import weather as W


def _frame() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=12, freq="D")
    rows = []
    for zone in ("N.Y.C.", "WEST"):
        for i, d in enumerate(days):
            base = 10.0 if zone == "N.Y.C." else 0.0
            rows.append(
                {
                    "date": d,
                    "zone": zone,
                    "tfc1_mean": base + i,
                    "tfc1_min": base + i - 3,
                    "tfc1_max": base + i + 3,
                    "tfc2_mean": base + 2 * i,
                    "tfc2_min": base + 2 * i - 3,
                    "tfc2_max": base + 2 * i + 3,
                }
            )
    return pd.DataFrame(rows)


def test_features_use_the_pre_cutoff_lead_and_lag_the_trailing_mean() -> None:
    feats = W.features_for(_frame(), "N.Y.C.", lead="d2")
    assert feats is not None
    assert list(feats.columns) == ["date", "temp_mean", "temp_min", "temp_max", "temp_dev"]
    assert list(feats["temp_mean"][:3]) == [10.0, 12.0, 14.0]  # tfc2, not tfc1
    # temp_dev on day i = tfc2_mean(i) - mean(tfc2_mean over the 7 days ending i-2)
    i = 10
    trailing = np.mean([10.0 + 2 * k for k in range(i - 8, i - 1)])
    assert feats["temp_dev"].iloc[i] == (10.0 + 2 * i) - trailing
    assert feats["temp_dev"].iloc[:4].isna().all()  # not enough history (min 3 days, shifted by 2)
    d1 = W.features_for(_frame(), "N.Y.C.", lead="d1")
    assert d1 is not None and list(d1["temp_mean"][:3]) == [10.0, 11.0, 12.0]


def test_features_absent_zone_or_weather_return_none() -> None:
    assert W.features_for(None, "N.Y.C.") is None
    assert W.features_for(_frame(), "NORTH") is None
    assert W.features_for(_frame().iloc[0:0], "N.Y.C.") is None


def test_with_nyca_is_load_weighted() -> None:
    out = W.with_nyca(_frame())
    nyca = out[out["zone"] == "NYCA"].sort_values("date")
    assert len(nyca) == 12
    w_nyc, w_west = W.ZONE_WEIGHT["N.Y.C."], W.ZONE_WEIGHT["WEST"]
    expected = (10.0 * w_nyc + 0.0 * w_west) / (w_nyc + w_west)
    assert abs(nyca["tfc1_mean"].iloc[0] - round(expected, 2)) < 1e-9
    assert set(out["zone"]) == {"N.Y.C.", "WEST", "NYCA"}


def test_load_weather_missing_file(tmp_path: Path) -> None:
    assert W.load_weather(tmp_path / "nope.csv") is None
    _frame().to_csv(tmp_path / "w.csv", index=False)
    loaded = W.load_weather(tmp_path / "w.csv")
    assert loaded is not None and str(loaded["date"].dtype).startswith("datetime64")


def test_hourly_features_pick_the_lead() -> None:
    ts = pd.date_range("2026-01-05 00:00", periods=6, freq="h", tz="UTC")
    hourly = pd.DataFrame({"ts_utc": ts, "zone": "N.Y.C.", "tfc1": range(6), "tfc2": range(10, 16)})
    feats = W.hourly_features_for(hourly, "N.Y.C.")
    assert feats is not None and list(feats.columns) == ["hour_utc", "temp_h"]
    assert list(feats["temp_h"]) == [10, 11, 12, 13, 14, 15]
    d1 = W.hourly_features_for(hourly, "N.Y.C.", lead="d1")
    assert d1 is not None and list(d1["temp_h"]) == [0, 1, 2, 3, 4, 5]
    assert (
        W.hourly_features_for(hourly, "WEST") is None
        and W.hourly_features_for(None, "N.Y.C.") is None
    )
    total = W.with_nyca(hourly, key="ts_utc")
    assert set(total["zone"]) == {"N.Y.C.", "NYCA"} and len(total) == 12
