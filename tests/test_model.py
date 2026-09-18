"""models: cutoff, DST grid, repair, lag alignment, leakage guard, real fits on synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import models as M
from app import nyiso
from tests.conftest import FALL_BACK_DAY, SPRING_FORWARD_DAY
from tests.synthetic import SyntheticNYISO

ET = "America/New_York"


def zone_slots(synth: SyntheticNYISO, zone: str = "N.Y.C.") -> pd.DataFrame:
    load = pd.concat(
        [nyiso.normalize_pal(synth.pal_csv(d)) for d in synth.days()], ignore_index=True
    )
    slots = nyiso.resample_slots(load, "load_mw")
    return slots[slots["zone"] == zone].reset_index(drop=True)


@pytest.fixture(scope="module")
def synth_long() -> SyntheticNYISO:
    return SyntheticNYISO(date(2025, 8, 1), date(2025, 11, 8), seed=11, irregular=False)


@pytest.fixture(scope="module")
def nyc_slots(synth_long: SyntheticNYISO) -> pd.DataFrame:
    return zone_slots(synth_long)


def test_cutoff_is_dst_aware() -> None:
    assert M.cutoff_for(date(2025, 10, 1)) == pd.Timestamp("2025-09-30 09:00", tz="UTC")  # EDT
    assert M.cutoff_for(date(2025, 12, 1)) == pd.Timestamp("2025-11-30 10:00", tz="UTC")  # EST
    assert M.cutoff_for(date(2025, 11, 3)) == pd.Timestamp(
        "2025-11-02 10:00", tz="UTC"
    )  # cutoff on the fall-back day


def test_day_grid_counts_and_tod() -> None:
    normal, fall, spring = (
        M.day_grid(date(2025, 10, 1)),
        M.day_grid(FALL_BACK_DAY),
        M.day_grid(SPRING_FORWARD_DAY),
    )
    assert (len(normal), len(fall), len(spring)) == (96, 100, 92)
    assert list(normal["slot"]) == list(range(96)) and list(normal["tod"]) == list(range(96))
    assert list(fall["tod"][:16]) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]  # 01:xx twice
    assert 8 not in set(spring["tod"]) and list(fall["slot"]) == list(range(100))
    assert (normal["date"] == pd.Timestamp("2025-10-01")).all()


def test_holidays_are_federal() -> None:
    days = pd.Series(pd.to_datetime(["2025-11-27", "2025-11-28", "2026-07-03", "2026-02-12"]))
    assert list(M.is_holiday(days)) == [
        1,
        0,
        1,
        0,
    ]  # Thanksgiving, Friday after, July 4 observed, Lincoln (NY only)


def test_regularize_and_repair(nyc_slots: pd.DataFrame) -> None:
    series = M.regularize(nyc_slots)
    assert series.isna().sum() == 0 and series.index.freq is not None
    n_days = (nyc_slots["ts_utc"].max() - nyc_slots["ts_utc"].min()) / pd.Timedelta(days=1)
    assert abs(len(series) / 96 - n_days) < 1.01
    broken = series.copy()
    broken.iloc[3000:3010] = 0.0  # meter-style dropout
    broken.iloc[5000] = broken.iloc[5000] * 10  # spike
    broken.iloc[6000:6004] = np.nan
    repaired, tod_means = M.repair(broken)
    assert repaired.isna().sum() == 0 and len(tod_means) == 96
    for sl in (slice(3000, 3010), slice(5000, 5001), slice(6000, 6004)):
        rel = (repaired.iloc[sl] - series.iloc[sl]).abs() / series.iloc[sl]
        assert rel.max() < 0.08, sl
    untouched = ~broken.index.isin(
        broken.index[list(range(3000, 3010)) + [5000] + list(range(6000, 6004))]
    )
    assert np.allclose(repaired[untouched], series[untouched])


def test_build_features_align_on_wall_clock(nyc_slots: pd.DataFrame) -> None:
    series = M.regularize(nyc_slots)
    train_day, target = pd.Timestamp("2025-10-20"), date(2025, 10, 22)
    rows, cols = M.build_features(series, [train_day], target)
    assert set(cols) <= set(rows.columns) and M.AGE_COL in cols
    pred = rows[rows["date"] == pd.Timestamp(target)].sort_values("slot")
    assert len(pred) == 96 and pred["y"].isna().all()
    for k in (2, 7):
        expected = series.reindex(pred["ts_utc"] - pd.Timedelta(days=k)).to_numpy()
        assert np.allclose(pred[f"lag_{k}d"].to_numpy(), expected)
    d1 = series[
        (series.index >= M.local_midnight_utc(date(2025, 10, 21)))
        & (series.index < M.local_midnight_utc(date(2025, 10, 21)) + pd.Timedelta(hours=5))
    ]
    assert len(d1) == 20 and abs(pred["d1_morning"].iloc[0] - d1.mean()) < 1e-9
    assert abs(pred["d1_last"].iloc[0] - d1.iloc[-1]) < 1e-9
    trained = rows[rows["date"] == train_day]
    assert np.allclose(trained["y"].to_numpy(), series.reindex(trained["ts_utc"]).to_numpy())
    assert (trained[M.AGE_COL] == 0).all() and (pred[M.AGE_COL] == 0).all()
    assert pred["dayofweek"].iloc[0] == 2  # Wednesday


def test_lags_across_fall_back_use_wall_clock(nyc_slots: pd.DataFrame) -> None:
    series = M.regularize(nyc_slots)
    target = FALL_BACK_DAY + timedelta(days=7)  # 2025-11-09: lag_7d comes from the 25-hour day
    rows, _ = M.build_features(series, [pd.Timestamp("2025-11-05")], target)
    pred = rows[rows["date"] == pd.Timestamp(target)]
    noon = pred[pred["tod"] == 48].iloc[0]
    src = series.reindex([pd.Timestamp("2025-11-02 12:00", tz=ET).tz_convert("UTC")]).iloc[0]
    assert abs(noon["lag_7d"] - src) < 1e-9  # wall-clock noon, not noon shifted by the extra hour
    assert len(rows[rows["date"] == pd.Timestamp("2025-11-05")]) == 96


def test_forecast_day_guard_shape_and_accuracy(nyc_slots: pd.DataFrame) -> None:
    target = date(2025, 11, 3)  # the Monday after fall-back; cutoff lies on the 25-hour day
    cutoff = M.cutoff_for(target)
    history = nyc_slots[nyc_slots["ts_utc"] + M.SLOT <= cutoff]
    fast = {"n_estimators": 150, "n_jobs": 2}
    out = M.forecast_day(history, target, alpha=0.7, model_overrides=fast)
    assert list(out.columns) == [
        "ts_utc",
        "date",
        "slot",
        "tod",
        "pred",
        "p10",
        "p90",
        "p_alpha",
        "alpha",
    ]
    assert (
        len(out) == 96 and (out["p10"] <= out["pred"]).all() and (out["pred"] <= out["p90"]).all()
    )
    actual = nyc_slots.merge(out[["ts_utc"]], on="ts_utc")["load_mw"].to_numpy()
    assert np.mean(np.abs(actual - out["pred"]) / actual) < 0.03  # smooth synthetic load
    assert (out["alpha"] == 0.7).all() and (out["p_alpha"] >= out["pred"] - 1e-6).mean() > 0.8
    with pytest.raises(M.LeakageError):
        M.forecast_day(nyc_slots[nyc_slots["ts_utc"] < cutoff + pd.Timedelta(hours=1)], target)
    with pytest.raises(ValueError, match="training days"):
        M.forecast_day(history[history["ts_utc"] >= cutoff - pd.Timedelta(days=5)], target)
    mw = M.forecast_day(history, target, alpha=0.5, model_overrides=fast, target_mode="mw")
    assert np.array_equal(mw["p_alpha"], mw["pred"]) and len(mw) == 96


def test_forecast_day_on_the_fall_back_day(nyc_slots: pd.DataFrame) -> None:
    history = nyc_slots[nyc_slots["ts_utc"] + M.SLOT <= M.cutoff_for(FALL_BACK_DAY)]
    out = M.forecast_day(
        history, FALL_BACK_DAY, quantiles=(), model_overrides={"n_estimators": 100, "n_jobs": 2}
    )
    assert len(out) == 100 and list(out["slot"]) == list(range(100))
    assert out["ts_utc"].is_monotonic_increasing and "p10" not in out.columns
