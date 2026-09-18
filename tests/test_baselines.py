"""src.baselines on the synthetic truth: wall-clock naive lags, isolf pre/post-close files."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app import nyiso
from src import baselines as B
from tests.conftest import FALL_BACK_DAY
from tests.synthetic import SyntheticNYISO


@pytest.fixture(scope="module")
def synth() -> SyntheticNYISO:
    return SyntheticNYISO(date(2025, 10, 10), date(2025, 11, 12), seed=12, irregular=False)


@pytest.fixture(scope="module")
def slots(synth: SyntheticNYISO) -> pd.DataFrame:
    load = pd.concat(
        [nyiso.normalize_pal(synth.pal_csv(d)) for d in synth.days()], ignore_index=True
    )
    return nyiso.resample_slots(load, "load_mw")


@pytest.fixture(scope="module")
def isolf(synth: SyntheticNYISO) -> pd.DataFrame:
    return pd.concat(
        [nyiso.normalize_isolf(synth.isolf_csv(d), d) for d in synth.days()], ignore_index=True
    )


def test_naive_forecasts_align_on_wall_clock(slots: pd.DataFrame) -> None:
    nyc = slots[slots["zone"] == "N.Y.C."]
    target = FALL_BACK_DAY + timedelta(days=7)
    out = B.naive_forecasts(nyc, [target])
    assert len(out) == 96 and list(out.columns) == [
        "ts_utc",
        "date",
        "slot",
        "tod",
        "persist_2d",
        "persist_7d",
        "mean_7_14",
    ]
    by_ts = nyc.set_index("ts_utc")["load_mw"]
    assert np.allclose(out["persist_2d"], by_ts.reindex(out["ts_utc"] - pd.Timedelta(days=2)))
    noon = out[out["tod"] == 48].iloc[0]
    src7 = by_ts[pd.Timestamp("2025-11-02 12:00", tz="America/New_York").tz_convert("UTC")]
    src14 = by_ts[pd.Timestamp("2025-10-26 12:00", tz="America/New_York").tz_convert("UTC")]
    assert noon["persist_7d"] == src7 and noon["mean_7_14"] == pytest.approx((src7 + src14) / 2)
    fall = B.naive_forecasts(nyc, [FALL_BACK_DAY])
    assert (
        len(fall) == 100 and fall["persist_7d"].notna().all()
    )  # both 01:xx hours get the 01:xx lag


def test_isolf_pre_and_post_close_files(isolf: pd.DataFrame, synth: SyntheticNYISO) -> None:
    target = date(2025, 11, 5)
    out = B.isolf_forecasts(isolf, "N.Y.C.", [target])
    assert len(out) == 96
    pre = synth.isolf_truth(target - timedelta(days=1))
    post = synth.isolf_truth(target)
    for col, truth in (("isolf_pre", pre), ("isolf_post", post)):
        t = truth[truth["zone"] == "N.Y.C."].set_index("ts_utc")["isolf_mw"]
        assert np.array_equal(
            out[col].to_numpy(), t.reindex(out["ts_utc"].dt.floor("h")).to_numpy()
        )
    assert not np.array_equal(out["isolf_pre"], out["isolf_post"])
    nyca = B.isolf_forecasts(isolf, "NYCA", [FALL_BACK_DAY])
    assert len(nyca) == 100 and nyca["isolf_post"].notna().all()


def test_all_baselines_carries_actual(slots: pd.DataFrame, isolf: pd.DataFrame) -> None:
    targets = [date(2025, 11, 1), date(2025, 11, 2), date(2025, 11, 3)]
    out = B.all_baselines(slots[slots["zone"] == "WEST"], isolf, "WEST", targets)
    assert list(out.columns) == ["zone", "ts_utc", "date", "slot", "tod", *B.ALL, "actual"]
    assert len(out) == 96 + 100 + 96 and (out["zone"] == "WEST").all()
    assert out["actual"].notna().all() and out[list(B.ALL)].notna().all().all()
    # the model's own persistence error on synthetic data is small but not zero
    err = ((out["persist_7d"] - out["actual"]).abs() / out["actual"]).mean()
    assert 0 < err < 0.1
