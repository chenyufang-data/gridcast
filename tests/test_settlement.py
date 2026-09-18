"""src.settlement: price join, newsvendor α as of the cutoff, imbalance sign conventions."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import models as M
from src import settlement as S


def _slots(start: str, n: int, zone: str = "N.Y.C.") -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="15min")
    return pd.DataFrame({"ts_utc": ts, "zone": zone})


def test_slot_prices_joins_the_hour() -> None:
    rt = _slots("2026-01-05 00:00", 8).assign(p_rt=[10, 12, 14, 16, 30, 30, 30, 30])
    da = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(["2026-01-05 00:00", "2026-01-05 02:00"], utc=True),
            "zone": "N.Y.C.",
            "p_da": [11.0, 99.0],
        }
    )
    out = S.slot_prices(rt, da)
    assert len(out) == 4  # 01:00 has no DA price -> dropped
    assert list(out.columns) == ["ts_utc", "zone", "p_rt", "p_da", "spread"]
    assert list(out["spread"]) == [-1.0, 1.0, 3.0, 5.0]


def _prices(spreads: list[float], start: str = "2026-01-01 05:00") -> pd.DataFrame:
    p = _slots(start, len(spreads))
    p["p_da"] = 30.0
    p["p_rt"] = 30.0 + np.asarray(spreads, dtype=float)
    p["spread"] = p["p_rt"] - p["p_da"]
    return p


def test_estimate_alpha_extremes_and_window() -> None:
    target = date(2026, 1, 20)
    assert S.estimate_alpha(_prices([5.0] * 96), "N.Y.C.", target) == 1.0
    assert S.estimate_alpha(_prices([-5.0] * 96), "N.Y.C.", target) == 0.0
    assert S.estimate_alpha(_prices([5.0, -5.0] * 48), "N.Y.C.", target) == 0.5
    assert S.estimate_alpha(_prices([0.0] * 96), "N.Y.C.", target) == 0.5
    assert S.estimate_alpha(_prices([]), "N.Y.C.", target) == 0.5
    # spreads at or after the cutoff (D-1 05:00 ET = 10:00Z) must not count
    cutoff = M.cutoff_for(target)
    before = _prices([-5.0] * 4, start=str((cutoff - pd.Timedelta(hours=1)).tz_convert(None)))
    after = _prices([500.0] * 8, start=str(cutoff.tz_convert(None)))
    assert S.estimate_alpha(pd.concat([before, after]), "N.Y.C.", target) == 0.0
    assert S.estimate_alpha(_prices([5.0] * 96), "WEST", target) == 0.5  # other zone: no data
    # window: 30 days by default, old spreads ignored
    old = _prices([500.0] * 96, start="2025-11-01 05:00")
    assert S.estimate_alpha(pd.concat([old, before]), "N.Y.C.", target) == 0.0


def test_alpha_series_indexed_by_target() -> None:
    s = S.alpha_series(
        _prices([5.0] * 96, start="2026-01-10 05:00"),
        "N.Y.C.",
        [date(2026, 1, 12), date(2026, 3, 1)],
    )
    assert list(s.index) == [date(2026, 1, 12), date(2026, 3, 1)] and list(s) == [1.0, 0.5]


def test_settle_sign_conventions() -> None:
    actual = _slots("2026-01-05 00:00", 4).assign(load_mw=100.0)
    prices = _prices([20.0, 20.0, -20.0, -20.0], start="2026-01-05 00:00")
    settled = S.settle(actual, np.array([90.0, 110.0, 90.0, 110.0]), prices)
    # short 10 MW (2.5 MWh) while RT > DA costs; long while RT > DA earns; mirrored when RT < DA
    assert list(settled["dev_mwh"]) == [2.5, -2.5, 2.5, -2.5]
    assert list(settled["imbalance_usd"]) == [50.0, -50.0, -50.0, 50.0]
    assert list(settled["da_cost_usd"]) == [
        90 * 0.25 * 30,
        110 * 0.25 * 30,
        90 * 0.25 * 30,
        110 * 0.25 * 30,
    ]
    total = S.summarize(settled)
    assert total["imbalance_usd"].iloc[0] == 0.0 and total["abs_imbalance_usd"].iloc[0] == 200.0
    assert total["energy_mwh"].iloc[0] == 100.0 and total["slots"].iloc[0] == 4
    by_zone = S.summarize(settled, by=["zone"])
    assert list(by_zone["zone"]) == ["N.Y.C."] and by_zone["usd_per_mwh"].iloc[0] == 0.0
    perfect = S.settle(actual, actual["load_mw"].to_numpy(), prices)
    assert (perfect["imbalance_usd"] == 0).all()


def test_hourly_bid_to_slots() -> None:
    grid = _slots("2026-01-05 00:00", 8)
    bid = pd.DataFrame(
        {
            "hour_utc": pd.to_datetime(["2026-01-05 00:00", "2026-01-05 01:00"], utc=True),
            "bid_mw": [10.0, 20.0],
        }
    )
    assert list(S.hourly_bid_to_slots(bid, grid)) == [10.0] * 4 + [20.0] * 4


@pytest.mark.parametrize("spreads, expected", [([3.0, 3.0, -1.0, -1.0], 0.75), ([1.0, -3.0], 0.25)])
def test_alpha_is_the_newsvendor_ratio(spreads: list[float], expected: float) -> None:
    assert S.estimate_alpha(_prices(spreads * 24), "N.Y.C.", date(2026, 1, 20)) == pytest.approx(
        expected
    )
