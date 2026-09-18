"""Naive and ISO baselines under the exact backtest protocol (data as of D-1 05:00 ET).

Every baseline forecasts the slots of target day ``D`` from data available at its
cutoff, aligned on wall-clock quarter hour like the model's lags:

- ``persist_2d``: same tod on D-2 (the most recent complete day)
- ``persist_7d``: same weekday one week back
- ``mean_7_14``: mean of D-7 and D-14
- ``isolf_pre``: NYISO's forecast in the file **named D-1** (posted on D-2 morning,
  so known before the cutoff): the fair, leakage-free ISO benchmark
- ``isolf_post``: NYISO's forecast in the file **named D** (posted on D-1 ~07:10-08:00
  ET, after the cutoff): the stronger post-close reference. Both are hourly and are
  expanded to slots by repetition.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from model import day_grid, local_fields

NAIVE = {"persist_2d": (2,), "persist_7d": (7,), "mean_7_14": (7, 14)}
ISO = ("isolf_pre", "isolf_post")
ALL = (*NAIVE, *ISO)


def _by_tod(zone_slots: pd.DataFrame) -> pd.Series:
    tbl = zone_slots[["ts_utc", "load_mw"]].copy()
    tbl[["date", "tod"]] = local_fields(tbl["ts_utc"])
    return tbl.groupby(["date", "tod"])["load_mw"].mean()


def naive_forecasts(zone_slots: pd.DataFrame, targets: list[date]) -> pd.DataFrame:
    """Naive slot forecasts: grid columns plus persist_2d, persist_7d, mean_7_14."""
    table = _by_tod(zone_slots)
    grids = pd.concat([day_grid(t) for t in targets], ignore_index=True)
    for name, offsets in NAIVE.items():
        cols = []
        for k in offsets:
            idx = pd.MultiIndex.from_arrays(
                [(grids["date"] - pd.Timedelta(days=k)).to_numpy(), grids["tod"].to_numpy()]
            )
            cols.append(table.reindex(idx).to_numpy(dtype=float))
        grids[name] = np.nanmean(np.column_stack(cols), axis=1) if len(cols) > 1 else cols[0]
    return grids


def isolf_forecasts(isolf: pd.DataFrame, zone: str, targets: list[date]) -> pd.DataFrame:
    """Slot-level ISO forecasts: ``ts_utc, isolf_pre, isolf_post`` (hourly values repeated)."""
    z = isolf[isolf["zone"] == zone]
    grids = pd.concat([day_grid(t) for t in targets], ignore_index=True)
    grids["hour_utc"] = grids["ts_utc"].dt.floor("h")
    for name, lag in (("isolf_pre", 1), ("isolf_post", 0)):
        issued = pd.to_datetime(grids["date"]).dt.date - timedelta(days=lag)
        key = pd.MultiIndex.from_arrays([issued.to_numpy(), grids["hour_utc"].to_numpy()])
        table = z.set_index(["issued", "ts_utc"])["isolf_mw"]
        table = table[~table.index.duplicated(keep="last")]
        grids[name] = table.reindex(key).to_numpy(dtype=float)
    return grids[["ts_utc", "date", "slot", "tod", "isolf_pre", "isolf_post"]]


def all_baselines(
    zone_slots: pd.DataFrame, isolf: pd.DataFrame, zone: str, targets: list[date]
) -> pd.DataFrame:
    """One frame with every baseline and the actual load (``actual``) per slot of `targets`."""
    naive = naive_forecasts(zone_slots, targets)
    iso = isolf_forecasts(isolf, zone, targets)
    out = naive.merge(iso[["ts_utc", *ISO]], on="ts_utc", how="left")
    actual = zone_slots[["ts_utc", "load_mw"]].rename(columns={"load_mw": "actual"})
    out = out.merge(actual, on="ts_utc", how="left")
    out.insert(0, "zone", zone)
    return out
