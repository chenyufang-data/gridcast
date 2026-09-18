"""Settlement in dollars: slot prices, imbalance cost, and the cost-aware bid quantile α.

Market rule (docs/plan.md §1): the DAM bid for hour *h* is a single MW quantity; every
deviation of the actual load from it is settled at the real-time price. Relative to a
perfect-foresight bid the extra cost of a slot is::

    imbalance_usd = (actual_mw - bid_mw) / 4 * (p_rt - p_da)

positive when the deviation cost money (under-bid while RT > DA, or over-bid while
RT < DA), negative when it happened to pay. ``p_rt`` is the 15-min mean of the 5-min RT
LBMPs (``rt_slots``), ``p_da`` the hour's DA LBMP.

The cost-aware bid quantile is the newsvendor ratio on the trailing spread::

    c_under = mean(max(p_rt - p_da, 0))   cost of being short 1 MWh
    c_over  = mean(max(p_da - p_rt, 0))   cost of being long 1 MWh
    alpha   = c_under / (c_under + c_over)

estimated on a window that ends strictly at the bid cutoff, per zone.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from models import SLOT, cutoff_for

MWH_PER_MW_SLOT = 0.25  # 15-min slot


def slot_prices(rt_slots: pd.DataFrame, da_hourly: pd.DataFrame) -> pd.DataFrame:
    """Join 15-min RT prices with the hour's DA price: ``ts_utc, zone, p_rt, p_da, spread``."""
    rt = rt_slots[["ts_utc", "zone", "p_rt"]].copy()
    rt["hour_utc"] = rt["ts_utc"].dt.floor("h")
    da = da_hourly.rename(columns={"ts_utc": "hour_utc"})[["hour_utc", "zone", "p_da"]]
    out = rt.merge(da, on=["hour_utc", "zone"], how="inner").drop(columns="hour_utc")
    out["spread"] = out["p_rt"] - out["p_da"]
    return out.sort_values(["ts_utc", "zone"]).reset_index(drop=True)


def estimate_alpha(prices: pd.DataFrame, zone: str, target: date, window_days: int = 30) -> float:
    """Newsvendor α for `zone` and `target`, from spreads of slots ending at or before the cutoff.

    Returns 0.5 when the window has no usable spread (no information either way).
    """
    cutoff = cutoff_for(target)
    start = cutoff - pd.Timedelta(days=window_days)
    p = prices[
        (prices["zone"] == zone) & (prices["ts_utc"] >= start) & (prices["ts_utc"] + SLOT <= cutoff)
    ]
    spread = p["spread"].dropna().to_numpy()
    if len(spread) == 0:
        return 0.5
    c_under = float(np.maximum(spread, 0).mean())
    c_over = float(np.maximum(-spread, 0).mean())
    if c_under + c_over <= 0:
        return 0.5
    return c_under / (c_under + c_over)


def alpha_series(
    prices: pd.DataFrame, zone: str, targets: list[date], window_days: int = 30
) -> pd.Series:
    """α per target day (index = target date), each estimated as of its own cutoff."""
    return pd.Series(
        {t: estimate_alpha(prices, zone, t, window_days) for t in targets}, name="alpha"
    )


def hourly_bid_to_slots(bid_hourly: pd.DataFrame, grid: pd.DataFrame) -> pd.Series:
    """Expand an hourly bid (``hour_utc, bid_mw``) onto a slot grid (``ts_utc``)."""
    hours = grid["ts_utc"].dt.floor("h")
    return hours.map(bid_hourly.set_index("hour_utc")["bid_mw"])


def settle(
    actual_slots: pd.DataFrame, bid_slots: pd.Series | np.ndarray, prices: pd.DataFrame
) -> pd.DataFrame:
    """Per-slot settlement of a bid against actuals and prices.

    `actual_slots`: ``ts_utc, zone, load_mw``; `bid_slots`: MW aligned with its rows.
    Returns the rows with ``bid_mw, p_rt, p_da, spread, dev_mwh, imbalance_usd,
    da_cost_usd`` (DA energy cost of the bid, the natural denominator for a % view).
    """
    out = actual_slots[["ts_utc", "zone", "load_mw"]].copy()
    out["bid_mw"] = np.asarray(bid_slots, dtype=float)
    out = out.merge(
        prices[["ts_utc", "zone", "p_rt", "p_da", "spread"]], on=["ts_utc", "zone"], how="left"
    )
    out["dev_mwh"] = (out["load_mw"] - out["bid_mw"]) * MWH_PER_MW_SLOT
    out["imbalance_usd"] = out["dev_mwh"] * out["spread"]
    out["da_cost_usd"] = out["bid_mw"] * MWH_PER_MW_SLOT * out["p_da"]
    return out


def summarize(settled: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Totals per group: imbalance, abs imbalance, DA cost, energy, $/MWh, % of DA cost, slots."""
    keys = by or []
    s = settled.dropna(subset=["imbalance_usd"]).copy()
    s["energy_mwh"] = s["load_mw"] * MWH_PER_MW_SLOT
    s["abs_imbalance_usd"] = s["imbalance_usd"].abs()
    agg = (
        s.groupby(keys)[["imbalance_usd", "abs_imbalance_usd", "da_cost_usd", "energy_mwh"]].sum()
        if keys
        else s[["imbalance_usd", "abs_imbalance_usd", "da_cost_usd", "energy_mwh"]]
        .sum()
        .to_frame()
        .T
    )
    agg["usd_per_mwh"] = agg["imbalance_usd"] / agg["energy_mwh"]
    agg["pct_of_da_cost"] = agg["imbalance_usd"] / agg["da_cost_usd"] * 100
    agg["slots"] = s.groupby(keys).size() if keys else len(s)
    return agg.reset_index() if keys else agg
