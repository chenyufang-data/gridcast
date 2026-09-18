"""Settlement in dollars for every forecast: baselines, ISO, ISO + α, model median, α-bid.

    python scripts/imbalance_report.py --name default

Bids are hourly (the DAM product): each slot forecast is averaged per hour, then the
hourly bid is settled per 15-min slot at (mean 5-min RT LBMP - DA LBMP) against the
actual load (src/settlement.py). Perfect foresight costs $0 by definition; a positive
total is money lost to forecast error, a negative one money gained by luck.

Strategies (all leakage-free: α and any adjustment use only days <= D-2):
  persist_2d, persist_7d, mean_7_14      naive volumes
  isolf_pre / isolf_post                 NYISO's forecast (pre-close fair / post-close ref)
  isolf_pre_alpha                        isolf_pre scaled by the trailing α-quantile of actual/isolf
  model                                  our median forecast
  model_alpha_bid                        our α-quantile forecast (quantile LightGBM at α)
  model_alpha_emp                        our median scaled by the trailing α-quantile of actual/pred

Headline rule (docs/plan.md §4): the α-bid may lead only if it beats every other
strategy here in pooled dollars over the full range; the last lines say whether it did.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.log import configure_logging  # noqa: E402
from src import dataset  # noqa: E402
from src.backtest import RESULTS_DIR, load_results  # noqa: E402
from src.config import NYCA  # noqa: E402
from src.settlement import settle, slot_prices, summarize  # noqa: E402

RATIO_WINDOW_DAYS = 30
STRATEGIES = [
    "persist_2d", "persist_7d", "mean_7_14", "isolf_pre", "isolf_post", "isolf_pre_alpha",
    "model", "model_alpha_bid", "model_alpha_emp",
]  # fmt: skip


def hourly_bids(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace each slot forecast by its hour's mean (one MW bid per hour)."""
    out = df.copy()
    hour = out["ts_utc"].dt.floor("h")
    for c in cols:
        out[c] = out.groupby(["zone", hour])[c].transform("mean")
    return out


def trailing_alpha_scale(
    df: pd.DataFrame, forecast_col: str, window_days: int = RATIO_WINDOW_DAYS
) -> pd.Series:
    """Per row: the α-quantile of actual/forecast over the `window_days` days ending at D-2.

    Hourly ratios are used (the bid is hourly). Rows without enough history get 1.0.
    """
    d = df[["zone", "ts_utc", "date", "alpha", forecast_col, "actual"]].copy()
    d["hour_utc"] = d["ts_utc"].dt.floor("h")
    hourly = d.groupby(["zone", "date", "hour_utc"], as_index=False).agg(
        f=(forecast_col, "mean"), a=("actual", "mean"), alpha=("alpha", "first")
    )
    hourly["ratio"] = hourly["a"] / hourly["f"]
    scale = pd.Series(1.0, index=df.index)
    for zone, z in hourly.groupby("zone"):
        by_day = {day: g["ratio"].dropna().to_numpy() for day, g in z.groupby("date")}
        days = sorted(by_day)
        day_scale: dict[pd.Timestamp, float] = {}
        for day in days:
            lo, hi = day - pd.Timedelta(days=window_days + 1), day - pd.Timedelta(days=2)
            hist = [by_day[x] for x in days if lo <= x <= hi]
            alpha = float(z.loc[z["date"] == day, "alpha"].iloc[0])
            if not hist:
                continue
            pooled = np.concatenate(hist)
            if len(pooled) >= 24 * 7:
                day_scale[day] = float(np.quantile(pooled, alpha))
        mask = df["zone"] == zone
        scale[mask] = df.loc[mask, "date"].map(day_scale).fillna(1.0)
    return scale


def build_strategies(results: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(
        baselines[
            ["zone", "ts_utc", "persist_2d", "persist_7d", "mean_7_14", "isolf_pre", "isolf_post"]
        ],
        on=["zone", "ts_utc"],
        how="inner",
    )
    df = df[df["zone"] != NYCA].copy()  # no zonal price for the statewide total
    df = df.rename(columns={"pred": "model", "p_alpha": "model_alpha_bid"})
    df = hourly_bids(df, ["persist_2d", "persist_7d", "mean_7_14", "model", "model_alpha_bid"])
    df["isolf_pre_alpha"] = df["isolf_pre"] * trailing_alpha_scale(df, "isolf_pre")
    df["model_alpha_emp"] = df["model"] * trailing_alpha_scale(df, "model")
    return df


def bootstrap_ci(daily_totals: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the pooled total from resampling days with replacement."""
    rng = np.random.default_rng(seed)
    vals = daily_totals.to_numpy(dtype=float)
    sums = rng.choice(vals, size=(n, len(vals)), replace=True).sum(axis=1)
    return float(np.quantile(sums, 0.025)), float(np.quantile(sums, 0.975))


def report(df: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = df[["ts_utc", "zone", "actual"]].rename(columns={"actual": "load_mw"})
    by_zone, pooled = [], []
    for name in STRATEGIES:
        s = settle(actual, df[name].to_numpy(), prices)
        z = summarize(s, by=["zone"])
        z.insert(0, "strategy", name)
        by_zone.append(z)
        p = summarize(s)
        p.insert(0, "strategy", name)
        daily = s.assign(day=s["ts_utc"].dt.tz_convert("America/New_York").dt.date)
        lo, hi = bootstrap_ci(daily.groupby("day")["imbalance_usd"].sum())
        p["ci95_low"], p["ci95_high"] = lo, hi
        pooled.append(p)
    return pd.concat(by_zone, ignore_index=True), pd.concat(pooled, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="default")
    args = parser.parse_args(argv)
    configure_logging()

    results = load_results(args.name)
    baselines_path = RESULTS_DIR / args.name / "baselines.csv"
    if not baselines_path.exists():
        print(
            f"{baselines_path} missing: run scripts/skill_baselines.py --name {args.name} first",
            file=sys.stderr,
        )
        return 2
    baselines = pd.read_csv(baselines_path)
    baselines["ts_utc"] = pd.to_datetime(baselines["ts_utc"], utc=True)
    data = dataset.load_all()
    prices = slot_prices(data["rt_slots"], data["da_hourly"])

    df = build_strategies(results, baselines)
    by_zone, pooled = report(df, prices)
    by_zone.to_csv(RESULTS_DIR / args.name / "imbalance_by_zone.csv", index=False)
    pooled.to_csv(RESULTS_DIR / args.name / "imbalance_pooled.csv", index=False)

    days = df["date"].nunique()
    print(
        f"imbalance cost vs perfect foresight, {df['date'].min().date()} .. {df['date'].max().date()} ({days} days, {df['zone'].nunique()} zones), hourly bids settled per 15-min slot\n"
    )
    view = pooled.set_index("strategy")[
        [
            "imbalance_usd",
            "ci95_low",
            "ci95_high",
            "usd_per_mwh",
            "pct_of_da_cost",
            "abs_imbalance_usd",
        ]
    ]
    view["usd_per_day"] = view["imbalance_usd"] / days
    money = ["imbalance_usd", "ci95_low", "ci95_high", "abs_imbalance_usd", "usd_per_day"]
    shown = view.copy()
    shown[money] = shown[money].round(0)
    shown[["usd_per_mwh", "pct_of_da_cost"]] = shown[["usd_per_mwh", "pct_of_da_cost"]].round(3)
    print(shown.to_string())
    print(
        "\nci95 = bootstrap over days (2000 resamples); overlapping intervals are not distinguishable"
    )
    alpha = df.groupby("zone")["alpha"].agg(["mean", "min", "max"]).round(3)
    print("\nα per zone (newsvendor ratio of the trailing 30-day spread, as of each cutoff):")
    print(alpha.to_string())

    best_other = view.drop(index="model_alpha_bid")["imbalance_usd"].idxmin()
    ours = view.loc["model_alpha_bid", "imbalance_usd"]
    other = view.loc[best_other, "imbalance_usd"]
    verdict = "BEATS" if ours < other else "does NOT beat"
    print(
        f"\nheadline check: model_alpha_bid ${ours:,.0f} {verdict} the strongest other strategy ({best_other}: ${other:,.0f})"
    )
    print("per-zone table written to results/<name>/imbalance_by_zone.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
