"""Compare backtest runs on exactly the same (zone, day) pairs, hourly MAPE.

    python scripts/compare_runs.py --base default --runs imp_extra imp_daytype ...
    python scripts/compare_runs.py --base lgbm24 --runs xgb24 tft24 --bootstrap

A run covering fewer zones or days than the base is scored on its own coverage and the
base is re-scored on the same subset, so partial experiments stay comparable.
``--bootstrap`` adds a paired bootstrap over days of the daily hourly-MAPE difference
(run minus base): mean, 95% interval, share of days the run is better, and the mean
difference per month, which is how the experiment log states whether a change is real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import RESULTS_DIR, load_results  # noqa: E402
from src.metrics import mape, to_hourly  # noqa: E402


def score(df: pd.DataFrame) -> pd.Series:
    h = to_hourly(df, ["pred", "actual"], keys=["zone"])
    out = {z: mape(g["actual"], g["pred"]) for z, g in h.groupby("zone")}
    out["pooled"] = mape(h["actual"], h["pred"])
    return pd.Series(out)


def daily_mape(df: pd.DataFrame) -> pd.Series:
    """Hourly MAPE of every (zone, day)."""
    h = to_hourly(df, ["pred", "actual"], keys=["zone", "date"])
    h["ape"] = (h["actual"] - h["pred"]).abs() / h["actual"] * 100
    return h.groupby(["zone", "date"])["ape"].mean()


def paired_bootstrap(
    base: pd.DataFrame, run: pd.DataFrame, draws: int = 4000, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Per zone and pooled: mean daily difference, 95% bootstrap CI, share of days better."""
    d = (daily_mape(run) - daily_mape(base)).rename("delta").reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for zone, g in [*d.groupby("zone"), ("pooled", d)]:
        v = g["delta"].to_numpy()
        boots = np.array([rng.choice(v, len(v)).mean() for _ in range(draws)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append((zone, len(v), v.mean(), lo, hi, (v < 0).mean() * 100))
    table = pd.DataFrame(
        rows, columns=["zone", "days", "mean_delta", "ci95_low", "ci95_high", "pct_days_better"]
    )
    by_month = d.groupby(pd.to_datetime(d["date"]).dt.to_period("M"))["delta"].mean()
    return table, by_month


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="default")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--bootstrap", action="store_true", help="paired bootstrap over days")
    args = parser.parse_args(argv)

    base = load_results(args.base)
    rows = {}
    boots = []
    for name in args.runs:
        path = RESULTS_DIR / name / "results.csv"
        if not path.exists():
            print(f"{name}: no results.csv yet", file=sys.stderr)
            continue
        run = load_results(name)
        keys = run[["zone", "date"]].drop_duplicates()
        sub = base.merge(keys, on=["zone", "date"], how="inner")
        b, r = score(sub), score(run)
        rows[f"{args.base} (same days)"] = b
        rows[name] = r
        rows[f"{name} Δ"] = r - b
        days = keys.groupby("zone").size().to_dict()
        print(f"{name}: {len(keys)} zone-days, days per zone {days}")
        if args.bootstrap:
            boots.append((name, *paired_bootstrap(sub, run)))
    if not rows:
        return 1
    table = pd.DataFrame(rows).round(2)
    print(
        "\nhourly MAPE (%) on the run's own (zone, day) coverage; Δ = run - base (negative is better)\n"
    )
    print(table.to_string())
    for name, table_b, by_month in boots:
        print(
            f"\n{name} - {args.base}: paired bootstrap of the daily hourly MAPE (negative = better)"
        )
        print(table_b.round(2).to_string(index=False))
        print(
            "mean difference by month: " + ", ".join(f"{m}: {v:+.2f}" for m, v in by_month.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
