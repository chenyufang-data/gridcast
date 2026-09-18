"""Empirical calibration of the P10-P90 band, before and after a conformal adjustment.

    python scripts/quantile_calibration.py --name default

The band comes from quantile LightGBM at α = 0.1 / 0.9 (nominal 80%). Tree quantiles are
typically too narrow out of sample, so this also evaluates a simple split-conformal
rescaling: for each day D, the 0.9-quantile of actual/P90 (and 0.1-quantile of
actual/P10) over the trailing 30 days ending at D-2 scales the band. Both are scored at
15-min and hourly resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.log import configure_logging  # noqa: E402
from src.backtest import RESULTS_DIR, load_results  # noqa: E402
from src.metrics import to_hourly  # noqa: E402

WINDOW_DAYS = 30


def conformal_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Per (zone, date): trailing-window quantile scale for p10 and p90 (1.0 without history)."""
    d = df[["zone", "date", "actual", "p10", "p90"]].copy()
    d["r10"] = d["actual"] / d["p10"]
    d["r90"] = d["actual"] / d["p90"]
    rows = []
    for zone, z in d.groupby("zone"):
        by_day = {day: g for day, g in z.groupby("date")}
        days = sorted(by_day)
        for day in days:
            lo, hi = day - pd.Timedelta(days=WINDOW_DAYS + 1), day - pd.Timedelta(days=2)
            hist = [by_day[x] for x in days if lo <= x <= hi]
            if len(hist) < 7:
                rows.append({"zone": zone, "date": day, "s10": 1.0, "s90": 1.0})
                continue
            h = pd.concat(hist)
            rows.append(
                {
                    "zone": zone,
                    "date": day,
                    "s10": float(np.nanquantile(h["r10"], 0.1)),
                    "s90": float(np.nanquantile(h["r90"], 0.9)),
                }
            )
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame, lo: str, hi: str) -> dict[str, float]:
    inside = (df[lo] <= df["actual"]) & (df["actual"] <= df[hi])
    return {
        "coverage": float(inside.mean() * 100),
        "below": float((df["actual"] < df[lo]).mean() * 100),
        "above": float((df["actual"] > df[hi]).mean() * 100),
        "width_pct": float(((df[hi] - df[lo]) / df["actual"]).mean() * 100),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="default")
    args = parser.parse_args(argv)
    configure_logging()

    df = load_results(args.name).dropna(subset=["actual"])
    scales = conformal_scales(df)
    df = df.merge(scales, on=["zone", "date"], how="left")
    df["p10_c"] = df["p10"] * df["s10"]
    df["p90_c"] = df["p90"] * df["s90"]
    hourly = to_hourly(df, ["actual", "p10", "p90", "p10_c", "p90_c"], keys=["zone"])

    rows = []
    for zone in [*sorted(df["zone"].unique()), "pooled"]:
        r = df if zone == "pooled" else df[df["zone"] == zone]
        h = hourly if zone == "pooled" else hourly[hourly["zone"] == zone]
        raw_s, con_s = coverage(r, "p10", "p90"), coverage(r, "p10_c", "p90_c")
        raw_h, con_h = coverage(h, "p10", "p90"), coverage(h, "p10_c", "p90_c")
        rows.append(
            {
                "zone": zone,
                "slot_raw": raw_s["coverage"],
                "slot_conformal": con_s["coverage"],
                "hourly_raw": raw_h["coverage"],
                "hourly_conformal": con_h["coverage"],
                "width_raw_pct": raw_s["width_pct"],
                "width_conformal_pct": con_s["width_pct"],
                "below_conf": con_s["below"],
                "above_conf": con_s["above"],
            }  # fmt: skip
        )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / args.name / "calibration.csv", index=False)
    print(
        f"P10-P90 coverage (%), nominal 80, {df['date'].min().date()} .. {df['date'].max().date()}; conformal = trailing-{WINDOW_DAYS}-day ratio quantiles as of D-2\n"
    )
    print(table.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
