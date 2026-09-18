"""Naive and ISO baselines under the exact backtest protocol, next to the model's MAPE.

    python scripts/skill_baselines.py --name default

Reads results/<name>/results.csv (zones and days of that run), computes every baseline
in src/baselines.py for the same (zone, day) pairs from data/processed, and prints the
hourly MAPE table. Two ISO benchmarks are shown (data/README.md): ``isolf_pre`` is the
file named D-1 (known before the D-1 05:00 ET cutoff, the fair comparison) and
``isolf_post`` the file named D (posted after the close). Writes
results/<name>/baselines.csv for the imbalance report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.log import configure_logging  # noqa: E402
from src import dataset  # noqa: E402
from src.backtest import RESULTS_DIR, load_results  # noqa: E402
from src.baselines import ALL, all_baselines  # noqa: E402
from src.metrics import score_table, to_hourly  # noqa: E402


def build_baselines(
    results: pd.DataFrame, load_slots: pd.DataFrame, isolf: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for zone, r in results.groupby("zone"):
        targets = sorted({d.date() for d in pd.to_datetime(r["date"])})
        frames.append(
            all_baselines(load_slots[load_slots["zone"] == zone], isolf, str(zone), targets)
        )
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="default")
    args = parser.parse_args(argv)
    configure_logging()

    results = load_results(args.name)
    load_slots, isolf = dataset.load("load_slots"), dataset.load("isolf")
    baselines = build_baselines(results, load_slots, isolf)
    out_path = RESULTS_DIR / args.name / "baselines.csv"
    baselines.to_csv(out_path, index=False)

    merged = baselines.merge(
        results[["zone", "ts_utc", "pred", "p_alpha"]], on=["zone", "ts_utc"], how="inner"
    )
    cols = ["pred", *ALL]
    hourly = to_hourly(merged, [*cols, "actual"], keys=["zone"])
    table = score_table(hourly, cols).rename(columns={"pred": "model"})
    table.to_csv(RESULTS_DIR / args.name / "baselines_summary.csv", index=False)
    print(
        f"hourly MAPE (%), {results['date'].min().date()} .. {results['date'].max().date()}, data as of D-1 05:00 ET\n"
    )
    print(table.round(2).to_string(index=False))
    pooled = table[table["zone"] == "pooled"].iloc[0]
    best_naive = min(pooled[c] for c in ("persist_2d", "persist_7d", "mean_7_14"))
    print(
        f"\nmodel vs best naive: {(best_naive - pooled['model']) / best_naive * 100:+.1f}% relative error reduction"
    )
    print(
        f"model vs isolf_pre (fair): {(pooled['isolf_pre'] - pooled['model']) / pooled['isolf_pre'] * 100:+.1f}%"
    )
    print(
        f"model vs isolf_post (post-close reference): {(pooled['isolf_post'] - pooled['model']) / pooled['isolf_post'] * 100:+.1f}%"
    )
    print(f"\nbaselines written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
