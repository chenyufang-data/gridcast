"""Rolling backtest of the day-ahead model under the exact bid protocol (D-1 05:00 ET).

    python scripts/run_backtest.py                              # all 11 zones + NYCA, 12 months, 12 workers
    python scripts/run_backtest.py --zones N.Y.C. --start 2025-09-01 --end 2025-09-30 --workers 1
    python scripts/run_backtest.py --name no_weather --no-weather
    python scripts/run_backtest.py --name w28 --window 28
    python scripts/run_backtest.py --name xgb --estimator xgb --zones NYCA N.Y.C. MHK VL
    python scripts/run_backtest.py --name swap --augment swap --swap-p 0.1
    python scripts/run_backtest.py --name h12 --history-start 2025-06-01   # 12 months of data

Requires data/processed (scripts/backfill.py) and, unless --no-weather, data/weather.csv
(scripts/fetch_weather.py). Results: results/<name>/results.csv (+ per-day cache) and
results/<name>/summary.csv with hourly and 15-min MAPE per zone. Every headline number
in the README must come from this script or its siblings (docs/plan.md §4).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import weather  # noqa: E402
from app.log import configure_logging  # noqa: E402
from src import dataset  # noqa: E402
from src.backtest import BacktestConfig, run, summarize_results  # noqa: E402
from src.config import BACKTEST_END, BACKTEST_START, NYCA, ZONES  # noqa: E402
from src.settlement import slot_prices  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="default")
    parser.add_argument(
        "--zones", nargs="+", default=["all"], help="zone names or 'all' (11 zones + NYCA)"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=BACKTEST_START)
    parser.add_argument("--end", type=date.fromisoformat, default=BACKTEST_END)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--window", type=int, default=365, help="training window (days)")
    parser.add_argument("--half-life", type=float, default=90.0, help="time-decay half-life (days)")
    parser.add_argument(
        "--alpha-window", type=int, default=30, help="trailing days for the α estimate"
    )
    parser.add_argument("--no-weather", action="store_true")
    parser.add_argument(
        "--no-hourly-weather",
        dest="hourly_weather",
        action="store_false",
        help="drop temp_h (forecast C at each hour); on by default",
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument(
        "--no-extra-weather",
        dest="extra_weather",
        action="store_false",
        help="apparent temp, dew point, humidity, cloud, wind, radiation per hour",
    )
    parser.add_argument(
        "--daytype", action="store_true", help="weekend/holiday type + same-type lag anchors"
    )
    parser.add_argument("--decay-floor", type=float, default=0.0, help="minimum sample weight")
    parser.add_argument(
        "--estimator", choices=["lgbm", "xgb"], default="lgbm", help="boosting library"
    )
    parser.add_argument(
        "--augment", choices=["swap"], default=None, help="training-row augmentation"
    )
    parser.add_argument("--swap-p", type=float, default=0.1, help="swap-noise cell probability")
    parser.add_argument("--swap-copies", type=int, default=1, help="augmented copies per row")
    parser.add_argument("--swap-weight", type=float, default=0.5, help="weight of a copy")
    parser.add_argument(
        "--swap-within",
        choices=["tod", "none"],
        default="tod",
        help="donor rows share this column (tod) or come from anywhere (none)",
    )
    parser.add_argument(
        "--history-start",
        type=date.fromisoformat,
        default=None,
        help="ignore load before this day, e.g. 2025-06-01 reproduces the 12-month data runs",
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--target-mode",
        choices=["mw", "ratio"],
        default="mw",
        help="fit MW directly (default) or y / 3-week same-slot mean",
    )
    parser.add_argument(
        "--weather-lead",
        choices=["d1", "d2"],
        default="d2",
        help="d2 = issued before the cutoff (default); d1 = optimistic experiment",
    )
    args = parser.parse_args(argv)

    configure_logging()
    zones = (*ZONES, NYCA) if args.zones == ["all"] else tuple(args.zones)
    cfg = BacktestConfig(
        name=args.name,
        start=args.start,
        end=args.end,
        zones=zones,
        window_days=args.window,
        half_life=args.half_life,
        alpha_window_days=args.alpha_window,
        weather_lead=None if args.no_weather else args.weather_lead,
        target_mode=args.target_mode,
        hourly_weather=args.hourly_weather,
        workers=args.workers,
        extra_weather=args.extra_weather,
        daytype=args.daytype,
        decay_floor=args.decay_floor,
        estimator=args.estimator,
        augment=args.augment,
        augment_params={
            "p": args.swap_p,
            "copies": args.swap_copies,
            "weight": args.swap_weight,
            "within": None if args.swap_within == "none" else args.swap_within,
        },
        history_start=args.history_start,
        model_overrides={
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
        },
    )
    data = dataset.load_all()
    prices = slot_prices(data["rt_slots"], data["da_hourly"])
    wdf = None if args.no_weather else weather.load_weather()
    if not args.no_weather and wdf is None:
        print(
            "data/weather.csv missing: run scripts/fetch_weather.py or pass --no-weather",
            file=sys.stderr,
        )
        return 2

    whourly = weather.load_weather_hourly() if args.hourly_weather else None
    if args.hourly_weather and whourly is None:
        print("data/weather_hourly.csv missing: run scripts/fetch_weather.py", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    results = run(cfg, data["load_slots"], prices, wdf, whourly)
    if results.empty:
        print("no results", file=sys.stderr)
        return 1
    summary = summarize_results(results)
    summary.to_csv(cfg.out_dir / "summary.csv", index=False)
    print(
        f"\nbacktest {cfg.name!r}: {cfg.start} .. {cfg.end}, window {cfg.window_days}d, "
        f"half-life {cfg.half_life}d, weather {cfg.weather_lead or 'off'}, target {cfg.target_mode}, "
        f"hourly weather {cfg.hourly_weather}, trees {cfg.model_overrides.get('n_estimators')} "
        f"@ lr {cfg.model_overrides.get('learning_rate')}, leaves {cfg.model_overrides.get('num_leaves')}, "
        f"extra weather {cfg.extra_weather}, daytype {cfg.daytype}, decay floor {cfg.decay_floor}, "
        f"estimator {cfg.estimator}, augment {cfg.augment or 'off'}"
        f"{' ' + str(cfg.augment_params) if cfg.augment else ''}, history from {cfg.history_start or 'all data'}"
    )
    print(
        f"protocol: retrain per zone and day on slots ending <= D-1 05:00 ET; {time.perf_counter() - t0:.0f}s wall\n"
    )
    print(summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
