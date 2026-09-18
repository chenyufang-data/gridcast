"""Backtest of the Temporal Fusion Transformer (models/tft.py) with periodic refits.

    python scripts/run_tft.py --name tft24 --zones NYCA N.Y.C. "MHK VL" --train-zones all
    python scripts/run_tft.py --name tft_w7 --refit-days 7 --epochs 60

Same protocol as run_backtest.py (targets, cutoff D-1 05:00 ET, α from src.settlement,
per-day cache under results/<name>/<zone>/, results.csv + summary.csv) with one
difference: the network is fitted once per `--refit-days` block of target days on the
`--window` ending at the block's first cutoff, then every day of the block is forecast
with an encoder that stops at that day's own cutoff. `--train-zones all` fits one global
model on every zone with data and forecasts only `--zones`. Needs torch
(requirements-research.txt); runs on CUDA when available.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app import weather  # noqa: E402
from app.log import configure_logging  # noqa: E402
from app.weather import hourly_features_for  # noqa: E402
from models import cutoff_for  # noqa: E402
from models import tft as T  # noqa: E402
from src import dataset  # noqa: E402
from src.backtest import (  # noqa: E402
    RESULT_COLUMNS,
    BacktestConfig,
    day_path,
    restrict_history,
    summarize_results,
)
from src.config import BACKTEST_END, BACKTEST_START, NYCA, ZONES  # noqa: E402
from src.settlement import alpha_series, slot_prices  # noqa: E402

log = logging.getLogger("run_tft")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="tft")
    parser.add_argument("--zones", nargs="+", default=["all"], help="zones to forecast")
    parser.add_argument(
        "--train-zones", nargs="+", default=["run"], help="'run' (= --zones), 'all', or names"
    )
    parser.add_argument("--start", type=date.fromisoformat, default=BACKTEST_START)
    parser.add_argument("--end", type=date.fromisoformat, default=BACKTEST_END)
    parser.add_argument("--window", type=int, default=365, help="training window (days)")
    parser.add_argument("--refit-days", type=int, default=30, help="days served by one fit")
    parser.add_argument("--enc-days", type=int, default=7, help="encoder length in days")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60, help="upper bound; early stopping")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-days", type=int, default=14, help="last days of the window")
    parser.add_argument("--alpha-window", type=int, default=30)
    parser.add_argument("--history-start", type=date.fromisoformat, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)

    configure_logging()
    zones = (*ZONES, NYCA) if args.zones == ["all"] else tuple(args.zones)
    if args.train_zones == ["run"]:
        train_zones = zones
    elif args.train_zones == ["all"]:
        train_zones = (*ZONES, NYCA)
    else:
        train_zones = tuple(dict.fromkeys([*args.train_zones, *zones]))
    cfg = BacktestConfig(
        name=args.name,
        start=args.start,
        end=args.end,
        zones=zones,
        window_days=args.window,
        alpha_window_days=args.alpha_window,
        estimator="tft",
        history_start=args.history_start,
    )
    data = dataset.load_all()
    load_slots = restrict_history(data["load_slots"], cfg.history_start)
    prices = slot_prices(data["rt_slots"], data["da_hourly"])
    whourly = weather.load_weather_hourly()
    if whourly is None:
        print("data/weather_hourly.csv missing: run scripts/fetch_weather.py", file=sys.stderr)
        return 2
    targets = cfg.targets()
    zd: dict[str, T.ZoneData] = {}
    for z in train_zones:
        zs = load_slots[load_slots["zone"] == z].reset_index(drop=True)
        if zs.empty:
            print(f"no load data for {z}", file=sys.stderr)
            return 2
        zd[z] = T.prepare_zone(z, zs, hourly_features_for(whourly, z, extra=True), cfg.end)
    alphas = {
        z: alpha_series(prices, z, targets, cfg.alpha_window_days).to_dict() if z != NYCA else {}
        for z in zones
    }
    actual = load_slots[["zone", "ts_utc", "load_mw"]].rename(columns={"load_mw": "actual"})
    forecaster = T.TFTForecaster(
        list(train_zones),
        enc_days=args.enc_days,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        val_days=args.val_days,
        device=args.device,
        seed=args.seed,
    )
    log.info(
        "tft backtest %s: forecast %s, train on %d zones, %d days, refit every %d days, %s",
        cfg.name,
        zones,
        len(train_zones),
        len(targets),
        args.refit_days,
        T.describe(forecaster),
    )
    t0 = time.perf_counter()
    frames = []
    fits = 0
    for block in T.periods(targets, args.refit_days):
        pending = [(z, d) for d in block for z in zones if not day_path(cfg, z, d).exists()]
        tod_means: dict[str, dict[int, float]] = {}
        if pending:
            d0 = block[0]
            c0 = cutoff_for(d0)
            fit_data = {}
            for z in train_zones:
                y0, tod_means[z] = T.series_asof(zd[z], c0)
                fit_data[z] = (zd[z], y0)
            train_days = [d0 - timedelta(days=k) for k in range(args.window + 1, 1, -1)]
            forecaster.fit(fit_data, train_days)
            fits += 1
        for d in block:
            for z in zones:
                path = day_path(cfg, z, d)
                if path.exists():
                    frames.append(pd.read_csv(path, parse_dates=["ts_utc", "date"]))
                    continue
                y_d, _ = T.series_asof(zd[z], cutoff_for(d), tod_means[z])
                try:
                    q = forecaster.predict(zd[z], y_d, d)
                except ValueError as exc:
                    log.warning("%s %s skipped: %s", z, d, exc)
                    continue
                out = T.to_result_rows(q, forecaster.quantiles, alphas[z].get(d, 0.5))
                out.insert(0, "zone", z)
                out = out.merge(
                    actual[actual["zone"] == z].drop(columns="zone"), on="ts_utc", how="left"
                )
                out = out[RESULT_COLUMNS]
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(path, index=False)
                frames.append(out)
        done = sum(1 for f in frames)
        log.info(
            "block %s .. %s done (%d zone-days so far, %d fits)", block[0], block[-1], done, fits
        )
    if not frames:
        print("no results", file=sys.stderr)
        return 1
    results = pd.concat(frames, ignore_index=True)
    results["ts_utc"] = pd.to_datetime(results["ts_utc"], utc=True)
    results = results.sort_values(["zone", "ts_utc"]).reset_index(drop=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.out_dir / "results.csv", index=False)
    summary = summarize_results(results)
    summary.to_csv(cfg.out_dir / "summary.csv", index=False)
    print(
        f"\ntft {cfg.name!r}: {cfg.start} .. {cfg.end}, window {cfg.window_days}d, refit every "
        f"{args.refit_days}d ({fits} fits this run), train zones {len(train_zones)}, "
        f"{T.describe(forecaster)}, history from {cfg.history_start or 'all data'}"
    )
    print(
        f"protocol: encoder stops at D-1 05:00 ET for every day; {time.perf_counter() - t0:.0f}s wall\n"
    )
    print(summary.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
