"""Fit the TFT on the latest window and export the served bundle (ONNX + constants).

    python scripts/export_tft.py                       # tomorrow's cutoff, all zones -> data/models/tft
    python scripts/export_tft.py --target 2026-09-20 --out data/models/tft

The fit uses the `--window` days before the cutoff of `--target` (default: tomorrow in
ET), on every zone, with the same settings as scripts/run_tft.py. The export pins eval
mode and is verified against torch before the file is written; the bundle is then
re-loaded with onnxruntime (no torch) and a forecast of `--target` is printed per zone
as a final check. Upload the two files of the bundle to the VM's data volume.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import weather  # noqa: E402
from app.log import configure_logging  # noqa: E402
from app.nyiso import PROJECT_ROOT  # noqa: E402
from app.weather import hourly_features_for  # noqa: E402
from models import SLOT, cutoff_for  # noqa: E402
from models import tft as T  # noqa: E402
from models.tft_onnx import OnnxTFT  # noqa: E402
from src import dataset  # noqa: E402
from src.config import MARKET_TZ, NYCA, ZONES  # noqa: E402

log = logging.getLogger("export_tft")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "models" / "tft")
    parser.add_argument(
        "--target",
        type=date.fromisoformat,
        default=None,
        help="first day the bundle will forecast (default: tomorrow, ET)",
    )
    parser.add_argument("--window", type=int, default=365)
    parser.add_argument("--enc-days", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    target = args.target or (datetime.now(MARKET_TZ).date() + timedelta(days=1))
    cutoff = cutoff_for(target)
    zones = (*ZONES, NYCA)
    data = dataset.load_all()
    whourly = weather.load_weather_hourly()
    if whourly is None:
        print("data/weather_hourly.csv missing: run scripts/fetch_weather.py", file=sys.stderr)
        return 2
    load_slots = data["load_slots"]
    last_slot = load_slots["ts_utc"].max()
    if last_slot + SLOT * 4 * 24 < cutoff:
        log.warning("load data end %s, more than a day before the cutoff %s", last_slot, cutoff)
    zd = {}
    fit_data = {}
    for z in zones:
        zs = load_slots[load_slots["zone"] == z].reset_index(drop=True)
        zd[z] = T.prepare_zone(z, zs, hourly_features_for(whourly, z, extra=True), target)
        fit_data[z] = (zd[z], T.series_asof(zd[z], cutoff)[0])
    forecaster = T.TFTForecaster(
        list(zones),
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
    train_days = [target - timedelta(days=k) for k in range(args.window + 1, 1, -1)]
    t0 = time.perf_counter()
    forecaster.fit(fit_data, train_days)
    forecaster.export(args.out, fit_cutoff=cutoff.isoformat(), window_days=args.window)
    served = OnnxTFT.load(args.out)
    print(
        f"\nbundle {args.out}: {served.version}, fit on {args.window} d before {cutoff} "
        f"({len(forecaster.history)} epochs, {time.perf_counter() - t0:.0f}s), "
        f"max |onnx - torch| {served.meta['max_abs_diff_vs_torch']:.1e}\n"
    )
    print(f"forecast of {target} from the bundle (peak MW of the median, P10-P90 at the peak):")
    for z in zones:
        y_d, _ = T.series_asof(zd[z], cutoff)
        rows = served.forecast(zd[z], y_d, target)
        i = int(rows["pred"].idxmax())
        print(
            f"  {z:7s} {rows['pred'].max():8.0f} MW at {rows['ts_utc'].iloc[i].tz_convert(MARKET_TZ):%H:%M} ET "
            f"[{rows['p10'].iloc[i]:.0f}, {rows['p90'].iloc[i]:.0f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
