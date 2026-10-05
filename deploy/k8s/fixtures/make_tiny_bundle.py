"""Train and export the tiny TFT bundle that kind and CI serve (laptop only: needs torch).

The bundle is trained on the synthetic NYISO generator (tests/synthetic.py), never on
NYISO data, so it can be committed: deploy/k8s/fixtures/tft-tiny/{tft.onnx,tft.json}. It
exists to exercise the serving path on Kubernetes (the model image, the init container,
the version pin in readiness), not to forecast well.

It is tied to models.features.FEATURE_VERSION on purpose: when the features change, the
loader rejects this bundle, CI's readiness check fails, and this script must be re-run.

    python deploy/k8s/fixtures/make_tiny_bundle.py               # the committed fixture
    python deploy/k8s/fixtures/make_tiny_bundle.py --seed 2 --out <dir>   # a second model
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app import nyiso  # noqa: E402
from models import cutoff_for  # noqa: E402
from models import tft as T  # noqa: E402
from models.tft_onnx import OnnxTFT  # noqa: E402
from tests.synthetic import SyntheticNYISO  # noqa: E402

FIRST, LAST = date(2025, 8, 1), date(2025, 9, 12)
FIT_TARGET = date(2025, 9, 10)
ZONES = ("N.Y.C.", "WEST")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "tft-tiny")
    args = parser.parse_args()

    synth = SyntheticNYISO(FIRST, LAST, seed=21)
    slots = pd.concat(
        [
            nyiso.resample_slots(nyiso.normalize_pal(synth.pal_csv(d)), "load_mw")
            for d in synth.days()
        ]
    ).drop_duplicates(["zone", "ts_utc"], keep="last")
    until = LAST + timedelta(days=1)
    cutoff = cutoff_for(FIT_TARGET)
    data = {}
    for zone in ZONES:
        hist = slots[(slots["zone"] == zone) & (slots["ts_utc"] < cutoff_for(until))]
        zd = T.prepare_zone(
            zone, hist[["ts_utc", "load_mw", "coverage"]].reset_index(drop=True), None, until
        )
        data[zone] = (zd, T.series_asof(zd, cutoff)[0])
    model = T.TFTForecaster(
        list(ZONES), enc_days=2, hidden=8, heads=2, epochs=2, batch_size=16, val_days=3,
        device="cpu", seed=args.seed,
    )  # fmt: skip
    model.fit(data, [date(2025, 8, 5) + timedelta(days=k) for k in range(34)])
    args.out.mkdir(parents=True, exist_ok=True)
    model.export(args.out, fit_cutoff=cutoff.isoformat(), window_days=34)
    bundle = OnnxTFT.load(args.out)
    size = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"wrote {args.out} ({size / 1024:.0f} KiB)")
    print(f"TFT_EXPECTED_VERSION={bundle.version}")


if __name__ == "__main__":
    main()
