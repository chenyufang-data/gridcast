"""TFT export bundle: ONNX + constants round trip, torch-free inference, hash and version checks."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from models import cutoff_for
from tests.synthetic import SyntheticNYISO
from tests.test_model import zone_slots

T = pytest.importorskip("models.tft")
pytest.importorskip("onnxruntime")
from models.tft_onnx import OnnxTFT  # noqa: E402


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, dict[str, Any], Path, date]:
    synth = SyntheticNYISO(date(2025, 8, 1), date(2025, 9, 12), seed=9, irregular=False)
    zones = {
        z: T.prepare_zone(z, zone_slots(synth, z), None, date(2025, 9, 12))
        for z in ("N.Y.C.", "WEST")
    }
    d0 = date(2025, 9, 10)
    c0 = cutoff_for(d0)
    data = {z: (zd, T.series_asof(zd, c0)[0]) for z, zd in zones.items()}
    f = T.TFTForecaster(
        list(zones),
        enc_days=2,
        hidden=8,
        heads=2,
        epochs=2,
        batch_size=16,
        val_days=3,
        device="cpu",
    )
    f.fit(data, [date(2025, 8, 5) + timedelta(days=k) for k in range(34)])
    out = tmp_path_factory.mktemp("bundle")
    f.export(out, fit_cutoff=c0.isoformat(), window_days=34)
    return f, zones, out, d0


@pytest.mark.slow
def test_bundle_round_trip(fitted: tuple[Any, dict[str, Any], Path, date]) -> None:
    f, zones, out, d0 = fitted
    meta = json.loads((out / "tft.json").read_text(encoding="utf-8"))
    assert meta["format"] == T.ONNX_FORMAT and meta["zones"] == ["N.Y.C.", "WEST"]
    assert set(meta["scale"]) == {"N.Y.C.", "WEST"} and meta["max_abs_diff_vs_torch"] < 1e-4
    assert meta["onnx_sha256"] == T.sha256_of(out / "tft.onnx") and meta["enc_len"] == 2 * 96
    served = OnnxTFT.load(out)
    assert served.version.startswith("tft-onnx:") and served.quantiles == T.QUANTILES
    for z in ("N.Y.C.", "WEST"):
        zd = zones[z]
        y_d, _ = T.series_asof(zd, cutoff_for(d0))
        ref = f.predict(zd, y_d, d0)
        got = served.predict(zd, y_d, d0)
        qcols = [c for c in ref.columns if c.startswith("q")]
        assert len(got) == 96 and list(got.columns) == list(ref.columns)
        assert np.abs(got[qcols].to_numpy() - ref[qcols].to_numpy()).max() < 1e-2  # MW
        rows = served.forecast(zd, y_d, d0, alpha=0.4)
        assert list(rows.columns) == [
            "ts_utc",
            "date",
            "slot",
            "tod",
            "pred",
            "p10",
            "p90",
            "p_alpha",
            "alpha",
        ]
        assert (rows["p10"] <= rows["pred"]).all() and (rows["pred"] <= rows["p90"]).all()
    # the encoder may not see anything at or after the cutoff, in the served path too
    leaky = y_d.copy()
    leaky[np.isnan(leaky)] = 1.0
    with pytest.raises(ValueError, match="past the cutoff"):
        served.predict(zones["WEST"], leaky, d0)


@pytest.mark.slow
def test_bundle_rejects_tampering_and_version(
    fitted: tuple[Any, dict[str, Any], Path, date], tmp_path: Path
) -> None:
    _, _, out, _ = fitted
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "tft.onnx").write_bytes((out / "tft.onnx").read_bytes() + b"\0")
    (bad / "tft.json").write_text((out / "tft.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        OnnxTFT.load(bad)
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "tft.onnx").write_bytes((out / "tft.onnx").read_bytes())
    meta = json.loads((out / "tft.json").read_text(encoding="utf-8"))
    meta["feature_version"] = "nyiso-fv0"
    (stale / "tft.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="features"):
        OnnxTFT.load(stale)
