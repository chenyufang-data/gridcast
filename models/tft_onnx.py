"""The served TFT: an ONNX file plus its constants, run with onnxruntime (no torch).

A bundle directory holds ``tft.onnx`` and ``tft.json`` as written by
:meth:`models.tft.TFTForecaster.export` on the laptop: the zone list and per-zone
scales, the known-input normalisation, window lengths, quantile grid, feature version
and the SHA-256 of the graph, which :meth:`OnnxTFT.load` verifies before it trusts the
file. Forecasting mirrors :meth:`models.tft.TFTForecaster.predict` exactly (same
:func:`models.tft_data.sample_arrays`, same :func:`models.tft_data.quantile_frame`).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.features import FEATURE_VERSION
from models.tft_data import (
    ONNX_FORMAT,
    ONNX_INPUTS,
    ZoneData,
    check_sample,
    quantile_frame,
    sample_arrays,
    sha256_of,
    to_result_rows,
)

log = logging.getLogger(__name__)


class OnnxTFT:
    """Inference-only TFT loaded from an export bundle."""

    def __init__(self, session: Any, meta: dict[str, Any], path: Path) -> None:
        self.session = session
        self.meta = meta
        self.path = path
        self.zones: list[str] = list(meta["zones"])
        self.zone_id = {z: i for i, z in enumerate(self.zones)}
        self.scale: dict[str, float] = {z: float(v) for z, v in meta["scale"].items()}
        self.enc_len = int(meta["enc_len"])
        self.quantiles: tuple[float, ...] = tuple(float(q) for q in meta["quantiles"])
        self.known_cols: list[str] = list(meta["known_cols"])
        self.wmean = np.asarray(meta["wmean"], dtype=np.float32)
        self.wstd = np.asarray(meta["wstd"], dtype=np.float32)

    @classmethod
    def load(cls, bundle_dir: Path | str) -> OnnxTFT:
        """Open a bundle; raises ``ValueError`` on a format, feature-version or hash mismatch."""
        import onnxruntime as ort

        bundle = Path(bundle_dir)
        meta = json.loads((bundle / "tft.json").read_text(encoding="utf-8"))
        if meta.get("format") != ONNX_FORMAT:
            raise ValueError(f"unknown bundle format {meta.get('format')!r}")
        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"bundle built for features {meta.get('feature_version')!r}, "
                f"code has {FEATURE_VERSION!r}"
            )
        onnx_path = bundle / "tft.onnx"
        digest = sha256_of(onnx_path)
        if digest != meta.get("onnx_sha256"):
            raise ValueError("tft.onnx does not match the hash in tft.json")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        log.info(
            "loaded ONNX TFT %s (fit cutoff %s, %d zones)",
            bundle,
            meta.get("fit_cutoff"),
            len(meta["zones"]),
        )
        return cls(session, meta, bundle)

    @property
    def version(self) -> str:
        """Short identity of the served weights for the forecast version hash."""
        return f"tft-onnx:{self.meta['onnx_sha256'][:12]}:{self.meta.get('fit_cutoff')}"

    def predict(self, zd: ZoneData, y_asof: np.ndarray, target: date) -> pd.DataFrame:
        """Quantile forecast of `target`'s slots for a zone the bundle was trained on."""
        if zd.zone not in self.zone_id:
            raise ValueError(f"zone {zd.zone!r} not in the bundle")
        if zd.known_cols != self.known_cols:
            raise ValueError(
                f"known inputs {zd.known_cols} differ from the bundle's {self.known_cols}"
            )
        zd.scale = self.scale[zd.zone]
        s = check_sample(zd, self.zone_id[zd.zone], y_asof, target, self.enc_len)
        arrays = sample_arrays(zd, y_asof, s, self.enc_len, self.wmean, self.wstd)
        feed = {k: arrays[k][None] for k in ONNX_INPUTS}
        out = self.session.run(None, feed)[0][0]  # [DEC_LEN, Q], scaled
        return quantile_frame(zd, s, out * zd.scale, target, self.quantiles)

    def forecast(
        self, zd: ZoneData, y_asof: np.ndarray, target: date, alpha: float = 0.5
    ) -> pd.DataFrame:
        """``pred, p10, p90, p_alpha, alpha`` rows for `target` (the backtest result layout)."""
        return to_result_rows(self.predict(zd, y_asof, target), self.quantiles, alpha)
