"""The served models: the ONNX TFT bundle (primary) and the LightGBM trees (fallback).

Decision (CLAUDE.md, 2026-09-18): the Temporal Fusion Transformer is fitted and exported
on the laptop (``scripts/export_tft.py``) and its two-file bundle is copied to the data
volume; the VM runs it with onnxruntime only. The trees are trained on demand on the VM
and take over whenever the bundle is missing, stale, rejected or lacks a zone.

``ModelRegistry`` owns the loaded bundle: it reloads when the file on disk changes, checks
its age against ``TFT_MAX_AGE_DAYS`` and reports a status block for ``/health`` and
``/models``. Every forecast records which model produced it (``tft-onnx:<sha12>:<fit
cutoff>`` or ``lgbm:<FEATURE_VERSION>``), so a served row is always attributable.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from models import FEATURE_VERSION, cutoff_for
from models.tft_onnx import OnnxTFT

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(os.environ.get("TFT_BUNDLE_DIR") or PROJECT_ROOT / "data" / "models" / "tft")
MAX_AGE_DAYS = int(os.environ.get("TFT_MAX_AGE_DAYS") or 60)
MODEL_THREADS = int(os.environ.get("MODEL_THREADS") or 2)

# the trees: the backtest defaults (src.backtest.BacktestConfig), the same everywhere
TREE_WINDOW_DAYS = 365
TREE_HALF_LIFE = 90.0
TREE_DECAY_FLOOR = 0.0
TREE_QUANTILES = (0.1, 0.9)
LGBM_IDENTITY = f"lgbm:{FEATURE_VERSION}"


def bundle_signature(bundle_dir: Path) -> str | None:
    """Cheap change detector for the bundle: sizes and mtimes of its two files."""
    parts = []
    for name in ("tft.onnx", "tft.json"):
        p = bundle_dir / name
        if not p.exists():
            return None
        st = p.stat()
        parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


class ModelRegistry:
    """Loads the TFT bundle lazily and keeps it while the files on disk are unchanged."""

    def __init__(self, bundle_dir: Path | None = None, max_age_days: int | None = None) -> None:
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else BUNDLE_DIR
        self.max_age_days = max_age_days if max_age_days is not None else MAX_AGE_DAYS
        self.tft: OnnxTFT | None = None
        self.error: str | None = None
        self._signature: str | None = None
        self._lock = threading.Lock()

    def reload(self, force: bool = False) -> dict[str, Any]:
        """(Re)load the bundle if it changed on disk; never raises."""
        with self._lock:
            sig = bundle_signature(self.bundle_dir)
            if sig is None:
                if self.tft is not None:
                    log.warning("TFT bundle removed from %s; serving the trees", self.bundle_dir)
                self.tft, self.error, self._signature = None, "bundle missing", None
                return self.status()
            if sig == self._signature and not force:
                return self.status()
            try:
                self.tft = OnnxTFT.load(self.bundle_dir)
                self.error = None
            except Exception as exc:
                self.tft = None
                self.error = f"rejected: {exc}"
                log.error("TFT bundle %s rejected: %s", self.bundle_dir, exc)
            self._signature = sig
            return self.status()

    # --- freshness --------------------------------------------------------------------
    def fit_cutoff(self) -> pd.Timestamp | None:
        if self.tft is None:
            return None
        raw = self.tft.meta.get("fit_cutoff")
        return pd.Timestamp(raw).tz_convert("UTC") if raw else None

    def age_days(self, target: date) -> float | None:
        fit = self.fit_cutoff()
        if fit is None:
            return None
        return float((cutoff_for(target) - fit) / pd.Timedelta(days=1))

    def tft_for(self, zone: str, target: date) -> tuple[OnnxTFT | None, str | None]:
        """The bundle if it may serve `zone` on `target`, else ``(None, reason)``."""
        self.reload()
        if self.tft is None:
            return None, self.error or "bundle missing"
        if zone not in self.tft.zone_id:
            return None, f"zone {zone!r} not in the bundle"
        age = self.age_days(target)
        if age is not None and age > self.max_age_days:
            return None, f"bundle is {age:.0f} days old (limit {self.max_age_days})"
        return self.tft, None

    def status(self, target: date | None = None) -> dict[str, Any]:
        tft: dict[str, Any] = {
            "bundle_dir": str(self.bundle_dir),
            "loaded": self.tft is not None,
            "error": self.error,
        }
        if self.tft is not None:
            fit = self.fit_cutoff()
            tft.update(
                version=self.tft.version,
                fit_cutoff=fit.isoformat() if fit is not None else None,
                zones=self.tft.zones,
                feature_version=self.tft.meta.get("feature_version"),
                max_age_days=self.max_age_days,
            )
            if target is not None:
                age = self.age_days(target)
                tft["age_days"] = None if age is None else round(age, 1)
                tft["stale"] = age is not None and age > self.max_age_days
        return {
            "tft": tft,
            "lgbm": {
                "identity": LGBM_IDENTITY,
                "window_days": TREE_WINDOW_DAYS,
                "half_life_days": TREE_HALF_LIFE,
                "threads": MODEL_THREADS,
            },
        }


registry = ModelRegistry()
