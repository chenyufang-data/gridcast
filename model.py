"""Modeling core (features, anomaly repair, LightGBM wrapper): ported in Phase 2.

Kept at the repo root and imported by both src/ (offline backtest) and app/ (service),
exactly like the source repo. The Dockerfile copies this file, so the layout is stable
from Phase 0 on.
"""

from __future__ import annotations
