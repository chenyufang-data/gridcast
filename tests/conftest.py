"""Shared test setup: isolated env before app modules import; synthetic archives.

APP_DB_PATH / NYISO_CACHE_DIR / TFT_BUNDLE_DIR / WEATHER_PATH must be set before anything
imports the app modules (they read the env at import time), which is why this happens at
conftest import.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault(
    "APP_DB_PATH", os.path.join(tempfile.mkdtemp(prefix="gridcast_test_"), "app.db")
)
os.environ.setdefault("NYISO_CACHE_DIR", tempfile.mkdtemp(prefix="gridcast_cache_"))
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("LOG_LEVEL", "WARNING")
# the service must never see the laptop's real bundle or weather files, refresh weather
# over the network, or start the scheduler thread under pytest
_isolated = tempfile.mkdtemp(prefix="gridcast_models_")
os.environ.setdefault("TFT_BUNDLE_DIR", os.path.join(_isolated, "tft"))
os.environ.setdefault("WEATHER_PATH", os.path.join(_isolated, "weather.csv"))
os.environ.setdefault("WEATHER_HOURLY_PATH", os.path.join(_isolated, "weather_hourly.csv"))
os.environ.setdefault("SCHEDULER_ENABLED", "0")
os.environ.setdefault("WEATHER_REFRESH", "0")

from tests.synthetic import SyntheticNYISO  # noqa: E402

# DST transition days inside the backtest range (docs/plan.md §2.2)
FALL_BACK_DAY = date(2025, 11, 2)  # 25 local hours: 01:xx listed twice (EDT then EST)
SPRING_FORWARD_DAY = date(2026, 3, 8)  # 23 local hours: no 02:xx


@pytest.fixture(scope="session")
def synth_autumn() -> SyntheticNYISO:
    """One week around the fall-back day, with off-schedule RTD stamps."""
    return SyntheticNYISO(date(2025, 10, 30), date(2025, 11, 5), seed=1)


@pytest.fixture(scope="session")
def synth_spring() -> SyntheticNYISO:
    """Five days around the spring-forward day, clean grid only."""
    return SyntheticNYISO(date(2026, 3, 6), date(2026, 3, 10), seed=2, irregular=False)


@pytest.fixture(scope="session")
def synth_archive(tmp_path_factory: pytest.TempPathFactory, synth_autumn: SyntheticNYISO) -> Path:
    """The autumn week written in the archive layout (daily files only)."""
    root = tmp_path_factory.mktemp("archive")
    synth_autumn.write_archive(root)
    return root
