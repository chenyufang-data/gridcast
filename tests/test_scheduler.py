"""Scheduler timing (ET wall clock, DST, monthly jobs, catch-up), the write rate limiter,
the weather splice/trim helpers and the store's timestamp conventions."""

from __future__ import annotations

from datetime import UTC, date, datetime
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import pytest

from app import db, scheduler, weather
from app.main import TokenBucket


def _job(at: str, monthly: bool = False) -> scheduler.Job:
    hh, mm = at.split(":")
    return scheduler.Job("t", dtime(int(hh), int(mm)), lambda: None, monthly=monthly)


def test_last_occurrence_is_et_wall_clock_across_dst() -> None:
    job = _job("04:30")
    # EDT: 04:30 ET = 08:30 UTC
    assert scheduler.last_occurrence(job, datetime(2025, 10, 1, 9, 0, tzinfo=UTC)) == datetime(
        2025, 10, 1, 8, 30, tzinfo=UTC
    )
    # before today's slot: yesterday's occurrence
    assert scheduler.last_occurrence(job, datetime(2025, 10, 1, 8, 0, tzinfo=UTC)) == datetime(
        2025, 9, 30, 8, 30, tzinfo=UTC
    )
    # EST: 04:30 ET = 09:30 UTC; the day after fall-back
    assert scheduler.last_occurrence(job, datetime(2025, 11, 3, 12, 0, tzinfo=UTC)) == datetime(
        2025, 11, 3, 9, 30, tzinfo=UTC
    )


def test_monthly_job_only_on_the_first() -> None:
    job = _job("07:00", monthly=True)
    got = scheduler.last_occurrence(job, datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
    assert got.astimezone(scheduler.MARKET_TZ).date() == date(2026, 3, 1)


def test_is_due_never_run_and_catch_up() -> None:
    job = _job("06:30")
    now = datetime(2025, 10, 1, 12, 0, tzinfo=UTC)  # 08:00 ET
    assert scheduler.is_due(job, now, None)
    assert scheduler.is_due(job, now, "2025-09-30 10:31:00")  # ran yesterday -> due
    assert not scheduler.is_due(job, now, "2025-10-01 10:31:00")  # ran today after 06:30 ET
    assert not scheduler.is_due(job, now, "2025-10-01 11:00:00")


def test_token_bucket_refills() -> None:
    b = TokenBucket(rate_per_min=60, burst=2)
    assert b.allow("a", now=0.0) and b.allow("a", now=0.0)
    assert not b.allow("a", now=0.0)
    assert b.allow("b", now=0.0)  # separate key
    assert b.allow("a", now=1.0)  # one token per second at 60/min


def test_weather_merge_and_trim(tmp_path: Path) -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-09-01", "2025-09-02", "2025-09-03"] * 2),
            "zone": ["WEST"] * 3 + ["NYCA"] * 3,
            "tfc2_mean": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    new = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-09-03", "2025-09-04"] * 2),
            "zone": ["WEST", "WEST", "NYCA", "NYCA"],
            "tfc2_mean": [30.0, 40.0, 30.0, 40.0],
        }
    )
    merged = weather._merge(daily, new, "date")
    assert len(merged) == 8  # 09-01, 09-02 kept; 09-03 replaced; 09-04 added
    assert (
        merged.loc[
            (merged["zone"] == "WEST") & (merged["date"] == "2025-09-03"), "tfc2_mean"
        ].item()
        == 30.0
    )

    path = tmp_path / "w.csv"
    merged.to_csv(path, index=False)
    hourly_path = tmp_path / "h.csv"
    pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-09-01 22:00", periods=8, freq="h", tz="UTC"),
            "zone": "WEST",
            "tfc2": range(8),
        }
    ).to_csv(hourly_path, index=False)
    removed = weather.trim(date(2025, 9, 2), path, hourly_path)
    # daily: 09-01 x two zones; hourly: 00:00 ET on 09-02 = 04:00 UTC -> six rows before it
    assert removed == 2 + 6
    daily_left, hourly_left = weather.load_weather(path), weather.load_weather_hourly(hourly_path)
    assert daily_left is not None and hourly_left is not None
    assert daily_left["date"].min() == pd.Timestamp("2025-09-02")
    assert hourly_left["ts_utc"].min() == pd.Timestamp("2025-09-02 04:00", tz="UTC")


def test_store_timestamps_are_utc_text() -> None:
    assert (
        db.ts_text(pd.Timestamp("2025-09-10 01:00", tz="America/New_York")) == "2025-09-10 05:00:00"
    )
    with pytest.raises(ValueError):
        db.ts_text(pd.Timestamp("2025-09-10 01:00"))
    back = db.ts_column(pd.Series(["2025-09-10 05:00:00"]))
    assert back.iloc[0] == pd.Timestamp("2025-09-10 05:00", tz="UTC")
