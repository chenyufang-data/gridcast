"""The synthetic archive must reproduce the observed NYISO layouts and gotchas.

Each gotcha in data/README.md has a case here, so the Phase 1 normalizers can be
tested against a generator that is itself verified.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.config import EXTERNAL_PTID, ISOLF_COLUMNS, PRICE_NAMES, ZONE_PTID, ZONES
from tests.conftest import FALL_BACK_DAY, SPRING_FORWARD_DAY
from tests.synthetic import SyntheticNYISO

NORMAL_DAY = date(2025, 10, 31)


def _read(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------- pal
def test_pal_layout_and_row_order(synth_spring: SyntheticNYISO) -> None:
    text = synth_spring.pal_csv(date(2026, 3, 6))
    lines = text.splitlines()
    assert lines[0] == '"Time Stamp","Time Zone","Name","PTID","Load"'
    assert lines[1].startswith('"03/06/2026 00:00:00","EST","CAPITL",61757,')
    df = _read(text)
    assert list(df.columns) == ["Time Stamp", "Time Zone", "Name", "PTID", "Load"]
    assert len(df) == 288 * 11
    assert list(df["Name"].iloc[:11]) == list(ZONES)  # alphabetical zones per stamp
    assert df.groupby("Name")["PTID"].first().to_dict() == ZONE_PTID
    assert df["Time Stamp"].iloc[-1] == "03/06/2026 23:55:00"
    assert (df["Load"] > 0).all()


def test_pal_fall_back_day_lists_one_am_twice(synth_autumn: SyntheticNYISO) -> None:
    df = _read(SyntheticNYISO(FALL_BACK_DAY, FALL_BACK_DAY, irregular=False).pal_csv(FALL_BACK_DAY))
    assert len(df) == 300 * 11  # 25 local hours
    one_am = df[df["Time Stamp"].str.startswith("11/02/2025 01:")]
    assert len(one_am) == 2 * 12 * 11  # 264 rows, as observed in the real file
    assert list(one_am["Time Zone"].unique()) == ["EDT", "EST"]  # EDT block comes first
    assert set(df["Time Zone"]) == {"EDT", "EST"}


def test_pal_spring_forward_day_skips_two_am(synth_spring: SyntheticNYISO) -> None:
    df = _read(synth_spring.pal_csv(SPRING_FORWARD_DAY))
    assert len(df) == 276 * 11  # 23 local hours
    assert not df["Time Stamp"].str.startswith("03/08/2026 02:").any()
    assert (df["Time Zone"].iloc[: 12 * 11 * 2] == "EST").all()
    assert (df["Time Zone"].iloc[-11:] == "EDT").all()


def test_pal_off_schedule_stamps_are_shared_with_realtime(synth_autumn: SyntheticNYISO) -> None:
    days_with_extra = 0
    for day in synth_autumn.days():
        pal = _read(synth_autumn.pal_csv(day))
        rt = _read(synth_autumn.realtime_zone_csv(day))
        pal_off = set(pal.loc[~pal["Time Stamp"].str.endswith(":00"), "Time Stamp"])
        rt_off = set(rt.loc[~rt["Time Stamp"].str.endswith(":00"), "Time Stamp"])
        assert pal_off == rt_off
        assert len(pal) == 11 * len(pal.drop_duplicates(["Time Stamp", "Time Zone"]))
        assert len(rt) % 15 == 0
        days_with_extra += bool(pal_off)
    assert days_with_extra > 0, "the default archive must contain off-schedule stamps"


def test_truth_load_is_clean_and_utc(synth_autumn: SyntheticNYISO) -> None:
    truth = synth_autumn.load_5min()
    assert list(truth.columns) == ["ts_utc", "zone", "load_mw"]
    assert str(truth["ts_utc"].dt.tz) == "UTC"
    per_zone = truth.groupby("zone")["ts_utc"].count()
    assert (per_zone == 7 * 288 + 12).all()  # the week contains the 25-hour fall-back day
    assert not truth.duplicated(["ts_utc", "zone"]).any()
    # the clean pal file for a normal day is exactly the truth slice for that day
    pal = _read(SyntheticNYISO(NORMAL_DAY, NORMAL_DAY, seed=1, irregular=False).pal_csv(NORMAL_DAY))
    nyc = pal[pal["Name"] == "N.Y.C."]["Load"].to_numpy()
    day_truth = truth[
        (truth["zone"] == "N.Y.C.")
        & (truth["ts_utc"].dt.tz_convert("America/New_York").dt.date == NORMAL_DAY)
    ]
    assert len(nyc) == len(day_truth) == 288
    assert abs(nyc - day_truth["load_mw"].to_numpy()).max() < 1e-9


# ---------------------------------------------------------------------------- prices
def test_damlbmp_layout_no_quotes_15_names(synth_autumn: SyntheticNYISO) -> None:
    text = synth_autumn.damlbmp_zone_csv(NORMAL_DAY)
    lines = text.splitlines()
    assert lines[0] == (
        "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),"
        "Marginal Cost Congestion ($/MWHr)"
    )
    assert '"' not in text
    df = _read(text)
    assert len(df) == 24 * 15
    assert list(df["Name"].iloc[:15]) == list(PRICE_NAMES)
    assert set(df["Name"]) == set(ZONE_PTID) | set(EXTERNAL_PTID)
    assert df["Time Stamp"].iloc[0] == "10/31/2025 00:00"
    assert "Time Zone" not in df.columns


def test_damlbmp_dst_days_have_25_and_23_hours(
    synth_autumn: SyntheticNYISO, synth_spring: SyntheticNYISO
) -> None:
    fall = _read(synth_autumn.damlbmp_zone_csv(FALL_BACK_DAY))
    assert len(fall) == 25 * 15
    stamps = fall["Time Stamp"].iloc[::15].tolist()  # first row of each 15-name block
    assert stamps.count("11/02/2025 01:00") == 2
    assert stamps[1] == stamps[2] == "11/02/2025 01:00"  # both 01:00 blocks adjacent, first = EDT
    spring = _read(synth_spring.damlbmp_zone_csv(SPRING_FORWARD_DAY))
    assert len(spring) == 23 * 15
    assert "03/08/2026 02:00" not in set(spring["Time Stamp"])


def test_realtime_is_interval_ending_and_quoted(synth_spring: SyntheticNYISO) -> None:
    text = synth_spring.realtime_zone_csv(date(2026, 3, 6))
    lines = text.splitlines()
    assert lines[0].startswith('"Time Stamp","Name","PTID","LBMP ($/MWHr)"')
    assert lines[1].startswith('"03/06/2026 00:05:00","CAPITL",61757,')
    df = _read(text)
    assert len(df) == 288 * 15
    assert df["Time Stamp"].iloc[-1] == "03/07/2026 00:00:00"


def test_rtlbmp_is_hourly_mean_of_realtime(synth_spring: SyntheticNYISO) -> None:
    day = date(2026, 3, 7)
    rt = _read(synth_spring.realtime_zone_csv(day))
    hourly = _read(synth_spring.rtlbmp_zone_csv(day))
    assert len(hourly) == 24 * 15
    assert hourly["Time Stamp"].iloc[0] == "03/07/2026 00:00"
    ends = pd.to_datetime(rt["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    rt["hour"] = (ends - pd.Timedelta(minutes=5)).dt.floor("h")
    for name in ("N.Y.C.", "PJM"):
        got = hourly[hourly["Name"] == name]["LBMP ($/MWHr)"].to_numpy()
        want = rt[rt["Name"] == name].groupby("hour")["LBMP ($/MWHr)"].mean().to_numpy()
        assert abs(got - want).max() < 0.011  # both sides rounded to cents


def test_price_truth_frames(synth_spring: SyntheticNYISO) -> None:
    da = synth_spring.da_prices_hourly()
    rt = synth_spring.rt_prices_5min()
    assert (
        list(da.columns) == list(rt.columns) == ["ts_utc", "name", "lbmp", "losses", "congestion"]
    )
    assert len(da) == (5 * 24 - 1) * 15  # five days, one of them 23 hours
    assert len(rt) == (5 * 288 - 12) * 15
    assert rt["ts_utc"].min() == da["ts_utc"].min() + pd.Timedelta(minutes=5)  # interval end
    assert not da.duplicated(["ts_utc", "name"]).any()


# ---------------------------------------------------------------------------- isolf
def test_isolf_layout_horizon_and_total(synth_autumn: SyntheticNYISO) -> None:
    text = synth_autumn.isolf_csv(NORMAL_DAY)
    lines = text.splitlines()
    assert lines[0] == '"Time Stamp",' + ",".join(f'"{c}"' for c in ISOLF_COLUMNS)
    assert lines[1].startswith('"10/31/2025 00:00",')
    df = _read(text)
    assert list(df.columns) == ["Time Stamp", *ISOLF_COLUMNS]
    assert len(df) == 6 * 24 + 1  # covers the fall-back day (25 h) two days later
    assert df["Time Stamp"].iloc[-1] == "11/05/2025 23:00"
    zone_cols = [c for c in ISOLF_COLUMNS if c != "NYISO"]
    assert (df[zone_cols].sum(axis=1) == df["NYISO"]).all()
    assert (df[zone_cols].dtypes == "int64").all()


def test_isolf_truth_matches_file_and_is_near_load(synth_autumn: SyntheticNYISO) -> None:
    issue = date(2025, 10, 30)
    truth = synth_autumn.isolf_truth(issue)
    assert set(truth["zone"]) == set(ZONES) | {"NYCA"}
    nyc_file = _read(synth_autumn.isolf_csv(issue))["N.Y.C."].to_numpy()
    nyc_truth = truth[truth["zone"] == "N.Y.C."].sort_values("ts_utc")["isolf_mw"].to_numpy()
    assert (nyc_file == nyc_truth).all()
    load = synth_autumn.load_5min()
    nyc_load = load[load["zone"] == "N.Y.C."].set_index("ts_utc")["load_mw"].resample("1h").mean()
    joined = (
        truth[truth["zone"] == "N.Y.C."]
        .set_index("ts_utc")["isolf_mw"]
        .to_frame()
        .join(nyc_load, how="inner")
    )
    rel_err = ((joined["isolf_mw"] - joined["load_mw"]).abs() / joined["load_mw"]).mean()
    assert 0 < rel_err < 0.08


# ---------------------------------------------------------------------------- archive
def test_write_archive_layout(synth_archive: Path, synth_autumn: SyntheticNYISO) -> None:
    for day in synth_autumn.days():
        ymd = f"{day:%Y%m%d}"
        assert (synth_archive / "pal" / f"{ymd}pal.csv").exists()
        assert (synth_archive / "damlbmp" / f"{ymd}damlbmp_zone.csv").exists()
        assert (synth_archive / "realtime" / f"{ymd}realtime_zone.csv").exists()
        assert (synth_archive / "rtlbmp" / f"{ymd}rtlbmp_zone.csv").exists()
        assert (synth_archive / "isolf" / f"{ymd}isolf.csv").exists()
    text = (synth_archive / "pal" / "20251102pal.csv").read_text(encoding="utf-8")
    assert text == synth_autumn.pal_csv(FALL_BACK_DAY)


def test_write_archive_monthly_zips(tmp_path: Path) -> None:
    import zipfile

    synth = SyntheticNYISO(date(2025, 8, 30), date(2025, 9, 30), seed=3, irregular=False)
    written = synth.write_archive(tmp_path, file_types=("pal",), monthly_zips=True)
    zips = [p for p in written if p.suffix == ".zip"]
    assert [p.name for p in zips] == ["20250901pal_csv.zip"]  # August is partial: no zip
    with zipfile.ZipFile(zips[0]) as zf:
        names = sorted(zf.namelist())
        assert names[0] == "20250901pal.csv" and names[-1] == "20250930pal.csv" and len(names) == 30
        assert zf.read("20250915pal.csv").decode() == synth.pal_csv(date(2025, 9, 15))


def test_generator_is_deterministic() -> None:
    a = SyntheticNYISO(date(2026, 1, 1), date(2026, 1, 2), seed=7)
    b = SyntheticNYISO(date(2026, 1, 1), date(2026, 1, 2), seed=7)
    c = SyntheticNYISO(date(2026, 1, 1), date(2026, 1, 2), seed=8)
    for ftype in ("pal", "damlbmp_zone", "realtime_zone", "rtlbmp_zone", "isolf"):
        assert a.csv_for(ftype, date(2026, 1, 2)) == b.csv_for(ftype, date(2026, 1, 2))
        assert a.csv_for(ftype, date(2026, 1, 2)) != c.csv_for(ftype, date(2026, 1, 2))


@pytest.mark.parametrize("day", [date(2025, 11, 2), date(2026, 3, 8)])
def test_dst_days_stamp_count_matches_utc_span(day: date) -> None:
    synth = SyntheticNYISO(day, day, irregular=False)
    hours = len(_read(synth.damlbmp_zone_csv(day))) / 15
    lo = pd.Timestamp(day.isoformat(), tz="America/New_York")
    hi = pd.Timestamp((day + timedelta(days=1)).isoformat(), tz="America/New_York")
    assert hours == (hi - lo) / pd.Timedelta(hours=1)


def test_values_depend_only_on_seed_and_timestamp() -> None:
    """A one-day generator and a week-long one agree on every file for a shared day."""
    day = date(2025, 11, 2)
    one = SyntheticNYISO(day, day, seed=5)
    week = SyntheticNYISO(day - timedelta(days=3), day + timedelta(days=3), seed=5)
    for ftype in ("pal", "damlbmp_zone", "realtime_zone", "rtlbmp_zone", "isolf"):
        assert one.csv_for(ftype, day) == week.csv_for(ftype, day), ftype
