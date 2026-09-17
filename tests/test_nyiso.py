"""app.nyiso against the synthetic archive: normalizers recover the truth frames, the
client caches and falls back the way the real archive behaves, DST days survive."""

from __future__ import annotations

import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import nyiso
from app.nyiso import ArchiveClient, NotAvailable
from src.config import NYISO_ARCHIVE_BASE, ZONES, daily_filename, monthly_zip_name
from tests.conftest import FALL_BACK_DAY, SPRING_FORWARD_DAY
from tests.synthetic import SyntheticNYISO

ET = "America/New_York"


def _client(root: Path, cache: Path, today: date) -> tuple[ArchiveClient, list[str]]:
    """Client whose fetch reads a synthetic archive on disk; returns it with the URL log."""
    calls: list[str] = []

    def fetch(url: str) -> bytes | None:
        calls.append(url)
        rel = url.removeprefix(NYISO_ARCHIVE_BASE + "/")
        path = root / rel
        return path.read_bytes() if path.exists() else None

    return ArchiveClient(cache_dir=cache, fetch=fetch, today=lambda: today), calls


def _local_dates(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert(ET).dt.date


# ---------------------------------------------------------------------------- pal
def test_normalize_pal_recovers_truth_on_clean_days(synth_spring: SyntheticNYISO) -> None:
    truth = synth_spring.load_5min()
    got = pd.concat(
        [nyiso.normalize_pal(synth_spring.pal_csv(d)) for d in synth_spring.days()],
        ignore_index=True,
    )
    assert list(got.columns) == ["ts_utc", "ts_end_utc", "zone", "load_mw"]
    pd.testing.assert_frame_equal(got[["ts_utc", "zone", "load_mw"]], truth, check_dtype=False)
    assert ((got["ts_end_utc"] - got["ts_utc"]) == pd.Timedelta(minutes=5)).all()


def test_normalize_pal_dst_days_and_dedupe() -> None:
    fall = nyiso.normalize_pal(
        SyntheticNYISO(FALL_BACK_DAY, FALL_BACK_DAY, irregular=False).pal_csv(FALL_BACK_DAY)
    )
    assert len(fall) == 300 * 11
    nyc = fall[fall["zone"] == "N.Y.C."]
    assert nyc["ts_utc"].is_monotonic_increasing and nyc["ts_utc"].is_unique
    one_am = nyc[nyc["ts_utc"].dt.tz_convert(ET).dt.hour == 1]
    assert len(one_am) == 24 and one_am["ts_utc"].iloc[0] == pd.Timestamp(
        "2025-11-02 05:00", tz="UTC"
    )
    assert one_am["ts_utc"].iloc[12] == pd.Timestamp("2025-11-02 06:00", tz="UTC")
    spring = nyiso.normalize_pal(
        SyntheticNYISO(SPRING_FORWARD_DAY, SPRING_FORWARD_DAY, irregular=False).pal_csv(
            SPRING_FORWARD_DAY
        )
    )
    assert len(spring) == 276 * 11
    # exact duplicate rows collapse to one; a re-posted value keeps the later row
    text = SyntheticNYISO(date(2026, 1, 5), date(2026, 1, 5)).pal_csv(date(2026, 1, 5))
    lines = text.splitlines()
    dup = nyiso.normalize_pal("\n".join([*lines, lines[1], lines[1].rsplit(",", 1)[0] + ",999.0"]))
    assert len(dup) == len(nyiso.normalize_pal(text))
    assert dup.loc[(dup["zone"] == "CAPITL"), "load_mw"].iloc[0] == 999.0


def test_normalize_pal_off_schedule_stamps_shorten_intervals(synth_autumn: SyntheticNYISO) -> None:
    day = next(d for d in synth_autumn.days() if len(synth_autumn.offgrid_stamps(d)))
    got = nyiso.normalize_pal(synth_autumn.pal_csv(day))
    nyc = got[got["zone"] == "N.Y.C."]
    assert len(nyc) == 288 + len(synth_autumn.offgrid_stamps(day)) + (
        12 if day == FALL_BACK_DAY else 0
    )
    lengths = nyc["ts_end_utc"] - nyc["ts_utc"]
    assert lengths.max() == pd.Timedelta(minutes=5) and lengths.min() > pd.Timedelta(0)
    assert (
        nyc["ts_end_utc"].iloc[:-1].to_numpy() == nyc["ts_utc"].iloc[1:].to_numpy()
    ).all()  # contiguous


def test_normalize_pal_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        nyiso.normalize_pal("a,b\n1,2\n")
    bad = (
        '"Time Stamp","Time Zone","Name","PTID","Load"\n'
        '"01/05/2026 00:00:00","PST","CAPITL",61757,1.0\n'
    )
    with pytest.raises(ValueError, match="time zone"):
        nyiso.normalize_pal(bad)


# ---------------------------------------------------------------------------- prices
def test_normalize_realtime_intervals_and_truth(synth_spring: SyntheticNYISO) -> None:
    truth = synth_spring.rt_prices_5min()
    truth = truth[truth["name"].isin(ZONES)].rename(columns={"name": "zone", "lbmp": "p_rt"})
    truth = truth[["ts_utc", "zone", "p_rt"]].sort_values(["ts_utc", "zone"]).reset_index(drop=True)
    got = pd.concat(
        [nyiso.normalize_realtime(synth_spring.realtime_zone_csv(d)) for d in synth_spring.days()],
        ignore_index=True,
    )
    assert list(got.columns) == ["ts_utc", "ts_end_utc", "zone", "p_rt"]
    pd.testing.assert_frame_equal(
        got[["ts_end_utc", "zone", "p_rt"]].rename(columns={"ts_end_utc": "ts_utc"}),
        truth,
        check_dtype=False,
    )
    assert ((got["ts_end_utc"] - got["ts_utc"]) == pd.Timedelta(minutes=5)).all()
    assert got["ts_utc"].min() == pd.Timestamp(
        "2026-03-06 00:00", tz=ET
    )  # first interval starts at local midnight


def test_normalize_realtime_fall_back_by_row_order() -> None:
    synth = SyntheticNYISO(FALL_BACK_DAY, FALL_BACK_DAY, seed=3)  # includes off-schedule stamps
    got = nyiso.normalize_realtime(synth.realtime_zone_csv(FALL_BACK_DAY))
    nyc = got[got["zone"] == "N.Y.C."]
    assert nyc["ts_end_utc"].is_monotonic_increasing and nyc["ts_end_utc"].is_unique
    assert nyc["ts_end_utc"].iloc[0] == pd.Timestamp("2025-11-02 04:05", tz="UTC")
    assert nyc["ts_end_utc"].iloc[-1] == pd.Timestamp("2025-11-03 05:00", tz="UTC")
    local = nyc["ts_end_utc"].dt.tz_convert(ET)
    assert (local.dt.hour == 1).sum() >= 24  # both local 01:xx hours present


def test_normalize_hourly_prices_dst_and_truth(
    synth_autumn: SyntheticNYISO, synth_spring: SyntheticNYISO
) -> None:
    da = nyiso.normalize_damlbmp(synth_autumn.damlbmp_zone_csv(FALL_BACK_DAY))
    assert list(da.columns) == ["ts_utc", "zone", "p_da"]
    assert len(da) == 25 * 11 and set(da["zone"]) == set(ZONES)
    nyc = da[da["zone"] == "N.Y.C."].reset_index(drop=True)
    assert nyc["ts_utc"].iloc[1] == pd.Timestamp("2025-11-02 05:00", tz="UTC")  # 01:00 EDT
    assert nyc["ts_utc"].iloc[2] == pd.Timestamp("2025-11-02 06:00", tz="UTC")  # 01:00 EST
    truth = synth_autumn.da_prices_hourly()
    truth_nyc = truth[
        (truth["name"] == "N.Y.C.") & (_local_dates(truth["ts_utc"]) == FALL_BACK_DAY)
    ]
    assert np.array_equal(nyc["p_da"].to_numpy(), truth_nyc["lbmp"].to_numpy())
    spring = nyiso.normalize_damlbmp(synth_spring.damlbmp_zone_csv(SPRING_FORWARD_DAY))
    assert len(spring) == 23 * 11
    hourly_rt = nyiso.normalize_rtlbmp(synth_spring.rtlbmp_zone_csv(date(2026, 3, 7)))
    assert (
        list(hourly_rt.columns) == ["ts_utc", "zone", "p_rt_hourly"] and len(hourly_rt) == 24 * 11
    )


def test_rtlbmp_matches_resampled_realtime(synth_spring: SyntheticNYISO) -> None:
    day = date(2026, 3, 7)
    rt = nyiso.normalize_realtime(synth_spring.realtime_zone_csv(day))
    hourly = nyiso.resample_slots(rt, "p_rt", minutes=60)
    ref = nyiso.normalize_rtlbmp(synth_spring.rtlbmp_zone_csv(day))
    merged = ref.merge(hourly, on=["ts_utc", "zone"])
    assert len(merged) == 24 * 11
    assert (merged["coverage"] == 1.0).all()
    assert (merged["p_rt"] - merged["p_rt_hourly"]).abs().max() < 0.011


# ---------------------------------------------------------------------------- isolf
def test_normalize_isolf_matches_truth_and_dst(synth_autumn: SyntheticNYISO) -> None:
    issued = date(2025, 10, 31)
    got = nyiso.normalize_isolf(synth_autumn.isolf_csv(issued), issued)
    truth = synth_autumn.isolf_truth(issued)
    assert list(got.columns) == ["issued", "ts_utc", "zone", "isolf_mw"]
    assert (got["issued"] == issued).all()
    pd.testing.assert_frame_equal(got[["ts_utc", "zone", "isolf_mw"]], truth, check_dtype=False)
    assert len(got) == (6 * 24 + 1) * 12  # includes the 25-hour fall-back day
    nyca = got[got["zone"] == "NYCA"].set_index("ts_utc")["isolf_mw"]
    zones_sum = got[got["zone"] != "NYCA"].groupby("ts_utc")["isolf_mw"].sum()
    assert (nyca == zones_sum).all()


# ---------------------------------------------------------------------------- resampling
def test_resample_slots_clean_grid_is_plain_mean(synth_spring: SyntheticNYISO) -> None:
    load = nyiso.normalize_pal(synth_spring.pal_csv(date(2026, 3, 6)))
    slots = nyiso.resample_slots(load, "load_mw")
    assert list(slots.columns) == ["ts_utc", "zone", "load_mw", "coverage"]
    assert len(slots) == 96 * 11 and (slots["coverage"] == 1.0).all()
    nyc = load[load["zone"] == "N.Y.C."]["load_mw"].to_numpy().reshape(96, 3).mean(axis=1)
    got = slots[slots["zone"] == "N.Y.C."]["load_mw"].to_numpy()
    assert np.allclose(got, nyc, atol=1e-9)


@pytest.mark.parametrize(("day", "n_slots"), [(FALL_BACK_DAY, 100), (SPRING_FORWARD_DAY, 92)])
def test_resample_slots_dst_day_slot_counts(day: date, n_slots: int) -> None:
    load = nyiso.normalize_pal(SyntheticNYISO(day, day, irregular=False).pal_csv(day))
    slots = nyiso.resample_slots(load, "load_mw")
    assert (slots.groupby("zone").size() == n_slots).all()
    assert (_local_dates(slots["ts_utc"]) == day).all()


def test_resample_slots_off_schedule_and_gaps(synth_autumn: SyntheticNYISO) -> None:
    day = next(
        d for d in synth_autumn.days() if len(synth_autumn.offgrid_stamps(d)) and d != FALL_BACK_DAY
    )
    irregular = nyiso.resample_slots(nyiso.normalize_pal(synth_autumn.pal_csv(day)), "load_mw")
    clean = nyiso.resample_slots(
        nyiso.normalize_pal(
            SyntheticNYISO(day, day, seed=synth_autumn.seed, irregular=False).pal_csv(day)
        ),
        "load_mw",
    )
    assert len(irregular) == len(clean) == 96 * 11 and (irregular["coverage"] == 1.0).all()
    rel = ((irregular["load_mw"] - clean["load_mw"]).abs() / clean["load_mw"]).max()
    assert 0 < rel < 0.01  # interpolated off-schedule values barely move the slot mean
    # drop 04:10 for every zone: the 04:00 slot keeps 10 of 15 minutes
    load = nyiso.normalize_pal(synth_autumn.pal_csv(day))
    gap = load[load["ts_utc"].dt.tz_convert(ET).dt.strftime("%H:%M:%S") != "04:10:00"]
    slots = nyiso.resample_slots(gap, "load_mw")
    hit = slots[slots["ts_utc"].dt.tz_convert(ET).dt.strftime("%H:%M") == "04:00"]
    assert len(hit) == 11 and np.allclose(hit["coverage"], 10 / 15)
    assert (slots.loc[~slots.index.isin(hit.index), "coverage"] == 1.0).all()


def test_add_nyca_sums_complete_timestamps(synth_spring: SyntheticNYISO) -> None:
    slots = nyiso.resample_slots(
        nyiso.normalize_pal(synth_spring.pal_csv(date(2026, 3, 9))), "load_mw"
    )
    partial = slots[~((slots["zone"] == "WEST") & (slots["ts_utc"] == slots["ts_utc"].iloc[0]))]
    total = nyiso.add_nyca(partial, "load_mw")
    nyca = total[total["zone"] == "NYCA"]
    assert len(nyca) == 95  # the first slot lacks WEST and gets no total
    check = slots[slots["ts_utc"] == nyca["ts_utc"].iloc[0]]["load_mw"].sum()
    assert abs(nyca["load_mw"].iloc[0] - check) < 1e-9
    assert (nyca["coverage"] == 1.0).all()
    assert len(total) == len(partial) + 95


# ---------------------------------------------------------------------------- client
def test_client_daily_files_are_cached_once(
    synth_archive: Path, synth_autumn: SyntheticNYISO, tmp_path: Path
) -> None:
    client, calls = _client(synth_archive, tmp_path / "cache", today=date(2025, 11, 10))
    text = client.day_csv("pal", FALL_BACK_DAY)
    assert text == synth_autumn.pal_csv(FALL_BACK_DAY)
    assert calls == [f"{NYISO_ARCHIVE_BASE}/pal/20251102pal.csv"]
    assert (tmp_path / "cache" / "pal" / "20251102pal.csv").exists()
    client.day_csv("pal", FALL_BACK_DAY)
    assert len(calls) == 1  # served from cache
    with pytest.raises(NotAvailable):
        client.day_csv("pal", date(2025, 9, 1))  # not in the synthetic archive at all
    with pytest.raises(ValueError):
        client.day_csv("nope", FALL_BACK_DAY)


def test_client_final_month_comes_from_one_zip(tmp_path: Path) -> None:
    synth = SyntheticNYISO(date(2025, 8, 30), date(2025, 9, 30), seed=4, irregular=False)
    root = tmp_path / "archive"
    synth.write_archive(root, file_types=("pal", "isolf"), monthly_zips=True)
    for d in synth.days():  # old daily files are gone from the real archive
        if d.month == 9:
            (root / "pal" / daily_filename("pal", d)).unlink()
            (root / "isolf" / daily_filename("isolf", d)).unlink()
    client, calls = _client(root, tmp_path / "cache", today=date(2025, 10, 5))
    assert client.day_csv("pal", date(2025, 9, 15)) == synth.pal_csv(date(2025, 9, 15))
    assert client.day_csv("pal", date(2025, 9, 16)) == synth.pal_csv(date(2025, 9, 16))
    assert calls == [f"{NYISO_ARCHIVE_BASE}/pal/20250901pal_csv.zip"]
    assert (tmp_path / "cache" / "pal" / "20250901pal_csv.zip").exists()
    frame, missing = client.frame("isolf", date(2025, 9, 28), date(2025, 10, 1))
    assert missing == [date(2025, 10, 1)]  # October: no zip, no daily file
    assert sorted(frame["issued"].unique()) == [
        date(2025, 9, 28),
        date(2025, 9, 29),
        date(2025, 9, 30),
    ]
    assert len(frame) == 3 * 144 * 12


def test_client_current_month_uses_partial_zip_then_daily(tmp_path: Path) -> None:
    today = date(2025, 9, 17)
    synth = SyntheticNYISO(date(2025, 9, 1), today, seed=5, irregular=False)
    root = tmp_path / "archive"
    synth.write_archive(root, file_types=("pal",))
    # the archive's current-month zip holds every day so far (including the partial today)
    with zipfile.ZipFile(root / "pal" / monthly_zip_name("pal", date(2025, 9, 1)), "w") as zf:
        for d in synth.days():
            zf.writestr(daily_filename("pal", d), synth.pal_csv(d))
    for d in synth.days()[:5]:  # only the last ~11 daily files survive
        (root / "pal" / daily_filename("pal", d)).unlink()
    client, calls = _client(root, tmp_path / "cache", today=today)

    assert client.day_csv("pal", date(2025, 9, 3)) == synth.pal_csv(date(2025, 9, 3))
    assert [u.rsplit("/", 1)[1] for u in calls] == ["20250903pal.csv", "20250901pal_csv.zip"]
    assert (tmp_path / "cache" / "pal" / "20250901pal_csv.zip.partial").exists()
    assert not (tmp_path / "cache" / "pal" / "20250901pal_csv.zip").exists()
    client.day_csv("pal", date(2025, 9, 4))
    client.day_csv("pal", date(2025, 9, 10))  # a recent day: still served from the partial zip
    assert len(calls) == 2
    # today is fetched fresh every time and never cached
    assert client.day_csv("pal", today) == synth.pal_csv(today)
    assert client.day_csv("pal", today) == synth.pal_csv(today)
    assert [u.rsplit("/", 1)[1] for u in calls[2:]] == ["20250917pal.csv", "20250917pal.csv"]
    assert not (tmp_path / "cache" / "pal" / "20250917pal.csv").exists()
    with pytest.raises(NotAvailable, match="not published"):
        client.day_csv("pal", today + timedelta(days=1))


def test_client_frame_concatenates_days(synth_spring: SyntheticNYISO, tmp_path: Path) -> None:
    root = tmp_path / "archive"
    synth_spring.write_archive(root, file_types=("pal", "realtime_zone", "damlbmp_zone"))
    client, _ = _client(root, tmp_path / "cache", today=date(2026, 3, 20))
    load, missing = client.frame("pal", synth_spring.start, synth_spring.end)
    assert missing == []
    pd.testing.assert_frame_equal(
        load[["ts_utc", "zone", "load_mw"]], synth_spring.load_5min(), check_dtype=False
    )
    slots = nyiso.add_nyca(nyiso.resample_slots(load, "load_mw"), "load_mw")
    per_day = (
        slots[slots["zone"] == "NYCA"]
        .groupby(_local_dates(slots[slots["zone"] == "NYCA"]["ts_utc"]))
        .size()
    )
    assert per_day.to_dict() == {
        date(2026, 3, 6): 96,
        date(2026, 3, 7): 96,
        SPRING_FORWARD_DAY: 92,
        date(2026, 3, 9): 96,
        date(2026, 3, 10): 96,
    }
    rt, _ = client.frame("realtime_zone", synth_spring.start, synth_spring.end)
    assert rt["ts_utc"].is_unique or rt.groupby("zone")["ts_utc"].apply(lambda s: s.is_unique).all()
    assert (
        rt.groupby("zone")["ts_utc"].diff().dropna() == pd.Timedelta(minutes=5)
    ).all()  # day files chain up
    da, _ = client.frame("damlbmp_zone", synth_spring.start, synth_spring.end)
    assert len(da) == (5 * 24 - 1) * 11
    empty, missing = client.frame("damlbmp_zone", date(2026, 4, 1), date(2026, 4, 2))
    assert empty.empty and list(empty.columns) == ["ts_utc", "zone", "p_da"] and len(missing) == 2


def test_http_fetch_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    attempts: list[int] = []

    def fake_get(url: str, timeout: float) -> object:
        attempts.append(1)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(nyiso.requests, "get", fake_get)
    monkeypatch.setattr(nyiso.time, "sleep", lambda s: None)
    with pytest.raises(requests.ConnectionError):
        nyiso.http_fetch("http://example.invalid/x", retries=3)
    assert len(attempts) == 3

    class Resp:
        status_code = 404
        content = b""

    monkeypatch.setattr(nyiso.requests, "get", lambda url, timeout: Resp())
    assert nyiso.http_fetch("http://example.invalid/x") is None


# ---------------------------------------------------------------------------- live
@pytest.mark.live
def test_live_recent_day_round_trip(tmp_path: Path) -> None:
    """One real day from the archive through every normalizer (network; run with -m live)."""
    client = ArchiveClient(cache_dir=tmp_path / "cache")
    day = nyiso.today_et() - timedelta(days=2)
    load = nyiso.normalize_pal(client.day_csv("pal", day))
    assert set(load["zone"]) == set(ZONES) and (load.groupby("zone").size() >= 288).all()
    slots = nyiso.add_nyca(nyiso.resample_slots(load, "load_mw"), "load_mw")
    nyca = slots[slots["zone"] == "NYCA"]
    assert len(nyca) == 96 and (nyca["coverage"] > 0.99).all()
    assert 8_000 < nyca["load_mw"].mean() < 30_000  # MW, statewide
    rt = nyiso.normalize_realtime(client.day_csv("realtime_zone", day))
    assert (rt.groupby("zone").size() >= 288).all()
    da = nyiso.normalize_damlbmp(client.day_csv("damlbmp_zone", day))
    assert (da.groupby("zone").size() == 24).all()
    hourly = nyiso.normalize_rtlbmp(client.day_csv("rtlbmp_zone", day))
    assert (hourly.groupby("zone").size() == 24).all()
    isolf = nyiso.normalize_isolf(client.day_csv("isolf", day), day)
    assert (isolf.groupby("zone").size() == 144).all() and set(isolf["zone"]) == set(ZONES) | {
        "NYCA"
    }
