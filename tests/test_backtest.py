"""src.backtest end to end on synthetic data: protocol, per-day cache, NYCA without prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app import nyiso
from src import backtest as BT
from src.settlement import slot_prices
from tests.synthetic import SyntheticNYISO


@pytest.mark.slow
def test_run_backtest_on_synthetic(tmp_path: Path) -> None:
    synth = SyntheticNYISO(date(2025, 8, 20), date(2025, 11, 4), seed=13, irregular=False)
    load = pd.concat(
        [nyiso.normalize_pal(synth.pal_csv(d)) for d in synth.days()], ignore_index=True
    )
    load_slots = nyiso.add_nyca(nyiso.resample_slots(load, "load_mw"), "load_mw")
    rt = pd.concat(
        [nyiso.normalize_realtime(synth.realtime_zone_csv(d)) for d in synth.days()],
        ignore_index=True,
    )
    da = pd.concat(
        [nyiso.normalize_damlbmp(synth.damlbmp_zone_csv(d)) for d in synth.days()],
        ignore_index=True,
    )
    prices = slot_prices(nyiso.resample_slots(rt, "p_rt"), da)

    cfg = BT.BacktestConfig(
        name="t",
        start=date(2025, 11, 1),
        end=date(2025, 11, 3),
        zones=("N.Y.C.", "NYCA"),
        workers=1,
        results_dir=tmp_path,
        weather_lead=None,
        model_overrides={"n_estimators": 200, "learning_rate": 0.05},
    )
    results = BT.run(cfg, load_slots, prices, None)
    assert list(results.columns) == BT.RESULT_COLUMNS
    assert set(results["zone"]) == {"N.Y.C.", "NYCA"}
    assert len(results) == 2 * (96 + 100 + 96)  # includes the fall-back day
    assert results["actual"].notna().all() and (results["p10"] <= results["p90"]).all()
    nyc, nyca = results[results["zone"] == "N.Y.C."], results[results["zone"] == "NYCA"]
    assert nyc["alpha"].between(0, 1).all() and (nyca["alpha"] == 0.5).all()
    assert (nyca["p_alpha"] == nyca["pred"]).all()
    assert ((nyc["actual"] - nyc["pred"]).abs() / nyc["actual"]).mean() < 0.05
    assert (tmp_path / "t" / "results.csv").exists() and (
        tmp_path / "t" / "N_Y_C" / "2025-11-02.csv"
    ).exists()

    # a second run serves every day from the cache without refitting
    cached = tmp_path / "t" / "NYCA" / "2025-11-01.csv"
    stamp = cached.stat().st_mtime_ns
    again = BT.run(cfg, load_slots, prices, None)
    assert cached.stat().st_mtime_ns == stamp
    pd.testing.assert_frame_equal(again, results, check_dtype=False)
    loaded = BT.load_results("t", tmp_path)
    assert len(loaded) == len(results) and str(loaded["ts_utc"].dt.tz) == "UTC"


def test_zone_dirname() -> None:
    assert BT.zone_dirname("N.Y.C.") == "N_Y_C" and BT.zone_dirname("HUD VL") == "HUD_VL"
