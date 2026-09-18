"""models.tft: windows, leakage guard, quantile post-processing, and a small real fit."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from models import SLOT, cutoff_for
from tests.synthetic import SyntheticNYISO
from tests.test_model import zone_slots

T = pytest.importorskip("models.tft")


@pytest.fixture(scope="module")
def two_zones() -> dict[str, Any]:
    synth = SyntheticNYISO(date(2025, 8, 1), date(2025, 9, 20), seed=7, irregular=False)
    return {
        z: T.prepare_zone(z, zone_slots(synth, z), None, date(2025, 9, 20))
        for z in ("N.Y.C.", "WEST")
    }


def test_prepare_zone_grid_and_windows(two_zones: dict[str, Any]) -> None:
    zd = two_zones["N.Y.C."]
    assert zd.index.freq is not None and zd.known_cols == T.CALENDAR_COLS
    assert zd.index[-1] == T.local_midnight_utc(date(2025, 9, 21)) - SLOT
    s = T.make_sample(zd, 0, date(2025, 9, 10), enc_len=3 * 96)
    assert s is not None
    cutoff = cutoff_for(date(2025, 9, 10))
    assert zd.index[s.enc.stop - 1] + SLOT == cutoff and zd.index[s.dec.start] == cutoff
    assert s.dec.stop - s.dec.start == 19 * 4 + 96
    assert T.make_sample(zd, 0, date(2025, 8, 2), enc_len=3 * 96) is None  # too little history
    y, means = T.series_asof(zd, cutoff)
    assert np.isnan(y[s.dec]).all() and not np.isnan(y[s.enc]).any() and len(means) == 96


def test_quantile_post_processing() -> None:
    q = (0.1, 0.5, 0.9)
    frame = pd.DataFrame({"ts_utc": [0, 1], "date": [0, 0], "slot": [0, 1], "tod": [0, 1]})
    frame["q10"], frame["q50"], frame["q90"] = [90.0, 95.0], [100.0, 100.0], [120.0, 105.0]
    out = T.to_result_rows(frame, q, alpha=0.7)
    assert list(out["pred"]) == [100.0, 100.0] and list(out["p10"]) == [90.0, 95.0]
    assert list(out["p_alpha"]) == [110.0, 102.5] and (out["alpha"] == 0.7).all()
    assert np.array_equal(T.to_result_rows(frame, q, alpha=0.5)["p_alpha"], out["pred"])
    assert T.periods([date(2025, 1, d) for d in range(1, 8)], 3) == [
        [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
        [date(2025, 1, 4), date(2025, 1, 5), date(2025, 1, 6)],
        [date(2025, 1, 7)],
    ]


@pytest.mark.slow
def test_fit_predict_and_leakage_guard(two_zones: dict[str, Any]) -> None:
    f = T.TFTForecaster(
        list(two_zones),
        enc_days=3,
        hidden=16,
        heads=2,
        epochs=12,
        patience=12,
        batch_size=16,
        val_days=3,
        device="cpu",
    )
    d0 = date(2025, 9, 18)
    c0 = cutoff_for(d0)
    fit_data = {z: (zd, T.series_asof(zd, c0)[0]) for z, zd in two_zones.items()}
    train_days = [date(2025, 8, 10) + pd.Timedelta(days=k).to_pytimedelta() for k in range(38)]
    f.fit(fit_data, train_days)
    assert f.model is not None and len(f.history) >= 3
    zd = two_zones["N.Y.C."]
    y_d, _ = T.series_asof(zd, c0)
    q = f.predict(zd, y_d, d0)
    assert len(q) == 96 and list(q.columns[:4]) == ["ts_utc", "date", "slot", "tod"]
    qcols = [c for c in q.columns if c.startswith("q")]
    assert (np.diff(q[qcols].to_numpy(), axis=1) >= 0).all()  # sorted quantiles
    actual = zone_slots(
        SyntheticNYISO(date(2025, 8, 1), date(2025, 9, 20), seed=7, irregular=False)
    )
    actual = actual.merge(q[["ts_utc"]], on="ts_utc")["load_mw"].to_numpy()
    assert np.mean(np.abs(actual - q["q50"]) / actual) < 0.15
    # the encoder may not see anything at or after the cutoff
    leaky = y_d.copy()
    leaky[np.isnan(leaky)] = 1.0
    with pytest.raises(ValueError, match="past the cutoff"):
        f.predict(zd, leaky, d0)
