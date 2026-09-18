"""models.tft: aggregate zones and residual block bootstrap as extra training series."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from models import cutoff_for
from tests.synthetic import SyntheticNYISO
from tests.test_model import zone_slots

T = pytest.importorskip("models.tft")


@pytest.fixture(scope="module")
def zones() -> dict[str, Any]:
    synth = SyntheticNYISO(date(2025, 8, 1), date(2025, 9, 10), seed=8, irregular=False)
    return {
        z: T.prepare_zone(z, zone_slots(synth, z), None, date(2025, 9, 10))
        for z in ("N.Y.C.", "WEST", "NORTH")
    }


def test_aggregate_zone_sums_loads(zones: dict[str, Any]) -> None:
    members = [zones["N.Y.C."], zones["WEST"]]
    agg = T.aggregate_zone("agg0:N.Y.C.+WEST", members, {"N.Y.C.": 6100, "WEST": 1730})
    assert agg.zone.startswith("agg0") and agg.index.equals(members[0].index)
    ok = ~np.isnan(agg.raw)
    assert np.allclose(agg.raw[ok], members[0].raw[ok] + members[1].raw[ok])
    assert np.array_equal(agg.tod, members[0].tod) and np.array_equal(agg.dow, members[0].dow)
    # calendar columns untouched (no weather in this fixture)
    assert np.array_equal(agg.known, members[0].known)


def test_block_bootstrap_keeps_level_and_cutoff(zones: dict[str, Any]) -> None:
    zd = zones["N.Y.C."]
    cutoff = cutoff_for(date(2025, 9, 5))
    y, _ = T.series_asof(zd, cutoff)
    boot = T.block_bootstrap(zd, y, np.random.default_rng(0))
    assert boot.shape == y.shape
    assert np.array_equal(np.isnan(boot), np.isnan(y))  # same observed range, same NaN tail
    ok = ~np.isnan(y)
    assert abs(boot[ok].mean() / y[ok].mean() - 1) < 0.02 and (boot[ok] > 0).all()
    assert not np.allclose(boot[ok], y[ok])
    # the daily profile survives: hour-of-day means stay close
    prof_y = np.array([y[ok][zd.tod[ok] == t].mean() for t in range(0, 96, 24)])
    prof_b = np.array([boot[ok][zd.tod[ok] == t].mean() for t in range(0, 96, 24)])
    assert np.allclose(prof_y, prof_b, rtol=0.05)


@pytest.mark.slow
def test_fit_with_replicas_and_aggregate(zones: dict[str, Any]) -> None:
    agg = T.aggregate_zone("agg0:WEST+NORTH", [zones["WEST"], zones["NORTH"]], {})
    names = [*zones, agg.zone]
    f = T.TFTForecaster(
        names, enc_days=2, hidden=8, heads=2, epochs=2, batch_size=16, val_days=3, device="cpu"
    )
    d0 = date(2025, 9, 8)
    c0 = cutoff_for(d0)
    data = {z: (zd, T.series_asof(zd, c0)[0]) for z, zd in zones.items()}
    data[agg.zone] = (agg, T.series_asof(agg, c0)[0])
    rng = np.random.default_rng(1)
    replicas = [(zd, T.block_bootstrap(zd, y, rng)) for zd, y in data.values()]
    train_days = [date(2025, 8, 5) + timedelta(days=k) for k in range(32)]
    f.fit(data, train_days, replicas=replicas)
    assert f.model is not None
    q = f.predict(zones["N.Y.C."], T.series_asof(zones["N.Y.C."], c0)[0], d0)
    assert len(q) == 96 and np.isfinite(q["q50"]).all()
