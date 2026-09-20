"""The Streamlit app against a fake backend, through streamlit.testing.AppTest.

Every view renders without an exception, the buttons and chips navigate, the chat
input routes through the keyword guide, and writes reach the fake API with the right
arguments. No network, no model, no browser.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend import api, llm, router

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")
TODAY = date(2026, 9, 20)
TARGET = date(2026, 9, 19)


def _slots(day: date, base: float) -> list[dict[str, Any]]:
    ts = pd.date_range(
        pd.Timestamp(day, tz="America/New_York"), periods=96, freq="15min"
    ).tz_convert("UTC")
    shape = base * (0.8 + 0.2 * np.sin(np.linspace(0, np.pi, 96)))
    return [
        {
            "slot": i,
            "ts_utc": t.isoformat(),
            "local": t.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
            "predicted": round(float(v), 1),
            "p10": round(float(v) * 0.95, 1),
            "p90": round(float(v) * 1.05, 1),
            "p_alpha": round(float(v) * 0.98, 1),
            "actual": round(float(v) * 1.01, 1),
            "isolf_pre": round(float(v) * 1.02, 1),
            "isolf_post": round(float(v) * 1.005, 1),
        }
        for i, (t, v) in enumerate(zip(ts, shape, strict=True))
    ]


class FakeApi(api.Api):
    """Canned answers shaped exactly like the backend's, plus a log of writes."""

    def __init__(self) -> None:
        super().__init__("http://fake", "tok")
        self.writes: list[tuple[str, Any]] = []
        self.reads: list[tuple[str, str, str | None]] = []  # forecast reads (zone, day, version)
        self.stored = {"N.Y.C.": {TARGET.isoformat()}, "LONGIL": {TARGET.isoformat()}}
        self.extra: dict[tuple[str, str], list[str]] = {}  # retrains per (zone, day)

    def request(self, method: str, path: str, timeout: float | None = None, **kwargs: Any) -> Any:
        raise AssertionError(f"unexpected raw request {method} {path}")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "now_et": "2026-09-20T10:00:00-04:00",
            "next_target": "2026-09-21",
            "models": {"tft": {"loaded": True, "version": "tft-onnx:e979ee4c9f6d:2026-09-19", "fit_cutoff": "2026-09-19T09:00:00+00:00", "age_days": 2.0, "stale": False}, "lgbm": {"identity": "lgbm:nyiso-fv1"}},
            "data": {"load_slots": {"first": "2024-09-01 04:00:00", "last": "2026-09-20 13:45:00"}},
        }  # fmt: skip

    def zones(self) -> list[dict[str, Any]]:
        cards = []
        for z in router.ZONES:
            has = z in self.stored
            cards.append(
                {
                    "zone": z,
                    "is_total": z == "NYCA",
                    "history_days": 750,
                    "latest_forecast": {"target_date": TARGET.isoformat(), "model": "tft-onnx:e979ee4c9f6d:x", "model_version": "abc", "created_at": "x"} if has else None,
                    "last_7d": {"days": 3 if has else 0, "mape_hour": 3.5 if has else None, "imbalance_usd": 1200.0 if has else None, "isolf_mape_hour": 3.9 if has else None, "isolf_imbalance_usd": 1500.0 if has else None, "band_coverage": 78.0 if has else None},
                    "alerts": [{"target_date": TARGET.isoformat(), "kind": "mape", "message": "hourly MAPE 12.0% exceeds 10%"}] if z == "WEST" else [],
                    "alert": z == "WEST",
                }
            )  # fmt: skip
        return cards

    def forecasts(self, zone: str, limit: int = 400) -> list[dict[str, Any]]:
        return [{"forecast_id": 1, "target_date": d, "model": "tft-onnx:e979ee4c9f6d:x", "model_version": "abc", "created_at": "x", "alpha": 0.43, "coverage": 1.0, "mape_hour": 3.5, "imbalance_usd": 900.0, "isolf_mape_hour": 3.9} for d in sorted(self.stored.get(zone, ()), reverse=True)]  # fmt: skip

    def forecast(self, zone: str, target: date | str, version: str | None = None) -> dict[str, Any]:
        t = str(target)
        self.reads.append((zone, t, version))
        if t not in self.stored.get(zone, ()):
            raise api.ApiError(404, "no forecast")
        extras = self.extra.get((zone, t), [])
        if version and version not in extras:
            raise api.ApiError(404, "no version")
        versions = [
            {"forecast_id": 1, "model": "tft-onnx:e979ee4c9f6d:x", "model_version": "abc", "requested": "auto", "created_at": "2026-09-18 08:30:00", "alpha": 0.43, "primary": True, "coverage": 1.0, "mape_hour": 3.5, "band_coverage": 78.0, "imbalance_usd": 900.0, "imbalance_alpha_usd": 850.0, "isolf_mape_hour": 3.9, "isolf_imbalance_usd": 1100.0},
            *({"forecast_id": 2 + k, "model": "lgbm:nyiso-fv1", "model_version": v, "requested": "lgbm", "created_at": "2026-09-20 03:05:00", "alpha": 0.43, "primary": False, "coverage": None, "mape_hour": None, "band_coverage": None, "imbalance_usd": None, "imbalance_alpha_usd": None, "isolf_mape_hour": None, "isolf_imbalance_usd": None} for k, v in enumerate(extras)),
        ]  # fmt: skip
        base = 6000.0 if zone == "N.Y.C." else 2300.0
        return {
            "forecast_id": 1, "zone": zone, "target_date": t, "model": "tft-onnx:e979ee4c9f6d:x" if not version else "lgbm:nyiso-fv1",
            "model_version": version or "abc", "requested": "lgbm" if version else "auto", "primary": not version,
            "cutoff_utc": "2026-09-18 09:00:00", "created_at": "2026-09-18 08:30:00",
            "alpha": 0.43, "band_scale_p10": 0.98, "band_scale_p90": 1.03, "history_start": "x", "history_end": "y", "weather": 1,
            "versions": versions,
            "score": {"coverage": 1.0, "mape_hour": 3.5, "mape_slot": 4.0, "mape_hour_alpha": 3.6, "band_coverage": 78.0, "imbalance_usd": 900.0, "imbalance_alpha_usd": 850.0, "da_cost_usd": 1e6, "isolf_mape_hour": 3.9, "isolf_imbalance_usd": 1100.0},
            "values": _slots(date.fromisoformat(t), base * (0.97 if version else 1.0)),
        }  # fmt: skip

    def create_forecast(
        self, zone: str, target: date | str | None = None, model: str = "auto"
    ) -> dict[str, Any]:
        t = str(target or "2026-09-21")
        self.writes.append(("forecast", (zone, t, model)))
        primary = True
        if model == "lgbm" and t in self.stored.get(zone, ()):
            self.extra.setdefault((zone, t), []).append("def")  # a retrain on a stored day
            primary = False
        self.stored.setdefault(zone, set()).add(t)
        return {"forecast_id": 2, "zone": zone, "target_date": t, "model": "lgbm:nyiso-fv1" if model == "lgbm" else "tft-onnx:e979ee4c9f6d:x", "model_version": "def" if not primary else "abc", "requested": model, "primary": primary, "new": True, "fallback_reason": None, "values": []}  # fmt: skip

    def scores(self, zone: str, limit: int = 400) -> list[dict[str, Any]]:
        return [{"target_date": (TARGET - timedelta(days=k)).isoformat(), "model": "tft-onnx:e979ee4c9f6d:x", "mape_hour": 3.0 + k / 10, "isolf_mape_hour": 3.5, "imbalance_usd": 500.0, "isolf_imbalance_usd": 600.0, "band_coverage": 77.0} for k in range(5)]  # fmt: skip

    def compare(
        self, zone: str, start: date | str, end: date | str, granularity: str = "hour"
    ) -> dict[str, Any]:
        rows = _slots(TARGET, 6000.0)
        pts = [{"ts_utc": r["ts_utc"], "local": r["local"], "predicted": r["predicted"], "p10": r["p10"], "p90": r["p90"], "p_alpha": r["p_alpha"], "actual": r["actual"], "isolf_pre": r["isolf_pre"], "ape": 1.0, "p_da": 40.0, "p_rt": 45.0, "spread": 5.0, "imbalance_usd": 12.5, "imbalance_alpha_usd": 11.0, "isolf_imbalance_usd": 14.0} for r in rows]  # fmt: skip
        return {"zone": zone, "granularity": granularity, "start": str(start), "end": str(end), "points": pts, "summary": {"days": 1, "mape_hour": 3.5, "mape_hour_alpha": 3.6, "isolf_mape_hour": 3.9, "imbalance_usd": 1200.0, "imbalance_alpha_usd": 1050.0, "isolf_imbalance_usd": 1300.0, "models": ["tft-onnx:e979ee4c9f6d:x"]}}  # fmt: skip

    def load(
        self, zone: str, start: date | str, end: date | str, granularity: str = "hour"
    ) -> dict[str, Any]:
        rows = _slots(TARGET, 6000.0)
        return {
            "zone": zone,
            "granularity": granularity,
            "points": [
                {"ts_utc": r["ts_utc"], "local": r["local"], "actual": r["actual"]} for r in rows
            ],
        }

    def prices(self, zone: str, start: date | str, end: date | str) -> dict[str, Any]:
        ts = pd.date_range("2026-09-19 04:00", periods=24, freq="h", tz="UTC")
        return {"zone": zone, "points": [{"ts_utc": t.isoformat(), "local": "x", "p_da": 40.0 + i, "p_rt": 42.0 + (i % 5) * 3 - 5, "spread": 2.0 + (i % 5) * 3 - 5} for i, t in enumerate(ts)]}  # fmt: skip

    def alpha(self, zone: str, target: date | str) -> dict[str, Any]:
        return {
            "zone": zone,
            "alpha": 0.43,
            "c_under_usd_per_mwh": 3.1,
            "c_over_usd_per_mwh": 4.0,
            "mean_abs_spread_usd_per_mwh": 7.1,
            "slots": 2880,
        }

    def schedules(self, zone: str) -> list[dict[str, Any]]:
        return [
            dict(
                w[1],
                schedule_id=1,
                created_at="x",
                mape_hour=None,
                model_mape_hour=3.5,
                imbalance_usd=None,
                model_imbalance_usd=900.0,
            )
            for w in self.writes
            if w[0] == "schedule"
        ]

    def schedule(self, zone: str, target: date | str) -> dict[str, Any]:
        saved = [w[1] for w in self.writes if w[0] == "schedule" and w[1]["zone"] == zone]
        if not saved:
            raise api.ApiError(404, "no schedule")
        s = saved[-1]
        return {"schedule_id": 1, "zone": zone, "target_date": s["target_date"], "model_version": "abc", "created_at": "x", "note": s["note"], "total_bid_mwh": sum(s["hourly_mw"]), "delta_pct": 1.0, "usd_at_risk": 50.0, "hourly": [{"hour": i, "bid_mw": v} for i, v in enumerate(s["hourly_mw"])], "score": None, "forecast_score": None}  # fmt: skip

    def create_schedule(
        self,
        zone: str,
        target: date | str,
        hourly_mw: list[float],
        note: str | None = None,
        adjustments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rec = {"zone": zone, "target_date": str(target), "hourly_mw": hourly_mw, "note": note}
        self.writes.append(("schedule", rec))
        return {
            "schedule_id": 1,
            "target_date": str(target),
            "total_bid_mwh": sum(hourly_mw),
            "usd_at_risk": 50.0,
        }


@pytest.fixture
def fake() -> Iterator[FakeApi]:
    f = FakeApi()
    api.set_client(f)
    llm.set_provider(None, "no chat model configured")
    import streamlit as st

    st.cache_data.clear()
    yield f
    api.set_client(api.Api())


def run(state: dict[str, Any] | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def test_overview_cards_and_open_button(fake: FakeApi) -> None:
    at = run()
    assert at.session_state["view"] == "overview"
    assert any("Zone overview" in m.value for m in at.markdown)
    labels = [b.label for b in at.button]
    assert "Open N.Y.C." in labels and "Open NYCA" in labels
    at.button(key="open_LONGIL").click().run()
    assert not at.exception
    assert at.session_state["view"] == "forecast" and at.session_state["zone"] == "LONGIL"
    assert any("Forecast · LONGIL" in m.value for m in at.markdown)


def test_forecast_view_actions(fake: FakeApi) -> None:
    at = run({"view": "forecast", "zone": "N.Y.C."})
    assert at.session_state["target"] == TARGET  # the stored day, tomorrow is not stored
    assert any("Hourly MAPE" in m.label for m in at.metric)
    labels = [b.label for b in at.button]
    assert "Forecast Sep 21 (served model)" in labels and "Retrain trees for this day" in labels
    at.button[0].click().run()  # forecast tomorrow with the served model
    assert not at.exception
    assert fake.writes[-1] == ("forecast", ("N.Y.C.", "2026-09-21", "auto"))
    assert at.session_state["target"] == date(2026, 9, 21)
    at.button[1].click().run()  # retrain the trees for the selected day
    assert not at.exception
    assert fake.writes[-1] == ("forecast", ("N.Y.C.", "2026-09-21", "lgbm"))
    assert any("extra version" in s.value for s in at.success)
    # the retrain is an extra version: drawn next to the primary, listed, and toggleable
    pills = at.pills(key="fc_versions_N.Y.C._2026-09-21_2")
    assert len(pills.options) == 1 and "Trees" in pills.options[0]
    assert fake.reads[-1] == ("N.Y.C.", "2026-09-21", "def")  # the overlay was fetched
    assert any("Every version of this day" in m.value for m in at.markdown)
    assert any("primary of 2 versions" in m.value for m in at.markdown)
    pills.set_value([]).run()
    assert not at.exception
    assert fake.reads[-1] == ("N.Y.C.", "2026-09-21", None)  # deselected: primary only


def test_schedule_view_saves_a_bid(fake: FakeApi) -> None:
    at = run({"view": "schedule", "zone": "N.Y.C."})
    assert any("Total bid" in m.label for m in at.metric)
    save = next(b for b in at.button if b.label == "Save schedule")
    save.click().run()
    assert not at.exception
    kind, rec = fake.writes[-1]
    assert kind == "schedule" and rec["zone"] == "N.Y.C." and len(rec["hourly_mw"]) == 24
    assert all(v > 0 for v in rec["hourly_mw"])
    assert any("Saved schedule #1" in s.value for s in at.success)
    assert any("Saved schedule #1" in m.value for m in at.markdown)


def test_compare_prices_and_load_views(fake: FakeApi) -> None:
    at = run({"view": "compare", "zone": "N.Y.C."})
    assert any("Hourly MAPE" in m.label for m in at.metric)
    assert any("no stored forecast" in w.value for w in at.warning)  # 6 of 7 days missing
    backfill = next(b for b in at.button if b.label.startswith("Backfill"))
    backfill.click().run()
    assert not at.exception
    assert sum(1 for w in fake.writes if w[0] == "forecast") == 6
    at = run({"view": "prices", "zone": "WEST"})
    assert any("α (bid quantile)" in m.label for m in at.metric)
    at = run({"view": "prices", "zone": "NYCA"})
    assert any("no zonal price" in i.value for i in at.info)
    at = run({"view": "load", "zone": "NORTH"})
    assert any("Peak" in m.label for m in at.metric)


class FakeProvider:
    """A chat model that always answers with the same JSON."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, messages: Any, timeout: float) -> str:
        return self.reply


def test_chat_bubble_opens_a_window_that_navigates(fake: FakeApi) -> None:
    at = run()
    assert not at.chat_input, "the chat is hidden until the bubble is clicked"
    at.button(key="chat_bubble_btn").click().run()
    assert not at.exception
    assert at.session_state["chat_open"] is True and at.chat_input
    assert at.session_state["chat"][0][1].startswith("👋")  # the greeting
    assert at.pills(key="option_pick").options[1].endswith("Forecast")  # emoji becomes an icon
    at.chat_input[0].set_value("how accurate was the long island forecast last week").run()
    assert not at.exception
    assert at.session_state["view"] == "compare" and at.session_state["zone"] == "LONGIL"
    assert at.session_state["start"] == TODAY - timedelta(days=7)
    assert "Forecast vs Actual" in at.session_state["chat"][-1][1]
    assert at.session_state["chat_open"] is False and not at.chat_input  # navigation closes it
    assert any("Forecast vs Actual" in t.value for t in at.toast)
    at.button(key="chat_bubble_btn").click().run()
    at.chat_input[0].set_value("help").run()
    assert not at.exception
    assert at.session_state["chat_open"] is True and at.chat_input  # an answer keeps it open
    assert "I can take you anywhere" in at.session_state["chat"][-1][1]
    at.pills(key="option_pick").set_value("💲 Prices").run()
    assert not at.exception
    assert at.session_state["view"] == "prices" and at.session_state["chat_open"] is False
    assert at.session_state["chat"][-2] == ("user", "💲 Prices")
    at = run()
    at.sidebar.selectbox[0].select("WEST").run()
    assert not at.exception
    assert at.session_state["zone"] == "WEST" and at.session_state["view"] == "forecast"
    at.sidebar.radio[0].set_value("load").run()
    assert not at.exception and at.session_state["view"] == "load"


def test_chat_limit_locks_the_input_and_points_at_the_options(fake: FakeApi) -> None:
    llm.set_provider(FakeProvider('{"action": "answer", "message": "Sure."}'))
    saved = llm.limiter
    llm.limiter = llm.ChatLimiter(per_user=1, global_cap=10)
    try:
        at = run({"chat_open": True})
        assert not at.chat_input[0].proto.disabled
        assert any("1 of 1 model replies left" in c.value for c in at.caption)
        at.chat_input[0].set_value("what is alpha").run()
        assert not at.exception
        chat = at.session_state["chat"]
        assert chat[-2] == ("assistant", "Sure.")
        assert chat[-1][1].startswith("⏸️ You've hit the chat limit")
        assert at.chat_input[0].proto.disabled
        assert at.chat_input[0].placeholder.startswith("Daily chat limit reached")
        at.pills(key="option_pick").set_value("🔮 Forecast").run()  # options still work
        assert not at.exception
        assert at.session_state["view"] == "forecast" and at.session_state["chat_open"] is False
    finally:
        llm.limiter = saved
        llm.set_provider(None, "no chat model configured")


def test_backend_down_is_a_message_not_a_crash() -> None:
    api.set_client(api.Api("http://127.0.0.1:9", "x", timeout=2))
    import streamlit as st

    st.cache_data.clear()
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert any("not reachable" in e.value for e in at.error)
    finally:
        api.set_client(api.Api())
