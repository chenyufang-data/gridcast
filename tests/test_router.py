"""The keyword guide: zones by alias, views by cue, dates and ranges, and the sanitizer."""

from __future__ import annotations

from datetime import date

from frontend import router

TODAY = date(2026, 9, 20)


def route(text: str) -> dict:
    out = router.keyword_route(text, router.ZONES, TODAY)
    assert out is not None, text
    return out


def test_zone_aliases_resolve_on_word_boundaries() -> None:
    assert route("show the nyc forecast")["zone"] == "N.Y.C."
    assert route("Long Island prices")["zone"] == "LONGIL"
    assert route("hudson valley load")["zone"] == "HUD VL"
    assert route("how did west do")["zone"] == "WEST"
    assert route("statewide forecast")["zone"] == "NYCA"
    assert route("Mohawk Valley bid sheet")["zone"] == "MHK VL"
    # "li" only as a word, never inside "Millwood" or "list"
    assert route("millwood forecast")["zone"] == "MILLWD"
    assert router.keyword_route("list zones", router.ZONES, TODAY) == {"action": "list_zones"}


def test_view_cues_in_priority_order() -> None:
    assert route("forecast vs actual for nyc")["view"] == "compare"
    assert route("how accurate was the west forecast")["view"] == "compare"
    assert route("open the bid sheet for nyc")["view"] == "schedule"
    assert route("dam schedule long island")["view"] == "schedule"
    assert route("west prices")["view"] == "prices"
    assert route("nyc load forecast")["view"] == "forecast"
    assert route("actual load in the city")["view"] == "load"
    assert route("north")["view"] == "forecast"  # a bare zone opens its forecast


def test_relative_and_absolute_dates() -> None:
    r = route("west accuracy last week")
    assert (r["start"], r["end"]) == ("2026-09-13", "2026-09-19")
    r = route("nyc prices last 3 days")
    assert (r["start"], r["end"]) == ("2026-09-17", "2026-09-19")
    r = route("long island forecast for tomorrow")
    assert r["target"] == "2026-09-21" and r["view"] == "forecast"
    r = route("nyc load 2026-09-01 to 2026-09-07 hourly")
    assert (r["start"], r["end"], r["granularity"]) == ("2026-09-01", "2026-09-07", "hour")
    r = route("west prices 9/10")
    assert (r["start"], r["end"]) == ("2026-09-10", "2026-09-10")  # one day on a range view
    r = route("capital load in august")
    assert (r["start"], r["end"]) == ("2026-08-01", "2026-08-31")
    r = route("genesee forecast Sep 5")
    assert r["target"] == "2026-09-05"
    assert route("nyc daily load")["granularity"] == "day"
    assert route("nyc 15-min load")["granularity"] == "slot"
    assert "granularity" not in route("nyc load yesterday")  # "yesterday" is not "daily"


def test_help_and_unknown() -> None:
    assert route("help")["action"] == "help"
    assert router.keyword_route("what a lovely day", router.ZONES, TODAY) is None
    assert router.keyword_route("", router.ZONES, TODAY) is None


def test_sanitize_model_output() -> None:
    raw = {
        "action": "show_zone",
        "zone": "Long Island",
        "view": "compare",
        "granularity": "hour",
        "start": "2026-09-01T00:00:00",
        "end": "bogus",
        "message": None,
    }
    out = router.sanitize(raw, router.ZONES)
    assert out == {"action": "show_zone", "zone": "LONGIL", "view": "compare", "granularity": "hour", "start": "2026-09-01"}  # fmt: skip
    assert router.sanitize({"action": "fly"}, router.ZONES)["action"] == "answer"
    assert router.sanitize({"action": "help"}, router.ZONES)["message"] == router.HELP_TEXT
