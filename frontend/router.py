"""Keyword guide: deterministic intent routing for the chat panel.

This is the fallback that always works, without any model: it resolves a zone (by
name or alias), a view, a granularity and a date range from plain English and returns
an *intent*, the same shape the LLM (``frontend/llm.py``) produces:

    {"action": "list_zones" | "show_zone" | "answer" | "help",
     "zone": "<canonical zone name or None>",
     "view": "forecast" | "schedule" | "compare" | "prices" | "load",
     "granularity": "slot" | "hour" | "day",
     "start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "target": "YYYY-MM-DD",
     "message": "<assistant text for answer/help>"}

The frontend imports nothing from ``app/`` or ``src/`` (it ships alone in its image),
so the zone list lives here too.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

ZONES: tuple[str, ...] = (
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL",
    "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST", "NYCA",
)  # fmt: skip
ZONE_LABELS: dict[str, str] = {
    "CAPITL": "Capital (Albany)",
    "CENTRL": "Central (Syracuse)",
    "DUNWOD": "Dunwoodie (Yonkers)",
    "GENESE": "Genesee (Rochester)",
    "HUD VL": "Hudson Valley",
    "LONGIL": "Long Island",
    "MHK VL": "Mohawk Valley (Utica)",
    "MILLWD": "Millwood",
    "N.Y.C.": "New York City",
    "NORTH": "North (Plattsburgh)",
    "WEST": "West (Buffalo)",
    "NYCA": "NYCA (statewide)",
}
# alias -> canonical; matched longest first on word boundaries
ZONE_ALIASES: dict[str, str] = {
    "n.y.c.": "N.Y.C.", "nyc": "N.Y.C.", "new york city": "N.Y.C.", "manhattan": "N.Y.C.",
    "the city": "N.Y.C.", "city": "N.Y.C.",
    "longil": "LONGIL", "long island": "LONGIL", "li": "LONGIL",
    "hud vl": "HUD VL", "hudson valley": "HUD VL", "hudson": "HUD VL",
    "mhk vl": "MHK VL", "mohawk valley": "MHK VL", "mohawk": "MHK VL", "utica": "MHK VL",
    "capitl": "CAPITL", "capital": "CAPITL", "albany": "CAPITL",
    "centrl": "CENTRL", "central": "CENTRL", "syracuse": "CENTRL",
    "dunwod": "DUNWOD", "dunwoodie": "DUNWOD", "yonkers": "DUNWOD",
    "genese": "GENESE", "genesee": "GENESE", "rochester": "GENESE",
    "millwd": "MILLWD", "millwood": "MILLWD",
    "north": "NORTH", "plattsburgh": "NORTH",
    "west": "WEST", "buffalo": "WEST",
    "nyca": "NYCA", "statewide": "NYCA", "state-wide": "NYCA", "whole state": "NYCA",
    "new york state": "NYCA", "total": "NYCA", "nyiso total": "NYCA",
}  # fmt: skip

VIEWS = ("forecast", "schedule", "compare", "prices", "load")
VIEW_LABELS = {
    "forecast": "Forecast",
    "schedule": "DAM Schedule",
    "compare": "Forecast vs Actual",
    "prices": "Prices",
    "load": "Load",
}
# checked in order: accuracy cues win, then the bid sheet, then prices, then forecast
# ("load forecast" is a forecast), then load history
VIEW_KEYWORDS: dict[str, list[str]] = {
    "compare": [
        "vs", "versus", "accuracy", "accurate", "error", "mape", "compare", "how good",
        "how well", "how did", "score", "imbalance", "cost", "settle", "settlement",
    ],
    "schedule": ["schedule", "bid", "dam ", "submit", "declar", "purchase"],
    "prices": ["price", "lbmp", "spread", "day-ahead price", "real-time price", "$/mwh"],
    "forecast": ["forecast", "predict", "tomorrow", "peak", "expected", "band", "p10", "p90"],
    "load": ["actual", "history", "consumption", "usage", "demand", "load"],
}  # fmt: skip
GRANULARITY_PATTERNS = {
    "slot": r"\b(15[- ]?min\w*|slot|quarter[- ]hour)\b",
    "hour": r"\b(hourly|per hour|by hour|hour)\b",
    "day": r"\b(daily|per day|by day)\b",
}
LIST_KEYWORDS = ["all zones", "overview", "home", "zone list", "list zones", "cards", "every zone"]
HELP_KEYWORDS = ["help", "what can you do", "how do i", "how to"]


def _today_et() -> date:
    """The market's calendar day when the caller gives none."""
    return datetime.now(ZoneInfo("America/New_York")).date()


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

HELP_TEXT = (
    "I can take you anywhere in the app. Try:\n"
    '- **Forecast**: "show the Long Island forecast for tomorrow"\n'
    '- **DAM schedule**: "open the bid sheet for NYC"\n'
    '- **Forecast vs actual**: "how accurate was the West forecast last week"\n'
    '- **Prices**: "Hudson Valley prices for the last 3 days"\n'
    '- **Load**: "statewide load history in September"\n'
    '- **Overview**: "all zones"\n\n'
    "Zones: Capital, Central, Dunwoodie, Genesee, Hudson Valley, Long Island, Mohawk "
    "Valley, Millwood, NYC, North, West and NYCA (statewide)."
)


def _find_zone(text: str, zone_names: list[str] | tuple[str, ...]) -> str | None:
    low = text.lower()
    candidates = {z.lower(): z for z in zone_names}
    candidates.update({a: z for a, z in ZONE_ALIASES.items() if z in zone_names})
    for alias in sorted(candidates, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", low):
            return candidates[alias]
    return None


def _find_view(text: str) -> str | None:
    low = text.lower()
    for view, words in VIEW_KEYWORDS.items():
        if any(w in low for w in words):
            return view
    return None


def _find_granularity(text: str) -> str | None:
    low = text.lower()
    for gran, pattern in GRANULARITY_PATTERNS.items():
        if re.search(pattern, low):
            return gran
    return None


def _month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def find_dates(text: str, today: date | None = None) -> dict[str, str]:
    """Absolute and relative dates in `text` -> ``{"start", "end"}`` and/or ``{"target"}``.

    Understands ISO dates, ``9/10`` and ``9/10/2026``, ``Sep 10`` / ``September 10``,
    ``today`` / ``tomorrow`` / ``yesterday``, ``last week``, ``last N days``,
    ``last month``, ``this month`` and a bare month name (``in September``).
    """
    today = today or _today_et()
    low = text.lower()
    out: dict[str, str] = {}

    def rng(a: date, b: date) -> None:
        out["start"], out["end"] = a.isoformat(), b.isoformat()

    if m := re.search(r"\blast (\d{1,3}) days?\b", low):
        n = int(m.group(1))
        rng(today - timedelta(days=n), today - timedelta(days=1))
    elif "last week" in low or "past week" in low:
        rng(today - timedelta(days=7), today - timedelta(days=1))
    elif "last month" in low:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        rng(*_month_range(last_prev.year, last_prev.month))
    elif "this month" in low:
        rng(today.replace(day=1), today)
    elif "yesterday" in low:
        out["target"] = (today - timedelta(days=1)).isoformat()
    elif "tomorrow" in low:
        out["target"] = (today + timedelta(days=1)).isoformat()
    elif re.search(r"\btoday\b", low):
        out["target"] = today.isoformat()

    found: list[date] = []
    for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", low):
        found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b", low):
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            found.append(date(year, int(m.group(1)), int(m.group(2))))
        except ValueError:
            continue
    for m in re.finditer(r"\b([a-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?(?:,? (\d{4}))?\b", low):
        if m.group(1) in MONTHS:
            year = int(m.group(3)) if m.group(3) else today.year
            try:
                found.append(date(year, MONTHS[m.group(1)], int(m.group(2))))
            except ValueError:
                continue
    if found:
        found.sort()
        if len(found) >= 2:
            rng(found[0], found[-1])
        else:
            out["target"] = found[0].isoformat()
            out.pop("start", None)
            out.pop("end", None)
        return out
    month_word = re.search(r"\b(in|for|during) ([a-z]{3,9})\b", low) if not out else None
    if month_word and month_word.group(2) in MONTHS:
        month = MONTHS[month_word.group(2)]
        year = today.year if month <= today.month else today.year - 1
        a, b = _month_range(year, month)
        rng(a, min(b, today))
    return out


def keyword_route(
    text: str, zone_names: list[str] | tuple[str, ...] = ZONES, today: date | None = None
) -> dict[str, Any] | None:
    """Deterministic router; None means the text needs the model (or a help reply)."""
    low = text.lower().strip()
    if not low:
        return None
    zone = _find_zone(low, zone_names)
    view = _find_view(low)
    gran = _find_granularity(low)
    dates = find_dates(low, today)
    if any(k in low for k in LIST_KEYWORDS) and not zone:
        return {"action": "list_zones"}
    if zone or view:
        intent: dict[str, Any] = {"action": "show_zone", "zone": zone, "view": view or "forecast"}
        if gran:
            intent["granularity"] = gran
        intent.update(dates)
        if intent["view"] in ("compare", "prices", "load") and "target" in intent:
            # a single day on a range view: show that day
            intent["start"] = intent["end"] = intent.pop("target")
        return intent
    if any(k in low for k in HELP_KEYWORDS):
        return {"action": "help", "message": HELP_TEXT}
    return None


def sanitize(
    intent: dict[str, Any], zone_names: list[str] | tuple[str, ...] = ZONES
) -> dict[str, Any]:
    """Coerce a model-produced intent into the contract; unknown things become an answer."""
    out: dict[str, Any] = {}
    action = intent.get("action")
    if action not in ("list_zones", "show_zone", "answer", "help"):
        return {
            "action": "answer",
            "message": str(intent.get("message") or "I did not get that. Try one of the chips."),
        }
    out["action"] = action
    zone = intent.get("zone")
    if zone:
        out["zone"] = _find_zone(str(zone), zone_names)
    if intent.get("view") in VIEWS:
        out["view"] = intent["view"]
    if intent.get("granularity") in ("slot", "hour", "day"):
        out["granularity"] = intent["granularity"]
    for key in ("start", "end", "target"):
        value = intent.get(key)
        if value:
            try:
                out[key] = date.fromisoformat(str(value)[:10]).isoformat()
            except ValueError:
                continue
    if intent.get("message"):
        out["message"] = str(intent["message"])
    if action == "help" and "message" not in out:
        out["message"] = HELP_TEXT
    return out
