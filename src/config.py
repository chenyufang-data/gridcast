"""Locked project constants: market rules, zones, archive layout, backtest window.

Every value here was decided in docs/plan.md (approved 2026-09-17). Change the plan
first, then this file. Zone names and PTIDs were verified against the live archive
files of 2026-09-15.
"""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

# --- market ------------------------------------------------------------------------
MARKET_TZ = ZoneInfo("America/New_York")
BID_CUTOFF_TIME = time(5, 0)  # DAM bids close 05:00 ET on D-1 (FERC guide, NYISO Manual 11)
SLOTS_PER_HOUR = 4  # 15-min settlement slots
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR  # nominal: 92 / 100 on DST days, never assume the grid
RT_INTERVAL_MIN = 5  # nominal RTD interval; off-schedule runs (e.g. 04:04:18) also appear

# --- zones -------------------------------------------------------------------------
ZONES: tuple[str, ...] = (
    "CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL",
    "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST",
)  # fmt: skip
NYCA = "NYCA"  # statewide total = sum of the 11 zones (not a row in pal; "NYISO" column in isolf)
ZONE_PTID: dict[str, int] = {
    "CAPITL": 61757, "CENTRL": 61754, "DUNWOD": 61760, "GENESE": 61753, "HUD VL": 61758,
    "LONGIL": 61762, "MHK VL": 61756, "MILLWD": 61759, "N.Y.C.": 61761, "NORTH": 61755,
    "WEST": 61752,
}  # fmt: skip
# external proxy buses present in every price file; filtered out by PTID
EXTERNAL_PTID: dict[str, int] = {"H Q": 61844, "NPX": 61845, "O H": 61846, "PJM": 61847}
PRICE_NAMES: tuple[str, ...] = tuple(sorted({**ZONE_PTID, **EXTERNAL_PTID}))  # file row order
ISOLF_COLUMNS: dict[str, str] = {
    "Capitl": "CAPITL", "Centrl": "CENTRL", "Dunwod": "DUNWOD", "Genese": "GENESE",
    "Hud Vl": "HUD VL", "Longil": "LONGIL", "Mhk Vl": "MHK VL", "Millwd": "MILLWD",
    "N.Y.C.": "N.Y.C.", "North": "NORTH", "West": "WEST", "NYISO": NYCA,
}  # fmt: skip
DEMO_DEFAULT_ZONE = "N.Y.C."

# --- data source (public MIS archive; fetched at runtime, never committed) ---------
NYISO_ARCHIVE_BASE = "http://mis.nyiso.com/public/csv"
NYISO_LEGAL_NOTICE_URL = "https://www.nyiso.com/legal-notice"
FILE_TYPES: tuple[str, ...] = ("pal", "damlbmp_zone", "realtime_zone", "rtlbmp_zone", "isolf")
ARCHIVE_DIRS: dict[str, str] = {
    "pal": "pal", "damlbmp_zone": "damlbmp", "realtime_zone": "realtime",
    "rtlbmp_zone": "rtlbmp", "isolf": "isolf",
}  # fmt: skip


def daily_filename(file_type: str, day: date) -> str:
    """`20260915pal.csv`, `20260915damlbmp_zone.csv`, ... (recent days only)."""
    return f"{day:%Y%m%d}{file_type}.csv"


def monthly_zip_name(file_type: str, month_start: date) -> str:
    """`20250901pal_csv.zip`: the month's daily files, available back to 2005."""
    return f"{month_start:%Y%m}01{file_type}_csv.zip"


def archive_url(file_type: str, name: str) -> str:
    """Full URL of a daily file or monthly zip under the public archive."""
    return f"{NYISO_ARCHIVE_BASE}/{ARCHIVE_DIRS[file_type]}/{name}"


# --- backtest (docs/plan.md §3) ----------------------------------------------------
WARMUP_START = date(2025, 6, 1)  # lags and alpha windows need history before the first bid
BACKTEST_START = date(2025, 9, 1)
BACKTEST_END = date(2026, 8, 31)
