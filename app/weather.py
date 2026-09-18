"""Weather provider: day-ahead-issued temperature forecasts per NYISO zone from Open-Meteo.

Leakage rule (docs/plan.md §2.4): a forecast for target day D may only use weather
information available at the cutoff D-1 05:00 ET. Open-Meteo's *previous runs* API
serves, for every hour, the value forecast ``N`` days earlier:

- ``temperature_2m_previous_day2`` (lead ~48 h) was issued on D-2 for every hour of D,
  always before the cutoff → the **default** feature source (``lead="d2"``);
- ``temperature_2m_previous_day1`` (lead ~24 h) was issued on D-1 at the valid hour,
  i.e. after the cutoff for hours past 05:00 ET → kept only for a labelled,
  optimistic experiment (``lead="d1"``).

Everything is best-effort and offline-safe: failures leave the existing CSVs untouched
and the model degrades to its no-weather feature set. Data: Open-Meteo
(https://open-meteo.com/, CC BY 4.0).

Files (``WEATHER_PATH`` / ``WEATHER_HOURLY_PATH``, default ``data/weather*.csv``):
daily ``date, zone, tfc1_mean, tfc1_min, tfc1_max, tfc2_mean, tfc2_min, tfc2_max`` (°C,
local ET days) and hourly ``ts_utc, zone, tfc1, tfc2``.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.config import MARKET_TZ, NYCA, ZONES

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_PATH = Path(os.environ.get("WEATHER_PATH") or PROJECT_ROOT / "data" / "weather.csv")
WEATHER_HOURLY_PATH = Path(
    os.environ.get("WEATHER_HOURLY_PATH") or PROJECT_ROOT / "data" / "weather_hourly.csv"
)
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# approximate load-weighted centroids (lat, lon)
ZONE_CENTROIDS: dict[str, tuple[float, float]] = {
    "CAPITL": (42.65, -73.75),  # Albany
    "CENTRL": (43.05, -76.15),  # Syracuse
    "DUNWOD": (40.93, -73.86),  # Yonkers
    "GENESE": (43.16, -77.61),  # Rochester
    "HUD VL": (41.70, -73.92),  # Poughkeepsie
    "LONGIL": (40.75, -73.20),  # Islip
    "MHK VL": (43.10, -75.23),  # Utica
    "MILLWD": (41.20, -73.80),  # Ossining
    "N.Y.C.": (40.71, -74.01),  # Manhattan
    "NORTH": (44.70, -73.45),  # Plattsburgh
    "WEST": (42.89, -78.88),  # Buffalo
}
# typical zone load (MW) used to weight the statewide (NYCA) temperature
ZONE_WEIGHT: dict[str, float] = {
    "CAPITL": 1350, "CENTRL": 1660, "DUNWOD": 660, "GENESE": 1080, "HUD VL": 1080,
    "LONGIL": 2380, "MHK VL": 790, "MILLWD": 250, "N.Y.C.": 6100, "NORTH": 630,
    "WEST": 1730,
}  # fmt: skip
VARIABLES = {"tfc1": "temperature_2m_previous_day1", "tfc2": "temperature_2m_previous_day2"}
LEADS = {"d1": "tfc1", "d2": "tfc2"}


def _frames(hourly: dict[str, list[object]], zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """API payload (UTC stamps) -> (daily aggregates on local ET days, hourly UTC rows)."""
    ts_utc = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    h = pd.DataFrame({"ts_utc": ts_utc, **{k: hourly[v] for k, v in VARIABLES.items()}})
    h.insert(1, "zone", zone)
    local_day = h["ts_utc"].dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_localize(None)
    parts = []
    for prefix in VARIABLES:
        agg = h.groupby(local_day)[prefix].agg(["mean", "min", "max"]).round(2)
        agg.columns = [f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max"]
        parts.append(agg)
    daily = pd.concat(parts, axis=1).rename_axis("date").reset_index()
    daily.insert(1, "zone", zone)
    return daily, h.dropna(subset=list(VARIABLES), how="all").reset_index(drop=True)


def fetch_zone(
    zone: str, start: date, end: date, timeout: float = 60.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily, hourly) forecast temperatures for one zone over `start`..`end`."""
    lat, lon = ZONE_CENTROIDS[zone]
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES.values()),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return _frames(resp.json()["hourly"], zone)


def with_nyca(df: pd.DataFrame, key: str = "date") -> pd.DataFrame:
    """Append the load-weighted statewide row per `key` (``date`` or ``ts_utc``)."""
    z = df[df["zone"] != NYCA].copy()
    z["w"] = z["zone"].map(ZONE_WEIGHT)
    cols = [c for c in z.columns if c.startswith("tfc")]
    weighted = z[cols].multiply(z["w"], axis=0)
    weighted[key] = z[key]
    weighted["w"] = z["w"]
    g = weighted.groupby(key).sum()
    total = g[cols].div(g["w"], axis=0).round(2).reset_index()
    total.insert(1, "zone", NYCA)
    out = pd.concat([z.drop(columns="w"), total], ignore_index=True)
    return out.sort_values([key, "zone"]).reset_index(drop=True)


def refresh(
    start: date,
    end: date | None = None,
    path: Path | None = None,
    hourly_path: Path | None = None,
    pause: float = 0.3,
) -> bool:
    """Rebuild both weather CSVs for `start`..`end` (default: 7 days ahead). False on failure."""
    path = path or WEATHER_PATH
    hourly_path = hourly_path or WEATHER_HOURLY_PATH
    end = end or (datetime.now(MARKET_TZ).date() + timedelta(days=7))
    daily_frames, hourly_frames = [], []
    try:
        for zone in ZONES:
            daily, hourly = fetch_zone(zone, start, end)
            daily_frames.append(daily)
            hourly_frames.append(hourly)
            time.sleep(pause)
    except Exception:
        log.exception("weather refresh failed; keeping %s", path)
        return False
    daily_out = with_nyca(pd.concat(daily_frames, ignore_index=True))
    hourly_out = with_nyca(pd.concat(hourly_frames, ignore_index=True), key="ts_utc")
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_out.to_csv(path, index=False)
    hourly_out.to_csv(hourly_path, index=False)
    log.info(
        "weather: %d zone-days -> %s, %d zone-hours -> %s",
        len(daily_out),
        path,
        len(hourly_out),
        hourly_path,
    )
    return True


def load_weather(path: Path | None = None) -> pd.DataFrame | None:
    path = path or WEATHER_PATH
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def load_weather_hourly(path: Path | None = None) -> pd.DataFrame | None:
    path = path or WEATHER_HOURLY_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def features_for(weather: pd.DataFrame | None, zone: str, lead: str = "d2") -> pd.DataFrame | None:
    """Day-level features for `zone`: ``date, temp_mean, temp_min, temp_max, temp_dev``.

    ``temp_dev`` = forecast mean minus the mean forecast of the seven days ending D-2:
    the "about to leave the recent level" signal, built from forecasts only so it is
    available at the cutoff and identical offline and live.
    """
    if weather is None or weather.empty:
        return None
    prefix = LEADS[lead]
    z = weather[weather["zone"] == zone].sort_values("date").copy()
    if z.empty:
        return None
    full = pd.date_range(z["date"].min(), z["date"].max(), freq="D")
    z = z.set_index("date").reindex(full)
    out = pd.DataFrame(index=full)
    out["temp_mean"] = z[f"{prefix}_mean"]
    out["temp_min"] = z[f"{prefix}_min"]
    out["temp_max"] = z[f"{prefix}_max"]
    trailing = out["temp_mean"].rolling(7, min_periods=3).mean().shift(2)
    out["temp_dev"] = out["temp_mean"] - trailing
    return out.rename_axis("date").reset_index()


def hourly_features_for(
    weather_hourly: pd.DataFrame | None, zone: str, lead: str = "d2"
) -> pd.DataFrame | None:
    """Hour-level feature for `zone`: ``hour_utc, temp_h`` (forecast °C valid at that hour)."""
    if weather_hourly is None or weather_hourly.empty:
        return None
    z = weather_hourly[weather_hourly["zone"] == zone]
    if z.empty:
        return None
    out = pd.DataFrame({"hour_utc": z["ts_utc"].dt.floor("h"), "temp_h": z[LEADS[lead]].to_numpy()})
    return out.drop_duplicates("hour_utc").sort_values("hour_utc").reset_index(drop=True)
