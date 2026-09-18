"""Weather provider: day-ahead-issued temperature forecasts per NYISO zone from Open-Meteo.

Leakage rule (docs/plan.md §2.4): a forecast for target day D may only use weather
information available at the cutoff D-1 05:00 ET. Open-Meteo's *previous runs* API
serves, for every hour, the value forecast ``N`` days earlier:

- ``temperature_2m_previous_day2`` (lead ~48 h) was issued on D-2 for every hour of D,
  always before the cutoff → the **default** feature source (``lead="d2"``);
- ``temperature_2m_previous_day1`` (lead ~24 h) was issued on D-1 at the valid hour,
  i.e. after the cutoff for hours past 05:00 ET → kept only for a labelled,
  optimistic experiment (``lead="d1"``).

Everything is best-effort and offline-safe: failures leave the existing CSV untouched
and the model degrades to its no-weather feature set. Data: Open-Meteo
(https://open-meteo.com/, CC BY 4.0).

CSV layout (``WEATHER_PATH``, default ``data/weather.csv``):
``date, zone, tfc1_mean, tfc1_min, tfc1_max, tfc2_mean, tfc2_min, tfc2_max`` (°C).
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


def _daily(hourly: dict[str, list[object]], var: str, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame({"ts": pd.to_datetime(hourly["time"]), "t": hourly[var]}).dropna()
    df["date"] = df["ts"].dt.normalize()
    agg = df.groupby("date")["t"].agg(["mean", "min", "max"]).round(2)
    agg.columns = [f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max"]
    return agg


def fetch_zone(zone: str, start: date, end: date, timeout: float = 60.0) -> pd.DataFrame:
    """Daily forecast temperatures for one zone: ``date, zone, tfc1_*, tfc2_*``."""
    lat, lon = ZONE_CENTROIDS[zone]
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES.values()),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": str(MARKET_TZ),
    }
    resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    parts = [_daily(hourly, var, prefix) for prefix, var in VARIABLES.items()]
    out = pd.concat(parts, axis=1).reset_index()
    out.insert(1, "zone", zone)
    return out


def with_nyca(df: pd.DataFrame) -> pd.DataFrame:
    """Append the load-weighted statewide row per date."""
    z = df[df["zone"] != NYCA].copy()
    z["w"] = z["zone"].map(ZONE_WEIGHT)
    cols = [c for c in z.columns if c.startswith("tfc")]
    weighted = z[cols].multiply(z["w"], axis=0)
    weighted["date"] = z["date"]
    weighted["w"] = z["w"]
    g = weighted.groupby("date").sum()
    total = g[cols].div(g["w"], axis=0).round(2).reset_index()
    total.insert(1, "zone", NYCA)
    return (
        pd.concat([z.drop(columns="w"), total], ignore_index=True)
        .sort_values(["date", "zone"])
        .reset_index(drop=True)
    )


def refresh(
    start: date, end: date | None = None, path: Path | None = None, pause: float = 0.3
) -> bool:
    """Rebuild the weather CSV for `start`..`end` (default: 7 days ahead). False on any failure."""
    path = path or WEATHER_PATH
    end = end or (datetime.now(MARKET_TZ).date() + timedelta(days=7))
    frames = []
    try:
        for zone in ZONES:
            frames.append(fetch_zone(zone, start, end))
            time.sleep(pause)
    except Exception:
        log.exception("weather refresh failed; keeping %s", path)
        return False
    out = with_nyca(pd.concat(frames, ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("weather: %d zone-days -> %s", len(out), path)
    return True


def load_weather(path: Path | None = None) -> pd.DataFrame | None:
    path = path or WEATHER_PATH
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def features_for(weather: pd.DataFrame | None, zone: str, lead: str = "d2") -> pd.DataFrame | None:
    """Model-ready day-level features for `zone`: ``date, temp_mean, temp_min, temp_max, temp_dev``.

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
