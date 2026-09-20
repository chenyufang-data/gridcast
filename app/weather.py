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
# extra hourly variables, 2-day lead only (columns x2_<key> in the hourly file)
EXTRA_VARIABLES = {
    "app": "apparent_temperature_previous_day2",
    "dew": "dew_point_2m_previous_day2",
    "rh": "relative_humidity_2m_previous_day2",
    "cloud": "cloud_cover_previous_day2",
    "wind": "wind_speed_10m_previous_day2",
    "rad": "shortwave_radiation_previous_day2",
}
LEADS = {"d1": "tfc1", "d2": "tfc2"}


def _frames(hourly: dict[str, list[object]], zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """API payload (UTC stamps) -> (daily aggregates on local ET days, hourly UTC rows)."""
    ts_utc = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    cols = {k: hourly[v] for k, v in VARIABLES.items()}
    cols.update({f"x2_{k}": hourly[v] for k, v in EXTRA_VARIABLES.items() if v in hourly})
    h = pd.DataFrame({"ts_utc": ts_utc, **cols})
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
        "hourly": ",".join([*VARIABLES.values(), *EXTRA_VARIABLES.values()]),
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
    cols = [c for c in z.columns if c.startswith("tfc") or c.startswith("x2_")]
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
    weather_hourly: pd.DataFrame | None, zone: str, lead: str = "d2", extra: bool = False
) -> pd.DataFrame | None:
    """Hour-level features for `zone`: ``hour_utc, temp_h`` and, with `extra`, the 2-day-lead
    apparent temperature, dew point, humidity, cloud cover, wind, radiation (``*_h``) plus
    ``temp_h_prev3`` (mean forecast temperature over the three preceding hours).
    """
    if weather_hourly is None or weather_hourly.empty:
        return None
    z = weather_hourly[weather_hourly["zone"] == zone]
    if z.empty:
        return None
    out = pd.DataFrame({"hour_utc": z["ts_utc"].dt.floor("h"), "temp_h": z[LEADS[lead]].to_numpy()})
    out = out.drop_duplicates("hour_utc").sort_values("hour_utc").reset_index(drop=True)
    if extra:
        zz = z.drop_duplicates("ts_utc").sort_values("ts_utc")
        for key in EXTRA_VARIABLES:
            col = f"x2_{key}"
            if col in zz.columns:
                out[f"{key}_h"] = zz[col].to_numpy()
        out["temp_h_prev3"] = out["temp_h"].shift(1).rolling(3, min_periods=1).mean()
    return out


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _merge(old: pd.DataFrame | None, new: pd.DataFrame, key: str) -> pd.DataFrame:
    """Replace the rows of `old` whose (`key`, zone) appear in `new`; append the rest."""
    if old is None or old.empty:
        return new.sort_values([key, "zone"]).reset_index(drop=True)
    old = old.copy()
    if key == "ts_utc":
        old[key] = pd.to_datetime(old[key], utc=True)
    else:
        old[key] = pd.to_datetime(old[key])
    lo, hi = new[key].min(), new[key].max()
    keep = old[(old[key] < lo) | (old[key] > hi)]
    out = pd.concat([keep, new], ignore_index=True)
    return out.sort_values([key, "zone"]).reset_index(drop=True)


def update(
    start: date,
    end: date,
    path: Path | None = None,
    hourly_path: Path | None = None,
    pause: float = 0.3,
) -> bool:
    """Incremental refresh: fetch `start`..`end` for every zone and splice it into the CSVs.

    Rows inside the fetched range are replaced (previous-run values fill in as model runs
    complete), everything outside it is kept, and each file is rewritten atomically.
    False on any failure, with the files untouched.
    """
    path = path or WEATHER_PATH
    hourly_path = hourly_path or WEATHER_HOURLY_PATH
    daily_frames, hourly_frames = [], []
    try:
        for zone in ZONES:
            daily, hourly = fetch_zone(zone, start, end)
            daily_frames.append(daily)
            hourly_frames.append(hourly)
            time.sleep(pause)
    except Exception:
        log.exception("weather update failed; keeping %s", path)
        return False
    daily_new = with_nyca(pd.concat(daily_frames, ignore_index=True))
    hourly_new = with_nyca(pd.concat(hourly_frames, ignore_index=True), key="ts_utc")
    daily_out = _merge(load_weather(path), daily_new, "date")
    hourly_out = _merge(load_weather_hourly(hourly_path), hourly_new, "ts_utc")
    _atomic_write(daily_out, path)
    _atomic_write(hourly_out, hourly_path)
    log.info(
        "weather update %s..%s: %d zone-days fetched, files now %d / %d rows",
        start,
        end,
        len(daily_new),
        len(daily_out),
        len(hourly_out),
    )
    return True


def trim(boundary: date, path: Path | None = None, hourly_path: Path | None = None) -> int:
    """Drop weather rows before `boundary` (retention); returns the number removed."""
    path = path or WEATHER_PATH
    hourly_path = hourly_path or WEATHER_HOURLY_PATH
    removed = 0
    daily = load_weather(path)
    if daily is not None and not daily.empty:
        keep = daily[daily["date"] >= pd.Timestamp(boundary)]
        removed += len(daily) - len(keep)
        if len(keep) != len(daily):
            _atomic_write(keep, path)
    hourly = load_weather_hourly(hourly_path)
    if hourly is not None and not hourly.empty:
        edge = pd.Timestamp(boundary, tz=MARKET_TZ).tz_convert("UTC")
        keep = hourly[hourly["ts_utc"] >= edge]
        removed += len(hourly) - len(keep)
        if len(keep) != len(hourly):
            _atomic_write(keep, hourly_path)
    return removed
