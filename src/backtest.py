"""Rolling day-ahead backtest: one fresh retrain per (zone, day) as of the D-1 05:00 ET cutoff.

Protocol (docs/plan.md §3): for every target day ``D`` in the range and every zone,
train on the zone's slots ending at or before the cutoff (the leakage guard in
:func:`model.forecast_day` enforces it), forecast the 96 (92/100) slots of ``D`` with
the median, the P10/P90 band and the α-quantile bid, where α is the newsvendor ratio
of the trailing price spread as of the same cutoff (:mod:`src.settlement`).

Results are cached per (zone, day) as CSV under ``results/<name>/<zone>/`` so a run can
be resumed, then concatenated to ``results/<name>/results.csv``. Work is split into
(zone, month) chunks over a process pool; LightGBM runs single-threaded per chunk.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.nyiso import PROJECT_ROOT
from app.weather import features_for, hourly_features_for
from model import DECAY_HALF_LIFE_DAYS, SLOT, LeakageError, cutoff_for, forecast_day
from src.config import BACKTEST_END, BACKTEST_START, NYCA, ZONES
from src.settlement import alpha_series

log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_COLUMNS = [
    "zone",
    "ts_utc",
    "date",
    "slot",
    "tod",
    "pred",
    "p10",
    "p90",
    "p_alpha",
    "alpha",
    "actual",
]


@dataclass(frozen=True)
class BacktestConfig:
    name: str = "default"
    start: date = BACKTEST_START
    end: date = BACKTEST_END
    zones: tuple[str, ...] = (*ZONES, NYCA)
    window_days: int = 120
    half_life: float = DECAY_HALF_LIFE_DAYS
    quantiles: tuple[float, ...] = (0.1, 0.9)
    alpha_window_days: int = 30
    weather_lead: str | None = "d2"  # None = no weather features
    target_mode: str = "mw"  # "mw" (load directly) or "ratio" (y / same-slot 3-week mean)
    hourly_weather: bool = True  # forecast temperature at each hour (temp_h); the sweep winner
    extra_weather: bool = (
        False  # apparent temp, dew point, humidity, cloud, wind, radiation per hour
    )
    daytype: bool = False  # weekend/holiday type and same-type lag anchors
    decay_floor: float = 0.0  # minimum sample weight (keeps last year's season in range)
    workers: int = 1
    results_dir: Path = RESULTS_DIR
    model_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return self.results_dir / self.name

    def targets(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]


def zone_dirname(zone: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", zone).strip("_")


def day_path(cfg: BacktestConfig, zone: str, target: date) -> Path:
    return cfg.out_dir / zone_dirname(zone) / f"{target.isoformat()}.csv"


def forecast_one(
    cfg: BacktestConfig,
    zone: str,
    target: date,
    zone_slots: pd.DataFrame,
    weather_feats: pd.DataFrame | None,
    alpha: float,
    hourly_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One (zone, day) forecast joined with the actual load; raises on thin history."""
    history = zone_slots[zone_slots["ts_utc"] + SLOT <= cutoff_for(target)]
    overrides = {
        "n_jobs": 1,
        "decay_half_life": cfg.half_life,
        "decay_floor": cfg.decay_floor,
        **cfg.model_overrides,
    }
    out = forecast_day(
        history,
        target,
        weather=weather_feats,
        weather_hourly=hourly_feats,
        quantiles=cfg.quantiles,
        alpha=alpha,
        window_days=cfg.window_days,
        model_overrides=overrides,
        target_mode=cfg.target_mode,
        daytype=cfg.daytype,
    )
    out.insert(0, "zone", zone)
    actual = zone_slots[["ts_utc", "load_mw"]].rename(columns={"load_mw": "actual"})
    out = out.merge(actual, on="ts_utc", how="left")
    return out[RESULT_COLUMNS]


def run_chunk(
    cfg: BacktestConfig,
    zone: str,
    targets: list[date],
    zone_slots: pd.DataFrame,
    weather_feats: pd.DataFrame | None,
    alphas: dict[date, float],
    hourly_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast every target of one zone, reusing cached days; returns the chunk's rows."""
    frames = []
    for target in targets:
        path = day_path(cfg, zone, target)
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["ts_utc", "date"]))
            continue
        t0 = time.perf_counter()
        try:
            out = forecast_one(
                cfg, zone, target, zone_slots, weather_feats, alphas.get(target, 0.5), hourly_feats
            )
        except LeakageError:
            raise
        except ValueError as exc:
            log.warning("%s %s skipped: %s", zone, target, exc)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        frames.append(out)
        log.debug("%s %s done in %.1fs", zone, target, time.perf_counter() - t0)
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    return out


def _chunks(targets: list[date]) -> list[list[date]]:
    by_month: dict[tuple[int, int], list[date]] = {}
    for t in targets:
        by_month.setdefault((t.year, t.month), []).append(t)
    return list(by_month.values())


def run(
    cfg: BacktestConfig,
    load_slots: pd.DataFrame,
    prices: pd.DataFrame | None,
    weather: pd.DataFrame | None,
    weather_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the whole backtest; returns the concatenated results (also saved as CSV)."""
    targets = cfg.targets()
    tasks = []
    for zone in cfg.zones:
        zone_slots = load_slots[load_slots["zone"] == zone].reset_index(drop=True)
        if zone_slots.empty:
            log.warning("no load data for zone %s", zone)
            continue
        feats = features_for(weather, zone, lead=cfg.weather_lead) if cfg.weather_lead else None
        if prices is not None and zone != NYCA:
            alphas = alpha_series(prices, zone, targets, cfg.alpha_window_days).to_dict()
        else:
            alphas = {}  # NYCA has no zonal price: α = 0.5, no settlement
        hfeats = None
        if cfg.hourly_weather and cfg.weather_lead:
            hfeats = hourly_features_for(
                weather_hourly, zone, lead=cfg.weather_lead, extra=cfg.extra_weather
            )
        for chunk in _chunks(targets):
            tasks.append((cfg, zone, chunk, zone_slots, feats, alphas, hfeats))
    log.info(
        "backtest %s: %d zones x %d days in %d chunks, %d workers",
        cfg.name,
        len(cfg.zones),
        len(targets),
        len(tasks),
        cfg.workers,
    )
    t0 = time.perf_counter()
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            frames = list(pool.map(_run_task, tasks))
    else:
        frames = [_run_task(t) for t in tasks]
    results = (
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if frames
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    results = results.sort_values(["zone", "ts_utc"]).reset_index(drop=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.out_dir / "results.csv", index=False)
    log.info(
        "backtest %s: %d rows in %.0fs -> %s",
        cfg.name,
        len(results),
        time.perf_counter() - t0,
        cfg.out_dir / "results.csv",
    )
    return results


def _run_task(task: tuple[Any, ...]) -> pd.DataFrame:
    return run_chunk(*task)


def load_results(name: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    path = results_dir / name / "results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `python scripts/run_backtest.py --name {name}` first"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df
