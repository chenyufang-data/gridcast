# scripts/ — one-command reports

Every headline number quoted in the README or the UI must be reproducible by one of
these scripts on the stated window (docs/plan.md §4). All of them run offline once the
two data scripts have populated `data/` (gitignored: NYISO and Open-Meteo data are
fetched, never redistributed).

| Script | Produces |
|---|---|
| `backfill.py` | `data/processed/*.pkl`: 15-min load slots (+ NYCA), 15-min RT prices, hourly DA / RT prices, isolf, from the NYISO archive (2025-06-01 → yesterday by default) |
| `fetch_weather.py` | `data/weather.csv`: day-ahead-issued temperature forecasts per zone centroid (Open-Meteo previous-runs API) |
| `run_backtest.py` | rolling daily retrain as of D−1 05:00 ET for every zone and day; `results/<name>/results.csv` (median, P10/P90, α-bid, actual per slot) and `summary.csv` (hourly and 15-min MAPE per zone); resumable per-day cache |
| `skill_baselines.py` | persistence D−2, D−7, mean(D−7, D−14) and NYISO's `isolf` (pre-close = file named D−1, post-close = file named D) on the same (zone, day) pairs; `results/<name>/baselines.csv` |
| `imbalance_report.py` | settlement in $ vs perfect foresight for every strategy incl. `isolf + α`, our median and our α-bid; per zone and pooled; prints the headline check |
| `quantile_calibration.py` | P10/P90 coverage at slot and hourly resolution, before and after a trailing-30-day conformal rescaling |

Typical sequence:

```powershell
.\.venv\Scripts\python.exe scripts\backfill.py
.\.venv\Scripts\python.exe scripts\fetch_weather.py
.\.venv\Scripts\python.exe scripts\run_backtest.py --name default            # ~30 min on 12 cores
.\.venv\Scripts\python.exe scripts\skill_baselines.py --name default
.\.venv\Scripts\python.exe scripts\imbalance_report.py --name default
.\.venv\Scripts\python.exe scripts\quantile_calibration.py --name default
```

`run_backtest.py` flags for the sensitivity sweep recorded in `docs/experiments.md`:
`--window`, `--half-life`, `--target-mode {mw,ratio}`, `--no-weather`,
`--weather-lead {d2,d1}`, `--alpha-window`, `--zones`, `--start/--end`.
