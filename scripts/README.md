# scripts/ — one-command reports

Every headline number quoted in the README or the UI must be reproducible by one of
these scripts on the stated window (docs/plan.md §4). All of them run offline once the
two data scripts have populated `data/` (gitignored: NYISO and Open-Meteo data are
fetched, never redistributed).

| Script | Produces |
|---|---|
| `backfill.py` | `data/processed/*.pkl`: 15-min load slots (+ NYCA), 15-min RT prices, hourly DA / RT prices, isolf, from the NYISO archive (`WARMUP_START` → yesterday by default) |
| `fetch_weather.py` | `data/weather.csv`: day-ahead-issued temperature forecasts per zone centroid (Open-Meteo previous-runs API) |
| `run_backtest.py` | rolling daily retrain as of D−1 05:00 ET for every zone and day; `results/<name>/results.csv` (median, P10/P90, α-bid, actual per slot) and `summary.csv` (hourly and 15-min MAPE per zone); resumable per-day cache |
| `skill_baselines.py` | persistence D−2, D−7, mean(D−7, D−14) and NYISO's `isolf` (pre-close = file named D−1, post-close = file named D) on the same (zone, day) pairs; `results/<name>/baselines.csv` |
| `imbalance_report.py` | settlement in $ vs perfect foresight for every strategy incl. `isolf + α`, our median and our α-bid; per zone and pooled; prints the headline check |
| `quantile_calibration.py` | P10/P90 coverage at slot and hourly resolution, before and after a trailing-30-day conformal rescaling |
| `compare_runs.py` | scores experiment runs against a base run on exactly the same (zone, day) pairs, so partial sweeps stay comparable; `--bootstrap` adds a paired bootstrap over days (95% CI, share of days better, mean by month) |
| `export_tft.py` | fits the TFT on the latest 365-day window (all zones, GPU) and writes the served bundle `data/models/tft/{tft.onnx, tft.json}` (constants, per-zone scales, feature version, SHA-256); the export pins eval mode, is verified against torch, and the bundle is re-loaded with onnxruntime for a smoke forecast of `--target` per zone. Upload both files to the VM's data volume once a month |
| `run_tft.py` | the Temporal Fusion Transformer (`models/tft.py`) under the same protocol, fitted once per `--refit-days` block on every zone (`--train-zones all`) and forecasting each day from an encoder that stops at that day's cutoff; same `results/<name>/` layout, so every report script works on it; needs torch (`requirements-research.txt`) |

Typical sequence:

```powershell
.\.venv\Scripts\python.exe scripts\backfill.py
.\.venv\Scripts\python.exe scripts\fetch_weather.py
.\.venv\Scripts\python.exe scripts\run_backtest.py --name default            # ~1.5 h with 14 workers (365-day window)
.\.venv\Scripts\python.exe scripts\skill_baselines.py --name default
.\.venv\Scripts\python.exe scripts\imbalance_report.py --name default
.\.venv\Scripts\python.exe scripts\quantile_calibration.py --name default
```

`run_backtest.py` flags for the sweeps recorded in `docs/experiments.md`:
`--window`, `--half-life`, `--decay-floor`, `--target-mode {mw,ratio}`, `--no-weather`,
`--no-hourly-weather`, `--no-extra-weather`, `--weather-lead {d2,d1}`, `--daytype`,
`--n-estimators`, `--learning-rate`, `--num-leaves`, `--min-child-samples`,
`--alpha-window`, `--zones`, `--start/--end`, and since the models/ package:
`--estimator {lgbm,xgb}`, `--augment swap` with `--swap-p`, `--swap-copies`,
`--swap-weight`, `--swap-within {tod,none}`, and `--history-start YYYY-MM-DD` (ignore
load before that day, e.g. `2025-06-01` reproduces the 12-month-data runs on the
24-month archive). The per-day cache is keyed by `--name` only, so a new configuration
always needs a new name.
