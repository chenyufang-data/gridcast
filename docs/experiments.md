# Experiment log

Every row was produced by `scripts/run_backtest.py` (and siblings) under the exact bid
protocol: one fresh retrain per zone and day on data ending at or before **D−1 05:00 ET**,
scored on the 12 months **2025-09-01 → 2026-08-31** unless a narrower window is stated.
MAPE is pooled over hours (hourly means of the 15-min slots), the only resolution at
which NYISO's hourly `isolf` can be compared fairly. Negative results are kept on
purpose; nothing here was tuned on the test range beyond the sweep listed in §2.

Data: NYISO public MIS archive (mis.nyiso.com/public/csv), fetched at runtime and not
redistributed; Open-Meteo previous-runs forecasts for temperature.

## 1. Fixed protocol

| Item | Value |
|---|---|
| Cutoff | D−1 05:00 ET; `model.forecast_day` raises on any slot ending later |
| Target | 15-min mean MW per zone (96 slots; 92 / 100 on DST days), hourly means for bidding and scoring |
| Training window | `[D−1−W, D−2]`, decay weight `0.5^(age/half-life)` |
| Features | same-tod lags 2/3/7/14/21 d and their daily levels; D−1 00:00–04:45 morning level and its ratios to D−2 / D−8 mornings; weekly aggregates, trends, shapes; weekday, US federal holiday (+ holiday tomorrow); temperature forecast (mean/min/max, deviation from the trailing 7 forecast days ending D−2) |
| Weather lead | `previous_day2` (issued on D−2 for every hour of D, before the cutoff). `previous_day1` is issued after the cutoff for hours past 05:00 ET and is only run as a labelled optimistic experiment |
| Model | LightGBM MAE objective, 600 trees, lr 0.02, 31 leaves, depth 5, subsample 0.7, colsample 0.6; quantile objective at 0.1 / 0.9 for the band and at α for the bid |
| α | newsvendor ratio `c_under / (c_under + c_over)` of the trailing 30-day RT−DA spread per zone, slots ending at or before the cutoff |
| Baselines | persistence D−2, D−7, mean(D−7, D−14); `isolf_pre` (file named D−1, known before the cutoff); `isolf_post` (file named D, posted ~07:10–08:00 ET on D−1, after the cutoff) |

## 2. Sensitivity sweep (N.Y.C. and WEST, full 12 months)

_Filled in from `results/sweep_*/summary.csv`; see §2 table below._

## 3. Full run (11 zones + NYCA)

_Filled in from `results/default/`._

## 4. Negative results and open items

_Filled in after the runs._
