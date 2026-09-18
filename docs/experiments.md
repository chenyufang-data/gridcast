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
| Training window | `[D−1−W, D−2]` with W = 120 days, decay weight `0.5^(age/32 d)` |
| Features | same-tod lags 2/3/7/14/21 d and their daily levels; D−1 00:00–04:45 morning level and its ratios to D−2 / D−8 mornings; weekly aggregates, trends, shapes; weekday, US federal holiday (+ holiday tomorrow); temperature forecast per day (mean/min/max, deviation from the trailing 7 forecast days ending D−2) and per hour (`temp_h`) |
| Weather lead | `previous_day2` (issued on D−2 for every hour of D, before the cutoff). `previous_day1` is issued after the cutoff for hours past 05:00 ET and is only run as a labelled optimistic experiment |
| Model | LightGBM MAE objective, 600 trees, lr 0.02, 31 leaves, depth 5, subsample 0.7, colsample 0.6; quantile objective at 0.1 / 0.9 for the band and at α for the bid |
| α | newsvendor ratio `c_under / (c_under + c_over)` of the trailing 30-day RT−DA spread per zone, slots ending at or before the cutoff |
| Baselines | persistence D−2, D−7, mean(D−7, D−14); `isolf_pre` (file named D−1, known before the cutoff); `isolf_post` (file named D, posted ~07:10–08:00 ET on D−1, after the cutoff) |

## 2. Sensitivity sweep (N.Y.C. and WEST, full 12 months)

Hourly MAPE (%) pooled over both zones; `results/sweep_<name>/summary.csv`. One knob
changes per row against the first row unless stated. Runs: `scripts/run_backtest.py
--zones N.Y.C. WEST --workers 12 <flags>`.

| Run | Flags | N.Y.C. | WEST | Pooled | P10–P90 coverage (raw) | Verdict |
|---|---|---|---|---|---|---|
| `mw56` | window 56, half-life 32, MW target, daily weather (d2) | 4.83 | 5.23 | 5.03 | 39% | baseline |
| `ratio56` | `--target-mode ratio` | 5.80 | 5.66 | 5.73 | 36% | **worse**: the ratio to the 3-week same-slot mean does not help trees extrapolate |
| `mw120` | `--window 120` | 4.74 | 5.20 | **4.97** | 42% | small gain: a longer window sees more of the temperature range |
| `mw120_hl64` | `--window 120 --half-life 64` | 4.71 | 5.24 | 4.97 | 44% | wash (N.Y.C. better, WEST worse) |
| `ratio120_hl64` | ratio target, window 120, half-life 64 | 5.54 | 5.66 | 5.60 | 40% | worse |
| `mw56_noweather` | `--no-weather` | 6.66 | 6.44 | 6.55 | 36% | weather is worth ~1.5 points |
| `mw56_d1` | `--weather-lead d1` (issued after the cutoff for most hours; optimistic) | 4.64 | 5.16 | 4.90 | 40% | only +0.13 over the leakage-free lead: **not used** |
| `mw120_fast` | window 120, 300 trees @ lr 0.04 | 4.79 | 5.25 | 5.02 | 43% | slightly worse; the 2× speed-up is not worth 0.05 |
| `mw120_hourly` | window 120, hourly `temp_h` added | 4.44 | 5.12 | **4.78** | 42% | **adopted**: the forecast temperature at each hour is the single best addition |

Default after the sweep: **window 120, half-life 32, MW target, daily + hourly weather at the
leakage-free lead, 600 trees @ lr 0.02** (`BacktestConfig` defaults).

Observations that shaped the default:

- The ratio target is a negative result in both windows.
- The optimistic weather lead buys almost nothing, so the honest lead costs little.
- The raw quantile band is badly under-dispersed (≈ 40% coverage for a nominal 80%):
  the conformal rescaling in `scripts/quantile_calibration.py` is mandatory, not optional.

## 3. Full run (11 zones + NYCA)

_Filled in from `results/default/`._

## 4. Negative results and open items

_Filled in after the runs._
