# Experiment log

Every row was produced by `scripts/run_backtest.py` (and siblings) under the exact bid
protocol: one fresh retrain per zone and day on data ending at or before **D−1 05:00 ET**,
scored on the 12 months **2025-09-01 → 2026-08-31** unless a narrower window is stated.
MAPE is pooled over hours (hourly means of the 15-min slots), the only resolution at
which NYISO's hourly `isolf` can be compared fairly. Negative results are kept on
purpose; nothing here was tuned on the test range beyond the sweep listed in §2.

Data: NYISO public MIS archive (mis.nyiso.com/public/csv), fetched at runtime and not
redistributed; Open-Meteo previous-runs forecasts for weather.

## 1. Fixed protocol

| Item | Value |
|---|---|
| Cutoff | D−1 05:00 ET; `model.forecast_day` raises on any slot ending later |
| Target | 15-min mean MW per zone (96 slots; 92 / 100 on DST days), hourly means for bidding and scoring |
| Training window | `[D−1−W, D−2]` with W = 365 days, decay weight `0.5^(age/90 d)` (first full run: 120 d / 32 d) |
| Features | same-tod lags 2/3/7/14/21 d and their daily levels; D−1 00:00–04:45 morning level and its ratios to D−2 / D−8 mornings; weekly aggregates, trends, shapes; weekday, US federal holiday (+ holiday tomorrow); temperature forecast per day (mean/min/max, deviation from the trailing 7 forecast days ending D−2) and per hour (`temp_h`, apparent temperature, dew point, humidity, cloud cover, wind, shortwave radiation, 3-h temperature mean) |
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

## 2b. Improvement sweep after the first full run (N.Y.C. and MHK VL, 2026-03-01 → 2026-08-31)

The error analysis of the first full run (§3.1) showed the gap to the ISO concentrated
in daytime hours, in spring/early summer regime shifts (a May heat wave under-forecast
by 15–21% with nothing that hot inside a 120-day window), on weekends (+3.5% bias) and
above 20 °C. Four candidate fixes were run on the two most informative zones over the
hard half of the year and scored with `scripts/compare_runs.py` against the default on
exactly the same days (default on those days: N.Y.C. 5.01, MHK VL 12.91, pooled 8.96).

| Run | Flags | N.Y.C. | MHK VL | Pooled | Δ vs default | Verdict |
|---|---|---|---|---|---|---|
| `imp_extra` | `--extra-weather` (apparent temperature, dew point, humidity, cloud, wind, radiation per hour + 3-h temperature mean) | 4.57 | 10.43 | 7.50 | **−1.46** | the single biggest gain, most of it in the hard zone |
| `imp_daytype` | `--daytype` (weekend/holiday type, same-type lag anchors) | 4.92 | 13.07 | 8.99 | +0.03 | **null** on its own |
| `imp_w365_hl90` | `--window 365 --half-life 90` | 4.43 | 12.61 | 8.52 | −0.44 | last summer in range helps the transition months |
| `imp_w365_floor` | `--window 365 --decay-floor 0.15` | 4.41 | 12.21 | 8.31 | −0.65 | a weight floor beats a slower decay |
| `imp_combo` | 365 d, half-life 90, extra weather, day-type | 3.97 | 10.10 | 7.04 | −1.92 | gains add up |
| `imp_combo_leaves` | combo + 63 leaves, 15 min child, 900 trees | 3.90 | 10.03 | 6.97 | −1.99 | +0.07 for 1.5× compute: not adopted |
| `imp_combo_floor` | 365 d, floor 0.15, extra weather, day-type | 3.92 | 10.05 | 6.99 | −1.97 | equal to the combo within noise |
| `imp_combo_nodaytype` | 365 d, half-life 90, extra weather | 4.03 | 10.02 | 7.02 | −1.93 | **adopted**: same result without the null feature group |

New default after this sweep: **window 365, half-life 32 → 90, MW target, daily + hourly
weather incl. apparent temperature / dew point / humidity / cloud / wind / radiation at
the leakage-free lead, 600 trees @ lr 0.02, no day-type features** (`--daytype` and
`--decay-floor` stay available as experiment flags). The first full run is kept as
`results/v1_w120` for the comparison column in §3.

## 3. Full run (11 zones + NYCA)

`scripts/run_backtest.py --name default --workers 14`, then `skill_baselines.py`,
`imbalance_report.py`, `quantile_calibration.py` with `--name default`. Committed tables:
`results/default/summary.csv`, `baselines_summary.csv`, `imbalance_pooled.csv`,
`imbalance_by_zone.csv`, `calibration.csv`.

### 3.1 Accuracy (hourly MAPE, %) and band coverage (15-min, nominal 80%)

| Zone | Model | persist D−2 | persist D−7 | mean(D−7, D−14) | isolf_pre | isolf_post | P10–P90 raw | conformal |
|---|---|---|---|---|---|---|---|---|
| CAPITL | 6.35 | 11.41 | 12.23 | 11.08 | 5.33 | 5.09 | 52% | 78% |
| CENTRL | 6.06 | 11.10 | 11.83 | 10.65 | 6.32 | 6.15 | 53% | 77% |
| DUNWOD | 4.72 | 10.09 | 10.59 | 10.07 | 3.75 | 3.40 | 52% | 78% |
| GENESE | 5.59 | 11.02 | 11.07 | 10.17 | 4.69 | 4.31 | 53% | 78% |
| HUD VL | 6.38 | 12.33 | 13.88 | 12.71 | 6.20 | 5.30 | 53% | 78% |
| LONGIL | 5.78 | 11.02 | 11.60 | 10.97 | 4.05 | 3.60 | 51% | 77% |
| MHK VL | 8.19 | 14.32 | 15.29 | 13.86 | 7.80 | 7.32 | 54% | 78% |
| MILLWD | 6.99 | 12.60 | 14.43 | 13.57 | 5.97 | 5.41 | 50% | 77% |
| N.Y.C. | 3.80 | 9.29 | 8.68 | 8.52 | 2.44 | 2.04 | 51% | 78% |
| NORTH | 4.43 | 5.85 | 6.93 | 6.24 | 5.39 | 4.37 | 55% | 78% |
| NYCA | 4.03 | 8.76 | 9.11 | 8.50 | 2.81 | 2.58 | 50% | 77% |
| WEST | 4.39 | 8.14 | 7.92 | 7.40 | 3.42 | 2.93 | 50% | 76% |
| pooled | 5.56 | 10.49 | 11.13 | 10.31 | 4.85 | 4.38 | 52% | 77% |

- The model beats the best naive baseline in every zone (pooled −46% relative error).
- NYISO's pre-close forecast is better in every zone except CENTRL, NORTH; post-close the model still wins CENTRL.
- Against the first full run (120-day window, temperature only; `results/v1_w120`), pooled hourly MAPE moved from 6.61 to 5.56; the hard upstate zones gained most (MHK VL 10.11 → 8.19).
- Raw quantile trees cover 52% of actuals; the trailing-30-day conformal rescaling
  reaches 77% (target 80%) with 12% below / 11% above.

### 3.2 Settlement in dollars (11 priced zones, hourly bids settled per 15-min slot)

| Strategy | Signed $ vs perfect foresight | 95% bootstrap CI | Σ \|dev × spread\| | $/MWh (signed) |
|---|---|---|---|---|
| `persist_2d` | $86M | [$31M, $143M] | $335M | 0.568 |
| `persist_7d` | $26M | [$-63M, $118M] | $432M | 0.175 |
| `mean_7_14` | $21M | [$-94M, $128M] | $446M | 0.141 |
| `isolf_pre` | $34M | [$8M, $60M] | $158M | 0.226 |
| `isolf_post` | $26M | [$-2M, $53M] | $148M | 0.171 |
| `isolf_pre_alpha` | $55M | [$35M, $77M] | $143M | 0.363 |
| `model` | $46M | [$19M, $74M] | $184M | 0.302 |
| `model_alpha_bid` | $46M | [$18M, $76M] | $186M | 0.307 |
| `model_alpha_emp` | $50M | [$18M, $81M] | $190M | 0.328 |

- α (newsvendor ratio of the trailing 30-day spread) averages 0.41–0.48 by zone:
  real-time prices sit below day-ahead most of the time, so the cost-aware bid is
  slightly *short* of the median.
- The signed totals are noise: every interval overlaps every other. **The α-bid does not
  meet the headline rule** (docs/plan.md §4) and stays a secondary feature of the product.
- Σ |dev × spread| (dollars at risk) follows accuracy: ISO $158M <
  model $184M < naive ≥ $335M.

## 4. Negative results and open items

Negative or null results (kept so nobody repeats them):

- Ratio target (`y / same-slot 3-week mean`): +0.6–0.7 points worse in both windows.
- 300 trees @ lr 0.04: 2× faster, +0.05 worse.
- Optimistic weather lead (`previous_day1`): +0.13 better but partly post-cutoff; unused.
- Half-life 64 vs 32 with the 120-day window: a wash.
- Day-type features (weekend/holiday type, same-type lag anchors): null on their own.
- Bigger trees (63 leaves, 900 trees): +0.07 for 1.5× compute.
- α-bid via quantile LightGBM: the under-dispersed quantiles move the bid by ~40 MW on a
  6,000 MW zone, and the empirical-ratio variant (which does move it) costs more, not less.
- Signed imbalance dollars over 12 months cannot rank strategies (see §3.2).

Open items, in the order they are likely to pay off:

1. A second weather station for the large zones, and zone-specific variables for the
   small upstate zones with industrial load (MHK VL, MILLWD).
2. The ISO forecast as a feature is excluded by decision (docs/plan.md Q8). A labelled
   experiment would almost certainly close most of the gap; it should stay a separate,
   clearly named variant if ever run.
3. Blending model and `isolf_pre` per zone with weights fitted on trailing days.
4. Band calibration per hour of day rather than one scale per day.
5. Two winters of data once the archive backfill reaches back to 2024 (the plan's Q2
   option), so both DST transitions and every holiday are seen twice.
