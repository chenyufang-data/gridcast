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
| Cutoff | D−1 05:00 ET; `models.forecast_day` raises on any slot ending later |
| Target | 15-min mean MW per zone (96 slots; 92 / 100 on DST days), hourly means for bidding and scoring |
| Training window | `[D−1−W, D−2]` with W = 365 days, decay weight `0.5^(age/90 d)` (first full run: 120 d / 32 d) |
| Features | same-tod lags 2/3/7/14/21 d and their daily levels; D−1 00:00–04:45 morning level and its ratios to D−2 / D−8 mornings; weekly aggregates, trends, shapes; weekday, US federal holiday (+ holiday tomorrow); temperature forecast per day (mean/min/max, deviation from the trailing 7 forecast days ending D−2) and per hour (`temp_h`, apparent temperature, dew point, humidity, cloud cover, wind, shortwave radiation, 3-h temperature mean) |
| Weather lead | `previous_day2` (issued on D−2 for every hour of D, before the cutoff). `previous_day1` is issued after the cutoff for hours past 05:00 ET and is only run as a labelled optimistic experiment |
| Model | LightGBM MAE objective, 600 trees, lr 0.02, 31 leaves, depth 5, subsample 0.7, colsample 0.6; quantile objective at 0.1 / 0.9 for the band and at α for the bid |
| α | newsvendor ratio `c_under / (c_under + c_over)` of the trailing 30-day RT−DA spread per zone, slots ending at or before the cutoff |
| Baselines | persistence D−2, D−7, mean(D−7, D−14); `isolf_pre` (file named D−1, known before the cutoff); `isolf_post` (file named D, posted ~07:10–08:00 ET on D−1, after the cutoff) |
| History | archive from **2024-09-01** (24 months; adopted in §2c). §2, §2b and §3 were run with the archive starting 2025-06-01, reproducible with `--history-start 2025-06-01` |
| Estimators | `models/`: LightGBM (default) and XGBoost per slot with time-decay weights, optional swap-noise augmentation; a Temporal Fusion Transformer (`scripts/run_tft.py`, global over zones, periodic refits) |

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

## 2c. Data length, estimators, augmentation and a sequence model (NYCA, N.Y.C., MHK VL, full 12 months)

With a 365-day window the first eight months of the backtest had trained on a partly
filled window (91 days on 2025-09-01), because the archive started on 2025-06-01. The
archive and the weather were backfilled to **2024-09-01** (24 months; 42 s and 30 s), and
the following runs were made on the three zones that matter most (statewide total,
largest zone, hardest zone), scored on identical zone-days against the `default` run of §3
with `scripts/compare_runs.py --base default --runs ... --bootstrap` (paired bootstrap over
days, 95% intervals).

| Run | Change | NYCA | N.Y.C. | MHK VL | Pooled | Δ vs default [95% CI] | Raw P10–P90 | Wall time |
|---|---|---|---|---|---|---|---|---|
| `default` (same days) | 12 months of data, LightGBM | 4.03 | 3.80 | 8.19 | 5.34 | — | 52% | — |
| `lgbm24` | 24 months of data | 3.85 | 3.55 | 7.85 | 5.08 | **−0.26 [−0.32, −0.20]** | 57% | 29 min, 12 workers |
| `xgb24` | + `--estimator xgb` | 3.81 | 3.51 | 7.86 | 5.06 | −0.28; vs `lgbm24` −0.02 [−0.07, +0.02] | 70% | 29 min |
| `swap24` | + `--augment swap` (p 0.1, one copy at weight 0.5, donors within the same tod) | 3.87 | 3.51 | 7.88 | 5.09 | −0.25; vs `lgbm24` +0.01 [−0.02, +0.03] | 58% | ~45 min |
| `tft24` | Temporal Fusion Transformer, one global model over the 12 zones, refit monthly | **2.91** | **2.94** | **7.36** | **4.40** | **−0.94 [−1.09, −0.78]**; vs `lgbm24` −0.68 [−0.82, −0.53] | 77% | 30 min on the GPU |
| `tft24_seed1` | same, seed 1 | 2.93 | 2.92 | 7.41 | 4.42 | −0.92 | 75% | 52 min (GPU shared) |
| `tft24_w7` | `--refit-days 7` (53 fits) | 2.88 | 2.87 | 7.33 | 4.36 | −0.98; vs `tft24` −0.04 [−0.13, +0.04] | 75% | 2.2 h (GPU shared) |
| NYISO `isolf_pre` (same days) | fair ISO benchmark | 2.81 | 2.44 | 7.80 | 4.35 | | | |
| NYISO `isolf_post` (same days) | post-close reference | 2.58 | 2.04 | 7.32 | 3.98 | | | |

Verdicts:

- **24 months of data: adopted** (`WARMUP_START = 2024-09-01`; `--history-start 2025-06-01`
  reproduces the old setting). The gain sits in September–April (−0.2 to −1.0 per month,
  December −0.96) and vanishes from May on, when both configurations already had a full
  window: exactly the mechanism expected. Cost: one forecast (four fits) takes a flat
  12.4 s instead of 3.7 → 12.4 s over the year, ≈ 1.55× the compute; a full 12-zone run
  should take ≈ 100–110 min instead of 72.
- **XGBoost: a wash** on accuracy at the same runtime; its quantile objective is better
  calibrated (raw coverage 70% vs 57%). Kept as `--estimator xgb`, not the default.
- **Swap noise: null** (+0.01 [−0.02, +0.03]). Kept as a flag for the record.
- **TFT: the strongest model by a wide margin**, robust to the seed (0.02 between seeds),
  level with the ISO's pre-close forecast pooled over these zones (4.40 vs 4.35), ahead of
  it on MHK VL (7.36 vs 7.80) and within 0.1 on NYCA. Its raw band already covers 77%,
  where the trees need the conformal rescaling to get there. One protocol difference,
  stated wherever the number is quoted: the weights are refit once per 30-day block on the
  365 days before the block's first cutoff, while the encoder of every forecast stops at
  that day's own cutoff; the leakage audit covered the encoder cut (asserted NaN beyond
  the cutoff), the training targets (all before the block's first cutoff), the per-zone
  scale and early-stopping split (training window only) and the weather (the same 2-day
  lead the trees use). 13 fits of 84–204 s on an RTX 5080 (torch 2.11 + cu128); inference
  is instant on a CPU.

- **Weekly refits: null** (−0.04 [−0.13, +0.04] against monthly, four times the fits).
  Monthly refits are the setting to carry forward.

Not yet done: the full 12-zone runs of `lgbm24` and `tft24` (the headline rules apply to
those, not to a three-zone table); the Phase 3 consequence (the VM would need torch for inference, with the
monthly refit on the laptop and the weights shipped as an artifact, or the trees stay the
served model).

## 2d. Data augmentation (NYCA, N.Y.C., MHK VL, full 12 months)

Five augmentation techniques suited to load series, each against the run it modifies on
identical zone-days (`compare_runs.py --bootstrap`). Trees: `--augment <kind>` of
`run_backtest.py` on the `lgbm24` configuration; TFT: `run_tft.py --aggregate` /
`--blockboot` on the `tft24` configuration. Implementation: `models/augment.py`,
`models/tft.py` (`aggregate_zone`, `block_bootstrap`).

| Run | Technique | Base | Pooled | Δ [95% CI] | Raw P10–P90 | Wall time | Verdict |
|---|---|---|---|---|---|---|---|
| `swap24` | swap noise (p 0.1, donors within the tod, one copy at weight 0.5) | `lgbm24` 5.08 | 5.09 | +0.01 [−0.02, +0.03] | 58% | ~45 min | null |
| `wnoise24` | weather-forecast-error injection: per-day temperature offset σ 1.55 °C + hourly jitter σ 1.58 °C (the spread of the 2-day-lead forecast updates), one copy at weight 0.5 | `lgbm24` 5.08 | 5.21 | **+0.13 [+0.09, +0.18]** | 58% | 51 min | worse, most in June–July (+0.26 / +0.40): noised temperatures make the trees under-use the forecast exactly when it matters most |
| `extreme24` | weight 3 on rows of hot / cold-decile days or days with a top-decile temperature swing (no new rows) | `lgbm24` 5.08 | 5.22 | **+0.13 [+0.09, +0.17]** | 58% | 26 min | worse in every month: the typical-day fit degrades more than the tail gains |
| `cmixup24` | C-Mixup within the tod, partner among the 5 nearest targets, λ ~ Beta(2, 2), one copy at weight 0.5 | `lgbm24` 5.08 | 5.33 | **+0.25 [+0.19, +0.31]** | 57% | 49 min | worse: interpolated rows blur the lag-to-load mapping the trees rely on |
| `tft24_agg` | 12 synthetic zones = sums of random 2–4 zone subsets (loads add exactly, weather load-weighted) as extra training series | `tft24` 4.40 | 4.48 | +0.08 [−0.01, +0.17] | 74% | 46 min (2× fit time) | null overall; N.Y.C. −0.21, MHK VL +0.32: the aggregates pull the shared weights toward large-zone behaviour |
| `tft24_boot` | one residual block-bootstrap replica per zone per fit (trend + weekday/tod profile + day-block resampled residual) | `tft24` 4.40 | 5.01 | **+0.61 [+0.48, +0.75]** | 84% | 48 min (2× fit time) | clearly worse: resampled residuals are pasted onto the wrong weather, so the network learns that weather explains less than it does |

Trade-offs, in one paragraph: none of the five buys accuracy, and the two that change the
weather–load coupling (`wnoise`, `blockboot`) are the most harmful, which is the useful
lesson. The load problem is not data-starved in the way augmentation fixes: every training
day already carries its true weather, and any synthetic row either repeats that information
(swap, aggregates: null) or corrupts it (noise, mixup, bootstrap: worse). The only
augmentation-like change that helped was more real data (§2c, 24 months, −0.26). Costs:
every copy-making kind doubles the training rows, so the tree runs take 1.7–2× longer
(26–51 min vs 29 min for the three zones) and the TFT fits 2×; code complexity is small
(one function per kind) and all kinds stay available behind flags for the record.
Frequency masking of the encoder was not run (lowest prior, and both TFT variants were
already null or negative).

## 3. Full runs (11 zones + NYCA, 24 months of history)

`scripts/run_tft.py --name tft_full --train-zones all` (monthly refits, GPU, 365 days × 12 zones)
and `scripts/run_backtest.py --name default --workers 14` (one fit per zone-day), then
`skill_baselines.py`, `imbalance_report.py`, `quantile_calibration.py` with each name. Committed
tables: `results/tft_full/` and `results/default/` (`summary.csv`, `baselines_summary.csv`,
`imbalance_pooled.csv`, `imbalance_by_zone.csv`, `calibration.csv`). Earlier full runs are kept
for the record: `results/v1_w120` (120-day window, temperature only, 12 months of history:
6.61 pooled) and `results/v2_12mo` (365-day window, hourly weather, 12 months of
history: 5.56).

### 3.1 Accuracy (hourly MAPE, %) with paired bootstraps against the ISO pre-close forecast

| Zone | TFT | Trees | persist D−2 | persist D−7 | mean(D−7, D−14) | isolf_pre | isolf_post | TFT − pre [95% CI] | Trees − pre [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| CAPITL | 5.43 | 6.08 | 11.41 | 12.23 | 11.08 | 5.33 | 5.09 | +0.10 [-0.21, +0.43] | +0.75 [+0.43, +1.10] |
| CENTRL | 5.11 | 5.78 | 11.10 | 11.83 | 10.65 | 6.32 | 6.15 | -1.21 [-1.57, -0.85] | -0.53 [-0.89, -0.17] |
| DUNWOD | 4.01 | 4.55 | 10.09 | 10.59 | 10.07 | 3.75 | 3.40 | +0.27 [-0.00, +0.55] | +0.80 [+0.54, +1.07] |
| GENESE | 4.66 | 5.33 | 11.02 | 11.07 | 10.17 | 4.69 | 4.31 | -0.03 [-0.32, +0.26] | +0.64 [+0.33, +0.96] |
| HUD VL | 5.24 | 6.07 | 12.33 | 13.88 | 12.71 | 6.20 | 5.30 | -0.95 [-1.31, -0.59] | -0.13 [-0.51, +0.23] |
| LONGIL | 4.44 | 5.56 | 11.02 | 11.60 | 10.97 | 4.05 | 3.60 | +0.39 [+0.13, +0.65] | +1.51 [+1.22, +1.80] |
| MHK VL | 7.46 | 7.85 | 14.32 | 15.29 | 13.86 | 7.80 | 7.32 | -0.33 [-0.79, +0.15] | +0.05 [-0.39, +0.52] |
| MILLWD | 6.29 | 6.61 | 12.60 | 14.43 | 13.57 | 5.97 | 5.41 | +0.32 [-0.07, +0.71] | +0.64 [+0.23, +1.06] |
| N.Y.C. | 2.93 | 3.55 | 9.29 | 8.68 | 8.52 | 2.44 | 2.04 | +0.49 [+0.30, +0.69] | +1.11 [+0.87, +1.35] |
| NORTH | 4.47 | 4.41 | 5.85 | 6.93 | 6.24 | 5.39 | 4.37 | -0.92 [-1.20, -0.65] | -0.98 [-1.29, -0.67] |
| NYCA | 2.94 | 3.85 | 8.76 | 9.11 | 8.50 | 2.81 | 2.58 | +0.13 [-0.09, +0.36] | +1.04 [+0.82, +1.27] |
| WEST | 4.58 | 4.32 | 8.14 | 7.92 | 7.40 | 3.42 | 2.93 | +1.16 [+0.86, +1.45] | +0.90 [+0.65, +1.17] |
| pooled | 4.80 | 5.33 | 10.49 | 11.13 | 10.31 | 4.85 | 4.38 | -0.05 [-0.15, +0.04] | +0.48 [+0.39, +0.58] |

- **TFT vs ISO pre-close, pooled: -0.05 [-0.15, +0.04]** — level. Significantly
  better in CENTRL, HUD VL, NORTH; significantly worse in LONGIL, N.Y.C., WEST; the rest within noise.
  Post-close the ISO is better by 0.42 pooled.
- Trees vs ISO pre-close, pooled: +0.48 [+0.39, +0.58]; better in
  CENTRL, NORTH. Trees vs best naive: −48% relative error.
- TFT vs trees on the same days: -0.53 pooled (§2c measured -0.94 on three zones with the
  12-month-history trees as the base; the trees gained 0.23 from the extra year).
- The TFT's α-bid at hourly resolution scores 4.88 (median 4.80); the
  trees' 5.39 (median 5.33).

### 3.2 Band coverage (15-min, nominal 80%), raw and after the trailing-30-day conformal rescaling

| Zone | TFT raw | TFT conformal | Trees raw | Trees conformal |
|---|---|---|---|---|
| CAPITL | 74% | 77% | 58% | 77% |
| CENTRL | 76% | 77% | 58% | 77% |
| DUNWOD | 74% | 77% | 58% | 78% |
| GENESE | 75% | 77% | 59% | 78% |
| HUD VL | 75% | 77% | 58% | 78% |
| LONGIL | 77% | 77% | 56% | 77% |
| MHK VL | 73% | 79% | 58% | 77% |
| MILLWD | 72% | 76% | 56% | 77% |
| N.Y.C. | 81% | 76% | 56% | 77% |
| NORTH | 78% | 77% | 60% | 78% |
| NYCA | 77% | 77% | 55% | 77% |
| WEST | 66% | 75% | 54% | 76% |
| pooled | 75% | 77% | 57% | 77% |

### 3.3 Settlement in dollars (11 priced zones, hourly bids settled per 15-min slot)

| Strategy | Signed $ vs perfect foresight | 95% bootstrap CI | Σ \|dev × spread\| | $/MWh (signed) |
|---|---|---|---|---|
| `persist_2d` | $86M | [$31M, $143M] | $335M | 0.568 |
| `persist_7d` | $26M | [$-63M, $118M] | $432M | 0.175 |
| `mean_7_14` | $21M | [$-94M, $128M] | $446M | 0.141 |
| `isolf_pre` | $34M | [$8M, $60M] | $158M | 0.226 |
| `isolf_post` | $26M | [$-2M, $53M] | $148M | 0.171 |
| `isolf_pre_alpha` | $55M | [$35M, $77M] | $143M | 0.363 |
| `model` (TFT) | $41M | [$26M, $57M] | $154M | 0.269 |
| `model_alpha_bid` (TFT) | $41M | [$23M, $61M] | $160M | 0.272 |
| `model_alpha_emp` (TFT) | $41M | [$22M, $60M] | $157M | 0.272 |
| `model` (trees) | $44M | [$13M, $74M] | $181M | 0.288 |
| `model_alpha_bid` (trees) | $46M | [$15M, $78M] | $182M | 0.305 |

- α (newsvendor ratio of the trailing 30-day spread) averages 0.41–0.48 by zone:
  real-time prices sit below day-ahead most of the time, so the cost-aware bid is
  slightly *short* of the median.
- The signed totals are noise: every interval overlaps every other. **The α-bid does not
  meet the headline rule** (docs/plan.md §4) and stays a secondary feature of the product.
- Σ |dev × spread| (dollars at risk) follows accuracy: ISO $158M ≈
  TFT $154M < trees $181M < naive ≥ $335M.

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
- XGBoost instead of LightGBM: −0.02 [−0.07, +0.02] on 24 months of data (§2c).
- Swap-noise augmentation of the training rows: +0.01 [−0.02, +0.03] (§2c).
- Weather-error injection +0.13, extreme-day weights +0.13, C-Mixup +0.25 (trees);
  aggregate zones +0.08 (null), residual block bootstrap +0.61 (TFT): §2d.

Open items, in the order they are likely to pay off:

1. A second weather station for the large zones, and zone-specific variables for the
   small upstate zones with industrial load (MHK VL, MILLWD).
2. The ISO forecast as a feature is excluded by decision (docs/plan.md Q8). A labelled
   experiment would almost certainly close most of the gap; it should stay a separate,
   clearly named variant if ever run.
3. Blending model and `isolf_pre` per zone with weights fitted on trailing days.
4. Band calibration per hour of day rather than one scale per day.
5. Band calibration of the TFT per hour of day (its raw band already covers 75%), and a
   per-zone blend of TFT and trees (they lose to the ISO in different zones).
6. Two winters of data: the archive now starts 2024-09-01, so the second winter arrives
   with the 2026–27 season; nothing to do until then.
