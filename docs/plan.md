# gridcast — Build Plan (NYISO day-ahead load forecasting)

> Status: **approved 2026-09-17** (answers to §10 recorded in CLAUDE.md); Phase 0 scaffold
> done the same day. Facts verified after approval live in `data/README.md`.
> Source of the architecture: github.com/chenyufang-data/gridcast-shanxi (Shanxi-market
> original). Every NYISO fact below marked ✅ was verified against the live archive on
> 2026-09-17; ⚠️ marks facts still to confirm.

## 0. Decisions already locked

| Decision | Value |
|---|---|
| Repo | `chenyufang-data/gridcast` (public, MIT). Local folder `Documents/gridcast-nyiso` until the folder swap (Q4) |
| Access model | public read without login; every write endpoint behind `X-Admin-Token`; no paid LLM key on the public site |
| Headline rules | MAPE is not the headline if NYISO's own forecast beats ours. The α-bid is the headline only if it is measured in $ (imbalance cost), beats the strongest baseline incl. `isolf + α`, and α is estimated strictly before each bid's cutoff; otherwise secondary |
| Domain | `https://gridcast.cyfang.org` (Caddy auto-HTTPS) |
| Demo video | < 2 minutes, English |

## 1. Market translation (Shanxi → NYISO)

| Concept | gridcast-shanxi | gridcast (NYISO) | Status |
|---|---|---|---|
| Bid deadline / data cutoff | actuals ~5 days late → train on ≤ D−6 | DAM bids due **05:00 ET on D−1** → train on actuals ≤ D−1 04:55 (last 5-min interval before close); D−1 is a partial day | ✅ [FERC guide](https://ferc.gov/introductory-guide-participation-new-york-iso-processes), [NYISO Manual 11](https://www.nyiso.com/documents/20142/2923301/dayahd_schd_mnl.pdf/0024bc71-4dd9-fa80-a816-f9f3e26ea53a) |
| Penalty | ±10% hourly band, breach hours | none; deviation settles at the real-time LBMP | ✅ market design |
| Cost of error | breach count | **imbalance cost = Σ (A − F) × (P_RT − P_DA)** vs perfect foresight; $/day, $/MWh, % of DA energy cost | design |
| Optimal bid | manual α fill, α=0.5 "pending prices" | α = c_under / (c_under + c_over), c_under = mean((P_RT−P_DA)+), c_over = mean((P_DA−P_RT)+) on a trailing window ending at the cutoff; bid = α-quantile model | design |
| Benchmarks | persistence D−6/D−7, mean(D−7,D−14) | persistence D−2, D−7, mean(D−7,D−14), **NYISO `isolf`**, `isolf + α` | design |
| Areas | province total + 126 meters | 11 zones (CAPITL, CENTRL, DUNWOD, GENESE, HUD VL, LONGIL, MHK VL, MILLWD, N.Y.C., NORTH, WEST) + NYCA total | ✅ zone list from `pal`/`isolf` |
| Actuals latency | days | 5-min load posts same day → score every forecast the next morning; scheduled fetch is the primary ingest | ✅ daily files exist for 2026-09-16 |
| Declaration | 24 hourly volumes → 96 slots | "DAM schedule": same mechanics, $-at-risk guardrails | design |
| Alerts | MAPE>10% or ≥8 breach hours | MAPE alert kept + imbalance-$ above trailing-30-day P90 | design |

## 2. Data plan — NYISO MIS public archive (`http://mis.nyiso.com/public/csv/<type>/`)

### 2.1 Datasets (all ✅ verified by download on 2026-09-17)

| Type | Daily file | Monthly zip | Cadence / rows per day | Exact columns |
|---|---|---|---|---|
| `pal` real-time actual load | `YYYYMMDDpal.csv` (recent days only) | `YYYYMM01pal_csv.zip` | 5-min; **rows vary: 3,169–3,301 observed** | `"Time Stamp","Time Zone","Name","PTID","Load"` — `MM/DD/YYYY HH:MM:SS`, tz `EDT`/`EST`, 11 zones, Load in MW |
| `damlbmp` day-ahead zonal price | `YYYYMMDDdamlbmp_zone.csv` | `YYYYMM01damlbmp_zone_csv.zip` | hourly; 361 rows (24 h × 15 names) | `Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)` — `MM/DD/YYYY HH:MM`, **no tz column** |
| `realtime` RT zonal price | `YYYYMMDDrealtime_zone.csv` | `YYYYMM01realtime_zone_csv.zip` | 5-min; 4,321 rows | same 6 columns, `HH:MM:SS` |
| `rtlbmp` RT hourly integrated price | `YYYYMMDDrtlbmp_zone.csv` | `YYYYMM01rtlbmp_zone_csv.zip` | hourly; 361 rows | same 6 columns |
| `isolf` ISO load forecast | `YYYYMMDDisolf.csv` | `YYYYMM01isolf_csv.zip` | hourly, wide; 145 rows | `"Time Stamp","Capitl","Centrl","Dunwod","Genese","Hud Vl","Longil","Mhk Vl","Millwd","N.Y.C.","North","West","NYISO"` |

Archive depth ✅: monthly zips exist from **2005** for `pal`, `damlbmp`, `isolf` (2001 absent).
Daily files ✅ for recent dates (2026-09-16 returned 200); older dates 404 → backfill from monthly zips, live fetch from daily files.

### 2.2 Gotchas found in the files (each becomes a unit test)

- **DST fall-back (2025-11-02)**: `pal` repeats `01:00:00–01:55:00` with `"EDT"` then `"EST"` (264 rows at 01:xx = 2 hours × 12 × 11) ✅. `damlbmp` has **25 hours with `01:00` twice and no tz column** (376 rows) ✅ → disambiguate by row order (first = EDT). Spring-forward days have 23 hours / 92 slots.
- Price files carry **15 names**: the 11 zones plus external proxies `H Q`, `NPX`, `O H`, `PJM` ✅ → filter to zones by PTID.
- `pal` row counts are not a clean 288 × 11 per day (3,169 on 09/01, 3,257 on 09/02, 3,301 on the DST day) ✅ → never assume the grid; dedupe on (timestamp, tz, zone), then time-weighted resample to 15-min mean MW; energy per slot = MW/4 MWh.
- `isolf` published on day X covers **X 00:00 → X+5 23:00 (6 days)** ✅ → the fair benchmark for day D is the file published on D−1, rows for D. ⚠️ Confirm the D−1 file's posting time is before 05:00 ET (NYISO posts the forecast each morning; if it lands after DAM close it is still the ISO's D−1 forecast, but the comparison must say so).
- Timestamps are local ET without offsets → store UTC internally, derive ET slots for display and for the DAM's hourly ET schedule.

### 2.3 Terms of use ⚠️ (decision needed, Q3)

NYISO's [Legal Notice](https://www.nyiso.com/legal-notice) grants no license, prohibits
stand-alone redistribution of images/video, and reserves trademarks; it says nothing
explicit about the CSV archive, which is served anonymously with no key. Plan:
**no NYISO bytes committed to git** — a fetch script + local cache (`data/cache/`,
gitignored); tests use deterministic synthetic fixtures; README attributes NYISO with a
link and uses no NYISO logo. The deployed demo shows derived aggregates and model outputs.

### 2.4 Pipeline

`app/nyiso.py`: `fetch_month(type, yyyymm)` / `fetch_day(type, date)` with retry + cache;
`normalize_pal`, `normalize_prices`, `normalize_isolf` → canonical frames
(`ts_utc, zone, load_mw` / `ts_utc, zone, p_da, p_rt` / `issued_date, ts_utc, zone, isolf_mw`).
Weather: Open-Meteo previous-runs API for each zone centroid, D−1-issued forecast only
(same leakage rule as the source). `data/README.md` documents all of the above.

## 3. Modeling plan

- **Cutoff** `BID_CUTOFF = D−1 05:00 ET` replaces `N_DELAY`; asserted in backtest, baseline,
  and α-estimation scripts (any feature timestamp ≥ cutoff fails the run).
- **Target**: 15-min mean MW for day D (96 slots; 92/100 on DST days), aggregated to the
  24 (23/25) hourly DAM schedule for settlement.
- **Features**: D−1 slots 00:00–04:45 (partial day) and its morning ramp stats; same-slot
  lags D−2, D−7, D−14, D−21; trailing 7-day same-slot mean; lag-group aggregates; shape
  (lag / day total); calendar (slot, weekday, US federal + NY holidays via `holidays`);
  weather (D−1-issued tmean/tmax/tmin for D and deviation from trailing 7-day actual).
  The ISO forecast as a *feature* is excluded by default (Q8).
- **Models**: LightGBM MAE + time-decay weights (port `DecayWeightedLGBMRegressor`);
  quantiles α=0.1/0.9 for the band; a third quantile model at the estimated α per zone for
  the bid. Sensitivity sweep, not tuning; experiment log with negative results.
- **Backtest**: rolling daily retrain, **2025-09-01 → 2026-08-31** (12 months, all four
  seasons; Q2), all 11 zones + NYCA, warm-up data from 2024-09-01 (a full 365-day window
  for every backtest day; adopted 2026-09-18 after the measurement in docs/experiments.md
  §2c, initially 2025-06-01).
  Compute estimate: 12 zones × 365 days × 3 fits × ~2 s ≈ 6 h single-core → parallelize
  by zone, cache per-day results.
- **Settlement metric**: per 15-min slot, deviation × (mean of the three 5-min RT LBMPs −
  DA LBMP); rolled to hourly and daily. Alternative: hourly deviation × `rtlbmp` (Q9).
- **Reports** (each a committed script): `run_backtest.py` (MAPE per zone/pooled),
  `skill_baselines.py` (persistence family + `isolf`), `imbalance_report.py` ($ for
  baselines, `isolf`, `isolf + α`, our median, our α-bid), `quantile_calibration.py`
  (+ conformal upper quantile if coverage < nominal).

## 4. Reporting rules (locked — applied to README, docs, UI, video)

1. If `isolf` beats our MAPE, MAPE appears in the results table but is not the lead.
2. The α-bid leads only if all hold: measured in $, beats the strongest of
   {persistence family, `isolf`, `isolf + α`}, α leakage-free and out-of-sample over the
   full range. Otherwise it is a secondary result and the lead is whatever is measured
   and true (calibration, the live scoring loop, the cost-aware schedule workflow).
3. Every headline number states its evaluation window and has a one-command script.

## 5. Product plan

- **Backend** (port of `app/`): zones as areas; `schedules` tables replace `declarations`;
  `forecast_scores` gain `imbalance_usd`, `da_cost_usd`; prices table; `X-Admin-Token`
  middleware on POST/PATCH/DELETE (`ADMIN_TOKEN` env); `logging` everywhere; type hints;
  pinned deps via a lock file.
- **Scheduler** (in-container, APScheduler or a loop thread): 04:30 ET → forecast + α-bid
  for tomorrow for all zones; 06:30 ET → fetch yesterday's `pal`/`realtime`/`damlbmp` and
  today's `isolf`, score yesterday's forecasts and schedules, refresh alerts. Nightly
  reset-to-seed = re-import from cache (optional, Q7).
- **Frontend** (EN-native): zone cards (7-day MAPE, 7-day imbalance $, alert); Forecast
  (median, P10–P90, α-bid line, `isolf` overlay); DAM Schedule (slider/per-hour edits,
  live total MWh, expected imbalance $, $-at-risk guardrails); Forecast vs Actual (error
  + DA/RT spread + $/hour); Prices; Load; chat panel with English persona, chips,
  GitHub-Models free tier. No i18n layer.
- **Rate limiting**: Caddy has no built-in limiter → simple in-app per-IP token bucket
  on write endpoints and LLM calls (Q7).

## 6. Deployment — recommendation: GCE VM, `docker-compose.prod.yml` as-is

| Option | Verdict | Why |
|---|---|---|
| **GCE e2-small** (2 vCPU shared, 2 GB) + 2 GB swap, 20 GB pd-balanced, static IP, Ubuntu 24.04 | **Recommended** | direct port of the EC2 runbook (t3.small/2 GB ran the Shanxi stack); ≈ $12.23/mo VM ([source](https://www.economize.cloud/resources/gcp/pricing/compute-engine/e2-small/)) + ~$2 disk + external-IP charge ⚠️ (~$3–4/mo, verify) ≈ **$18/mo** |
| GCE e2-medium (4 GB) | fallback if OOM during backfill/training | ≈ $24.46/mo ([source](https://www.economize.cloud/resources/gcp/pricing/compute-engine/e2-medium/)) |
| Cloud Run | rejected | SQLite needs a persistent disk (Cloud SQL/Filestore = re-architecture); in-process scheduler incompatible with scale-to-zero; min-instances cost more than the VM |
| Azure B2s VM | acceptable fallback | same compose stack; ≈ $30/mo |

Runbook (`deploy/RUNBOOK.md`, English): project + billing check → firewall (22 from
my IP, 80/443 public) → static external IP → DNS `A gridcast.cyfang.org → IP` (TTL 300)
→ `git clone`, `.env` on the VM only (`ADMIN_TOKEN`, `GITHUB_TOKEN`, `SITE_ADDRESS`)
→ `docker compose -f docker-compose.prod.yml up -d --build` → Caddy obtains the
certificate → `deploy/seed.py` backfills 15 months from monthly zips → health checks →
verify the scheduler ran → logs (`docker compose logs`) → teardown / snapshot / stop
schedule if idle. Region: us-east1 or us-east4 (Q6).

## 7. Demo video (< 2:00) — beat budget

| mm:ss | Beat | Screen |
|---|---|---|
| 0:00–0:15 | problem: bid by 05:00 ET for tomorrow; errors settle at real-time price | title card → home |
| 0:15–0:35 | N.Y.C. card → Forecast: median, P10–P90, α-bid line, `isolf` overlay | forecast view |
| 0:35–0:55 | DAM Schedule: move a slider, total MWh and expected imbalance $ update live | schedule view |
| 0:55–1:20 | Forecast vs Actual on the highest DA/RT-spread day: $/hour, where the bid saved money | compare view |
| 1:20–1:35 | typed chat: "show last week for Long Island" → navigates | chat |
| 1:35–1:55 | close on the headline chosen by §4, window on screen; repo URL | results card |

Full shot list + ≤ 260-word speech notes ship with the app (`docs/demo_video_runbook.md`).

## 8. Phases and effort

| Phase | Scope | Effort |
|---|---|---|
| 0 | scaffold (layout, LICENSE, lock file, CI, .gitignore, synthetic test fixtures, logging/typing conventions) | S |
| 1 | NYISO fetch + normalize + cache + DST tests + `data/README.md` | M |
| 2 | model port with cutoff timestamp, features, backtest CLI, baselines, `isolf`, α estimation, imbalance report, calibration/conformal, experiment log | L |
| 3 | backend port: zones, schedules, $ scoring, admin token, scheduler, tests (adapters, e2e with real training) | M |
| 4 | frontend EN: six views, chat persona, mocked-router tests | M |
| 5 | GCE deploy, DNS, HTTPS, seed, runbook | S |
| 6 | demo video runbook + speech notes; README written under §4 | S |

## 9. Risks

- Data terms are silent on CSV reuse → mitigated by fetch-not-redistribute; residual risk accepted or cleared with NYISO (Q3).
- `isolf` will likely beat a 56-day LightGBM on MAPE → §4 already handles it; the product story does not depend on it.
- α may be ≈ 0.5 in zones with symmetric spreads → α-bid becomes secondary by rule.
- DST bugs → dedicated tests on 2025-11-02 and 2026-03-08 files.
- Row-count irregularities in `pal` → dedupe + resample, never positional indexing.
- Let's Encrypt rate limits during repeated rebuilds → keep `caddy_data` volume.

## 10. Open questions (answer in one batch; defaults in brackets)

1. Zones: backtest all 11 + NYCA, demo default N.Y.C.? [yes]
2. Backtest range 2025-09-01 → 2026-08-31 (12 months)? [yes; 24 months if you want two winters]
3. Data terms: proceed under fetch-not-redistribute + attribution, or email NYISO first? [proceed]
4. Local folder swap (`gridcast` → `gridcast-shanxi`, `gridcast-nyiso` → `gridcast`): now or after the first push? [after]
5. GitHub: you create an empty public `chenyufang-data/gridcast` (no README), or install `gh` so I can? [you create]
6. GCE region and size: us-east4 (N. Virginia) e2-small with swap? [yes]
7. Rate limiting via in-app per-IP bucket, and nightly reset-to-seed on/off? [in-app; reset off]
8. ISO forecast as a model feature: excluded, or allowed as a labeled experiment? [excluded]
9. Settlement granularity: 15-min slot × mean RT LBMP, or hourly × `rtlbmp`? [15-min]
10. Video: you record voice-over from the speech notes, or TTS? [you]
