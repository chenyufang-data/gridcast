# CLAUDE.md — session handoff for `gridcast` (NYISO successor)

Read this first, then `docs/plan.md` (approved) and `CONTRIBUTING.md` (conventions).
Written 2026-09-17; nothing below is speculative — every fact was verified or decided
by the user.

## What this repo is

The US-native successor of **gridcast-shanxi** (github.com/chenyufang-data/gridcast-shanxi,
locally `Documents/gridcast-shanxi`). Same architecture, invariants, evaluation culture,
test layout and deployment stack, rebuilt on NYISO data and NYISO market rules
(D−1 05:00 ET bid cutoff, errors settled at real-time price, no ±10% band).
Full design: `docs/plan.md`. Source-repo references worth reading before porting:
`gridcast-shanxi/docs/specification.md` (module map) and
`gridcast-shanxi/docs/project_audit_report.md` (audit findings fixed from day one here:
lock file, logging, type hints, LICENSE, data provenance, year-proof holidays).

## Status and next action

- **Phase 0 (scaffold) and Phase 1 (NYISO data layer) are done.** Phase 0: layout,
  MIT LICENSE, universal lock files, CI (ruff + mypy + pytest + both Docker builds),
  logging/typing conventions, `tests/synthetic.py` (NYISO-shaped synthetic archive
  with truth frames), `data/README.md`. Phase 1: `app/nyiso.py` — `ArchiveClient`
  (daily files / final month zips / refreshed partial current-month zip, cache under
  `NYISO_CACHE_DIR`, injectable `fetch` and `today`), normalizers to canonical UTC
  frames, `resample_slots` (time-weighted 15-min grid with `coverage`), `add_nyca`;
  `tests/test_nyiso.py` (20 offline cases against the synthetic truth frames + one
  `-m live` round trip, which passed against the real archive on 2026-09-17).
- **Phase 2 (model + backtest) is done.** `model.py` (cutoff guard, wall-clock lag
  alignment, repair, features, decay-weighted LightGBM, `forecast_day`),
  `app/weather.py` (Open-Meteo previous-runs, daily + hourly, leakage-free `d2` lead),
  `src/dataset.py`, `src/settlement.py` (slot prices, newsvendor α, imbalance $),
  `src/baselines.py` (persistence family, `isolf_pre` / `isolf_post`),
  `src/backtest.py` (per-(zone, month) process pool, per-day CSV cache), `src/metrics.py`,
  scripts `backfill`, `fetch_weather`, `run_backtest`, `skill_baselines`,
  `imbalance_report`, `quantile_calibration`. 75 offline tests. Data lives in
  `data/processed/*.pkl`, `data/weather*.csv`, per-day results under `results/<name>/`
  (all gitignored except the small summary tables of `results/default/`).
  **Findings (numbers in `docs/experiments.md` and the README results section):**
  NYISO's pre-close forecast still beats the model on hourly MAPE (pooled 4.85 vs 5.56;
  NYCA 2.81 vs 4.03; the model wins CENTRL and NORTH), so MAPE is not the headline;
  in signed dollars no bidding strategy is distinguishable over one year (bootstrap CIs
  ± tens of $M), so the α-bid is secondary by rule; raw quantile bands cover 52%,
  conformal rescaling 77%. Two sweeps (§2, §2b of the experiment log): weather ~1.5
  points, hourly temperature 0.2, extra hourly weather (apparent temp, dew point,
  humidity, cloud, wind, radiation) another ~1.5 on the hard half-year, a 365-day window
  ~0.5; ratio target, day-type features, bigger trees and the post-cutoff weather lead
  are negative/null results. **Current defaults: window 365, half-life 90, MW target,
  daily + hourly + extra weather at the leakage-free `d2` lead, 600 trees @ lr 0.02**
  (`BacktestConfig`); the first full run (120 d, temperature only) is archived as
  `results/v1_w120`. `scripts/compare_runs.py` scores experiments against a base run on
  identical zone-days; the per-day cache is keyed by run name only, so every new
  configuration needs a new `--name`. A full 12-zone run takes ~72 min on 14 workers.
- **Model comparison pass (2026-09-18, after the improvement pass).** `model.py` became the
  `models/` package: `features.py` (unchanged feature code, `FEATURE_VERSION`), `tabular.py`
  (`DecayWeighted` around any sklearn-style regressor, `get_model(estimator, quantile=...)`
  for `lgbm` / `xgb`, `forecast_day(..., estimator, augment_kind, augment_params)`),
  `augment.py` (swap noise), `tft.py` (Temporal Fusion Transformer in plain torch, global
  over zones, `scripts/run_tft.py` with periodic refits, same `results/<name>/` layout).
  `run_backtest.py` gained `--estimator`, `--augment swap`, `--history-start`;
  `compare_runs.py --bootstrap` gives paired CIs; `summarize_results` moved to
  `src/backtest.py`. Dependencies: xgboost in the dev lock; torch is an unlocked research
  extra (`requirements-research.txt`, installed from the cu128 index; the laptop has an
  RTX 5080 and torch 2.11+cu128 is in the venv; tests skip without it). The archive and
  weather were backfilled to 2024-09-01 (`WARMUP_START`). **Findings on NYCA + N.Y.C. +
  MHK VL, 12 months, identical zone-days (docs/experiments.md §2c):** 24 months of data
  −0.26 pooled [−0.32, −0.20] (adopted; ≈ 1.55× compute, full run ≈ 100–110 min); XGBoost
  a wash; swap noise null; **TFT −0.94 [−1.09, −0.78], pooled 4.40 vs isolf_pre 4.35,
  ahead of the ISO on MHK VL, 77% raw band, seed-robust**, monthly refits of 84–204 s on
  the GPU (weekly refits: null, −0.04 [−0.13, +0.04]). **Augmentation (§2d): all five
  null or worse** — swap +0.01, weather-error injection +0.13, extreme-day weights +0.13,
  C-Mixup +0.25 (trees vs `lgbm24`); aggregate zones +0.08, block bootstrap +0.61 (TFT vs
  `tft24`); the kinds stay behind flags for the record, nothing adopted. The 12-zone tables in the README are still the 12-month-data LightGBM run.
- **Full 12-zone runs on 24 months (2026-09-19; README results section, experiments §3).**
  `results/tft_full` (TFT, monthly refits, 30 min GPU) and `results/default` (trees, 109 min
  on 14 workers; the 12-month-history trees moved to `results/v2_12mo`). **TFT pooled 4.80 vs
  isolf_pre 4.85: level (paired daily −0.05 [−0.15, +0.04]), significantly better in CENTRL,
  HUD VL, NORTH, worse in LONGIL, N.Y.C., WEST; isolf_post better by 0.42.** Trees 5.33
  (12-month history: 5.56), better than the ISO in CENTRL and NORTH. Headline wording,
  by the rule: the TFT *matches* NYISO's pre-close forecast, never *beats*; α-bid secondary
  (signed $ noise; the α-bid does not beat `mean_7_14`). Bands: TFT raw 75% → conformal 77%,
  trees 57% → 77%. Served path built: `models/tft_data.py` (torch-free), `models/tft_onnx.py`
  (`OnnxTFT.load` verifies format, `FEATURE_VERSION`, SHA-256), `scripts/export_tft.py`
  (fit latest window → `data/models/tft/{tft.onnx,tft.json}`, torch-vs-ORT check ≤ 1e-4);
  `onnxruntime` is in requirements.txt.
  Gotcha: a 12-worker run once died with `BrokenProcessPool` near its end while the
  machine was loaded; re-running the same `--name` resumed from the per-day cache in
  two minutes.
- **Phase 3 (backend port) is done (2026-09-20).** `app/db.py` (SQLite on the data volume,
  UTC text stamps; tables `load_slots`, `rt_slots`, `da_hourly`, `rt_hourly`, `isolf`,
  `ingest_log`, `forecasts` + `forecast_values` (raw and conformal band, α-bid), `schedules` +
  `schedule_values`, `forecast_scores` / `schedule_scores` (hourly MAPE, band coverage,
  `imbalance_usd`, `da_cost_usd`, the ISO's pre-close MAPE and $ on the same day), `alerts`,
  `jobs`), `app/serving.py` (`ModelRegistry`: reloads the ONNX bundle when the files change,
  `TFT_MAX_AGE_DAYS` = 60 staleness, status for `/health` and `/models`), `app/service.py`
  (per-day ingest with `ingest_log` catch-up and retry of recently missing days; today's
  partial files are stored but never logged as complete; α as of the cutoff from
  `src.settlement`; split-conformal band scales from the trailing 30 scored days of the same
  model family; immutable forecasts versioned by a hash of data window + model identity +
  settings, `model` = `tft-onnx:<sha12>:<fit cutoff>` or `lgbm:nyiso-fv1`; `auto` = TFT when
  valid else trees with `fallback_reason`; scoring on complete UTC hours and per-slot
  settlement of the hourly-mean bid; alerts (MAPE > 10 %, imbalance $ above the trailing P90
  after 10 scored days); zone cards; compare / load / prices / isolf / alpha views; DAM
  schedules = one flat MW bid per hour pinned to the latest forecast, `usd_at_risk` guardrail;
  retention pruning on whole months, never below 14, never touching forecasts/scores),
  `app/scheduler.py` (daemon thread, ET wall clock, a job runs when its last scheduled
  occurrence is newer than its last run, so missed runs catch up at startup: 04:30
  `forecast_all`, 06:30 `ingest_score`, 08:30 `isolf_refresh`, monthly `retention`),
  `app/main.py` (`X-Admin-Token` + per-IP token bucket on every write; `/health`, `/models`,
  `/jobs`, `/zones`, `/zones/{zone}/{forecasts,scores,compare,load,prices,isolf,alpha,
  schedules}`, `/alerts`, `/admin/{ingest,score,weather/refresh,models/reload,jobs/{name}/run}`;
  zone names resolve loosely: `nyc`, `hud_vl`), `app/weather.update` (incremental splice,
  atomic write) and `weather.trim`, `deploy/seed.py` (runs inside the backend container:
  ingest → weather → trailing forecasts with the served model → scores). The CSV-upload
  adapter path of the source repo was dropped: the archive fetch is the only ingest.
  Tests: `tests/test_service.py` (e2e on a synthetic archive with monthly zips + daily files:
  ingest, catch-up, token guard, trees, NYCA, scoring = compare view, schedules, conformal
  after 8 days, alerts, a tiny served TFT bundle with zone-missing / stale / corrupted
  fallbacks, jobs, retention; ~30 s) and `tests/test_scheduler.py`; conftest isolates
  `TFT_BUNDLE_DIR`, `WEATHER_PATH*`, `SCHEDULER_ENABLED=0`, `WEATHER_REFRESH=0`.
  **Measured on real data (laptop, scratch store):** 14 months ingest from the cache 33 s →
  179 MB SQLite; TFT `forecast_all` 12 zones 2.6 s; one tree fit (365-day window, weather)
  10.6–11 s per zone on `MODEL_THREADS=2`, so budget ~1 min per zone on e2-small and ~10 min
  for a full tree pass. A real bundle exists locally: `data/models/tft/` (gitignored),
  `tft-onnx:e979ee4c9f6d:2026-09-19T09:00:00+00:00`, verified 6e-7 vs torch.
  Not done locally: the Docker builds (Docker Desktop was not running); CI builds both images.
- **Phase 4 (frontend) is done (2026-09-20).** `frontend/api.py` (typed HTTP client; the
  admin token stays in the frontend process; `set_client` for tests), `frontend/router.py`
  (the keyword guide: zone aliases on word boundaries, view cues in priority order
  compare > schedule > prices > forecast > load, dates incl. relative phrases and bare
  months, `sanitize` for model output; the zone list is duplicated here because the
  frontend image ships alone), `frontend/llm.py` (providers `VertexGemini` via the
  google-genai SDK with ADC, `GitHubModels`; `LLM_PROVIDER=auto|vertex|github|none`;
  `ChatLimiter` per visitor (`X-Forwarded-For` / IP, else session) and global per UTC day;
  every failure, limit or garbage reply falls back to the keyword guide with a visible
  note), `frontend/charts.py` (Plotly on the data-viz reference palette: forecast blue,
  actual orange, NYISO aqua, bid yellow dotted, dollars blue/red diverging around zero,
  hairline solid grid, unified crosshair tooltip, one axis per chart), `frontend/app.py`
  (sidebar = zone + view + served-model badge + chat budget; left pane = chips + chat;
  right pane = Overview cards / Forecast / DAM Schedule (data editor, α-bid or median
  start, scale %, $ at risk, hours outside the band, save, CSV, history) / Forecast vs
  Actual (backfill button) / Prices / Load; every chart has a table expander; widget
  state pattern: state keys are the truth, each widget is seeded right before creation
  and writes back via `on_change` — never set a widget key after it exists). Tests:
  `tests/test_router.py`, `tests/test_llm.py` (fake provider, limits, fallbacks),
  `tests/test_frontend.py` (`streamlit.testing.AppTest` on a `FakeApi`: every view, the
  buttons, chips, chat input, backend-down message). Smoke-tested against the real-data
  backend on :8011 (every view, the served-model and retrain buttons, a saved schedule).
  `google-genai` was added to the frontend requirements; all three locks regenerated.
  Streamlit 1.64: `use_container_width` is deprecated, use `width="stretch"`.
- **Next action: Phase 5 — GCE deployment.** `deploy/RUNBOOK.md` (EN): project + billing,
  firewall (22 from the user's IP, 80/443 public), static IP, DNS A record for
  `gridcast.cyfang.org`, **VM service account with `roles/aiplatform.user` and the Vertex
  AI API enabled** (the chat model needs no key), `git clone`, `.env` on the VM only
  (`ADMIN_TOKEN`, `VERTEX_PROJECT`, `SITE_ADDRESS`, optional `GITHUB_TOKEN`),
  `docker compose -f docker-compose.prod.yml up -d --build`, copy `data/models/tft/` to
  the volume (`docker compose cp`), `docker compose exec backend python deploy/seed.py`,
  health checks, verify the 04:30 / 06:30 / 08:30 ET jobs ran, snapshot / stop schedule.
  The frontend container must forward the visitor IP: Caddy sets `X-Forwarded-For`, and
  Streamlit exposes it through `st.context.headers`. Then Phase 6 (demo video + README
  under §4).
- GitHub: `origin` = https://github.com/chenyufang-data/gridcast (private), `main`
  pushed and tracking. CI status must be checked on the web (no `gh`).
- Folder swap is done: this repo is `Documents/gridcast`, the source is
  `Documents/gridcast-shanxi`.

## Decisions (locked) and answered questions

| Topic | Decision |
|---|---|
| Zones | backtest all 11 zones + NYCA; demo default N.Y.C. |
| Backtest range | 2025-09-01 → 2026-08-31, warm-up data from **2024-09-01** (24 months; changed from 2025-06-01 on 2026-09-18 after the §2c measurement, `--history-start` reproduces the old setting) |
| Data terms | proceed: fetch-not-redistribute (no NYISO bytes in git), **cite the original source with a link everywhere data appears** (README, `data/README.md`, UI footer), no NYISO logo |
| Access model | public read, `X-Admin-Token` on every write endpoint, in-app per-IP rate limiting, nightly reset-to-seed **off** |
| Headline rules | MAPE is not the headline if NYISO's `isolf` beats ours. The α-bid leads only if measured in $, beats the strongest baseline incl. `isolf + α`, and α is estimated strictly before each cutoff; otherwise secondary. Every headline has a window and a one-command script |
| ISO forecast as feature | excluded |
| Settlement granularity | 15-min slot × mean of the three 5-min RT LBMPs − DA LBMP |
| Cloud | GCE **e2-small + 2 GB swap**, 20 GB pd-balanced, static IPv4, **us-east1** recommended (same price tier as us-central1; us-east4 ~10–15% more). ≈ $19/mo. The user has a **$300 GCP new-user credit expiring late November 2026** → deploy early (target: site live well before mid-November); e2-medium is affordable under the credit if the VM ever OOMs |
| Domain | `https://gridcast.cyfang.org`; user adds the DNS A record; Caddy auto-HTTPS |
| Chat model (decided 2026-09-20) | **Gemini via Vertex AI**, authenticated by the VM's service account (no key file anywhere; `roles/aiplatform.user`), default `gemini-2.5-flash-lite` in `us-central1`; per-visitor (20) and global (300) daily request limits; the keyword guide is the fallback for no model / over limit / failures; GitHub Models stays an optional second provider |
| Demo video | < 2:00, English, **the user's own voice + burned-in captions**; deliver speech notes (~150 wpm, ≤ 260 words) and an SRT file |
| Heavy compute | the 12-zone × 12-month backtest runs on the user's laptop; the VM only does daily incremental fetch + one forecast per zone |
| Served models (decided 2026-09-18) | **TFT exported to ONNX on the laptop and uploaded monthly** (export pins eval mode and is verified against torch on fresh inputs: 1.3e-6 agreement, dynamic batch, 79 ms for 12 zones on CPU, 2.2 MB file); the VM runs it with `onnxruntime` only, never torch. **LightGBM trees stay the fallback and the model the demo retrains live.** VM stays **e2-small**. A monthly TFT refit on the VM is ruled out by measurement (50 min on two fast laptop threads, 4.2 GB resident) |

## Verified NYISO facts (2026-09-17; full table and gotchas in `data/README.md`)

- Archive `http://mis.nyiso.com/public/csv/<dir>/`: dirs `pal`, `damlbmp`, `realtime`,
  `rtlbmp`, `isolf`; daily `YYYYMMDD<type>.csv` (recent dates only), monthly
  `YYYYMM01<type>_csv.zip` back to 2005. Helpers: `src/config.py`
  (`daily_filename`, `monthly_zip_name`, `archive_url`, zone PTIDs verified).
- `pal`: 5-min, tz column `EDT`/`EST`, 11 zones, stamps 00:00:00–23:55:00 (interval
  start). `realtime_zone`: stamps are **interval ends** 00:05:00 → next day 00:00:00.
  Both carry the **same off-schedule RTD stamps** (e.g. `04:04:18`; 296 stamps on
  2026-09-15, not 288) → dedupe on (ts, tz, zone) then resample; never index by
  position. Quoting differs per type (`damlbmp` unquoted, the rest quoted strings).
- Price files: no tz column, 15 names (11 zones + `H Q`, `NPX`, `O H`, `PJM`), DST
  fall-back day lists `01:00` twice in every hourly file (first = EDT; `isolf` repeats
  an identical row). `isolf` file named for day X covers X…X+5 hourly, integer MW,
  `NYISO` = sum of zones, and is **posted on X−1 between ~07:10 and ~08:00 ET, after
  the 05:00 ET DAM close** (absent at 05:03 ET). Leakage-free ISO benchmark for D =
  the file named D−1 (day-D rows); the D-named file is the post-close reference.
  `damlbmp` for D also appears on D−1 after the close. DAM bids due 05:00 ET on D−1
  (FERC intro guide; NYISO Manual 11).
- Daily files live ~11 days; monthly zips are flat daily files, final on the 1st of the
  next month, and the current month's zip is rebuilt ~05:00 ET daily including today's
  partial day. `ArchiveClient` encodes all of this.
- Legal notice grants no license and is silent on CSVs → the fetch-not-redistribute plan.

## Port-size estimate (what is new vs reused from gridcast-shanxi, ~5,100 LOC)

| Module | Source LOC | Plan | Est. new/rewritten |
|---|---|---|---|
| `models/` (was `model.py`) | 481 | done: `features.py` (cutoff-timestamp lags, DST grid, `holidays`), `tabular.py` (decay wrapper, LightGBM + XGBoost, `forecast_day`), `augment.py`, `tft.py` | ~900 |
| `src/` backtest + config | 182 | rewrite: cutoff, per-zone, parallel, CLI (`scripts/run_backtest.py`); `src/config.py` done | ~250 |
| `app/nyiso.py` (new) | — | fetch/cache/normalize incl. DST + dedupe | ~300 |
| `app/adapters.py` | 290 | keep as secondary CSV path | ~30 |
| `app/db.py` | 135 | done: zones, market tables, `schedules`, $ score columns, alerts, jobs | 268 |
| `app/service.py` | 1,058 | done (rewritten): ingest/catch-up, both models, α, conformal band, versioning, $ scoring, alerts, views, schedules, retention; + `app/serving.py` (142), `app/scheduler.py` (176) | 1,661 |
| `app/main.py` | 320 | done: admin-token + rate-limit middleware, zones/forecasts/schedules/scores/compare/load/prices/isolf/alpha/alerts/jobs/admin endpoints | 370 |
| `app/weather.py` | 82 | zone centroids | ~30 |
| `frontend/router.py` + `llm.py` | 416 | done: keyword guide (aliases, cues, dates), Gemini/GitHub providers, daily limits | ~570 |
| `frontend/app.py` + `charts.py` + `api.py` | 1,161 | done: six views, chat pane, served-model badge, DAM editor, palette charts | ~1,100 |
| `scripts/` | 165 | + `isolf` baseline, `imbalance_report.py`, conformal upper quantile | ~350 |
| `tests/` | 753 | keep adapter/router suites; rewrite service e2e for zones/prices; new DST, α, imbalance tests; synthetic generator done (`tests/synthetic.py`) | ~600 |
| deploy | ~100 + md | compose/Caddyfile ported; `seed.py` done (97); EN GCP runbook pending (Phase 5) | ~250 |
| **Total** | ~5,100 | | **≈ 3,000 new/rewritten (~45–50%), ≈ 3,000 reused with light edits** |

## Working conventions carried over (details in `CONTRIBUTING.md`)

- Commit style `type(scope): summary`; end commit messages with the attribution line the
  harness provides. `ruff check`, `ruff format --check`, `mypy`, `pytest` must pass
  before commit; CI runs them plus both Docker builds on every push.
- No secrets in git; `.env` only; `.env.sample` documents every variable (English).
- Dev environment: per-project `.venv` installed from `requirements-dev.lock`
  (`.\.venv\Scripts\python.exe -m pytest`). Local Python 3.13, CI/Docker 3.11; the
  locks are universal (`uv pip compile --universal`, commands in CONTRIBUTING.md).
  Regenerate all three locks whenever a `requirements*.txt` changes.
- Logging via `app.log.configure_logging()` + `logging.getLogger(__name__)`; type hints
  on all public functions; UTC internally, ET only at the edges; never assume the grid.
- Windows tooling: `gh` is not installed (repo creation/rename = user, web UI). The Bash
  tool truncates commands over ~8 KB — write long files with the Write tool, not heredocs.
  PowerShell 5.1: no `&&`; quote-heavy commit messages break — avoid `"` inside `-m`.
  Docker Desktop must be running for local image builds (`docker info` to check).
- Reporting to the user: lead with the outcome; cite files and line numbers; label anything
  unverified.
