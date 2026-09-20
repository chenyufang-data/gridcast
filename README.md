# ⚡ gridcast — NYISO day-ahead zonal load forecasting

[![CI](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml/badge.svg)](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml)

Day-ahead load forecasts for the eleven NYISO zones and the statewide total, built
for the way the market actually settles: bids are due **05:00 ET on D−1**, and every
MWh of forecast error is settled at the **real-time price**. The product turns a
15-minute forecast (a Temporal Fusion Transformer served as ONNX, LightGBM trees as the
fallback) with P10–P90 bands into a cost-aware DAM bid, scores both in dollars against
NYISO's own forecast, and fronts it all with a chat-driven web UI.
PyTorch → ONNX Runtime · LightGBM · FastAPI · SQLite · Streamlit · Gemini on Vertex AI · Docker · Caddy.

> **Status: model, backtest, service and UI are done; deployment to GCE (Phase 5) is next.**
> The build plan and the rules for what may become a headline number are in
> [docs/plan.md](docs/plan.md); every number below has a one-command script in
> [scripts/](scripts/README.md) and an entry in [docs/experiments.md](docs/experiments.md).
> This is the US-native successor of
> [gridcast-shanxi](https://github.com/chenyufang-data/gridcast-shanxi).

## Results (12 months, 2025-09-01 → 2026-08-31, all 11 zones + NYCA)

Two models under one protocol, scored on the same 4,380 zone-days at hourly
resolution, the DAM product and the only fair comparison with NYISO's hourly forecast:

- **TFT** (`models/tft.py`, the served model): one Temporal Fusion Transformer over all 12
  zones, refit monthly on the 365 days before the block's first cutoff; every forecast
  uses an encoder that stops at that day's **D−1 05:00 ET** bid cutoff.
- **Trees** (`models/tabular.py`, the fallback and the model the demo retrains live): one
  fresh LightGBM fit per zone and day on the 365 days before the cutoff. The code raises
  if anything later than the cutoff leaks into either model.

Two ISO benchmarks are shown because of a fact verified in this repo: the NYISO forecast
file named for day D is posted on D−1 between 07:10 and 08:00 ET, *after* the bid close.
`isolf_pre` (file named D−1) is what a bidder had at the cutoff; `isolf_post` (file
named D) is the stronger post-close reference. Data: 24 months of history (2024-09-01 on).

Hourly MAPE (%), lower is better; `scripts/skill_baselines.py --name tft_full` and `--name default`; last column from `scripts/compare_runs.py`-style paired bootstrap over days:

| Zone | **TFT** (served) | Trees (fallback) | Best naive | NYISO pre-close (fair) | NYISO post-close | TFT − pre-close, daily [95% CI] |
|---|---|---|---|---|---|---|
| CAPITL | 5.43 | 6.08 | 11.08 | 5.33 | 5.09 | +0.10 [-0.21, +0.43] level |
| CENTRL | 5.11 | 5.78 | 10.65 | 6.32 | 6.15 | -1.21 [-1.57, -0.85] better |
| DUNWOD | 4.01 | 4.55 | 10.07 | 3.75 | 3.40 | +0.27 [-0.00, +0.55] level |
| GENESE | 4.66 | 5.33 | 10.17 | 4.69 | 4.31 | -0.03 [-0.32, +0.26] level |
| HUD VL | 5.24 | 6.07 | 12.33 | 6.20 | 5.30 | -0.95 [-1.31, -0.59] better |
| LONGIL | 4.44 | 5.56 | 10.97 | 4.05 | 3.60 | +0.39 [+0.13, +0.65] worse |
| MHK VL | 7.46 | 7.85 | 13.86 | 7.80 | 7.32 | -0.33 [-0.79, +0.15] level |
| MILLWD | 6.29 | 6.61 | 12.60 | 5.97 | 5.41 | +0.32 [-0.07, +0.71] level |
| N.Y.C. | 2.93 | 3.55 | 8.52 | 2.44 | 2.04 | +0.49 [+0.30, +0.69] worse |
| NORTH | 4.47 | 4.41 | 5.85 | 5.39 | 4.37 | -0.92 [-1.20, -0.65] better |
| NYCA | 2.94 | 3.85 | 8.50 | 2.81 | 2.58 | +0.13 [-0.09, +0.36] level |
| WEST | 4.58 | 4.32 | 7.40 | 3.42 | 2.93 | +1.16 [+0.86, +1.45] worse |
| **pooled (11 zones + NYCA)** | **4.80** | **5.33** | **10.31** | **4.85** | **4.38** | -0.05 [-0.15, +0.04] level |

Dollars at risk over the year, Σ |deviation × (RT − DA)| across the 11 priced zones (`scripts/imbalance_report.py`): NYISO pre-close $158M, TFT $154M, trees $181M, best naive $335M. Signed totals against perfect foresight have 95% bootstrap intervals about ±$26M wide, so they do not rank strategies.

What the table says, under the reporting rules in the plan:

- **The TFT matches NYISO's own day-ahead forecast.** Pooled over 12 zones and 12 months it
  is level with the ISO's pre-close forecast (4.80 vs 4.85; paired daily
  difference -0.05 with a 95% interval of [-0.15, +0.04]),
  significantly ahead in CENTRL, HUD VL, NORTH and behind in LONGIL, N.Y.C., WEST;
  post-close the ISO is better by 0.42. The rule says MAPE is not the headline when the ISO beats us; here
  neither beats the other, so the claim is "matches", stated with its window, never "beats".
- **The trees are the fallback**, at 5.33 pooled (+0.48 against the ISO's pre-close,
  ahead in CENTRL, NORTH), 48% better than the best naive
  baseline, and the model the demo can retrain in seconds.
- **The cost-aware α-bid is a secondary result.** In signed dollars against perfect
  foresight no strategy is distinguishable over one year: the 95% bootstrap intervals
  of the yearly totals are several times wider than the differences between strategies,
  because real-time price spikes dominate the sum. The dollars *at risk*
  (Σ |deviation × spread|) do track accuracy, which is the number a desk can act on.
- **Bands.** The TFT's raw P10–P90 band covers 75% of 15-min actuals
  (77% after the trailing conformal rescaling, target 80%); the trees'
  quantile objective needs the rescaling to get from 57% to 77%.
- **How it got here** (pooled hourly MAPE, same window): first trees 6.61 (120-day
  window, temperature only) → 5.56 (365-day window, hourly weather) → 5.33
  (24 months of history) → TFT 4.80. Every step is in `docs/experiments.md`.

## Data

**Data: NYISO public MIS archive (<http://mis.nyiso.com/public/csv/>), fetched at
runtime and not redistributed.** No NYISO files are committed or shipped; the demo
shows derived aggregates and model outputs. NYISO does not sponsor or endorse this
project. Details, verified file layouts, and the DST gotchas: [data/README.md](data/README.md).

## Layout

```
models/         modeling package shared by src/ and app/: cutoff guard, features, LightGBM / XGBoost, swap-noise augmentation, TFT
src/            constants, dataset builder, backtest engine, settlement, baselines, metrics
app/            NYISO archive client, weather, SQLite store, service (ingest / forecast / settle / score), scheduler, FastAPI
frontend/       Streamlit UI (api client, keyword router, Gemini/GitHub chat layer, charts, app)
scripts/        one-command data pulls and reports behind every headline number
tests/          offline suite; tests/synthetic.py = NYISO-shaped synthetic archive
deploy/         Caddyfile, seed script (backfill + first forecasts), GCE runbook (Phase 5)
data/           README only; fetched data is cached here and gitignored
docs/           plan, experiment log, conventions
results/        per-run backtest outputs (gitignored except the default run's summaries)
```

## Quickstart

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload     # http://127.0.0.1:8000/docs
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py     # http://localhost:8501
```

Or with Docker (backend on :8010, frontend on :8510):

```powershell
docker compose up -d --build
docker compose exec backend sh -c "mkdir -p /data/models/tft"           # served TFT bundle
docker compose cp data/models/tft/tft.onnx backend:/data/models/tft/tft.onnx
docker compose cp data/models/tft/tft.json backend:/data/models/tft/tft.json
docker compose exec backend python deploy/seed.py --months 3 --forecast-days 3   # ~1-2 min
```

Environment variables are documented in [.env.sample](.env.sample); working conventions in
[CONTRIBUTING.md](CONTRIBUTING.md).

## The service

The backend (`app/`) runs the same code as the backtest, live, as of each day's bid
cutoff (D−1 05:00 ET), and keeps everything in one SQLite file on the data volume:

- **Ingest.** `ArchiveClient` fetches one day of one file type at a time (daily files for
  recent days, monthly zips for the rest, cached on the volume), normalizes it and
  upserts 15-min load and real-time prices, hourly day-ahead prices and NYISO's own
  forecast. Re-running never duplicates; a day the archive still lacks is retried.
- **Two models, one identity per row.** The served model is the Temporal Fusion
  Transformer exported to ONNX on the laptop (`scripts/export_tft.py`, uploaded to the
  volume monthly, run with onnxruntime only); the LightGBM trees train on demand and take
  over when the bundle is missing, stale or rejected, or for the demo's retrain button.
  Every forecast row records `tft-onnx:<sha12>:<fit cutoff>` or `lgbm:<feature version>`
  and a version hash of its data window, so "forecast vs actual" always shows what the
  model said at the time.
- **Bid and band.** α is the newsvendor ratio of the trailing 30 days of (RT − DA)
  spreads as of the cutoff; the P10–P90 band is rescaled with the trailing 30 days of
  scored forecasts of the same model (split conformal, as in the backtest).
- **Scoring.** Each morning the previous day's forecasts and DAM schedules are scored in
  hourly MAPE and in dollars (deviation × (RT − DA) per 15-min slot), next to NYISO's
  pre-close forecast on the same day; alerts fire on MAPE above 10% or an imbalance
  cost above the trailing P90.
- **Schedule.** 04:30 ET forecast tomorrow (weather refresh + today's partial load first),
  06:30 ET catch up and score, 08:30 ET fetch tomorrow's ISO forecast and DA prices,
  monthly pruning past the retention window (24 months; forecasts and scores are kept).
- **Access.** Every GET is public; POST/PATCH/DELETE need `X-Admin-Token` and pass a
  per-IP rate limit. `GET /health` shows the data coverage, the served model and the
  last job runs; the full API is at `/docs`.

```powershell
$env:ADMIN_TOKEN = "dev-token"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# seed a store from the archive (24 months by default; --months 2 for a quick local run)
.\.venv\Scripts\python.exe deploy\seed.py --months 2 --forecast-days 7
```

## The demo UI

`frontend/` is a Streamlit app that talks to the API only. The sidebar picks the zone
and the view and shows which model is serving; the chat guide sits behind a floating
bubble at the bottom right; the page is one of six views: **Overview** (zone cards: 7-day MAPE and dollars next to
NYISO's own forecast, alerts), **Forecast** (median, P10–P90, α-bid, NYISO overlay,
actuals; buttons to forecast the next bid day with the served model or retrain the
trees live; a day keeps every version: the served model's forecast is the primary that
the cards and scores count, a retrain is drawn as an extra line on the same chart and
scored on the same actuals, a click on a legend entry hides any line), **DAM Schedule** (start from the α-bid or the median, scale, edit any hour,
see the $ at risk and the hours outside the band, save, export CSV, and later the
schedule's own score), **Forecast vs Actual** (MAPE and dollars per hour, backfill
missing days), **Prices** (DA vs RT, the spread, α and its costs) and **Load**.

The bubble opens a message window with a greeting, option pills (never a model call) and
a text box. Typed text goes to **Gemini on Vertex AI** when configured
(`LLM_PROVIDER=vertex`; the GCE VM's service account authenticates, so there is no key
anywhere), within a per-visitor and a global daily limit; when the limit is reached the
box locks and the guide asks the visitor to pick an option; when no model is configured
or the model fails, the **keyword guide** answers with the same navigation. A reply that
navigates closes the window and opens the view. Every chart follows one validated
palette (forecast blue, actual orange, NYISO aqua, your bid yellow, dollars blue/red
around zero), one axis per chart, and has a table view.

## License

MIT (see [LICENSE](LICENSE)). The license covers this code only, not NYISO data.
