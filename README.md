# ⚡ gridcast — NYISO day-ahead zonal load forecasting

[![CI](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml/badge.svg)](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml)

Day-ahead load forecasts for the eleven NYISO zones and the statewide total, built
for the way the market actually settles: bids are due **05:00 ET on D−1**, and every
MWh of forecast error is settled at the **real-time price**. The product turns a
15-minute LightGBM forecast with P10–P90 bands into a cost-aware DAM bid, scores both
in dollars against NYISO's own forecast, and fronts it all with a chat-driven web UI.
LightGBM · FastAPI · SQLite · Streamlit · Docker · Caddy.

> **Status: Phase 2 (model + backtest) done; service, UI and deployment follow.**
> The build plan and the rules for what may become a headline number are in
> [docs/plan.md](docs/plan.md); every number below has a one-command script in
> [scripts/](scripts/README.md) and an entry in [docs/experiments.md](docs/experiments.md).
> This is the US-native successor of
> [gridcast-shanxi](https://github.com/chenyufang-data/gridcast-shanxi).

## Results (12 months, 2025-09-01 → 2026-08-31, all 11 zones + NYCA)

Protocol: one fresh LightGBM retrain per zone and day on data ending at or before the
**D−1 05:00 ET** bid cutoff (the code raises if anything later leaks in), scored at
hourly resolution, the DAM product and the only fair comparison with NYISO's hourly
forecast. Two ISO benchmarks are shown because of a fact verified in this repo: the
NYISO forecast file named for day D is posted on D−1 between 07:10 and 08:00 ET, *after*
the bid close. `isolf_pre` (file named D−1) is what a bidder had at the cutoff;
`isolf_post` (file named D) is the stronger post-close reference.

Hourly MAPE (%), lower is better; `scripts/skill_baselines.py --name default`:

| Zone | Model | First full run (120 d, temperature only) | Best naive | NYISO pre-close (fair) | NYISO post-close |
|---|---|---|---|---|---|
| CAPITL | 6.35 | 7.67 | 11.08 | 5.33 | 5.09 |
| CENTRL | 6.06 | 7.51 | 10.65 | 6.32 | 6.15 |
| DUNWOD | 4.72 | 5.49 | 10.07 | 3.75 | 3.40 |
| GENESE | 5.59 | 6.88 | 10.17 | 4.69 | 4.31 |
| HUD VL | 6.38 | 7.65 | 12.33 | 6.20 | 5.30 |
| LONGIL | 5.78 | 7.19 | 10.97 | 4.05 | 3.60 |
| MHK VL | 8.19 | 10.11 | 13.86 | 7.80 | 7.32 |
| MILLWD | 6.99 | 7.67 | 12.60 | 5.97 | 5.41 |
| N.Y.C. | 3.80 | 4.44 | 8.52 | 2.44 | 2.04 |
| NORTH | 4.43 | 4.70 | 5.85 | 5.39 | 4.37 |
| NYCA | 4.03 | 4.95 | 8.50 | 2.81 | 2.58 |
| WEST | 4.39 | 5.12 | 7.40 | 3.42 | 2.93 |
| **pooled (11 zones + NYCA)** | **5.56** | **6.61** | **10.31** | **4.85** | **4.38** |

Dollars at risk over the year, Σ |deviation × (RT − DA)| across the 11 priced zones (`scripts/imbalance_report.py --name default`): NYISO pre-close $158M, model $184M, best naive $335M. Signed totals against perfect foresight range from $21M to $86M with 95% bootstrap intervals about ±$29M wide, so they do not rank strategies.

What the table says, under the reporting rules in the plan:

- **NYISO's own forecast is more accurate than this model** in every zone except CENTRL and NORTH, so accuracy
  is not the headline. The model beats the best naive baseline by 46% relative
  error; hourly weather (temperature, humidity, cloud, wind, radiation) and a one-year
  training window are what closed the gap from the first full run. The remaining gap
  is the ISO's richer weather feeds and decades of tuning versus one station per zone.
- **The cost-aware α-bid is a secondary result.** In signed dollars against perfect
  foresight no strategy is distinguishable over one year: the 95% bootstrap intervals
  of the yearly totals are several times wider than the differences between strategies,
  because real-time price spikes dominate the sum. The dollars *at risk*
  (Σ |deviation × spread|) do track accuracy, which is the number a desk can act on.
- **What is measured and true:** a leakage-honest day-ahead protocol on public data,
  P10–P90 bands that reach 77% coverage after a trailing conformal
  rescaling (raw quantile trees: 52%), and a settlement layer that prices
  every forecast the way the market does.

Since this run, a three-zone comparison (`docs/experiments.md` §2c: NYCA, N.Y.C., MHK VL,
same 12 months) found two things the table above does not yet include: 24 months of
history instead of 12 (−0.26 pooled MAPE, adopted as the default), and a Temporal Fusion
Transformer (`models/tft.py`, −0.94; pooled 4.40 vs the ISO's 4.35 on those zones, ahead of
the ISO on MHK VL). XGBoost and swap-noise augmentation were null results. The full
12-zone re-run with both is the next step; until then the table above is the headline.

## Data

**Data: NYISO public MIS archive (<http://mis.nyiso.com/public/csv/>), fetched at
runtime and not redistributed.** No NYISO files are committed or shipped; the demo
shows derived aggregates and model outputs. NYISO does not sponsor or endorse this
project. Details, verified file layouts, and the DST gotchas: [data/README.md](data/README.md).

## Layout

```
models/         modeling package shared by src/ and app/: cutoff guard, features, LightGBM / XGBoost, swap-noise augmentation, TFT
src/            constants, dataset builder, backtest engine, settlement, baselines, metrics
app/            NYISO archive client, weather provider, logging, FastAPI (service: Phase 3)
frontend/       Streamlit UI, HTTP client of the API only (Phase 4)
scripts/        one-command data pulls and reports behind every headline number
tests/          offline suite; tests/synthetic.py = NYISO-shaped synthetic archive
deploy/         Caddyfile, GCE runbook and seed script (Phase 5)
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

Or `docker compose up --build` (backend on :8010, frontend on :8510). Environment
variables are documented in [.env.sample](.env.sample); working conventions in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT (see [LICENSE](LICENSE)). The license covers this code only, not NYISO data.
