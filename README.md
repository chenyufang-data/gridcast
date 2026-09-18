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

_RESULTS_TABLE_

What the table says, under the reporting rules in the plan:

- **NYISO's own forecast is more accurate than this model** by a wide margin, so
  accuracy is not the headline. The model beats every naive baseline and closes part of
  the gap with weather; the remaining gap is the ISO's weather feeds and decades of
  tuning versus a 120-day window and one temperature series per zone.
- **The cost-aware α-bid is a secondary result.** In signed dollars against perfect
  foresight no strategy is distinguishable over one year: the 95% bootstrap intervals
  of the yearly totals are several times wider than the differences between strategies,
  because real-time price spikes dominate the sum. The dollars *at risk*
  (Σ |deviation × spread|) do track accuracy, which is the number a desk can act on.
- **What is measured and true:** a leakage-honest day-ahead protocol on public data,
  P10–P90 bands that reach _CONF_COVERAGE_% coverage after a trailing conformal
  rescaling (raw quantile trees: _RAW_COVERAGE_%), and a settlement layer that prices
  every forecast the way the market does.

## Data

**Data: NYISO public MIS archive (<http://mis.nyiso.com/public/csv/>), fetched at
runtime and not redistributed.** No NYISO files are committed or shipped; the demo
shows derived aggregates and model outputs. NYISO does not sponsor or endorse this
project. Details, verified file layouts, and the DST gotchas: [data/README.md](data/README.md).

## Layout

```
model.py        modeling core shared by src/ and app/ (Phase 2)
src/            constants (src/config.py), offline backtest harness (Phase 2)
app/            FastAPI service: logging, health; zones/forecasts/schedules (Phase 3)
frontend/       Streamlit UI, HTTP client of the API only (Phase 4)
scripts/        one-command reports behind every headline number (Phase 2)
tests/          offline suite; tests/synthetic.py = NYISO-shaped synthetic archive
deploy/         Caddyfile, GCE runbook and seed script (Phase 5)
data/           README only; fetched data is cached here and gitignored
docs/           plan, conventions, demo runbook
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
