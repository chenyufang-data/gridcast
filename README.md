# ⚡ gridcast — NYISO day-ahead zonal load forecasting

[![CI](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml/badge.svg)](https://github.com/chenyufang-data/gridcast/actions/workflows/ci.yml)

Day-ahead load forecasts for the eleven NYISO zones and the statewide total, built
for the way the market actually settles: bids are due **05:00 ET on D−1**, and every
MWh of forecast error is settled at the **real-time price**. The product turns a
15-minute LightGBM forecast with P10–P90 bands into a cost-aware DAM bid, scores both
in dollars against NYISO's own forecast, and fronts it all with a chat-driven web UI.
LightGBM · FastAPI · SQLite · Streamlit · Docker · Caddy.

> **Status: Phase 0 (scaffold).** No model, backtest, or results yet. The build plan
> and the rules for what may become a headline number are in
> [docs/plan.md](docs/plan.md). This is the US-native successor of
> [gridcast-shanxi](https://github.com/chenyufang-data/gridcast-shanxi).

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
