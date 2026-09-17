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

- **Phase 0 (scaffold) is done** — layout, MIT LICENSE, universal lock files, CI
  (ruff + mypy + pytest + both Docker builds), logging/typing conventions,
  `tests/synthetic.py` (NYISO-shaped synthetic archive with truth frames),
  `data/README.md`. Suite: 19 offline tests, all green; both Docker images build.
- **Next action: Phase 1** — `app/nyiso.py`: `fetch_day` / `fetch_month` with retry and
  cache under `NYISO_CACHE_DIR` (mirror archive paths), zip extraction, and
  `normalize_pal` / `normalize_prices` / `normalize_isolf` → canonical UTC frames
  (`ts_utc, zone, load_mw` / `ts_utc, zone, p_da, p_rt` / `issued_date, ts_utc, zone,
  isolf_mw`). Test each against `SyntheticNYISO` truth frames (fall-back week fixture
  `synth_autumn`, spring fixture `synth_spring`), then one `-m live` test on a real
  recent day. Resolve the ⚠️ items in `data/README.md` (zip layout, isolf/rtlbmp
  fall-back hour, isolf posting time vs 05:00 ET) with real downloads.
- GitHub: the user creates a **private** `chenyufang-data/gridcast` (public later, when
  the README has backtest results and the site is live). When it exists:
  `git remote add origin https://github.com/chenyufang-data/gridcast && git push -u origin main`.
  Check `git remote -v` first — no remote was set as of Phase 0.
- Folder swap is done: this repo is `Documents/gridcast`, the source is
  `Documents/gridcast-shanxi`.

## Decisions (locked) and answered questions

| Topic | Decision |
|---|---|
| Zones | backtest all 11 zones + NYCA; demo default N.Y.C. |
| Backtest range | 2025-09-01 → 2026-08-31, warm-up data from 2025-06-01 |
| Data terms | proceed: fetch-not-redistribute (no NYISO bytes in git), **cite the original source with a link everywhere data appears** (README, `data/README.md`, UI footer), no NYISO logo |
| Access model | public read, `X-Admin-Token` on every write endpoint, in-app per-IP rate limiting, nightly reset-to-seed **off** |
| Headline rules | MAPE is not the headline if NYISO's `isolf` beats ours. The α-bid leads only if measured in $, beats the strongest baseline incl. `isolf + α`, and α is estimated strictly before each cutoff; otherwise secondary. Every headline has a window and a one-command script |
| ISO forecast as feature | excluded |
| Settlement granularity | 15-min slot × mean of the three 5-min RT LBMPs − DA LBMP |
| Cloud | GCE **e2-small + 2 GB swap**, 20 GB pd-balanced, static IPv4, **us-east1** recommended (same price tier as us-central1; us-east4 ~10–15% more). ≈ $19/mo. The user has a **$300 GCP new-user credit expiring late November 2026** → deploy early (target: site live well before mid-November); e2-medium is affordable under the credit if the VM ever OOMs |
| Domain | `https://gridcast.cyfang.org`; user adds the DNS A record; Caddy auto-HTTPS |
| Demo video | < 2:00, English, **the user's own voice + burned-in captions**; deliver speech notes (~150 wpm, ≤ 260 words) and an SRT file |
| Heavy compute | the 12-zone × 12-month backtest runs on the user's laptop; the VM only does daily incremental fetch + one forecast per zone |

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
  fall-back day lists `01:00` twice (first = EDT). `isolf` file of day X covers
  X…X+5 hourly, integer MW, `NYISO` = sum of zones → benchmark for D is the D−1 file's
  day-D rows (⚠️ posting time vs 05:00 ET still unverified). DAM bids due 05:00 ET
  on D−1 (FERC intro guide; NYISO Manual 11).
- Legal notice grants no license and is silent on CSVs → the fetch-not-redistribute plan.

## Port-size estimate (what is new vs reused from gridcast-shanxi, ~5,100 LOC)

| Module | Source LOC | Plan | Est. new/rewritten |
|---|---|---|---|
| `model.py` | 481 | keep repair, decay wrapper, `get_model`, shape features; replace day-lag scheme with cutoff-timestamp lags, DST-aware grid, `holidays` lib | ~200 |
| `src/` backtest + config | 182 | rewrite: cutoff, per-zone, parallel, CLI (`scripts/run_backtest.py`); `src/config.py` done | ~250 |
| `app/nyiso.py` (new) | — | fetch/cache/normalize incl. DST + dedupe | ~300 |
| `app/adapters.py` | 290 | keep as secondary CSV path | ~30 |
| `app/db.py` | 135 | zones, prices table, `schedules`, $ score columns | ~60 |
| `app/service.py` | 1,058 | keep quality/flags/versioning/scoring core; add cutoff windows, $ scoring, α estimation, schedules, scheduler hooks | ~400 |
| `app/main.py` | 320 | admin-token middleware, prices/isolf endpoints, logging (health + logging done) | ~100 |
| `app/weather.py` | 82 | zone centroids | ~30 |
| `frontend/router.py` | 416 | English persona, zone aliases | ~80 |
| `frontend/app.py` | 1,161 | English strings, α-bid line, `isolf` overlay, $ annotations, Prices view | ~500 |
| `scripts/` | 165 | + `isolf` baseline, `imbalance_report.py`, conformal upper quantile | ~350 |
| `tests/` | 753 | keep adapter/router suites; rewrite service e2e for zones/prices; new DST, α, imbalance tests; synthetic generator done (`tests/synthetic.py`) | ~600 |
| deploy | ~100 + md | compose/Caddyfile ported (no basic auth, `ADMIN_TOKEN`); new EN GCP runbook, `seed.py`, scheduler | ~250 |
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
