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
- **Next action: Phase 2** — model port (`model.py`) with the cutoff timestamp
  (D−1 05:00 ET), DST-aware slot grid from `resample_slots`, `holidays` lib,
  features per `docs/plan.md` §3; `src/backtest.py` + `scripts/run_backtest.py`
  (per-zone, parallel, per-day cache under `results/`); baselines incl. **two isolf
  benchmarks** (D−1-named file = pre-close, D-named file = post-close, see
  `data/README.md`); α estimation; `imbalance_report.py`; calibration/conformal;
  experiment log. Backfill 2025-06-01 → today first via `ArchiveClient.frame`
  (monthly zips; ~15 months × 5 types).
- GitHub: `origin` = https://github.com/chenyufang-data/gridcast (private), `main`
  pushed and tracking. CI status must be checked on the web (no `gh`).
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
