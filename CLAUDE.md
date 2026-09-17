# CLAUDE.md — session handoff for `gridcast` (NYISO successor)

Read this first, then `docs/plan.md`. Written 2026-09-17 at the end of the planning
session; nothing below is speculative — every fact was verified or decided by the user.

## What this repo is

The US-native successor of **gridcast-shanxi** (github.com/chenyufang-data/gridcast-shanxi,
locally `Documents/gridcast-shanxi` after the folder swap). Same architecture, invariants,
evaluation culture, test layout and deployment stack, rebuilt on NYISO data and NYISO
market rules (D−1 05:00 ET bid cutoff, errors settled at real-time price, no ±10% band).
Full design: `docs/plan.md` (committed, approved). Source-repo references worth reading
before porting: `gridcast-shanxi/docs/specification.md` (module map) and
`gridcast-shanxi/docs/project_audit_report.md` (audit findings to fix from day one).

## Status and next action

- `docs/plan.md` is **approved** with the answers below. **Phase 0 (scaffold) has not
  started.** Next action: Phase 0 — repo layout mirroring the source, MIT LICENSE, pinned
  dependency lock file, `.gitignore` (data cache, `.env`, `*.db`), CI workflow, logging
  and type-hint conventions, synthetic test fixtures, `data/README.md` with NYISO
  attribution. Then Phases 1–6 as planned.
- GitHub: the user creates a **private** `chenyufang-data/gridcast` (public later, when the
  README has backtest results and the site is live). When it exists:
  `git remote add origin https://github.com/chenyufang-data/gridcast && git push -u origin main`.
  Check `git remote -v` first — it may already be set.
- Folder swap: the user planned to rename `Documents/gridcast` → `gridcast-shanxi` and
  `Documents/gridcast-nyiso` → `gridcast` after closing the previous session. Check `pwd`;
  either name may be current.

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

## Verified NYISO facts (2026-09-17; details and gotchas in `docs/plan.md` §2)

- Archive `http://mis.nyiso.com/public/csv/<type>/`: monthly zips `YYYYMM01<type>_csv.zip`
  back to 2005 for `pal`, `damlbmp_zone`, `realtime_zone`, `rtlbmp_zone`, `isolf`; daily
  `YYYYMMDD<type>.csv` only for recent dates.
- `pal`: `"Time Stamp","Time Zone","Name","PTID","Load"`, 5-min, tz EDT/EST, 11 zones,
  rows/day vary (3,169–3,301) → dedupe on (ts, tz, zone) then resample; never index by
  position. Price files: no tz column, 15 names (11 zones + `H Q`, `NPX`, `O H`, `PJM`),
  DST fall-back day lists `01:00` twice (first = EDT). `isolf` file of day X covers
  X…X+5 → benchmark for D is the D−1 file's day-D rows (⚠️ posting time vs 05:00 ET
  still unverified). DAM bids due 05:00 ET on D−1 (FERC intro guide; NYISO Manual 11).
- Legal notice grants no license and is silent on CSVs → the fetch-not-redistribute plan.

## Port-size estimate (what is new vs reused from gridcast-shanxi, ~5,100 LOC)

| Module | Source LOC | Plan | Est. new/rewritten |
|---|---|---|---|
| `model.py` | 481 | keep repair, decay wrapper, `get_model`, shape features; replace day-lag scheme with cutoff-timestamp lags, DST-aware grid, `holidays` lib | ~200 |
| `src/` backtest + config | 182 | rewrite: cutoff, per-zone, parallel, CLI (`scripts/run_backtest.py`) | ~250 |
| `app/nyiso.py` (new) | — | fetch/cache/normalize incl. DST + dedupe | ~300 |
| `app/adapters.py` | 290 | keep as secondary CSV path | ~30 |
| `app/db.py` | 135 | zones, prices table, `schedules`, $ score columns | ~60 |
| `app/service.py` | 1,058 | keep quality/flags/versioning/scoring core; add cutoff windows, $ scoring, α estimation, schedules, scheduler hooks | ~400 |
| `app/main.py` | 320 | admin-token middleware, prices/isolf endpoints, logging | ~100 |
| `app/weather.py` | 82 | zone centroids | ~30 |
| `frontend/router.py` | 416 | English persona, zone aliases | ~80 |
| `frontend/app.py` | 1,161 | English strings, α-bid line, `isolf` overlay, $ annotations, Prices view | ~500 |
| `scripts/` | 165 | + `isolf` baseline, `imbalance_report.py`, conformal upper quantile | ~350 |
| `tests/` | 753 | keep adapter/router suites; rewrite service e2e for zones/prices; new DST, α, imbalance tests; synthetic fixture generator | ~600 |
| deploy | ~100 + md | reuse compose/Caddyfile; new EN GCP runbook, `seed.py`, scheduler | ~250 |
| **Total** | ~5,100 | | **≈ 3,000 new/rewritten (~45–50%), ≈ 3,000 reused with light edits** |

## Working conventions carried over

- Commit style `type(scope): summary`; end commit messages with the attribution line the
  harness provides. Tests must pass before commit; CI runs the offline suite on every push.
- No secrets in git; `.env` only; `.env.sample` documents every variable (English here).
- Dev environment: per-project `.venv` from `requirements-dev.txt`
  (`.\.venv\Scripts\python.exe -m pytest`). CI uses Python 3.11.
- Windows tooling: `gh` is not installed (repo creation/rename = user, web UI). The Bash
  tool truncates commands over ~8 KB — write long files with the Write tool, not heredocs.
  PowerShell 5.1: no `&&`; quote-heavy commit messages break — avoid `"` inside `-m`.
- Reporting to the user: lead with the outcome; cite files and line numbers; label anything
  unverified.
