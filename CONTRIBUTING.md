# Contributing / working conventions

## Setup

```powershell
py -3.13 -m venv .venv                       # any Python >= 3.11; CI and Docker use 3.11
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pytest         # offline suite (network tests need -m live)
.\.venv\Scripts\python.exe -m ruff check . ; .\.venv\Scripts\python.exe -m ruff format .
.\.venv\Scripts\python.exe -m mypy
```

Copy `.env.sample` to `.env` for local secrets (`ADMIN_TOKEN`, `GITHUB_TOKEN`). Never
commit `.env`, a database, or any file fetched from NYISO (see `data/README.md`).

## Dependencies

Loose specs live in `requirements*.txt`; exact versions in the matching
`requirements*.lock`, which is what CI, Docker, and developers install.
`requirements-research.txt` (torch, onnx, onnxscript: training and exporting the TFT in
`models/tft.py`) is the one exception: it is installed by hand from the CUDA index that
matches the GPU and has no lock; nothing in the images or CI depends on it, and its tests
skip when torch is missing. Serving the exported TFT needs only `onnxruntime`, a normal
backend dependency (`models/tft_data.py` + `models/tft_onnx.py` import no torch). The locks are
*universal* (one file for Windows dev, Linux CI, and the images) and consistent with
each other. After editing a `.txt`, regenerate all three:

```powershell
$uv = ".\.venv\Scripts\uv.exe"
& $uv pip compile --universal --python-version 3.11 -o requirements-dev.lock requirements-dev.txt
& $uv pip compile --universal --python-version 3.11 -c requirements-dev.lock -o requirements.lock requirements.txt
& $uv pip compile --universal --python-version 3.11 -c requirements-dev.lock -o requirements-frontend.lock requirements-frontend.txt
```

## Code conventions (enforced by `ruff` and `mypy` in CI)

- **Type hints** on every public function and method; `from __future__ import
  annotations` at the top of each module; Python 3.11 syntax (`X | None`, `list[str]`).
  mypy runs with `check_untyped_defs`; add `# mypy: disallow-untyped-defs` to a module
  once it is fully annotated.
- **Logging, not print.** `log = logging.getLogger(__name__)` per module; entry points
  call `app.log.configure_logging()` once. `print()` is allowed only in `scripts/`
  (their report) and `frontend/` (ruff rule T20 elsewhere). Log at `INFO` what an
  operator needs to see (fetches, training runs, scheduler ticks), `WARNING` for
  degraded-but-continuing paths (weather missing, retry), `ERROR` with `exc_info`
  when a background task fails.
- **Datetimes.** Store and compute in UTC (`ts_utc`, tz-aware). Convert to
  `America/New_York` only at the edges: parsing NYISO files, building the bid cutoff,
  displaying. Naive datetimes fail ruff's `DTZ` rules outside `tests/`.
- **Never assume the grid.** Rows per day vary, DST days have 92/100 slots, off-schedule
  stamps exist. Dedupe, then resample; never index by position.
- **Leakage guard.** Any feature or window that touches data at or after the bid cutoff
  (D−1 05:00 ET) is a bug; backtest and scoring code assert it.
- Layering as in the source repo: `models/` is shared by `src/` and `app/`;
  `frontend/` talks to the backend over HTTP only and imports nothing from `app/`.

## Tests

- `tests/synthetic.py` generates NYISO-shaped CSVs and their clean truth frames;
  use it instead of any real file. Session fixtures in `tests/conftest.py` cover a
  fall-back week (with off-schedule stamps) and a spring-forward week.
- Markers: `live` (network; excluded by default), `slow` (real LightGBM training or
  multi-day backtests; deselect with `-m "not slow"`).
- Tests must pass before a commit; CI runs `ruff`, `mypy`, `pytest`, and both Docker
  builds on every push and pull request.

## Commits and reporting

- Commit style `type(scope): summary` (`feat`, `fix`, `docs`, `test`, `ci`, `chore`,
  `refactor`).
- Every headline number in README or docs states its evaluation window and has a
  one-command script under `scripts/` (the reporting rules below).

## Reporting rules

What may become a headline number, in the README, the docs, the UI and any video:

- **Accuracy.** Hourly MAPE against NYISO's own pre-close forecast (`isolf_pre`, the file
  named D−1) on identical zone-days. If the ISO's forecast is better, MAPE is not the
  headline; if the paired difference is not significant, the claim is "matches", never
  "beats".
- **Dollars.** The cost-aware α-bid may lead only when it is measured in settled dollars,
  beats the strongest baseline including `isolf + α`, and α was estimated strictly before
  each day's cutoff. Otherwise it is a secondary result.
- **Window and script.** Every headline states its evaluation window and is reproduced by
  one command under `scripts/`, with an entry in `docs/experiments.md`.
- **Data.** Wherever data appears, credit the source with a link (NYISO public MIS
  archive; Open-Meteo); never redistribute NYISO bytes; never use the NYISO logo.
- **The ISO forecast is never a feature.** It is the benchmark; a model that reads it
  would be a different experiment and must be named as such.
