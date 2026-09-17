# scripts/ — one-command reports

Every headline number quoted in the README or the UI must be reproducible by one of
these scripts on the stated window (docs/plan.md §4). Planned for Phase 2:

| Script | Produces |
|---|---|
| `run_backtest.py` | rolling daily retrain, 2025-09-01 → 2026-08-31, all 11 zones + NYCA: MAPE per zone and pooled; per-day cache under `results/` |
| `skill_baselines.py` | persistence D−2, D−7, mean(D−7, D−14) and NYISO `isolf` on the same protocol |
| `imbalance_report.py` | settlement in $ (15-min slot × (mean RT LBMP − DA LBMP)) for the baselines, `isolf`, `isolf + α`, our median and our α-bid |
| `quantile_calibration.py` | P10/P90 coverage, conformal upper quantile if coverage < nominal |
| `alpha_estimate.py` | α = c_under / (c_under + c_over) on a trailing window ending strictly before each cutoff |

All scripts take `--start/--end/--zones`, print a table, and exit non-zero if any
feature timestamp is at or after the bid cutoff (leakage guard).
