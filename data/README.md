# data/ — source, terms, and layout

Nothing in this directory is committed except this file. Everything else is fetched
at runtime and gitignored (`data/cache/`, `*.csv`, `*.zip`, `*.db`).

## Source and attribution

All market data comes from the **New York Independent System Operator (NYISO)**
public MIS archive: <http://mis.nyiso.com/public/csv/>. Each file type has its own
directory, for example <http://mis.nyiso.com/public/csv/pal/> for actual load.

- The data is retrieved directly from NYISO when the application or a script needs
  it. **No NYISO bytes are redistributed** in this repository, in Docker images, or
  through the public demo, which shows derived aggregates and model outputs only.
- NYISO's [legal notice](https://www.nyiso.com/legal-notice) grants no license and is
  silent on the CSV archive, hence this fetch-not-redistribute policy. NYISO does not
  sponsor or endorse this project; no NYISO logos or marks are used.
- Wherever data appears (README, UI footer, reports) it is credited as
  *"Data: NYISO public MIS archive (mis.nyiso.com/public/csv), fetched at runtime."*

Weather features use the [Open-Meteo](https://open-meteo.com/) previous-runs API
(CC BY 4.0), credited the same way.

## Archive facts (verified on 2026-09-17; sample day 2026-09-15)

| Type | Directory | Daily file | Monthly zip | Rows / day | Notes |
|---|---|---|---|---|---|
| `pal` actual load | `pal/` | `YYYYMMDDpal.csv` | `YYYYMM01pal_csv.zip` | 11 zones × (288 + off-schedule stamps) | `"Time Stamp","Time Zone","Name","PTID","Load"`; `MM/DD/YYYY HH:MM:SS` local ET; tz `EDT`/`EST`; strings quoted, numbers bare; stamps 00:00:00 .. 23:55:00 |
| `damlbmp_zone` day-ahead price | `damlbmp/` | `YYYYMMDDdamlbmp_zone.csv` | `YYYYMM01damlbmp_zone_csv.zip` | 15 names × 24 h (23 / 25 on DST days) | unquoted; `MM/DD/YYYY HH:MM`; **no tz column**: fall-back day lists `01:00` twice, first = EDT |
| `realtime_zone` RT price | `realtime/` | `YYYYMMDDrealtime_zone.csv` | `YYYYMM01realtime_zone_csv.zip` | 15 names × (288 + off-schedule stamps) | quoted; stamps are **interval ends**: 00:05:00 .. next day 00:00:00 |
| `rtlbmp_zone` RT hourly integrated | `rtlbmp/` | `YYYYMMDDrtlbmp_zone.csv` | `YYYYMM01rtlbmp_zone_csv.zip` | 15 names × 24 h | quoted; `MM/DD/YYYY HH:MM`, 00:00 .. 23:00 |
| `isolf` ISO load forecast | `isolf/` | `YYYYMMDDisolf.csv` | `YYYYMM01isolf_csv.zip` | 6 days × 24 h, wide | `"Time Stamp","Capitl",…,"West","NYISO"`; integer MW; the file issued on day X covers X 00:00 .. X+5 23:00; `NYISO` = sum of the zones |

Price columns: `Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)`.

Zones and PTIDs: CAPITL 61757, CENTRL 61754, DUNWOD 61760, GENESE 61753, HUD VL 61758,
LONGIL 61762, MHK VL 61756, MILLWD 61759, N.Y.C. 61761, NORTH 61755, WEST 61752.
Price files also carry the external proxies H Q 61844, NPX 61845, O H 61846, PJM 61847.

Gotchas that every normalizer must handle (each has a unit test):

- **Off-schedule RTD stamps.** Besides the 5-minute grid, `pal` and `realtime_zone`
  share extra stamps such as `04:04:18` (8 of them on 2026-09-15, giving 296 stamps
  instead of 288). Never index by position: dedupe on (timestamp, tz, zone), then
  resample to the 15-minute slot mean.
- **DST.** Fall-back (2025-11-02): 25 local hours, `01:xx` appears twice, `pal` labels
  them `EDT` then `EST`, the price files have no tz column and rely on row order.
  Spring-forward (2026-03-08): 23 hours, no `02:xx`. Store UTC internally.
- **Two stamping conventions.** `pal` stamps the interval start; `realtime_zone`
  stamps the interval end.
- **Daily files live about 11 days** (on 2026-09-17: 09-06 → 200, 09-04 → 404). The
  monthly zip is the durable source: flat daily files inside (`20251102isolf.csv`, …),
  final on the 1st of the next month (~00:15 ET), and for the **current month rebuilt
  every morning (~05:00 ET) with every day so far, including today's partial day**.
- **Fall-back hour in the hourly files** (2025-11-02, from the November zips):
  `damlbmp` and `rtlbmp` list `01:00` twice with different values (EDT first);
  `isolf` lists `01:00` twice with **identical** values (145 rows that day).
- **`isolf` posting time**: the file named for day D (covering D…D+5) is last modified
  on D−1 between ~07:10 and ~08:00 ET, and does not exist at 05:00 ET on D−1
  (checked 2026-09-17 05:03 ET: `20260918isolf.csv` → 404). So NYISO's forecast for D
  in the D-named file was **not available before the DAM close**; the leakage-free ISO
  benchmark for day D is the file named D−1 (its day-D rows, a 2-day-ahead forecast),
  and the D-named file is the stronger post-close reference. Both must be reported.
- `damlbmp` for day D is published on D−1 after the DAM clears (not present at
  05:03 ET on D−1); DA prices for D are therefore unknown at the bid cutoff.
- Line endings vary (`isolf` LF, price files CRLF inside the zips); parsers must not care.

## Local layout (created at runtime by `app/nyiso.py`)

```
data/
  README.md          this file (the only committed item)
  cache/<dir>/…      raw daily CSVs and monthly zips, mirroring the archive paths;
                     `<zip>.partial` = the current month's zip (refreshed when a
                     requested day is missing); days >= today are never cached
  processed/*.pkl    normalized tables for offline work (scripts/backfill.py)
  app.db             SQLite (APP_DB_PATH)
  weather.csv        Open-Meteo day-ahead forecasts per zone, daily (WEATHER_PATH)
  weather_hourly.csv the same per hour (WEATHER_HOURLY_PATH); scripts/fetch_weather.py
```

Canonical frames produced by the normalizers (all timestamps tz-aware UTC):
`normalize_pal` → `ts_utc` (interval start), `ts_end_utc`, `zone`, `load_mw`;
`normalize_realtime` → the same with `p_rt` (the file stamps interval ends; the start
is the previous stamp, capped at 5 min); `normalize_damlbmp` / `normalize_rtlbmp` →
hourly `p_da` / `p_rt_hourly`; `normalize_isolf` → `issued`, `ts_utc`, `zone`,
`isolf_mw` including `NYCA`. `resample_slots` turns interval observations into a
time-weighted 15-min grid with a `coverage` column, and `add_nyca` appends the
statewide total where all 11 zones are present.

Tests never touch this directory: they use the deterministic synthetic archive in
`tests/synthetic.py`, which reproduces the layouts and gotchas above.
