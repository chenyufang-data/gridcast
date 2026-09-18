"""Accuracy metrics and the hourly view shared by the backtest, baselines and reports.

Two resolutions are reported everywhere:

- **hourly** (the DAM product and the only fair comparison with NYISO's hourly
  ``isolf``): slots averaged per UTC hour, keyed on ``hour_utc`` so the fall-back
  day's repeated local hour stays two distinct hours;
- **slot** (15-min): our own forecasts only.

MAPE is pooled over slots/hours (mean of absolute percentage errors), the same
definition as the source repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p) & (a != 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(a[ok] - p[ok]) / np.abs(a[ok])) * 100)


def mae(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(a[ok] - p[ok]))) if ok.any() else float("nan")


def to_hourly(
    df: pd.DataFrame, value_cols: list[str], keys: list[str] | None = None
) -> pd.DataFrame:
    """Average slot columns per UTC hour (and per `keys`, e.g. ``["zone"]``)."""
    keys = keys or []
    d = df.copy()
    d["hour_utc"] = d["ts_utc"].dt.floor("h")
    out = d.groupby([*keys, "hour_utc"], sort=True)[value_cols].mean().reset_index()
    return out


def score_table(
    df: pd.DataFrame, pred_cols: list[str], actual_col: str = "actual", by: str = "zone"
) -> pd.DataFrame:
    """MAPE (%) of each prediction column per `by` group plus a pooled row."""
    rows = []
    for key, g in df.groupby(by, sort=True):
        rows.append(
            {
                by: key,
                "n": int(g[actual_col].notna().sum()),
                **{c: mape(g[actual_col], g[c]) for c in pred_cols},
            }
        )
    rows.append(
        {
            by: "pooled",
            "n": int(df[actual_col].notna().sum()),
            **{c: mape(df[actual_col], df[c]) for c in pred_cols},
        }
    )
    return pd.DataFrame(rows)
