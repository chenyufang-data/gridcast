"""Training-set augmentation for the tabular estimators.

``swap_noise`` (Jahrer's tabular denoising trick, used here as plain augmentation): copies
of the training rows in which every feature cell is replaced, with probability ``p``, by
the same feature of another row. The target stays with the row, so the estimator learns
to reach the same answer with a randomly corrupted view of the features - a regulariser
against over-trusting any single lag or weather column.

Two knobs keep the corruption plausible for load data:

- ``within="tod"`` draws the donor row from the same quarter hour, so a night-time lag is
  never swapped into a noon row (the marginal per slot is preserved);
- calendar columns and the decay-age column are never swapped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from models.features import AGE_COL

CALENDAR_COLUMNS = ("tod", "dayofweek", "is_holiday", "holiday_tomorrow", "is_weekend", AGE_COL)
KINDS = ("swap",)


def swap_noise(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    p: float = 0.1,
    copies: int = 1,
    weight: float = 0.5,
    within: str | None = "tod",
    exclude: tuple[str, ...] = CALENDAR_COLUMNS,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Append `copies` swap-noised copies of ``(X, y)``.

    Returns ``(X_aug, y_aug, sample_weight)``: the originals first with weight 1, then
    the copies with weight `weight` (multiplied into the decay weights by the wrapper).
    `within` names a column whose value the donor row must share (``None`` = any row).
    """
    if copies < 1 or p <= 0:
        return X, np.asarray(y, dtype=float), np.ones(len(X))
    rng = rng or np.random.default_rng(0)
    cols = list(X.columns)
    swap_idx = np.array([i for i, c in enumerate(cols) if c not in exclude])
    base = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    n = len(base)
    if within is not None and within in cols:
        groups = [np.flatnonzero(X[within].to_numpy() == g) for g in np.unique(X[within])]
    else:
        groups = [np.arange(n)]
    outs, ys, ws = [base], [yv], [np.ones(n)]
    for _ in range(copies):
        donor = np.arange(n)
        for idx in groups:
            donor[idx] = rng.choice(idx, size=len(idx), replace=True)
        mask = rng.random((n, len(swap_idx))) < p
        copy = base.copy()
        sub = copy[:, swap_idx]
        copy[:, swap_idx] = np.where(mask, base[donor][:, swap_idx], sub)
        outs.append(copy)
        ys.append(yv)
        ws.append(np.full(n, weight))
    X_aug = pd.DataFrame(np.vstack(outs), columns=cols)
    return X_aug, np.concatenate(ys), np.concatenate(ws)


def apply(
    kind: str | None,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Dispatch by name; ``None`` returns the inputs unchanged (weight ``None``)."""
    if kind is None:
        return X, np.asarray(y, dtype=float), None
    if kind == "swap":
        return swap_noise(X, y, rng=np.random.default_rng(seed), **(params or {}))
    raise ValueError(f"unknown augmentation {kind!r}; known: {KINDS}")
