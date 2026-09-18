"""Per-slot gradient-boosting regressors on the cutoff feature table, and ``forecast_day``.

Estimators (``ESTIMATORS``): ``"lgbm"`` (LightGBM, the default) and ``"xgb"`` (XGBoost,
same hyper-parameters translated name for name; imported lazily so the backend image
needs only LightGBM). Both are wrapped in :class:`DecayWeighted`, which turns the
``_age_days`` column into time-decay sample weights ``0.5 ** (age / half_life)``.

:func:`forecast_day` is the one entry point shared by the backtest and the service: it
regularises and repairs the history, builds the feature table as of the cutoff, fits the
median model, the band quantiles and the α-bid quantile, and returns the chronological
grid of the target day.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from models import augment
from models.features import (
    AGE_COL,
    MIN_TRAIN_DAYS,
    SLOT,
    LeakageError,
    build_features,
    cutoff_for,
    local_fields,
    regularize,
    repair,
    training_targets,
)

log = logging.getLogger(__name__)

DECAY_HALF_LIFE_DAYS = 32
ESTIMATORS = ("lgbm", "xgb")
# shared hyper-parameters in LightGBM's vocabulary; translated for XGBoost below
DEFAULT_PARAMS: dict[str, Any] = dict(
    n_estimators=600,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=5,
    min_child_samples=30,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,
)


class Estimator(Protocol):
    def fit(self, X: pd.DataFrame, y: Any, sample_weight: np.ndarray | None = None) -> Any: ...

    def predict(self, X: pd.DataFrame) -> Any: ...


class DecayWeighted:
    """Time-decay sample weights ``0.5 ** (age_days / half_life)`` around any regressor.

    Softer than a hard window: data just outside the window fades instead of vanishing,
    recent behaviour dominates, older weekly structure is kept. `decay_floor` bounds the
    weight from below; extra per-row weights (augmented copies) multiply in.
    """

    def __init__(
        self,
        model: Estimator,
        decay_half_life: float = DECAY_HALF_LIFE_DAYS,
        decay_floor: float = 0.0,
    ) -> None:
        self.model = model
        self.decay_half_life = decay_half_life
        self.decay_floor = decay_floor

    @staticmethod
    def _pop_age(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        if AGE_COL in X.columns:
            return X.drop(columns=[AGE_COL]), X[AGE_COL].to_numpy(dtype=float)
        return X, None

    def weights(
        self, X: pd.DataFrame, sample_weight: np.ndarray | None = None
    ) -> np.ndarray | None:
        _, age = self._pop_age(X)
        weight = None
        if age is not None and self.decay_half_life:
            weight = np.maximum(0.5 ** (age / self.decay_half_life), self.decay_floor)
        if sample_weight is not None:
            weight = sample_weight if weight is None else weight * sample_weight
        return weight

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, sample_weight: np.ndarray | None = None
    ) -> DecayWeighted:
        weight = self.weights(X, sample_weight)
        X, _ = self._pop_age(X)
        self.model.fit(X, y, sample_weight=weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X, _ = self._pop_age(X)
        return np.asarray(self.model.predict(X), dtype=float)


def _lgbm(quantile: float | None, params: dict[str, Any]) -> Estimator:
    p: dict[str, Any] = dict(objective="mae", subsample_freq=1, verbose=-1, **params)
    if quantile is not None:
        p.update(objective="quantile", alpha=quantile)
    return LGBMRegressor(**p)


def _xgb(quantile: float | None, params: dict[str, Any]) -> Estimator:
    from xgboost import XGBRegressor  # research extra (requirements-dev.txt)

    p: dict[str, Any] = dict(
        objective="reg:absoluteerror",
        tree_method="hist",
        grow_policy="lossguide",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        # L1 and quantile objectives use a unit hessian, so this is a row count as in LightGBM
        min_child_weight=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
        verbosity=0,
    )
    if quantile is not None:
        p.update(objective="reg:quantileerror", quantile_alpha=quantile)
    return XGBRegressor(**p)


BUILDERS = {"lgbm": _lgbm, "xgb": _xgb}


def get_model(
    estimator: str = "lgbm",
    *,
    quantile: float | None = None,
    decay_half_life: float = DECAY_HALF_LIFE_DAYS,
    decay_floor: float = 0.0,
    **overrides: Any,
) -> DecayWeighted:
    """A decay-weighted median model (or the `quantile` model) of the named estimator.

    `overrides` use LightGBM's parameter names (``n_estimators``, ``learning_rate``,
    ``num_leaves``, ``min_child_samples``, ...) for every estimator.
    """
    if estimator not in BUILDERS:
        raise ValueError(f"unknown estimator {estimator!r}; known: {ESTIMATORS}")
    params = {**DEFAULT_PARAMS, **overrides}
    return DecayWeighted(
        BUILDERS[estimator](quantile, params),
        decay_half_life=decay_half_life,
        decay_floor=decay_floor,
    )


# ---------------------------------------------------------------------------- one forecast
def _ratio_denominator(rows: pd.DataFrame) -> pd.Series:
    """Level the ratio target is expressed against: same-slot mean of the last three weeks."""
    return rows["lag_week_mean"].fillna(rows["lag_all_mean"])


def forecast_day(
    history: pd.DataFrame,
    target: date,
    *,
    weather: pd.DataFrame | None = None,
    weather_hourly: pd.DataFrame | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.9),
    alpha: float | None = None,
    window_days: int = 365,
    model_overrides: dict[str, Any] | None = None,
    target_mode: str = "mw",
    daytype: bool = False,
    estimator: str = "lgbm",
    augment_kind: str | None = None,
    augment_params: dict[str, Any] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Train on `history` (one zone, slots ending at or before the cutoff) and forecast `target`.

    Returns the chronological grid of `target` with ``pred`` (median), one column per
    requested quantile (``p10``, ``p90``, ...) and, when `alpha` is given, ``p_alpha``
    (the α-quantile bid, or the median when α is 0.5).

    `target_mode`: ``"ratio"`` fits ``y / lag_week_mean`` (the same slot's mean over the
    three previous weeks) and rescales the predictions, which lets trees follow level
    shifts they never saw in the window; ``"mw"`` fits the load directly.
    `estimator` picks the boosting library; `augment_kind` / `augment_params` add
    augmented training rows (:mod:`models.augment`), shared by every quantile fit.
    """
    cutoff = cutoff_for(target)
    if history.empty:
        raise ValueError("empty history")
    last_end = history["ts_utc"].max() + SLOT
    if last_end > cutoff:
        raise LeakageError(f"history ends {last_end}, after the cutoff {cutoff} for {target}")

    series = regularize(history)
    repaired, _ = repair(series)
    fields = local_fields(pd.Series(repaired.index, index=repaired.index))
    targets = training_targets(target, window_days, pd.Index(fields["date"].unique()))
    if len(targets) < MIN_TRAIN_DAYS:
        raise ValueError(
            f"only {len(targets)} training days before {target}; need {MIN_TRAIN_DAYS}"
        )

    rows, cols = build_features(repaired, targets, target, weather, weather_hourly, daytype)
    train = rows[rows["y"].notna() & rows["lag_7d"].notna()]
    pred_rows = (
        rows[rows["date"] == pd.Timestamp(target)].sort_values("slot").reset_index(drop=True)
    )
    if train.empty or pred_rows.empty:
        raise ValueError("no usable training rows")
    if target_mode == "ratio":
        denom_train = _ratio_denominator(train)
        denom_pred = _ratio_denominator(pred_rows)
        keep = denom_train.notna() & (denom_train > 0)
        train, denom_train = train[keep], denom_train[keep]
        y: pd.Series | np.ndarray = train["y"] / denom_train
        scale = denom_pred.fillna(denom_pred.mean()).to_numpy()
    elif target_mode == "mw":
        y = train["y"]
        scale = np.ones(len(pred_rows))
    else:
        raise ValueError(f"unknown target_mode {target_mode!r}")
    X, y, weight = augment.apply(augment_kind, train[cols], y, augment_params, seed)
    Xp = pred_rows[cols]
    overrides = dict(model_overrides or {})

    def fit_predict(q: float | None) -> np.ndarray:
        model = get_model(estimator, quantile=q, **overrides)
        return model.fit(X, y, sample_weight=weight).predict(Xp) * scale

    out = pred_rows[["ts_utc", "date", "slot", "tod"]].copy()
    out["pred"] = fit_predict(None)
    fitted: dict[float, np.ndarray] = {}
    for q in quantiles:
        fitted[q] = fit_predict(q)
        out[f"p{int(round(q * 100))}"] = fitted[q]
    if alpha is not None:
        if abs(alpha - 0.5) < 1e-9:
            out["p_alpha"] = out["pred"].to_numpy()
        elif alpha in fitted:
            out["p_alpha"] = fitted[alpha]
        else:
            out["p_alpha"] = fit_predict(float(alpha))
        out["alpha"] = float(alpha)
    # quantile crossing: keep the band ordered around the median
    if {"p10", "p90"} <= set(out.columns):
        lo = np.minimum(out["p10"], out["pred"])
        hi = np.maximum(out["p90"], out["pred"])
        out["p10"], out["p90"] = lo, hi
    return out
