"""Modeling package shared by ``src/`` (offline backtest) and ``app/`` (service).

- :mod:`models.features` - cutoff, DST-aware slot grid, repair, the feature table as of
  the bid cutoff (``FEATURE_VERSION``);
- :mod:`models.tabular` - per-slot gradient boosting (LightGBM default, XGBoost) with
  time-decay weights, :func:`forecast_day`;
- :mod:`models.augment` - swap-noise augmentation of the training rows;
- :mod:`models.tft` - a Temporal Fusion Transformer trained once per refit period over
  all zones (needs ``torch``; imported explicitly, never here).

The public names below are what the rest of the repo imports.
"""

from models.features import (
    AGE_COL,
    FEATURE_VERSION,
    MIN_TRAIN_DAYS,
    SLOT,
    LeakageError,
    build_features,
    cutoff_for,
    day_grid,
    feature_columns,
    is_holiday,
    local_fields,
    local_midnight_utc,
    regularize,
    repair,
    training_targets,
)
from models.tabular import (
    DECAY_HALF_LIFE_DAYS,
    DEFAULT_PARAMS,
    ESTIMATORS,
    DecayWeighted,
    forecast_day,
    get_model,
)

__all__ = [
    "AGE_COL",
    "DECAY_HALF_LIFE_DAYS",
    "DEFAULT_PARAMS",
    "ESTIMATORS",
    "FEATURE_VERSION",
    "MIN_TRAIN_DAYS",
    "SLOT",
    "DecayWeighted",
    "LeakageError",
    "build_features",
    "cutoff_for",
    "day_grid",
    "feature_columns",
    "forecast_day",
    "get_model",
    "is_holiday",
    "local_fields",
    "local_midnight_utc",
    "regularize",
    "repair",
    "training_targets",
]
