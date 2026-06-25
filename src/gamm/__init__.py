"""GaMM: GARCH-assisted MLP-Mixer utilities for volatility forecasting."""

from .data import (
    ChronologicalSplits,
    GaMMData,
    chronological_split,
    chronological_splits,
    create_sequence_dataset,
    generate_synthetic_5m_close,
    make_gamm_dataset,
    make_gamm_dataset_with_garch_features,
    gamm_defaults,
    volatility_frame,
)
from .metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    normalized_mean_absolute_error,
    quasi_likelihood,
    root_mean_squared_error,
    value_at_risk,
)

__version__ = "0.1.0"

__all__ = [
    "GaMMData",
    "ChronologicalSplits",
    "chronological_split",
    "chronological_splits",
    "create_sequence_dataset",
    "generate_synthetic_5m_close",
    "make_gamm_dataset",
    "make_gamm_dataset_with_garch_features",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "normalized_mean_absolute_error",
    "gamm_defaults",
    "quasi_likelihood",
    "root_mean_squared_error",
    "value_at_risk",
    "volatility_frame",
]
