from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    chronological_splits,
    generate_synthetic_5m_close,
    make_gamm_dataset_with_garch_features,
    gamm_defaults,
)
from .metrics import (
    mean_absolute_error,
    normalized_mean_absolute_error,
    quasi_likelihood,
    root_mean_squared_error,
    value_at_risk,
)
from .model import compile_gamm_model
from .tuning import tune_gamm_with_validation


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    rmse: float
    mae: float
    qlike: float
    nmae: float
    var: float
    best_params: dict[str, Any] | None = None


GARCH_MODEL_COLUMNS = {
    "GARCH": "CV_GARCH",
    "AVGARCH": "CV_AVGARCH",
    "FIGARCH": "CV_FIGARCH",
    "GJR-GARCH": "CV_GJR_GARCH",
    "ThGARCH": "CV_ThGARCH",
}


def run_synthetic_benchmark(
    n_days: int = 30,
    seed: int = 0,
    max_evals: int = 50,
    tuning_epochs: int = 20,
    epochs: int = 100,
    verbose: int = 0,
) -> pd.DataFrame:
    """Run standard GaMM and GARCH baseline evaluation on synthetic 5-min data."""

    defaults = gamm_defaults()
    prices = generate_synthetic_5m_close(n_days=n_days, seed=seed)
    data = make_gamm_dataset_with_garch_features(
        prices,
        lag=int(defaults["lag"]),
        lag_sd=int(defaults["lag_sd"]),
        timestep=int(defaults["timestep"]),
    )
    splits = chronological_splits(len(data.X), 0.7, 0.1, 0.2)

    rows = _garch_baseline_results(data.frame, splits.test, alpha=float(defaults["alpha"]))

    tuned = tune_gamm_with_validation(
        data.X[splits.train],
        data.y[splits.train],
        data.X[splits.validation],
        data.y[splits.validation],
        max_evals=max_evals,
        learning_rate=float(defaults["learning_rate"]),
        epochs=tuning_epochs,
        batch_size=int(defaults["batch_size"]),
        random_state=seed,
        verbose=verbose,
    )
    model = compile_gamm_model(
        input_shape=(data.X.shape[1], data.X.shape[2]),
        learning_rate=float(defaults["learning_rate"]),
        pred_len=1,
        norm_type="B",
        activation=tuned.best_params["activation_function"],
        dropout=tuned.best_params["dropout"],
        n_block=tuned.best_params["n_block"],
        ff_dim=tuned.best_params["ff_dim"],
        target_slice=None,
    )
    model.fit(
        data.X[splits.full_train],
        data.y[splits.full_train],
        epochs=epochs,
        batch_size=int(defaults["batch_size"]),
        verbose=verbose,
    )
    pred = model.predict(data.X[splits.test], verbose=verbose).reshape(-1)
    rows.append(_result("GaMM", data.y[splits.test], pred, float(defaults["alpha"]), tuned.best_params))
    return pd.DataFrame([row.__dict__ for row in rows])


def _garch_baseline_results(frame: pd.DataFrame, test_index: np.ndarray, alpha: float) -> list[BenchmarkResult]:
    y_true = frame["TrueSD"].to_numpy(dtype=float)[test_index]
    rows = []
    for model_name, column in GARCH_MODEL_COLUMNS.items():
        if column in frame:
            rows.append(_result(model_name, y_true, frame[column].to_numpy(dtype=float)[test_index], alpha, None))
    return rows


def _result(model: str, y_true, y_pred, alpha: float, best_params: dict[str, Any] | None) -> BenchmarkResult:
    pred = np.maximum(np.asarray(y_pred, dtype=float).reshape(np.asarray(y_true).shape), 1e-12)
    return BenchmarkResult(
        model=model,
        rmse=root_mean_squared_error(y_true, pred),
        mae=mean_absolute_error(y_true, pred),
        qlike=quasi_likelihood(y_true, pred),
        nmae=normalized_mean_absolute_error(y_true, pred),
        var=float(np.mean(value_at_risk(alpha, pred))),
        best_params=best_params,
    )
