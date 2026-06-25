from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import compile_gamm_model


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    history: list[dict[str, Any]]


def gamm_tuning_space() -> dict[str, Any]:
    """Hyperopt space from the GaMM reference workflow."""

    hp = _hyperopt()[2]
    return {
        "activation_function": hp.choice("activation_function", ["relu", "sigmoid", "tanh", "gelu"]),
        "ff_dim": hp.randint("ff_dim", 20, 201),
        "dropout": hp.uniform("dropout", 0, 0.4),
        "n_block": hp.quniform("n_block", 1, 8, 1),
    }


def tune_gamm(
    X: np.ndarray,
    y: np.ndarray,
    space: dict[str, Any] | None = None,
    max_evals: int = 50,
    validation_fraction: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 20,
    batch_size: int = 128,
    random_state: int = 0,
    verbose: int = 0,
) -> TuningResult:
    """Tune GaMMixer hyperparameters with Hyperopt/TPE."""

    fmin, tpe, _, Trials, space_eval, STATUS_OK = _hyperopt()
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    X_train, X_val, y_train, y_val = _split(X_arr, y_arr, validation_fraction)
    return tune_gamm_with_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        space=space,
        max_evals=max_evals,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        verbose=verbose,
    )


def tune_gamm_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    space: dict[str, Any] | None = None,
    max_evals: int = 50,
    learning_rate: float = 0.001,
    epochs: int = 20,
    batch_size: int = 128,
    random_state: int = 0,
    verbose: int = 0,
) -> TuningResult:
    """Tune GaMMixer with explicit chronological train and validation sets."""

    fmin, tpe, _, Trials, space_eval, STATUS_OK = _hyperopt()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    search_space = gamm_tuning_space() if space is None else space
    history: list[dict[str, Any]] = []

    def objective(params):
        clean = _clean_params(params)
        model = compile_gamm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            learning_rate=learning_rate,
            pred_len=1,
            norm_type="B",
            activation=clean["activation_function"],
            dropout=clean["dropout"],
            n_block=clean["n_block"],
            ff_dim=clean["ff_dim"],
            target_slice=None,
        )
        fit = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        score = float(fit.history["val_loss"][-1])
        history.append({"score": score, **clean})
        return {"loss": score, "status": STATUS_OK}

    trials = Trials()
    best_raw = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        show_progressbar=False,
    )
    best_params = _clean_params(space_eval(search_space, best_raw))
    return TuningResult(
        best_params=best_params,
        best_score=min(item["score"] for item in history),
        history=history,
    )


def _split(X: np.ndarray, y: np.ndarray, validation_fraction: float):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    n_val = max(1, int(round(len(X) * validation_fraction)))
    if n_val >= len(X):
        raise ValueError("not enough samples for the requested validation split.")
    return X[:-n_val], X[-n_val:], y[:-n_val], y[-n_val:]


def _clean_params(params):
    clean = dict(params)
    if "ff_dim" in clean:
        clean["ff_dim"] = int(clean["ff_dim"])
    if "n_block" in clean:
        clean["n_block"] = int(clean["n_block"])
    if "dropout" in clean:
        clean["dropout"] = float(clean["dropout"])
    return clean


def _hyperopt():
    try:
        from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
    except ImportError as exc:
        raise ImportError("Hyperopt is required for tuning. Install with `pip install gamm[tuning]`.") from exc
    return fmin, tpe, hp, Trials, space_eval, STATUS_OK
