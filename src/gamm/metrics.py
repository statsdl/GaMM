from __future__ import annotations

import statistics

import numpy as np


def root_mean_squared_error(y_true, y_pred) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float).reshape(np.asarray(y_true).shape)
    return float(np.sqrt(np.mean(err**2)))


def mean_absolute_error(y_true, y_pred) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float).reshape(np.asarray(y_true).shape)
    return float(np.mean(np.abs(err)))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float).reshape(truth.shape)
    denom = np.maximum(np.abs(truth), 1e-12)
    return float(np.mean(np.abs((truth - pred) / denom)))


def normalized_mean_absolute_error(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float).reshape(truth.shape)
    scale = max(float(truth.max() - truth.min()), 1e-12)
    return float(np.mean(np.abs(truth - pred)) / scale)


def quasi_likelihood(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.maximum(np.asarray(y_pred, dtype=float).reshape(truth.shape), 1e-12)
    variance = pred**2
    return float(np.mean(np.log(variance) + (truth**2 / variance)))


def value_at_risk(alpha: float, volatility_forecast, returns_mean: float = 0.0) -> np.ndarray:
    """Normal VaR used by the GaMM reference workflow."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    quantile = statistics.NormalDist().inv_cdf(alpha)
    volatility = np.asarray(volatility_forecast, dtype=float)
    return -returns_mean - volatility * quantile
