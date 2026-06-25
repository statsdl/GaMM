from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaMMData:
    X: np.ndarray
    y: np.ndarray
    index: pd.DatetimeIndex
    frame: pd.DataFrame


@dataclass(frozen=True)
class ChronologicalSplits:
    train: np.ndarray
    validation: np.ndarray
    full_train: np.ndarray
    test: np.ndarray


def gamm_defaults() -> dict[str, float | int]:
    """Return the main data/model defaults used by the GaMM reference workflow."""

    return {
        "lag": 1,
        "lag_sd": 12,
        "timestep": 96,
        "learning_rate": 0.001,
        "epochs": 100,
        "alpha": 0.05,
        "batch_size": 128,
    }


def generate_synthetic_5m_close(
    n_days: int = 30,
    start: str = "2021-01-01",
    seed: int = 0,
    initial_price: float = 100.0,
    periods_per_day: int = 288,
) -> pd.DataFrame:
    """Create synthetic 5-minute close prices with volatility clustering."""

    if n_days <= 0:
        raise ValueError("n_days must be positive.")
    if periods_per_day <= 0:
        raise ValueError("periods_per_day must be positive.")

    rng = np.random.default_rng(seed)
    n_obs = int(n_days * periods_per_day)
    minute_of_day = np.arange(n_obs) % periods_per_day
    intraday = 0.65 + 0.35 * np.cos(2 * np.pi * minute_of_day / periods_per_day) ** 2

    variance = np.empty(n_obs, dtype=float)
    returns = np.empty(n_obs, dtype=float)
    variance[0] = 2.0e-6
    shock = rng.standard_t(df=6, size=n_obs)
    for i in range(n_obs):
        if i > 0:
            variance[i] = 1.0e-7 + 0.08 * returns[i - 1] ** 2 + 0.9 * variance[i - 1]
        returns[i] = np.sqrt(variance[i]) * intraday[i] * shock[i]

    close = initial_price * np.exp(np.cumsum(returns))
    index = pd.date_range(start=start, periods=n_obs, freq="5min")
    return pd.DataFrame({"close": close.astype(float)}, index=index)


def make_gamm_dataset(
    prices: pd.DataFrame,
    lag: int = 1,
    lag_sd: int = 12,
    timestep: int = 96,
    extra_features: pd.DataFrame | None = None,
) -> GaMMData:
    """Build the supervised GaMM volatility dataset from 5-minute close prices."""

    frame = volatility_frame(prices, lag=lag, lag_sd=lag_sd)
    if extra_features is not None:
        aligned = extra_features.reindex(frame.index).ffill().bfill()
        frame = pd.concat([frame, aligned], axis=1).dropna()

    feature_frame = frame.drop(columns=["TrueSD"])
    X_raw = feature_frame.to_numpy(dtype=np.float32)
    X_scaled = _standardize(X_raw)
    y = frame["TrueSD"].to_numpy(dtype=np.float32)
    X, y_seq = create_sequence_dataset(timestep, X_scaled, y)
    return GaMMData(X=X, y=y_seq, index=frame.index[timestep - 1 :], frame=frame.iloc[timestep - 1 :])


def make_gamm_dataset_with_garch_features(
    prices: pd.DataFrame,
    lag: int = 1,
    lag_sd: int = 12,
    timestep: int = 96,
    include_figarch: bool = True,
) -> GaMMData:
    """Build the GaMM dataset with GARCH-family volatility features."""

    from .garch import fit_garch_volatility_features

    frame = volatility_frame(prices, lag=lag, lag_sd=lag_sd)
    garch_features = fit_garch_volatility_features(frame["DailyReturns"], include_figarch=include_figarch)
    return make_gamm_dataset(
        prices,
        lag=lag,
        lag_sd=lag_sd,
        timestep=timestep,
        extra_features=garch_features,
    )


def volatility_frame(prices: pd.DataFrame, lag: int = 1, lag_sd: int = 12) -> pd.DataFrame:
    """Compute returns, lagged returns, trailing SD, and forward realized SD."""

    if "close" not in prices:
        raise ValueError("prices must contain a 'close' column.")
    if lag <= 0 or lag_sd <= 1:
        raise ValueError("lag must be positive and lag_sd must be greater than one.")

    close = prices["close"].astype(float)
    returns = np.log(close.shift(-lag)) - np.log(close)
    daily_returns_old = returns.shift(1)
    sd = returns.rolling(window=lag_sd).std(ddof=1)
    true_sd = returns.rolling(window=lag_sd).std(ddof=1).shift(-(lag_sd - 1))
    data = pd.DataFrame(
        {
            "DailyReturns": returns,
            "SD": sd,
            "TrueSD": true_sd,
            "DailyReturnsOld": daily_returns_old,
        },
        index=prices.index,
    )
    return data.dropna()


def create_sequence_dataset(
    timestep: int,
    X_data: np.ndarray | Iterable[Iterable[float]],
    y_data: np.ndarray | Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create `[sample, timestep, feature]` sequences for GaMMixer training."""

    if timestep <= 0:
        raise ValueError("timestep must be positive.")
    X_arr = np.asarray(X_data, dtype=np.float32)
    y_arr = np.asarray(y_data, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("X_data must be a 2D array.")
    if y_arr.ndim != 1:
        raise ValueError("y_data must be a 1D array.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X_data and y_data must have the same length.")

    n_samples = len(X_arr) - timestep + 1
    if n_samples <= 0:
        raise ValueError("not enough observations for the requested timestep.")
    X = np.zeros((n_samples, timestep, X_arr.shape[1]), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        X[i] = X_arr[i : i + timestep]
        y[i] = y_arr[timestep + i - 1]
    return X, y


def chronological_split(n_samples: int, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    train_len = int(n_samples * train_fraction)
    if train_len <= 0 or train_len >= n_samples:
        raise ValueError("not enough samples for the requested split.")
    return np.arange(train_len), np.arange(train_len, n_samples)


def chronological_splits(
    n_samples: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> ChronologicalSplits:
    """Return 70/10/20 train, validation, full-train, and test indexes."""

    if any(value <= 0 for value in (train_fraction, validation_fraction, test_fraction)):
        raise ValueError("split fractions must be positive.")
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("split fractions must sum to one.")

    train_len = int(n_samples * train_fraction)
    val_len = int(n_samples * validation_fraction)
    test_len = n_samples - train_len - val_len
    if min(train_len, val_len, test_len) <= 0:
        raise ValueError("not enough samples for the requested split.")

    train = np.arange(train_len)
    validation = np.arange(train_len, train_len + val_len)
    full_train = np.arange(train_len + val_len)
    test = np.arange(train_len + val_len, n_samples)
    return ChronologicalSplits(train=train, validation=validation, full_train=full_train, test=test)


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return ((values - mean) / scale).astype(np.float32)
