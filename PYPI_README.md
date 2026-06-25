# Project description

[![Feature - Volatility Forecasting](https://img.shields.io/badge/Feature-Volatility%20Forecasting-blue)](https://github.com/statsdl/GaMM)
[![GitHub last commit](https://img.shields.io/github/last-commit/statsdl/GaMM)](https://github.com/statsdl/GaMM/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/statsdl/GaMM)](https://github.com/statsdl/GaMM/issues)
[![GitHub stars](https://img.shields.io/github/stars/statsdl/GaMM)](https://github.com/statsdl/GaMM/stargazers)
[![Python Version](https://img.shields.io/pypi/pyversions/gamm)](https://pypi.org/project/gamm/)
[![License](https://img.shields.io/pypi/l/gamm)](https://github.com/statsdl/GaMM/blob/main/LICENSE)

A rich documentation is available in the GitHub repository.

# GaMM: GARCH-Assisted Neural Heteroscedasticity Model

GaMM is a Python package for high-frequency volatility forecasting and risk assessment using neural networks-based heteroscedasticity modeling.

The package provides utilities for synthetic 5-minute financial price generation, volatility-frame construction, GARCH-family feature extraction, GaMMixer model construction, hyperparameter tuning, and volatility-risk metrics.

GaMM combines econometric volatility features from GARCH-family models with neural feature learning through a mixer-style architecture. This makes it useful for volatility forecasting, high-frequency time-series experiments, and Value-at-Risk based risk assessment.

This package provides two primary components:

* `GaMM` data utilities: tools for converting high-frequency close prices into supervised volatility forecasting datasets.

* `GaMMixer` model utilities: Keras-based model-building helpers with configurable normalization, activation, dropout, feed-forward dimension, and mixer blocks.

# Key Features

* Volatility Forecasting: Designed for high-frequency financial volatility prediction.
* GARCH-Assisted Features: Supports GARCH, GJR-GARCH, ThGARCH, AVGARCH, and FIGARCH volatility features.
* Neural Model Construction: Provides GaMMixer model-building and compilation utilities.
* Hyperparameter Tuning: Includes Hyperopt/TPE-based search utilities.
* Risk Assessment: Includes a normal Value-at-Risk helper for volatility forecasts.
* Forecasting Metrics: Provides RMSE, MAE, MAPE, NMAE, and QLIKE metrics.
* Reproducible Examples: Includes synthetic 5-minute close-price generation and benchmark helpers.

# Installation

Downloading locally and installing

    git clone https://github.com/statsdl/GaMM.git
    cd GaMM

Install dependencies:

    pip install -r requirements.txt

Install the package:

    pip install -e .

Using pip install from GitHub

    pip install git+https://github.com/statsdl/GaMM.git

Using pip install from PyPI

    pip install gamm

Optional full installation

    pip install "gamm[full]"

Development installation

    pip install -e ".[dev]"

# Usage

## 1. Dataset preparation

Example

    from gamm import generate_synthetic_5m_close, make_gamm_dataset, gamm_defaults

    defaults = gamm_defaults()
    prices = generate_synthetic_5m_close(n_days=30, seed=0)

    data = make_gamm_dataset(
        prices,
        lag=int(defaults["lag"]),
        lag_sd=int(defaults["lag_sd"]),
        timestep=int(defaults["timestep"]),
    )

    print(data.X.shape, data.y.shape)

## 2. GaMMixer model

Example

    from gamm.model import compile_gamm_model

    model = compile_gamm_model(
        input_shape=(data.X.shape[1], data.X.shape[2]),
        learning_rate=0.001,
        activation="relu",
        n_block=2,
        dropout=0.1,
        ff_dim=64,
    )

## 3. Synthetic benchmark

Example

    python examples/run_synthetic_benchmark.py

The benchmark compares GaMM with GARCH-family volatility estimates and reports volatility forecasting metrics.

# API Reference

## Data utilities

`generate_synthetic_5m_close`: Generates synthetic 5-minute close prices with volatility clustering.

`volatility_frame`: Computes returns, lagged returns, trailing standard deviation, and realized standard deviation.

`make_gamm_dataset`: Converts high-frequency prices into supervised GaMM forecasting tensors.

`make_gamm_dataset_with_garch_features`: Adds GARCH-family volatility features to the supervised dataset.

`chronological_splits`: Creates chronological train, validation, full-train, and test indexes.

## Model utilities

`build_gamm_model`: Builds a GaMMixer Keras model.

`compile_gamm_model`: Builds and compiles the GaMMixer model with Adam and MSE loss.

`gamm_tuning_space`: Returns the default Hyperopt/TPE search space.

`tune_gamm`: Tunes GaMMixer hyperparameters.

## Metrics

`root_mean_squared_error`: Computes RMSE.

`mean_absolute_error`: Computes MAE.

`mean_absolute_percentage_error`: Computes MAPE.

`normalized_mean_absolute_error`: Computes range-normalized MAE.

`quasi_likelihood`: Computes QLIKE.

`value_at_risk`: Computes normal Value-at-Risk from volatility forecasts.

# Dataset Details

| Component | Description |
|---|---|
| Input data | High-frequency close prices |
| Frequency example | 5-minute observations |
| Target | Realized volatility / standard deviation |
| Optional features | GARCH-family conditional volatilities |
| Split type | Chronological train-validation-test split |
| Metrics | RMSE, MAE, MAPE, NMAE, QLIKE, VaR |

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Citation

If you are using this package in your research, please consider citing the following paper:

High frequency volatility forecasting and risk assessment using neural networks-based heteroscedasticity model: Aryan Bhambu, Koushik Bera, Selvaraju Natarajan, Ponnuthurai Nagaratnam Suganthan.

    @article{bhambu2025highfrequency,
      title={High frequency volatility forecasting and risk assessment using neural networks-based heteroscedasticity model},
      author={Bhambu, Aryan and Bera, Koushik and Natarajan, Selvaraju and Suganthan, Ponnuthurai Nagaratnam},
      journal={Engineering Applications of Artificial Intelligence},
      volume={149},
      pages={110397},
      year={2025},
      doi={10.1016/j.engappai.2025.110397}
    }

# References

Bhambu, A., Bera, K., Natarajan, S., and Suganthan, P. N. (2025). High frequency volatility forecasting and risk assessment using neural networks-based heteroscedasticity model. Engineering Applications of Artificial Intelligence, 149, 110397.
