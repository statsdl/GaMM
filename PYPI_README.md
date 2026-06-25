[![Feature - Volatility Forecasting](https://img.shields.io/badge/Feature-Volatility%20Forecasting-blue)](https://github.com/statsdl/GaMM)
[![GitHub last commit](https://img.shields.io/github/last-commit/statsdl/GaMM)](https://github.com/statsdl/GaMM/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/statsdl/GaMM)](https://github.com/statsdl/GaMM/issues)
[![GitHub stars](https://img.shields.io/github/stars/statsdl/GaMM)](https://github.com/statsdl/GaMM/stargazers)
[![Python Version](https://img.shields.io/pypi/pyversions/volatility-gamm)](https://pypi.org/project/volatility-gamm/)
[![License](https://img.shields.io/pypi/l/volatility-gamm)](https://github.com/statsdl/GaMM/blob/main/LICENSE)

A rich documentation is available in the GitHub repository.

# volatility-gamm

GARCH-based MLP Mixer for high-frequency volatility forecasting and risk assessment.

volatility-gamm is a Python package for volatility forecasting using GARCH-based MLP Mixer models.

The package combines GARCH-family volatility information with MLP-Mixer style neural feature learning.

This package provides two primary components:

- **GARCH-based volatility features**: Utilities for using GARCH, GJR-GARCH, Threshold GARCH, AVGARCH, and FIGARCH conditional-volatility estimates as model inputs.
- **GaMMixer neural model**: A configurable MLP-Mixer style architecture for volatility forecasting.

## Key Features

- **GARCH-based MLP Mixer**: Combines GARCH-family volatility features with neural MLP-Mixer modeling.
- **High-Frequency Volatility Forecasting**: Designed for intraday financial volatility forecasting tasks.
- **Econometric Feature Integration**: Supports GARCH-family conditional-volatility features.
- **Neural Forecasting Model**: Provides Keras-based GaMMixer model construction and compilation utilities.
- **Hyperparameter Tuning**: Includes Hyperopt/TPE-based tuning helpers.
- **Risk Assessment**: Includes Value-at-Risk utilities.
- **Forecasting Metrics**: Provides RMSE, MAE, MAPE, NMAE, and QLIKE.
- **Research-Oriented Design**: Suitable for volatility forecasting and financial risk modeling.

## Installation

Downloading locally and installing:

```bash
git clone https://github.com/statsdl/GaMM.git
cd GaMM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install -e .
```

Using pip install from GitHub:

```bash
pip install git+https://github.com/statsdl/GaMM.git
```

Using pip install from PyPI:

```bash
pip install volatility-gamm
```

Optional full installation:

```bash
pip install "volatility-gamm[full]"
```

Development installation:

```bash
pip install -e ".[dev]"
```

## Usage

### 1. GaMMixer model

```python
import numpy as np
from gamm.model import compile_gamm_model

X_train = np.random.randn(200, 20, 6).astype("float32")
y_train = np.random.rand(200).astype("float32")

model = compile_gamm_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    learning_rate=0.001,
    activation="relu",
    n_block=2,
    dropout=0.1,
    ff_dim=64,
)

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

predictions = model.predict(X_train[:10], verbose=0)
print("Volatility forecasts:", predictions.reshape(-1))
```

### 2. Volatility forecasting metrics

```python
import numpy as np
from gamm.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    normalized_mean_absolute_error,
    quasi_likelihood,
    value_at_risk,
)

y_true = np.array([0.12, 0.15, 0.09, 0.18])
y_pred = np.array([0.11, 0.14, 0.10, 0.17])

print("RMSE:", root_mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
print("NMAE:", normalized_mean_absolute_error(y_true, y_pred))
print("QLIKE:", quasi_likelihood(y_true, y_pred))
print("VaR:", value_at_risk(0.05, y_pred))
```

### 3. Synthetic benchmark

```bash
python examples/run_synthetic_benchmark.py
```

## API Reference

### Model utilities

- `build_gamm_model`: Builds a GARCH-based MLP Mixer model.
- `compile_gamm_model`: Builds and compiles the GaMMixer model with Adam optimizer and MSE loss.

### Tuning utilities

- `gamm_tuning_space`: Returns the default Hyperopt/TPE search space.
- `tune_gamm`: Tunes GaMMixer hyperparameters.

### GARCH utilities

- `append_garch_features`: Adds GARCH-family conditional-volatility features.

### Metrics

- `root_mean_squared_error`: Computes RMSE.
- `mean_absolute_error`: Computes MAE.
- `mean_absolute_percentage_error`: Computes MAPE.
- `normalized_mean_absolute_error`: Computes range-normalized MAE.
- `quasi_likelihood`: Computes QLIKE.
- `value_at_risk`: Computes normal Value-at-Risk from volatility forecasts.

## Dataset Details

| Component | Description |
|---|---|
| Input type | High-frequency financial time-series features |
| Forecasting target | Volatility / realized volatility |
| Optional features | GARCH-family conditional volatilities |
| Model type | GARCH-based MLP Mixer |
| Split type | Chronological train-validation-test split |
| Metrics | RMSE, MAE, MAPE, NMAE, QLIKE, VaR |

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{bhambu2025highfrequency,
  title={High frequency volatility forecasting and risk assessment using neural networks-based heteroscedasticity model},
  author={Bhambu, Aryan and Bera, Koushik and Natarajan, Selvaraju and Suganthan, Ponnuthurai Nagaratnam},
  journal={Engineering Applications of Artificial Intelligence},
  volume={149},
  pages={110397},
  year={2025},
  doi={10.1016/j.engappai.2025.110397}
}

