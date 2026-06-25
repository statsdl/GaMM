# GaMM: GARCH-Assisted Neural Heteroscedasticity Model

![GaMM Banner](https://img.shields.io/badge/GaMM-High--Frequency%20Volatility%20Forecasting-blue)

⚡ Fast GaMMixer utilities for high-frequency volatility forecasting and financial risk assessment ⚡

**Features • Installation • Quick Start • Synthetic Benchmark • Citation**

[![Tests](https://github.com/statsdl/GaMM/actions/workflows/test.yml/badge.svg)](https://github.com/statsdl/GaMM/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/gamm)](https://pypi.org/project/gamm/)
[![Python Version](https://img.shields.io/pypi/pyversions/gamm)](https://pypi.org/project/gamm/)
[![License](https://img.shields.io/pypi/l/gamm)](https://github.com/statsdl/GaMM/blob/main/LICENSE)

## 🎯 Use Cases & Applications

### 📈 Financial Volatility Forecasting

- High-frequency volatility prediction
- Realized-volatility modeling
- Volatility clustering analysis

### ⚠️ Risk Management

- Value-at-Risk estimation
- Short-horizon financial risk assessment
- Market uncertainty monitoring

### 🔬 Research Experiments

- GARCH-assisted neural forecasting
- Neural heteroscedasticity modeling
- Synthetic 5-minute price benchmarks

GaMM provides a clean Python package for GARCH-assisted high-frequency volatility forecasting. It combines GARCH-family volatility features with a GaMMixer neural architecture for regression-style volatility prediction and risk evaluation.

The package is designed for reproducible experiments, lightweight imports, optional model-training extras, and clean integration into forecasting workflows.

## 🚀 Key Features

### 📊 Forecasting Workflow

- ✅ Synthetic 5-minute close-price generation
- ✅ Volatility-frame construction from high-frequency prices
- ✅ Chronological train, validation, and test splitting
- ✅ GARCH-family feature integration

### 🧠 Neural Model Utilities

- ✅ GaMMixer model construction with lazy TensorFlow imports
- ✅ Batch-normalization and layer-normalization options
- ✅ Dropout, feed-forward dimension, and block-count configuration
- ✅ Hyperopt/TPE tuning utilities

### 📉 Risk and Error Metrics

- ✅ RMSE, MAE, MAPE, NMAE, and QLIKE
- ✅ Normal Value-at-Risk helper
- ✅ Benchmark utilities for GARCH-family comparisons

## 📦 Installation

Install from PyPI:

```bash
pip install gamm
```

Install from GitHub:

```bash
pip install git+https://github.com/statsdl/GaMM.git
```

Install locally for development:

```bash
git clone https://github.com/statsdl/GaMM.git
cd GaMM
pip install -e ".[dev]"
pytest
```

Optional model-training dependencies:

```bash
pip install "gamm[full]"
```

## ⚡ Quick Start

Create a synthetic high-frequency volatility dataset:

```python
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
```

Build a Keras GaMMixer model:

```python
from gamm.model import compile_gamm_model

model = compile_gamm_model(
    input_shape=(data.X.shape[1], data.X.shape[2]),
    learning_rate=0.001,
    activation="relu",
    n_block=2,
    dropout=0.1,
    ff_dim=64,
)
```

## 🔬 Synthetic Benchmark

Install full dependencies:

```bash
pip install "gamm[full]"
```

Run the synthetic benchmark:

```bash
python examples/run_synthetic_benchmark.py
```

The benchmark reports results for GARCH, AVGARCH, FIGARCH, GJR-GARCH, ThGARCH, and GaMM.

## 📚 Main API

### Data utilities

- `generate_synthetic_5m_close`
- `volatility_frame`
- `make_gamm_dataset`
- `make_gamm_dataset_with_garch_features`
- `chronological_splits`
- `gamm_defaults`

### Model utilities

- `build_gamm_model`
- `compile_gamm_model`
- `gamm_tuning_space`
- `tune_gamm`

### Metrics

- `root_mean_squared_error`
- `mean_absolute_error`
- `mean_absolute_percentage_error`
- `normalized_mean_absolute_error`
- `quasi_likelihood`
- `value_at_risk`

## 🛠️ Development

```bash
git clone https://github.com/statsdl/GaMM.git
cd GaMM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📚 Citation

If you use GaMM in your research, please cite:

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
```

## 🏆 Acknowledgments

GaMM builds on the Python scientific computing ecosystem, including NumPy, Pandas, TensorFlow/Keras, Hyperopt, and ARCH.

⭐ Star the repository if GaMM helps your volatility forecasting research.
