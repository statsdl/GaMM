import numpy as np

import pandas as pd

from gamm import chronological_splits, create_sequence_dataset, generate_synthetic_5m_close, make_gamm_dataset


def test_synthetic_5m_close_shape_and_frequency():
    frame = generate_synthetic_5m_close(n_days=2, seed=0)

    assert frame.shape == (576, 1)
    assert frame.index.freqstr == "5min"
    assert np.isfinite(frame["close"]).all()


def test_make_gamm_dataset_uses_default_shapes():
    prices = generate_synthetic_5m_close(n_days=5, seed=1)
    data = make_gamm_dataset(prices, lag=1, lag_sd=12, timestep=96)

    assert data.X.ndim == 3
    assert data.X.shape[1:] == (96, 3)
    assert data.y.shape[0] == data.X.shape[0]
    assert data.frame.index.equals(data.index)


def test_make_gamm_dataset_accepts_garch_feature_frame():
    prices = generate_synthetic_5m_close(n_days=5, seed=1)
    base = make_gamm_dataset(prices, lag=1, lag_sd=12, timestep=96)
    extra = pd.DataFrame(
        {
            "CV_GARCH": np.linspace(0.1, 0.2, len(base.frame)),
            "CV_ThGARCH": np.linspace(0.2, 0.3, len(base.frame)),
        },
        index=base.frame.index,
    )
    data = make_gamm_dataset(prices, lag=1, lag_sd=12, timestep=96, extra_features=extra)

    assert data.X.shape[2] == 5


def test_chronological_splits_are_70_10_20_and_full_train_is_80():
    splits = chronological_splits(100)

    assert len(splits.train) == 70
    assert len(splits.validation) == 10
    assert len(splits.full_train) == 80
    assert len(splits.test) == 20


def test_create_sequence_dataset_values():
    X, y = create_sequence_dataset(3, np.arange(10).reshape(5, 2), np.arange(5))

    assert X.shape == (3, 3, 2)
    assert y.tolist() == [2.0, 3.0, 4.0]
