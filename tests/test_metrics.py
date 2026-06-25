import numpy as np

from gamm import mean_absolute_error, root_mean_squared_error, value_at_risk


def test_metrics_are_finite():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])

    assert root_mean_squared_error(y_true, y_pred) > 0
    assert mean_absolute_error(y_true, y_pred) > 0


def test_value_at_risk_shape():
    var = value_at_risk(0.05, np.array([0.1, 0.2]))

    assert var.shape == (2,)
    assert np.all(var > 0)
