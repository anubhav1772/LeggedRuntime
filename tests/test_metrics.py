import numpy as np

from legged_runtime.evaluation.metrics import rmse


def test_rmse_zero_error():
    actual = np.array([0.5, 0.5, 0.5])
    target = np.array([0.5, 0.5, 0.5])

    np.testing.assert_allclose(rmse(actual, target), 0.0)


def test_rmse_stationary_robot():
    actual = np.array([0.0, 0.0, 0.0])
    target = np.array([0.2, 0.2, 0.2])

    np.testing.assert_allclose(rmse(actual, target), 0.2)


def test_rmse_multiple_velocity_components():
    actual = np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])

    target = np.array([
        [0.5, 0.2, 0.0],
        [0.5, 0.2, 0.0]
    ])

    np.testing.assert_allclose(rmse(actual, target, axis=0), [0.0, 0.2, 0.0])
