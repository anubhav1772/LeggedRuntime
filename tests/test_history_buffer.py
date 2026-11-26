import numpy as np

from legged_runtime.utils.history_buffer import HistoryBuffer


def test_history_buffer():
    history = HistoryBuffer(history_length=3, obs_dim=2)

    history.append([1, 2])
    np.testing.assert_array_equal(history.get(), [1, 2, 1, 2, 1, 2])

    history.append([3, 4])
    np.testing.assert_array_equal(history.get(), [1, 2, 1, 2, 3, 4])

    history.append([5, 6])
    np.testing.assert_array_equal(history.get(), [1, 2, 3, 4, 5, 6])

    history.append([7, 8])
    np.testing.assert_array_equal(history.get(), [3, 4, 5, 6, 7, 8])

    history.reset()
    history.append([9, 10])
    np.testing.assert_array_equal(history.get(), [9, 10, 9, 10, 9, 10])
