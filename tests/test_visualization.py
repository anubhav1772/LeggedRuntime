import numpy as np
import pytest

from legged_runtime.evaluation.visualization import create_figures, export_figures, load_evaluation


def trajectory():
    n = 20
    t = np.arange(1, n + 1) * 0.02
    return {
        "time": t,
        "observation_time": t - 0.02,
        "actual_velocity": np.zeros((n, 3)),
        "velocity_command": np.tile([0.2, 0.0, 0.0], (n, 1)),
        "base_position": np.column_stack([t, t * 0, np.full(n, 0.75)]),
        "base_tilt": np.full(n, 3.0),
        "torso_height": np.full(n, 0.79),
        "torso_height_command": np.full(n, 0.8),
        "inference_latency": np.full(n, 0.15),
    }


def test_full_trajectory(tmp_path):
    data = trajectory()
    np.savez_compressed(tmp_path / "trajectory.npz", **data)
    loaded, metrics = load_evaluation(tmp_path)
    figures = create_figures(loaded, metrics)
    assert set(figures) == {"velocity_tracking", "velocity_error", "base_stability", "torso_height_tracking", "xy_trajectory", "inference_latency"}
    first = export_figures(figures, tmp_path, dpi=60)
    second = export_figures(figures, tmp_path, dpi=60)
    assert first != second
    assert len(list(first.glob("*.png"))) == 6
    assert len(list(second.glob("*.png"))) == 6


def test_optional_fields():
    data = trajectory()
    for key in ("torso_height", "torso_height_command", "velocity_command", "base_tilt"):
        del data[key]
    figures = create_figures(data)
    assert "torso_height_tracking" not in figures
    assert "velocity_error" not in figures
    assert "velocity_tracking" in figures
    assert "base_stability" in figures


def test_malformed_velocity():
    data = trajectory()
    data["actual_velocity"] = np.zeros((20, 2))
    with pytest.raises(ValueError, match="Invalid shape"):
        create_figures(data)
