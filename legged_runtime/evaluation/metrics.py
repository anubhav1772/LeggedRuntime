import numpy as np


def rmse(actual, target, axis=None):
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    return np.sqrt(np.mean(np.square(actual - target), axis=axis))


def compute_metrics(data, wall_time, cfg):
    """Compute configured evaluation metrics from recorded trajectory data."""

    time = np.asarray(data["time"])

    if len(time) == 0:
        raise ValueError("Cannot evaluate an empty trajectory")

    warmup = cfg["warmup_s"]
    metric_cfg = cfg["metrics"]
    fall_cfg = cfg["fall_detection"]

    start_time = data["observation_time"][0] if "observation_time" in data else 0.0

    elapsed_time = time - start_time
    mask = elapsed_time >= warmup

    duration = float(elapsed_time[-1])

    metrics = {
        "simulation_duration_s": duration,
        "wall_time_s": float(wall_time),
        "num_samples": int(len(time)),
        "num_evaluation_samples": int(mask.sum()),
        "warmup_s": float(warmup),
    }

    # Runtime performance.
    if metric_cfg["real_time_factor"]:
        metrics["real_time_factor"] = float(duration / wall_time) if wall_time > 0 else None

    # Fall detection over the entire rollout.
    if fall_cfg["enabled"]:
        min_height = fall_cfg["min_base_height"]
        max_tilt = fall_cfg["max_base_tilt_deg"]

        fall_mask = (
            (data["base_position"][:, 2] < min_height)
            | (data["base_tilt"] > max_tilt)
        )

        fall_indices = np.flatnonzero(fall_mask)

        metrics["fall_detection"] = {
            "fall_detected": bool(fall_mask.any()),
            "time_to_fall_s": float(time[fall_indices[0]] - start_time) if len(fall_indices) else None,
            "min_base_height_m": float(min_height),
            "max_base_tilt_deg": float(max_tilt),
        }

    # Inference performance over the entire rollout.
    if metric_cfg["inference_latency"]:
        latency = data["inference_latency"]

        metrics["inference_latency"] = {
            "mean_ms": float(np.mean(latency)),
            "std_ms": float(np.std(latency)),
            "p95_ms": float(np.percentile(latency, 95)),
            "max_ms": float(np.max(latency)),
        }

    # No samples available after warmup.
    if not mask.any():
        metrics["evaluation_valid"] = False
        return metrics

    metrics["evaluation_valid"] = True

    # Velocity tracking.
    if metric_cfg["velocity_tracking"]:
        actual = data["actual_velocity"][mask]
        command = data["velocity_command"][mask]

        error = actual - command

        velocity_rmse = rmse(actual, command, axis=0)
        velocity_bias = np.mean(error, axis=0)

        metrics["velocity_tracking"] = {
            "vx_rmse_mps": float(velocity_rmse[0]),
            "vy_rmse_mps": float(velocity_rmse[1]),
            "wz_rmse_radps": float(velocity_rmse[2]),

            "vx_bias_mps": float(velocity_bias[0]),
            "vy_bias_mps": float(velocity_bias[1]),
            "wz_bias_radps": float(velocity_bias[2]),

            "mean_actual_velocity": actual.mean(axis=0).tolist(),
            "mean_commanded_velocity": command.mean(axis=0).tolist(),
        }

    # Base postural stability.
    if metric_cfg["base_stability"]:
        tilt = data["base_tilt"][mask]
        height = data["base_position"][mask, 2]

        metrics["base_stability"] = {
            "mean_tilt_deg": float(np.mean(tilt)),
            "std_tilt_deg": float(np.std(tilt)),
            "max_tilt_deg": float(np.max(tilt)),
            "mean_height_m": float(np.mean(height)),
            "min_height_m": float(np.min(height)),
            "std_height_m": float(np.std(height)),
        }

    # Torso height tracking.
    if metric_cfg["torso_height_tracking"]:
        actual_height = data["torso_height"][mask]
        target_height = data["torso_height_command"][mask]

        metrics["torso_height_tracking"] = {
            "rmse_m": float(rmse(actual_height, target_height)),
            "bias_m": float(np.mean(actual_height - target_height)),
            "mean_actual_height_m": float(np.mean(actual_height)),
            "mean_commanded_height_m": float(np.mean(target_height)),
        }

    return metrics
