import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from legged_runtime.evaluation.metrics import compute_metrics


class TrajectoryRecorder:

    def __init__(self, cfg):
        self.cfg = cfg
        self.records = defaultdict(list)

    def reset(self):
        self.records.clear()

    def record(self, **sample):
        """Record a single trajectory sample."""

        if not sample:
            raise ValueError("Cannot record an empty sample")

        # Ensure a consistent trajectory schema throughout the rollout.
        if self.records and set(sample) != set(self.records):
            raise ValueError("Trajectory sample keys do not match previous records")

        sample = {key: np.array(value, copy=True) for key, value in sample.items()}

        # Ensure that each recorded quantity retains its shape.
        for key, value in sample.items():
            if key in self.records and value.shape != self.records[key][0].shape:
                raise ValueError(f"Inconsistent shape for trajectory field '{key}'")

        for key, value in sample.items():
            self.records[key].append(value)

    def get_data(self):
        """Convert recorded samples into NumPy arrays."""

        return {key: np.stack(values, axis=0) for key, values in self.records.items()}

    def save(self, output_dir, wall_time, termination="duration"):
        """Save the recorded trajectory and quantitative evaluation results."""

        data = self.get_data()

        if not data:
            print("[WARNING] No trajectory data recorded.")
            return

        metrics = compute_metrics(data, wall_time, self.cfg)
        metrics["termination"] = termination

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        np.savez_compressed(output_dir / "trajectory.npz", **data)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"\n[INFO] Evaluation saved to: {output_dir}")
        print(json.dumps(metrics, indent=4))
