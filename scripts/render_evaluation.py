"""Export a synchronized simulation and diagnostics video from an evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from legged_runtime.evaluation.replay import export_replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Render a recorded evaluation to MP4.")
    parser.add_argument("--input", type=Path, required=True, help="Recorded experiment directory.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--distance", type=float, default=2.8, help="Camera distance in meters.")
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-15.0)
    args = parser.parse_args()

    input_dir = ROOT / args.input
    with open(input_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with np.load(input_dir / "trajectory.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metrics_path = input_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

    output_dir = export_replay(data, cfg, metrics, input_dir, ROOT / cfg["model_path"], width=args.width, height=args.height, fps=args.fps, distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
    print(f"[INFO] Video and thumbnail saved to: {output_dir}")


if __name__ == "__main__":
    main()
