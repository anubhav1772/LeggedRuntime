# LeggedRuntime

A modular runtime for sim-to-sim validation, quantitative evaluation, diagnostic visualization, and future ROS 2 deployment of learned policies for humanoid and quadruped robots.

LeggedRuntime provides a unified interface between robot learning frameworks and lightweight simulation environments, enabling trained policies to be executed, evaluated, and analyzed independently of their original training infrastructure.

## Features

- MuJoCo-based physics simulation and visualization.
- ONNX-based policy inference.
- Configurable PD joint position control with effort limits.
- Robot-configurable joint and actuator mapping.
- Policy-specific observation and action adapters with temporal observation history.
- Isaac Lab deployment metadata validation.
- Velocity-conditioned whole-body policy execution.
- Headless simulation and YAML-based deployment configurations.
- Configurable quantitative evaluation of velocity tracking, base stability, torso height tracking, fall detection, and runtime performance.
- Trajectory recording with compressed NumPy data, evaluation metrics, and experiment configuration export.
- Interactive, tabbed evaluation viewer with timestamped PNG export.
- Synchronized MuJoCo trajectory replay and diagnostic visualization with MP4 video and thumbnail export, without rerunning policy inference.

## Supported Robots

| Robot | Policy | Status |
|---|---|---|
| Unitree G1 (29 DoF) | Whole-body control | Supported |
| Unitree AlienGo | Locomotion | Planned |
| Additional humanoids and quadrupeds | — | Planned |

## Architecture

```text
                         LeggedRuntime
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
 Deployment Inputs                             Command Interface
        │                                             │
 ├─ Robot assets                               ├─ CLI commands
 ├─ Environment assets                         ├─ Programmatic commands*
 ├─ Trained policies                           ├─ ROS 2 interface*
 ├─ Deployment metadata                        └─ High-level planners*
 └─ Runtime configuration
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                       Runtime Execution
                               │
     ┌──────────────┬──────────┴─────────┬──────────────┐
     │              │                    │              │
 Simulation    Policy Adapter     Inference Engine   Actuation
     │              │                    │              │
 ├─ Physics    ├─ Joint mapping   ├─ ONNX Runtime   ├─ PD control
 ├─ State      ├─ Observations    └─ TorchScript*   ├─ Action scaling
 ├─ Contacts*  ├─ Commands                           ├─ Effort limits
 ├─ Sensors*   └─ History                            └─ Joint targets
 └─ Rendering
                               │
                               ▼
                     Evaluation & Diagnostics
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
 Trajectory Recording   Quantitative Evaluation   Visualization & Replay
        │                      │                      │
 ├─ Robot state         ├─ Velocity tracking      ├─ Interactive GUI
 ├─ Policy outputs      ├─ Base stability         ├─ PNG export
 ├─ Commands            ├─ Height tracking        ├─ Simulation replay
 └─ Timing              ├─ Fall detection         └─ Diagnostic video export
                        └─ Runtime performance

 * Planned functionality (including dedicated contact and sensor interfaces)
```

Simulation, policy inference, observation processing, actuator control, and evaluation are intentionally decoupled to support different robot embodiments and learned policy architectures. Evaluation consumes recorded physical quantities and standardized command fields rather than hardcoded robot joint counts or policy observation indices.

## Installation

### Core runtime

Requirements: Python 3.10+, MuJoCo, ONNX Runtime, NumPy, and PyYAML.

```bash
git clone https://github.com/anubhav1772/LeggedRuntime.git
cd LeggedRuntime
python -m pip install -e .
```

### Visualization and video export

```bash
python -m pip install matplotlib imageio imageio-ffmpeg
```

The interactive evaluation viewer also requires Tkinter. Check its availability with:

```bash
python -m tkinter
```

If needed, install `python3-tk` for Ubuntu system Python or `tk` in a Conda-managed environment. Tkinter is not required for headless plot export or offline video generation.

## Usage

### Standalone Simulation

Launch the MuJoCo simulation without executing a learned policy:

```bash
python scripts/simulate.py
```

### Policy Deployment

Place the exported ONNX policy and its deployment metadata in the corresponding checkpoint directory:

```text
checkpoints/g1/flat/
├── policy.onnx
└── deployment_info.json
```

The deployment configuration, including body names, joint mappings, actuator parameters, observation settings, control frequencies, and evaluation criteria, is defined in `configs/g1_whole_body.yaml`.

### Execution Examples

```bash
# Standing and balance
python scripts/play.py --duration 20

# Forward and lateral locomotion
python scripts/play.py --vx 0.5
python scripts/play.py --vy 0.5

# Turning and combined velocity commands
python scripts/play.py --wz 0.5
python scripts/play.py --vx 0.3 --vy 0.2 --wz 0.3

# Headless execution
python scripts/play.py --headless --duration 20

# Custom deployment configuration
python scripts/play.py --config configs/g1_whole_body.yaml
```

#### Command-Line Arguments

| Argument | Description | Unit |
|---|---|---|
| `--config` | Deployment configuration file | — |
| `--vx` | Forward velocity command | m/s |
| `--vy` | Lateral velocity command | m/s |
| `--wz` | Yaw angular velocity command | rad/s |
| `--duration` | Simulation duration | s |
| `--headless` | Disable the interactive MuJoCo viewer | — |
| `--record` | Directory for trajectory and metric export | — |
| `--warmup` | Override the configured warmup duration excluded from tracking metrics | s |

### Policy Evaluation and Diagnostics

LeggedRuntime supports an offline analysis workflow that separates policy execution from subsequent visualization and replay:

```text
Policy execution → Trajectory recording → Quantitative metrics
                           │
                           ├─ Interactive evaluation viewer / PNG export
                           └─ Synchronized simulation replay / MP4 export
```

#### 1. Record an evaluation

```bash
# Standing baseline
python scripts/play.py --headless --duration 20 --record outputs/evaluations/g1_standing

# Combined locomotion experiment
python scripts/play.py --headless --duration 20 --vx 0.3 --vy 0.2 --wz 0.3 --record outputs/evaluations/g1_combined
```

Each experiment saves `trajectory.npz`, `metrics.json`, and `config.yaml`. Use a new experiment directory for each run to avoid overwriting previous results. The saved configuration includes the effective velocity commands used during execution.

| Evaluation category | Metrics |
|---|---|
| Velocity tracking | Component-wise RMSE, tracking bias, mean actual and commanded velocities |
| Base stability | Tilt and height statistics |
| Whole-body tracking | Torso height RMSE and bias (when applicable) |
| Failure detection | Configurable height and inclination thresholds, time to detected fall |
| Runtime performance | Inference latency statistics and real-time factor |

Metrics and fall criteria are configurable for each robot and task. Tracking metrics exclude the configured warmup interval, while fall detection covers all recorded samples.

#### 2. Interactive visualization and PNG export

```bash
# Open the interactive evaluation viewer and export PNG plots
python scripts/plot_evaluation.py --input outputs/evaluations/g1_combined

# Open the viewer without automatically exporting plots
python scripts/plot_evaluation.py --input outputs/evaluations/g1_combined --no-save

# Export publication-resolution plots without opening a GUI
python scripts/plot_evaluation.py --input outputs/evaluations/g1_combined --headless --dpi 300
```

The viewer provides a metrics summary and interactive tabs for commanded versus actual velocities, tracking errors, base stability, torso height tracking, XY trajectory, and inference latency. Available plots depend on the recorded trajectory fields. The GUI supports zooming, panning, switching experiments, and exporting plots to timestamped folders.

#### 3. Synchronized simulation replay and diagnostic video

Render the recorded robot motion alongside synchronized diagnostic plots, including velocity and torso height tracking, simulation time, control frequency, base tilt, and inference latency when available.

```bash
# Export a 720p diagnostic video for quick inspection
python scripts/render_evaluation.py --input outputs/evaluations/g1_combined --width 1280 --height 720 --fps 30

# Export a 1080p diagnostic video with a custom camera
python scripts/render_evaluation.py --input outputs/evaluations/g1_combined --width 1920 --height 1080 --fps 30 --distance 2.8 --azimuth 135 --elevation -15
```

The replay engine restores recorded generalized positions and velocities, reconstructs robot motion in MuJoCo, and synchronizes the rendered frames with recorded diagnostics. It does not rerun the trained policy or advance physics, and does not require the ONNX checkpoint. The original compatible MuJoCo scene and robot assets must remain available.

| Argument | Description | Default |
|---|---|---|
| `--input` | Recorded experiment directory | Required |
| `--width` | Video width in pixels | 1920 |
| `--height` | Video height in pixels | 1080 |
| `--fps` | Video frame rate | 30 |
| `--distance` | Camera distance in meters | 2.8 |
| `--azimuth` | Camera azimuth in degrees | 135 |
| `--elevation` | Camera elevation in degrees | -15 |

The output is an H.264 MP4 and a PNG thumbnail suitable for research reports, presentations, and supplementary videos. Offscreen rendering requires a working OpenGL context. On compatible headless systems, select EGL for GPU rendering or OSMesa for software rendering, with the necessary system libraries installed:

```bash
MUJOCO_GL=egl python scripts/render_evaluation.py --input outputs/evaluations/g1_combined
```

Replay currently reconstructs recorded generalized states, not the complete simulator state; future dynamic environments may require additional recorded state for faithful reconstruction.

#### Evaluation outputs

```text
outputs/evaluations/g1_combined/
├── config.yaml
├── metrics.json
├── trajectory.npz
├── plots/
│   └── YYYYMMDD_HHMMSS_ffffff/
│       ├── velocity_tracking.png
│       ├── velocity_error.png
│       ├── base_stability.png
│       ├── torso_height_tracking.png
│       ├── xy_trajectory.png
│       └── inference_latency.png
└── videos/
    └── YYYYMMDD_HHMMSS_ffffff/
        ├── evaluation.mp4
        └── thumbnail.png
```

Each plot or video export creates a separate timestamped directory, preserving earlier exports. Generated files depend on the available trajectory fields.

### Validation and Inspection

```bash
# Inspect robot joints and actuators
python scripts/inspect_robot.py

# Inspect ONNX policy inputs and outputs
python scripts/inspect_onnx_policy.py

# Validate nominal posture and position controller
python scripts/test_standing.py

# Run unit tests
python -m pytest tests/ -v
```

## Repository Structure

```text
LeggedRuntime/
├── assets/
│   ├── robots/
│   └── scenes/
├── checkpoints/
├── configs/
├── legged_runtime/
│   ├── controllers/
│   ├── envs/
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── recorder.py
│   │   ├── state_extractor.py
│   │   ├── visualization.py
│   │   └── replay.py
│   ├── policies/
│   │   └── adapters/
│   ├── robots/
│   └── utils/
├── scripts/
│   ├── simulate.py
│   ├── play.py
│   ├── plot_evaluation.py
│   └── render_evaluation.py
├── tests/
├── outputs/                      # Generated evaluation artifacts
├── pyproject.toml
└── README.md
```

## Current Development Status

### Phase 1 — Core Runtime
- [x] MuJoCo simulation infrastructure
- [x] ONNX policy inference
- [x] Configurable joint position control
- [x] Cross-simulator joint and actuator mapping
- [x] Observation history buffer
- [x] Whole-body policy adapter
- [x] Isaac Lab deployment metadata validation
- [x] Closed-loop Unitree G1 policy execution
- [x] Preliminary standing and locomotion validation
- [x] Headless simulation

### Phase 2 — Policy Evaluation
- [x] Quantitative policy evaluation
- [x] Trajectory recording and experiment configuration export
- [x] Interactive performance visualization and PNG export
- [x] Synchronized simulation replay and diagnostic video export
- [ ] Programmatic command scheduling

### Phase 3 — Simulation Environments
- [ ] Configurable terrains and dynamic environments
- [ ] Contact force and sensor interfaces
- [ ] Egocentric camera integration

### Phase 4 — Policy and Robot Generalization
- [ ] TorchScript policy support
- [ ] Multiple robot embodiments and policy adapters

### Phase 5 — Deployment
- [ ] ROS 2 integration
- [ ] RViz visualization

### Phase 6 — Advanced Control
- [ ] Hierarchical policy execution
- [ ] High-level policy and world-model integration
- [ ] Contact-rich whole-body control evaluation

## Acknowledgements

LeggedRuntime builds upon the following open-source projects:

- [MuJoCo](https://github.com/google-deepmind/mujoco) — physics simulation.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — robot models and assets.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — robot learning and policy training.
- [Ego-VCP](https://github.com/HybridRobotics/Ego-VCP) — pretrained humanoid whole-body policy used for initial sim-to-sim validation.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — neural network inference.

LeggedRuntime is under active development.
