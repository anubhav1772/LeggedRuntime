# LeggedRuntime

A modular runtime for sim-to-sim validation, interactive evaluation, and ROS 2 deployment of learned policies for humanoid and quadruped robots.

LeggedRuntime aims to provide a unified interface between robot learning frameworks and lightweight simulation and deployment environments, enabling trained policies to be evaluated independently of their original training infrastructure.

## Features

- MuJoCo-based physics simulation and visualization.
- ONNX-based policy inference.
- Configurable joint position controllers with PD gains and effort limits.
- Robot-independent joint and actuator mapping.
- Policy-specific observation and action adapters.
- Temporal observation history management.
- Isaac Lab deployment metadata validation.
- Velocity-conditioned whole-body policy execution.
- Headless simulation and evaluation.
- YAML-based deployment configurations.

## Supported Robots

| Robot | Policy | Status |
|---|---|---|
| Unitree G1 (29 DoF) | Whole-body control | Supported |
| Unitree AlienGo | Locomotion | Planned |
| Additional humanoids and quadrupeds | — | Planned |

## Architecture
                            LeggedRuntime
                                  │
           ┌──────────────────────┴──────────────────────┐
           │                                             │
    Deployment Inputs                             Command Interface
           │                                             │
    ├── Robot assets                             ├── CLI commands
    ├── Environment assets                       ├── Programmatic commands
    ├── Trained policies                         ├── ROS 2 interface*
    ├── Deployment metadata                     └── High-level planners*
    └── Runtime configuration
           │                                             │
           └──────────────────────┬──────────────────────┘
                                  │
                          Runtime Execution
                                  │
        ┌──────────────┬──────────┴──────────┬──────────────┐
        │              │                     │              │
    Simulation    Policy Adapter      Inference Engine   Actuation
        │              │                     │              │
    ├─ Physics    ├─ Joint mapping    ├─ ONNX Runtime   ├─ PD control
    ├─ State      ├─ Observations     └─ TorchScript*   ├─ Action scaling
    ├─ Contacts   ├─ Command encoding                   ├─ Effort limits
    ├─ Sensors*   └─ Temporal history                   └─ Joint targets
    └─ Rendering
                                  │
                                  ▼
                       Evaluation & Monitoring
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
         Runtime Diagnostics              Performance Evaluation*
                │                                   │
         ├─ Robot state                     ├─ Command tracking
         ├─ Policy outputs                  ├─ Stability and robustness
         └─ Simulation status              ├─ Energy efficiency
                                            └─ Contact and gait analysis

    * Planned functionality

The simulation, policy inference, observation processing, and actuator control components are intentionally decoupled to support different robot embodiments and learned policy architectures.

## Installation

Requirements:

- Python 3.10+
- MuJoCo
- ONNX Runtime
- NumPy
- PyYAML

Install the package in editable mode:

```bash
git clone https://github.com/anubhav1772/LeggedRuntime.git
cd LeggedRuntime

python -m pip install -e .
```

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

The deployment configuration, including joint mappings, actuator parameters, observation settings, and control frequencies, is defined in `configs/g1_whole_body.yaml`.

### Execution Examples

```bash
# Standing and balance
python scripts/play.py --duration 20

# Forward locomotion
python scripts/play.py --vx 0.5

# Lateral locomotion
python scripts/play.py --vy 0.5

# Turning
python scripts/play.py --wz 0.5

# Combined locomotion commands
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
| `--headless` | Disable visualization | — |

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
│   ├── policies/
│   │   └── adapters/
│   ├── robots/
│   └── utils/
├── scripts/
├── tests/
├── pyproject.toml
└── README.md
```

## Current Development Status

- [x] MuJoCo simulation infrastructure
- [x] ONNX policy inference
- [x] Configurable joint position control
- [x] Cross-simulator joint mapping
- [x] Observation history buffer
- [x] Whole-body policy adapter
- [x] Isaac Lab deployment metadata validation
- [x] Closed-loop Unitree G1 policy execution
- [x] Preliminary standing and locomotion validation
- [x] Headless simulation
- [ ] Interactive keyboard teleoperation
- [ ] Quantitative policy evaluation
- [ ] Configurable terrains and dynamic environments
- [ ] TorchScript policy support
- [ ] Multiple robot embodiments
- [ ] ROS 2 integration
- [ ] RViz visualization
- [ ] High-level policy and world-model integration

## Acknowledgements

LeggedRuntime builds upon the following open-source projects:

- [MuJoCo](https://github.com/google-deepmind/mujoco) — physics simulation.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — robot models and assets.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — robot learning and policy training.
- [Ego-VCP](https://github.com/HybridRobotics/Ego-VCP) — pretrained humanoid whole-body policy used for initial sim-to-sim validation.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — neural network inference.

LeggedRuntime is under active development.
