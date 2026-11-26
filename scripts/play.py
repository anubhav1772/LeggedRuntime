import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from legged_runtime.envs.mujoco_env import MujocoEnv
from legged_runtime.controllers.position_controller import PositionController
from legged_runtime.policies.onnx_policy import ONNXPolicy
from legged_runtime.policies.adapters.whole_body import WholeBodyPolicyAdapter


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Run a trained policy in MuJoCo.")
parser.add_argument("--config", type=str, default="configs/g1_whole_body.yaml")
parser.add_argument("--vx", type=float, default=None)
parser.add_argument("--vy", type=float, default=None)
parser.add_argument("--wz", type=float, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--duration", type=float, default=20.0)

args = parser.parse_args()


def validate_metadata(cfg, metadata):
    """Validate the configured joint mappings and controller parameters."""

    joint_cfg = cfg["actuators"]["joints"]

    assert cfg["observations"]["joint_names"] == metadata["observation_joint_names"]
    assert list(joint_cfg.keys()) == metadata["action_joint_names"]

    action_ids = np.asarray(metadata["action_joint_ids"], dtype=np.int32)

    params = np.asarray(list(joint_cfg.values()))

    for i, key in enumerate(["default_joint_pos", "stiffness", "damping", "effort_limits"]):
        expected = np.asarray(metadata[key])[action_ids]
        np.testing.assert_allclose(params[:, i], expected, atol=1e-6)

    np.testing.assert_allclose(cfg["simulation"]["dt"], metadata["physics_dt"])

    assert cfg["simulation"]["decimation"] == metadata["decimation"]
    assert cfg["policy"]["history_length"] == metadata["history_length"]
    assert cfg["policy"]["action_scale"] == metadata["action_scale"]
    assert cfg["policy"]["clip_actions"] == metadata["clip_actions"]
    assert cfg["policy"]["clip_observations"] == metadata["clip_observations"]
    assert cfg["observations"]["scales"] == metadata["observation_scales"]
    assert not metadata["action_delay"]["enabled"]

    print("[INFO] Deployment metadata validation passed.")


def main():
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    with open(ROOT / cfg["metadata_path"]) as f:
        metadata = json.load(f)

    validate_metadata(cfg, metadata)

    dt = cfg["simulation"]["dt"]
    decimation = cfg["simulation"]["decimation"]
    policy_dt = dt * decimation

    env = MujocoEnv(ROOT / cfg["model_path"], sim_dt=dt)

    controller = PositionController(env.model, env.data, cfg["actuators"])
    policy = ONNXPolicy(ROOT / cfg["policy_path"])
    adapter = WholeBodyPolicyAdapter(env.model, env.data, cfg)

    # Override velocity commands using CLI arguments.
    for i, value in enumerate([args.vx, args.vy, args.wz]):
        if value is not None:
            adapter.velocity_command[i] = value

    # Initialize robot.
    env.reset()

    env.data.qpos[:3] = cfg["initial_state"]["base_pos"]
    env.data.qpos[3:7] = cfg["initial_state"]["base_quat"]

    controller.reset()
    adapter.reset()

    mujoco.mj_forward(env.model, env.data)

    # Validate policy dimensions.
    assert policy.obs_dim == cfg["policy"]["obs_dim"]
    assert policy.action_dim == cfg["policy"]["action_dim"]

    print("\n--- POLICY INFORMATION ---")
    print("Observation dimension:", policy.obs_dim)
    print("Action dimension:", policy.action_dim)
    print("Physics frequency:", 1.0 / dt)
    print("Policy frequency:", 1.0 / policy_dt)

    print("\n--- JOINT MAPPING ---")
    print("Observation qpos indices:", adapter.obs_mapping.qpos_ids)
    print("Observation qvel indices:", adapter.obs_mapping.qvel_ids)
    print("Action actuator indices:", controller.mapping.actuator_ids)

    # Verify the initial observation.
    obs = adapter.compute_current_observation()

    print("\n--- INITIAL OBSERVATION ---")
    print("Angular velocity:", obs[:3])
    print("Projected gravity:", obs[3:6])
    print("Velocity command:", obs[6:9])
    print("Whole-body command:", obs[9:19])
    print("Joint position error:", obs[19:48])
    print("Joint velocity error:", obs[48:77])
    print("Previous action:", obs[77:106])

    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-5)
    np.testing.assert_allclose(obs[19:77], 0.0, atol=1e-5)

    step = 0

    try:
        if not args.headless:
            env.render()

        while env.data.time < args.duration:
            if not args.headless and not env.is_running:
                break

            start = time.perf_counter()

            # Refresh derived state quantities.
            mujoco.mj_forward(env.model, env.data)

            # Construct stacked policy observation.
            obs = adapter.compute_observation()

            # Run policy inference.
            action = policy(obs)

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Policy produced non-finite actions")

            # Convert policy action to joint position targets.
            target_pos = controller.process_action(
                action,
                scale=cfg["policy"]["action_scale"],
                clip=cfg["policy"]["clip_actions"]
            )

            # Apply targets and advance simulation.
            controller.apply(target_pos)
            env.step(n_steps=decimation)

            # Save raw actions for the next observation.
            adapter.update_action(action)

            if not args.headless:
                env.render()

            if step % 50 == 0:
                print(
                    f"Time: {env.data.time:.2f} | "
                    f"Height: {env.data.qpos[2]:.3f} | "
                    f"Action norm: {np.linalg.norm(action):.3f}"
                )

            step += 1

            # Real-time synchronization for visualization.
            if not args.headless:
                elapsed = time.perf_counter() - start

                if elapsed < policy_dt:
                    time.sleep(policy_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
