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

from legged_runtime.evaluation.recorder import TrajectoryRecorder
from legged_runtime.evaluation.state_extractor import MujocoStateExtractor


ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Run a trained policy in MuJoCo.")
parser.add_argument("--config", type=str, default="configs/g1_whole_body.yaml")
parser.add_argument("--vx", type=float, default=None)
parser.add_argument("--vy", type=float, default=None)
parser.add_argument("--wz", type=float, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--record", type=str, default=None, help="Directory for saving evaluation results.")
parser.add_argument("--warmup", type=float, default=None, help="Warmup duration excluded from tracking metrics.")

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

    recorder = None
    state_extractor = None

    if args.record is not None:
        evaluation_cfg = cfg["evaluation"]

        if args.warmup is not None:
            evaluation_cfg["warmup_s"] = args.warmup

        state_extractor = MujocoStateExtractor(env.model, env.data, cfg)
        recorder = TrajectoryRecorder(evaluation_cfg)

    # Override velocity commands using CLI arguments.
    for i, value in enumerate([args.vx, args.vy, args.wz]):
        if value is not None:
            adapter.velocity_command[i] = value

    # Preserve the effective commands in the experiment configuration.
    cfg["commands"]["velocity"] = adapter.velocity_command.tolist()

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
    termination = "duration"
    rollout_start = time.perf_counter()

    try:
        if not args.headless:
            env.render()

        rollout_start = time.perf_counter()

        while env.data.time < args.duration:
            if not args.headless and not env.is_running:
                termination = "viewer_closed"
                break

            start = time.perf_counter()
            observation_time = env.data.time

            # Construct policy observation.
            obs = adapter.compute_observation()

            # Measure inference latency.
            inference_start = time.perf_counter()
            action = policy(obs)
            inference_latency = (time.perf_counter() - inference_start) * 1000.0

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Policy produced non-finite actions")

            target_pos = controller.process_action(
                action,
                scale=cfg["policy"]["action_scale"],
                clip=cfg["policy"]["clip_actions"]
            )

            # Advance simulation.
            controller.apply(target_pos)
            env.step(n_steps=decimation)

            # Refresh derived quantities for the resulting state.
            mujoco.mj_forward(env.model, env.data)

            adapter.update_action(action)

            # Record evaluation data.
            if recorder is not None:
                recorder.record(
                    observation_time=observation_time,
                    observation=obs,
                    action=action,
                    target_pos=target_pos,
                    inference_latency=inference_latency,
                    **adapter.get_evaluation_commands(),
                    **state_extractor.get_state()
                )

            if not args.headless:
                env.render()

            if step % 50 == 0:
                print(
                    f"Time: {env.data.time:.2f} | "
                    f"Height: {env.data.qpos[2]:.3f} | "
                    f"Action norm: {np.linalg.norm(action):.3f}"
                )

            step += 1

            if not args.headless:
                elapsed = time.perf_counter() - start

                if elapsed < policy_dt:
                    time.sleep(policy_dt - elapsed)

    except KeyboardInterrupt:
        termination = "interrupted"
        print("\n[INFO] Simulation interrupted by user.")

    finally:
        rollout_end = time.perf_counter()
        env.close()

    if recorder is not None and recorder.records:
        wall_time = rollout_end - rollout_start
        output_dir = ROOT / args.record

        recorder.save(output_dir, wall_time, termination=termination)

        # Preserve the deployment configuration used for this experiment.
        with open(output_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
