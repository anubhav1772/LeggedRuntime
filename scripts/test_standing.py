import time
from pathlib import Path

import mujoco
import yaml

from legged_runtime.envs.mujoco_env import MujocoEnv
from legged_runtime.controllers.position_controller import PositionController


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs/ego_vcp_g1.yaml") as f:
    cfg = yaml.safe_load(f)


def main():
    dt = cfg["simulation"]["dt"]
    decimation = cfg["simulation"]["decimation"]

    env = MujocoEnv(ROOT / cfg["model_path"], sim_dt=dt)

    controller = PositionController(env.model, env.data, cfg["actuators"])

    env.reset()

    env.data.qpos[:3] = cfg["initial_state"]["base_pos"]
    env.data.qpos[3:7] = cfg["initial_state"]["base_quat"]

    controller.reset()

    mujoco.mj_forward(env.model, env.data)

    print("Position error:", controller.default_joint_pos - controller.mapping.get_positions(env.data))
    print("Actuator gains:", env.model.actuator_gainprm[controller.mapping.actuator_ids, 0])
    print("Actuator damping:", -env.model.actuator_biasprm[controller.mapping.actuator_ids, 2])
    print("Actuator forces:", env.data.actuator_force)

    print("Joint position indices:", controller.mapping.qpos_ids)
    print("Joint velocity indices:", controller.mapping.qvel_ids)
    print("Actuator indices:", controller.mapping.actuator_ids)
    print("Default joint positions:", controller.default_joint_pos)

    try:
        env.render()

        while env.is_running:
            start = time.perf_counter()

            controller.apply(controller.default_joint_pos)

            env.step(n_steps=decimation)
            env.render()

            elapsed = time.perf_counter() - start
            remaining = dt * decimation - elapsed

            if remaining > 0:
                time.sleep(remaining)

    finally:
        env.close()


if __name__ == "__main__":
    main()
