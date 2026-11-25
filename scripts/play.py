import time
from pathlib import Path

from legged_runtime.envs.mujoco_env import MujocoEnv


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "assets/robots/unitree_g1/scene_white.xml"


def main():
    env = MujocoEnv(MODEL_PATH, sim_dt=0.005)

    try:
        env.render()

        while env.is_running:
            start = time.perf_counter()

            env.step()
            env.render()

            elapsed = time.perf_counter() - start
            remaining = env.sim_dt - elapsed

            if remaining > 0:
                time.sleep(remaining)

    finally:
        env.close()


if __name__ == "__main__":
    main()
