import mujoco
import mujoco.viewer
import numpy as np
import time

class MujocoEnv:

    def __init__(self, model_path, sim_dt=0.005):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = sim_dt
        self.viewer = None

        self.reset()

    @property
    def sim_dt(self):
        return self.model.opt.timestep

    @property
    def sim_time(self):
        return self.data.time

    @property
    def is_running(self):
        return self.viewer is not None and self.viewer.is_running()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl=None, n_steps=1):
        if ctrl is not None:
            self.data.ctrl[:] = ctrl

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            time.sleep(0.5)
            self.viewer = None
