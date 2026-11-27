import mujoco
import numpy as np


class MujocoStateExtractor:

    def __init__(self, model, data, cfg):
        self.model = model
        self.data = data

        body_cfg = cfg["bodies"]
        metric_cfg = cfg["evaluation"]["metrics"]

        self.base_id = self.get_body_id(body_cfg["base"])

        self.torso_id = None

        if metric_cfg["torso_height_tracking"]:
            self.torso_id = self.get_body_id(body_cfg["torso"])

    def get_body_id(self, body_name):
        if body_name is None:
            raise ValueError("Required body name is not configured")

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in robot model")

        return body_id

    def get_velocity(self):
        """Return yaw-aligned linear velocity and world-frame yaw angular velocity."""

        velocity = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, velocity, 0)

        angular_velocity_w = velocity[:3]
        linear_velocity_w = velocity[3:]

        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        c, s = np.cos(yaw), np.sin(yaw)

        vx = c * linear_velocity_w[0] + s * linear_velocity_w[1]
        vy = -s * linear_velocity_w[0] + c * linear_velocity_w[1]
        wz = angular_velocity_w[2]

        return np.array([vx, vy, wz])

    def get_base_tilt(self):
        """Return base inclination relative to the gravity-aligned vertical."""

        rotation = self.data.xmat[self.base_id].reshape(3, 3)

        gravity = self.model.opt.gravity
        gravity = gravity / np.linalg.norm(gravity)

        projected_gravity = rotation.T @ gravity

        return float(np.degrees(np.arccos(np.clip(-projected_gravity[2], -1.0, 1.0))))

    def get_state(self):
        """Extract the current physical state."""

        state = {
            "time": float(self.data.time),
            "actual_velocity": self.get_velocity(),
            "base_position": self.data.xpos[self.base_id].copy(),
            "base_quaternion": self.data.xquat[self.base_id].copy(),
            "base_tilt": self.get_base_tilt(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

        if self.torso_id is not None:
            state["torso_height"] = float(self.data.xpos[self.torso_id, 2])

        return state
