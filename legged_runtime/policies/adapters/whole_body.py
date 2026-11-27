import mujoco
import numpy as np

from legged_runtime.utils.history_buffer import HistoryBuffer
from legged_runtime.utils.joint_mapping import JointMapping


class WholeBodyPolicyAdapter:

    def __init__(self, model, data, cfg):
        self.model = model
        self.data = data

        self.policy_cfg = cfg["policy"]
        self.obs_cfg = cfg["observations"]
        self.scales = self.obs_cfg["scales"]

        # Joint mapping in Isaac Lab observation order.
        self.obs_mapping = JointMapping(model, self.obs_cfg["joint_names"])

        joint_cfg = cfg["actuators"]["joints"]

        self.default_joint_pos = np.array([
            joint_cfg[name][0] for name in self.obs_mapping.joint_names
        ], dtype=np.float32)

        self.default_joint_vel = np.zeros(len(self.obs_mapping.joint_names), dtype=np.float32)

        # Previous action remains in policy output order.
        self.last_action = np.zeros(self.policy_cfg["action_dim"], dtype=np.float32)

        self.history = HistoryBuffer(
            self.policy_cfg["history_length"],
            self.policy_cfg["single_obs_dim"]
        )

        self.velocity_command = np.asarray(cfg["commands"]["velocity"], dtype=np.float32)
        self.wb_command = np.asarray(cfg["commands"]["whole_body"], dtype=np.float32)

        base_name = cfg["bodies"]["base"]

        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)

        if self.base_id < 0:
            raise ValueError(f"Base body '{base_name}' not found")

    def reset(self):
        self.last_action.fill(0.0)
        self.history.reset()

    def get_base_ang_vel(self):
        velocity = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
            self.base_id, velocity, 1
        )

        return velocity[:3]

    def get_projected_gravity(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)

        gravity = self.model.opt.gravity
        gravity = gravity / np.linalg.norm(gravity)

        return rotation.T @ gravity

    @property
    def torso_height_command(self):
        return float(self.wb_command[-1])

    def set_commands(self, velocity_command, wb_command):
        self.velocity_command[:] = velocity_command
        self.wb_command[:] = wb_command

    def compute_current_observation(self):
        ang_vel = self.get_base_ang_vel()
        projected_gravity = self.get_projected_gravity()

        joint_pos = self.obs_mapping.get_positions(self.data) - self.default_joint_pos
        joint_vel = self.obs_mapping.get_velocities(self.data) - self.default_joint_vel

        obs = np.concatenate([
            ang_vel * self.scales["ang_vel"],
            projected_gravity * self.scales["projected_gravity"],
            self.velocity_command * self.scales["commands"],
            self.wb_command * self.scales["commands"],
            joint_pos * self.scales["joint_pos"],
            joint_vel * self.scales["joint_vel"],
            self.last_action * self.scales["actions"],
        ]).astype(np.float32)

        expected_shape = (self.policy_cfg["single_obs_dim"],)

        if obs.shape != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, got {obs.shape}")

        return obs

    def compute_observation(self):
        obs = self.compute_current_observation()
        self.history.append(obs)

        clip = self.policy_cfg["clip_observations"]

        return np.clip(self.history.get(), -clip, clip)

    def update_action(self, action):
        self.last_action[:] = action

    def get_evaluation_commands(self):
        return {
            "velocity_command": self.velocity_command.copy(),
            "whole_body_command": self.wb_command.copy(),
            "torso_height_command": self.torso_height_command,
        }
