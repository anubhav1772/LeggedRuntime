import numpy as np

from legged_runtime.utils.joint_mapping import JointMapping


class PositionController:

    def __init__(self, model, data, cfg):
        self.model = model
        self.data = data

        joint_cfg = cfg["joints"]

        self.mapping = JointMapping(model, list(joint_cfg.keys()))

        params = np.asarray(list(joint_cfg.values()), dtype=np.float64)

        self.default_joint_pos = params[:, 0]
        self.kp = params[:, 1]
        self.kd = params[:, 2]
        self.effort_limits = params[:, 3]

        self.configure_actuators(cfg["armature"])

    def configure_actuators(self, armature):
        ids = self.mapping.actuator_ids

        # Position actuator:
        # force = kp * ctrl - kp * position - kd * velocity

        self.model.actuator_gainprm[ids, 0] = self.kp

        self.model.actuator_biasprm[ids, 0] = 0.0
        self.model.actuator_biasprm[ids, 1] = -self.kp
        self.model.actuator_biasprm[ids, 2] = -self.kd

        self.model.actuator_forcerange[ids, 0] = -self.effort_limits
        self.model.actuator_forcerange[ids, 1] = self.effort_limits
        self.model.actuator_forcelimited[ids] = True

        self.model.dof_armature[self.mapping.qvel_ids] = armature

    def reset(self):
        self.data.qpos[self.mapping.qpos_ids] = self.default_joint_pos
        self.data.qvel[self.mapping.qvel_ids] = 0.0

        self.apply(self.default_joint_pos)

    def apply(self, target_pos):
        target_pos = np.asarray(target_pos, dtype=np.float64)

        if target_pos.shape != self.default_joint_pos.shape:
            raise ValueError(f"Invalid target shape: {target_pos.shape}")

        self.data.ctrl[self.mapping.actuator_ids] = target_pos

    def process_action(self, action, scale=0.25, clip=100.0):
        action = np.clip(action, -clip, clip)

        return self.default_joint_pos + scale * action
