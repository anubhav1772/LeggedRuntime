import mujoco
import numpy as np


class JointMapping:

    def __init__(self, model, joint_names):
        self.joint_names = list(joint_names)

        self.joint_ids = np.array([
            self._get_joint_id(model, name)
            for name in self.joint_names
        ], dtype=np.int32)

        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = model.jnt_dofadr[self.joint_ids]

        self.actuator_ids = np.array([
            self._get_actuator_id(model, joint_id)
            for joint_id in self.joint_ids
        ], dtype=np.int32)

    @staticmethod
    def _get_joint_id(model, name):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if joint_id < 0:
            raise ValueError(f"Joint not found: {name}")

        return joint_id

    @staticmethod
    def _get_actuator_id(model, joint_id):
        matches = np.flatnonzero(
            (model.actuator_trnid[:, 0] == joint_id)
            & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        )

        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for joint {joint_id}, found {len(matches)}")

        return matches[0]

    def get_positions(self, data):
        return data.qpos[self.qpos_ids].copy()

    def get_velocities(self, data):
        return data.qvel[self.qvel_ids].copy()
