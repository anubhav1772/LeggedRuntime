from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/robots/unitree_g1/scene_white.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

print("\n--- MODEL INFORMATION ---")
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("njnt:", model.njnt)
print("nbody:", model.nbody)

print("\n--- JOINTS ---")

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

    print(
        f"{i:02d} | {name:30s} | "
        f"qpos: {model.jnt_qposadr[i]:2d} | "
        f"qvel: {model.jnt_dofadr[i]:2d}"
    )

print("\n--- ACTUATORS ---")

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

    print(
        f"{i:02d} | {name:30s} | "
        f"target: {model.actuator_trnid[i, 0]:2d} | "
        f"force range: {model.actuator_forcerange[i]}"
    )
