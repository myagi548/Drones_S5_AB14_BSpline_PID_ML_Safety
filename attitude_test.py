
import csv
from pathlib import Path

import mujoco
import numpy as np

DT = 0.01
DURATION = 5.0
MASS = 1.2
GRAVITY = 9.81

root = Path(__file__).resolve().parent
model = mujoco.MjModel.from_xml_path(str(root / "drone_model.xml"))
model.opt.timestep = DT
data = mujoco.MjData(model)

drone_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "drone"
)

joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "root"
)
qvel_start = int(model.jnt_dofadr[joint_id])

rows = []

for step in range(int(DURATION / DT)):
    time = step * DT

    # Apply upward force to balance gravity.
    data.xfrc_applied[drone_id, :3] = [
        0.0, 0.0, MASS * GRAVITY
    ]

    # No external torque in this initial level-attitude test.
    data.xfrc_applied[drone_id, 3:] = 0.0

    mujoco.mj_step(model, data)

    q = data.xquat[drone_id].copy()
    w, x, y, z = q

    roll = np.degrees(np.arctan2(
        2 * (w * x + y * z),
        1 - 2 * (x * x + y * y)
    ))

    pitch = np.degrees(np.arcsin(np.clip(
        2 * (w * y - z * x), -1.0, 1.0
    )))

    rows.append({
        "time": time,
        "x": data.xpos[drone_id, 0],
        "y": data.xpos[drone_id, 1],
        "z": data.xpos[drone_id, 2],
        "roll_deg": roll,
        "pitch_deg": pitch,
        "angular_speed": np.linalg.norm(
            data.qvel[qvel_start + 3:qvel_start + 6]
        )
    })

output = root / "results" / "attitude_test.csv"
output.parent.mkdir(exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Level-attitude diagnostic finished.")
print("Initial position: x=0, y=0, z=1")
print("Final measurements:")
for key, value in rows[-1].items():
    print(f"{key}: {value:.6f}")

print(f"CSV saved to: {output}")
