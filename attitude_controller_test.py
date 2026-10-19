
import csv
from pathlib import Path

import mujoco
import numpy as np

DT = 0.01
DURATION = 5.0
MASS = 1.2
GRAVITY = 9.81

# Test attitude controller gains
KP = np.array([1.0, 1.0, 0.5])
KD = np.array([0.15, 0.15, 0.10])
TORQUE_LIMITS = np.array([1.5, 1.5, 0.8])

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

# Start with a 20-degree pitch disturbance.
angle = np.radians(20.0)
data.qpos[3:7] = [
    np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0
]
mujoco.mj_forward(model, data)

rows = []

for step in range(int(DURATION / DT)):
    time = step * DT

    q = data.xquat[drone_id].copy()
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q

    # Quaternion vector error relative to level orientation.
    attitude_error = 2.0 * q[1:4]

    # Free-joint angular velocity is in body coordinates.
    omega_body = data.qvel[
        qvel_start + 3:qvel_start + 6
    ].copy()

    # Transform both error and angular velocity to world coordinates.
    rotation = data.xmat[drone_id].reshape(3, 3)
    attitude_error_world = rotation @ attitude_error
    omega_world = rotation @ omega_body

    torque_requested = (
        -KP * attitude_error_world - KD * omega_world
    )
    torque = np.clip(
        torque_requested, -TORQUE_LIMITS, TORQUE_LIMITS
    )

    # Balance gravity while testing attitude control.
    data.xfrc_applied[drone_id, :3] = [
        0.0, 0.0, MASS * GRAVITY
    ]
    data.xfrc_applied[drone_id, 3:] = torque

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
        "roll_deg": roll,
        "pitch_deg": pitch,
        "torque_roll": torque[0],
        "torque_pitch": torque[1],
        "angular_speed": np.linalg.norm(omega_body)
    })

    mujoco.mj_step(model, data)

output = root / "results" / "attitude_controller_test.csv"
output.parent.mkdir(exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Attitude controller test finished.")
print("Initial pitch disturbance: 20 degrees")
print("Final measurements:")
for key, value in rows[-1].items():
    print(f"{key}: {value:.6f}")
print(f"CSV saved to: {output}")
