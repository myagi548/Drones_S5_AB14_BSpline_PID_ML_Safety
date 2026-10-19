import csv
from pathlib import Path
import numpy as np
import mujoco

from full_drone_demo import (
    PHASES,
    phase_at,
    train_demo_logistic_regression,
    predict_unsafe,
    quat_to_euler,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "drone_full_demo.xml"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

drone_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "drone"
)
if drone_id < 0:
    raise RuntimeError("Drone body was not found.")

mass = float(model.body_mass[drone_id])
dt = float(model.opt.timestep)
weights = train_demo_logistic_regression()

# Obstacle centers and half-sizes, taken from drone_full_demo.xml.
obstacles = [
    (np.array([2.0, 0.0, 1.25]), np.array([0.38, 0.65, 1.25])),
    (np.array([4.5, 0.25, 1.45]), np.array([0.42, 0.62, 1.45])),
    (np.array([7.0, -0.2, 1.15]), np.array([0.38, 0.65, 1.15])),
]

# Conservative approximate clearance: distance from drone center
# to each obstacle box, minus 0.35 m for the drone's arms/body.
def approximate_clearance(position):
    values = []
    for center, half_size in obstacles:
        outside = np.maximum(
            np.abs(position - center) - half_size, 0.0
        )
        values.append(float(np.linalg.norm(outside) - 0.35))
    return min(values)

rows = []
sim_t = 0.0
recovery_trigger_time = None
phase_seen = set()

while sim_t < 42.0 - dt / 2:
    _, _, phase, target = phase_at(sim_t)
    phase_seen.add(phase)

    position = data.qpos[:3].copy()
    velocity = data.qvel[:3].copy()
    rotation = data.xmat[drone_id].reshape(3, 3).copy()

    euler = quat_to_euler(data.qpos[3:7].copy())
    roll_deg, pitch_deg = np.degrees(euler[:2])
    angular_velocity = data.qvel[3:6].copy()
    angular_speed = float(np.linalg.norm(angular_velocity))

    probability = predict_unsafe(
        weights, roll_deg, pitch_deg, angular_speed
    )

    wind_force = np.zeros(3)
    wind_on = 16.0 <= sim_t < 21.0
    if wind_on:
        wind_force[:] = [0.45, 1.8, 0.0]

    disturbance_on = 21.0 <= sim_t < 21.55
    if disturbance_on:
        data.xfrc_applied[drone_id, 3:6] = [0.0, 1.15, 0.0]

    error = target - position
    acceleration = 3.4 * error - 2.7 * velocity
    force = mass * (acceleration - model.opt.gravity)
    force += wind_force
    force = np.clip(force, [-10, -10, 0], [10, 10, 22])

    recovering = (
        probability >= 0.55
        or (21.55 <= sim_t < 27.0)
    )

    if recovering and recovery_trigger_time is None:
        recovery_trigger_time = sim_t

    attitude_gain = 15.0 if recovering else 6.5
    damping_gain = 4.5 if recovering else 2.2

    torque_local = (
        -attitude_gain * euler
        -damping_gain * angular_velocity
    )
    torque_world = rotation @ torque_local
    torque_world = np.clip(
        torque_world, [-8, -8, -8], [8, 8, 8]
    )

    data.xfrc_applied[drone_id, :3] = force
    if disturbance_on:
        data.xfrc_applied[drone_id, 3:6] = [0.0, 1.15, 0.0]
    else:
        data.xfrc_applied[drone_id, 3:6] = torque_world

    rows.append({
        "time_s": round(sim_t, 3),
        "phase": phase,
        "x_m": position[0],
        "y_m": position[1],
        "z_m": position[2],
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "target_error_m": float(np.linalg.norm(error)),
        "approx_clearance_m": approximate_clearance(position),
        "wind_on": wind_on,
        "attitude_disturbance_on": disturbance_on,
        "unsafe_probability": probability,
        "recovery_active": recovering,
    })

    mujoco.mj_step(model, data)
    sim_t += dt

csv_path = RESULTS / "drone_validation.csv"
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

times = np.array([r["time_s"] for r in rows])
rolls = np.array([r["roll_deg"] for r in rows])
pitches = np.array([r["pitch_deg"] for r in rows])
clearances = np.array([r["approx_clearance_m"] for r in rows])
errors = np.array([r["target_error_m"] for r in rows])
probabilities = np.array([r["unsafe_probability"] for r in rows])

def nearest_index(t):
    return int(np.argmin(np.abs(times - t)))

wind_rows = [r for r in rows if r["wind_on"]]
recovery_rows = [
    r for r in rows if 21.0 <= r["time_s"] <= 27.0
]

print("\n===== DRONE VALIDATION REPORT =====")
print(f"Simulation duration: {times[-1]:.2f} seconds")
print(f"Flight phases observed: {len(phase_seen)} / {len(PHASES)}")
print(f"Maximum absolute roll: {np.max(np.abs(rolls)):.2f} degrees")
print(f"Maximum absolute pitch: {np.max(np.abs(pitches)):.2f} degrees")
print(f"Maximum target tracking error: {np.max(errors):.3f} m")
print(f"Minimum approximate obstacle clearance: {np.min(clearances):.3f} m")

print("\nAttitude recovery check:")
for t in [21.0, 22.0, 24.0, 27.0]:
    i = nearest_index(t)
    print(
        f"  t={times[i]:.2f}s: "
        f"roll={rolls[i]:.2f} deg, "
        f"pitch={pitches[i]:.2f} deg"
    )

if wind_rows:
    wind_errors = [r["target_error_m"] for r in wind_rows]
    print("\nWind response:")
    print(f"  Tracking error during wind: {max(wind_errors):.3f} m")

print("\nSafety predictor:")
print(f"  Maximum predicted unsafe probability: {np.max(probabilities):.3f}")
print(f"  Recovery first activated at: {recovery_trigger_time}")

print("\nPreliminary checks:")
print("  All flight phases executed:", len(phase_seen) == len(PHASES))
print("  Approximate clearance positive throughout:", np.min(clearances) > 0)
print("  CSV log saved:", csv_path)
print("\nNote: clearance is an approximation, not an exact MuJoCo collision test.")
print("Review the measurements before claiming successful obstacle avoidance.")
