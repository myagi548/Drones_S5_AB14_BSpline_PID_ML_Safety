
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Load the existing drone and obstacle model.
MODEL_PATH = Path(__file__).resolve().parent / "drone_model.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

drone_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "drone"
)

if drone_id < 0:
    raise RuntimeError("Drone body not found in drone_model.xml")

mass = model.body_mass[drone_id]
dt = model.opt.timestep

# Fly upward first, pass over the obstacle, then descend to the target.
waypoints = [
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, 2.6]),
    np.array([1.0, 0.0, 2.6]),
    np.array([3.0, 0.0, 2.6]),
    np.array([4.0, 0.0, 2.6]),
    np.array([4.0, 0.0, 1.0]),
]

waypoint_index = 0
simulation_time = 0.0
duration = 30.0

print("Starting MuJoCo 3D drone simulation...")
print("Blue drone | Red obstacle | Green target")
print("Close the viewer window to stop.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        loop_start = time.perf_counter()

        position = data.qpos[:3].copy()
        rotation = data.xmat[drone_id].reshape(3, 3).copy()

        # Convert drone velocity into world coordinates.
        velocity = rotation @ data.qvel[:3].copy()

        # Move to the next waypoint when close enough.
        target = waypoints[waypoint_index]
        distance = np.linalg.norm(target - position)

        if distance < 0.18 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            target = waypoints[waypoint_index]
            print(f"Waypoint {waypoint_index + 1} reached")

        # Position PD controller.
        acceleration = 3.0 * (target - position) - 2.4 * velocity

        force = mass * acceleration
        force[2] += mass * abs(model.opt.gravity[2])

        force = np.clip(
            force,
            [-8.0, -8.0, 0.0],
            [8.0, 8.0, 24.0]
        )

        # Stabilize roll and pitch and damp angular motion.
        q = data.qpos[3:7].copy()
        w, x, y, z = q

        roll = np.arctan2(
            2 * (w * x + y * z),
            1 - 2 * (x * x + y * y)
        )
        pitch = np.arcsin(
            np.clip(2 * (w * y - z * x), -1.0, 1.0)
        )
        yaw = np.arctan2(
            2 * (w * z + x * y),
            1 - 2 * (y * y + z * z)
        )

        angular_velocity = rotation @ data.qvel[3:6].copy()
        attitude_error = rotation @ np.array([roll, pitch, yaw])

        torque = -8.0 * attitude_error - 2.5 * angular_velocity
        torque = np.clip(torque, [-8, -8, -8], [8, 8, 8])

        # Apply forces and step the MuJoCo physics.
        data.xfrc_applied[drone_id, :3] = force
        data.xfrc_applied[drone_id, 3:] = torque

        mujoco.mj_step(model, data)
        simulation_time += dt

        # Keep the viewer updated and approximately real-time.
        viewer.sync()

        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)

        # Once the route is complete, hold the final target.
        if simulation_time >= duration:
            if waypoint_index == len(waypoints) - 1:
                print("Route time completed; keeping viewer open.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                break
            else:
                print("Time limit reached before all waypoints.")
                break

print("MuJoCo viewer closed.")
