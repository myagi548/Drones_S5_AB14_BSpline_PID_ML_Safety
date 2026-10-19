
import argparse
import csv
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from control.pid import PID
from navigation.trajectory import detour_reference


DT = 0.01
DURATION = 16.0
MASS = 1.2
GRAVITY = 9.81

FORCE_LIMIT_XY = 8.0
FORCE_LIMIT_Z = 24.0
TORQUE_LIMITS = np.array([1.5, 1.5, 0.8])

POSITION_KP = 3.0
POSITION_KI = 0.08
POSITION_KD = 2.2

ATTITUDE_KP = np.array([0.8, 0.8, 0.4])
ATTITUDE_KD = np.array([0.12, 0.12, 0.10])


def quaternion_to_euler_degrees(q):
    w, x, y, z = q

    roll = np.arctan2(
        2 * (w * x + y * z),
        1 - 2 * (x * x + y * y)
    )

    sin_pitch = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

    yaw = np.arctan2(
        2 * (w * z + x * y),
        1 - 2 * (y * y + z * z)
    )

    return np.degrees([roll, pitch, yaw])


def quaternion_attitude_error(q):
    """Quaternion vector error relative to level orientation."""
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)

    if q[0] < 0:
        q = -q

    return 2.0 * q[1:4]


class SimplePositionKalmanFilter:
    def __init__(self, dt):
        self.state = np.zeros(6)
        self.P = np.eye(6)
        self.initialized = False

        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3) * dt

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        self.Q = np.eye(6) * 0.01
        self.R = np.eye(3) * 0.04

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=float)

        if not self.initialized:
            self.state[:3] = measurement
            self.initialized = True
            return (
                self.state[:3].copy(),
                self.state[3:].copy()
            )

        predicted = self.F @ self.state
        predicted_P = self.F @ self.P @ self.F.T + self.Q

        innovation = measurement - self.H @ predicted
        S = self.H @ predicted_P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ predicted_P).T

        self.state = predicted + K @ innovation

        I = np.eye(6)
        A = I - K @ self.H
        self.P = A @ predicted_P @ A.T + K @ self.R @ K.T

        return (
            self.state[:3].copy(),
            self.state[3:].copy()
        )


def run_simulation(wind_on=True, noise_on=True):
    root = Path(__file__).resolve().parent
    xml_path = root / "drone_model.xml"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"MuJoCo XML not found: {xml_path}"
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = DT
    data = mujoco.MjData(model)

    drone_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone"
    )

    if drone_id < 1:
        raise ValueError(
            "Could not find drone body in drone_model.xml"
        )

    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "root"
    )

    if joint_id < 0:
        raise ValueError(
            "Could not find free joint named 'root'"
        )

    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("'root' must be a free joint")

    qvel_start = int(model.jnt_dofadr[joint_id])

    rng = np.random.default_rng(42)

    position_pids = [
        PID(
            POSITION_KP,
            POSITION_KI,
            POSITION_KD,
            2.0
        )
        for _ in range(3)
    ]

    kf = SimplePositionKalmanFilter(DT)
    logs = []
    steps = int(DURATION / DT)

    for step in range(steps):
        t = step * DT

        reference = np.asarray(
            detour_reference(t), dtype=float
        )

        # Read the drone's world position and translational velocity.
        position = data.xpos[drone_id].copy()
        velocity = data.qvel[
            qvel_start:qvel_start + 3
        ].copy()

        # Simulate noisy position measurements if enabled.
        measurement = position.copy()
        if noise_on:
            measurement += rng.normal(
                0.0, 0.02, size=3
            )

        estimated_position, estimated_velocity = kf.update(
            measurement
        )

        # Position PID controller.
        position_acceleration = np.array([
            position_pids[i].update(
                reference[i],
                estimated_position[i],
                DT
            )
            for i in range(3)
        ])

        position_acceleration -= 1.5 * estimated_velocity

        # Convert desired acceleration into world-frame force.
        force = MASS * position_acceleration
        force[2] += MASS * GRAVITY

        force = np.clip(
            force,
            [-FORCE_LIMIT_XY, -FORCE_LIMIT_XY, 0.0],
            [FORCE_LIMIT_XY, FORCE_LIMIT_XY, FORCE_LIMIT_Z]
        )

        # Optional horizontal wind disturbance.
        wind = np.zeros(3)
        if wind_on and 4.0 <= t <= 9.0:
            wind = np.array([0.45, 0.15, 0.0])

        # Attitude feedback: hold level orientation.
        quaternion = data.xquat[drone_id].copy()
        attitude_error = quaternion_attitude_error(
            quaternion
        )

        # Rotation matrix maps body-frame vectors to world frame.
        rotation_matrix = data.xmat[
            drone_id
        ].reshape(3, 3)

        # Express attitude error in world coordinates.
        attitude_error_world = (
            rotation_matrix @ attitude_error
        )

        # Free-joint angular velocity is expressed in body coordinates.
        angular_velocity_body = data.qvel[
            qvel_start + 3:qvel_start + 6
        ].copy()

        # Convert angular velocity into world coordinates.
        angular_velocity_world = (
            rotation_matrix @ angular_velocity_body
        )

        # Calculate corrective torque in world coordinates.
        torque_requested = (
            -ATTITUDE_KP * attitude_error_world
            -ATTITUDE_KD * angular_velocity_world
        )

        torque_saturated = bool(
            np.any(
                np.abs(torque_requested) > TORQUE_LIMITS
            )
        )

        torque = np.clip(
            torque_requested,
            -TORQUE_LIMITS,
            TORQUE_LIMITS
        )

        # Apply world-frame force and torque to the drone body.
        data.xfrc_applied[drone_id, :3] = force + wind
        data.xfrc_applied[drone_id, 3:] = torque

        angles = quaternion_to_euler_degrees(quaternion)

        logs.append({
            "time": t,
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "vx": velocity[0],
            "vy": velocity[1],
            "vz": velocity[2],
            "roll_deg": angles[0],
            "pitch_deg": angles[1],
            "yaw_deg": angles[2],
            "ref_x": reference[0],
            "ref_y": reference[1],
            "ref_z": reference[2],
            "fx": force[0],
            "fy": force[1],
            "fz": force[2],
            "wind_x": wind[0],
            "wind_y": wind[1],
            "wind_z": wind[2],
            "est_x": estimated_position[0],
            "est_y": estimated_position[1],
            "est_z": estimated_position[2],
            "est_vx": estimated_velocity[0],
            "est_vy": estimated_velocity[1],
            "est_vz": estimated_velocity[2],
            "torque_roll": torque[0],
            "torque_pitch": torque[1],
            "torque_yaw": torque[2],
            "torque_saturated": torque_saturated
        })

        mujoco.mj_step(model, data)

    # Calculate position tracking metrics.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    actual = np.array([
        [r["x"], r["y"], r["z"]]
        for r in logs
    ])

    desired = np.array([
        [r["ref_x"], r["ref_y"], r["ref_z"]]
        for r in logs
    ])

    errors = actual - desired

    rmse = float(
        np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
    )

    max_error = float(
        np.max(np.linalg.norm(errors, axis=1))
    )

    saturated_count = sum(
        r["torque_saturated"] for r in logs
    )

    # Save simulation log.
    csv_path = results_dir / f"drone_log_{stamp}.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(logs[0].keys())
        )
        writer.writeheader()
        writer.writerows(logs)

    times = np.array([
        r["time"] for r in logs
    ])

    # Plot desired and actual trajectories.
    trajectory_path = (
        results_dir / f"trajectory_{stamp}.png"
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        *desired.T,
        label="Desired trajectory"
    )
    ax.plot(
        *actual.T,
        label="Actual trajectory"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Drone trajectory tracking")
    ax.legend()

    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=150)
    plt.close(fig)

    # Plot roll, pitch, and yaw.
    attitude_path = (
        results_dir / f"attitude_{stamp}.png"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for name in (
        "roll_deg",
        "pitch_deg",
        "yaw_deg"
    ):
        ax.plot(
            times,
            [r[name] for r in logs],
            label=name
        )

    ax.axhline(
        0, linestyle="--", linewidth=1
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Drone attitude")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(attitude_path, dpi=150)
    plt.close(fig)

    print(f"Position tracking RMSE: {rmse:.4f} m")
    print(f"Maximum position error: {max_error:.4f} m")
    print(
        f"Torque saturated samples: "
        f"{saturated_count}/{len(logs)}"
    )
    print(f"CSV: {csv_path}")
    print(f"Trajectory plot: {trajectory_path}")
    print(f"Attitude plot: {attitude_path}")
    print(f"Wind: {'ON' if wind_on else 'OFF'}")
    print(f"Sensor noise: {'ON' if noise_on else 'OFF'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-wind", action="store_true"
    )
    parser.add_argument(
        "--no-noise", action="store_true"
    )

    args = parser.parse_args()

    run_simulation(
        wind_on=not args.no_wind,
        noise_on=not args.no_noise
    )
