
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "drone_full_demo.xml"
RESULTS_DIR = ROOT / "results"

DURATION = 46.0
SAFETY_THRESHOLD = 0.55

POSITION_KP = 3.4
POSITION_KD = 2.7

NORMAL_ATTITUDE_GAIN = 6.5
NORMAL_DAMPING_GAIN = 2.2
RECOVERY_ATTITUDE_GAIN = 15.0
RECOVERY_DAMPING_GAIN = 4.5

MAX_FORCE = np.array([10.0, 10.0, 22.0])
MIN_FORCE = np.array([-10.0, -10.0, 0.0])
MAX_TORQUE = 8.0

PHASES = [
    (0.0, 3.0, "TAKEOFF"),
    (3.0, 5.0, "HOVER / STABILIZE"),
    (5.0, 8.0, "CLIMB ABOVE OBSTACLES"),
    (8.0, 12.0, "TARGET 1 / NAVIGATION"),
    (12.0, 16.0, "TARGET 2 / NAVIGATION"),
    (16.0, 21.0, "WIND: HOLD POSITION"),
    (21.0, 27.0, "ATTITUDE SAFETY / RECOVERY"),
    (27.0, 34.0, "FINAL TARGET NAVIGATION"),
    (34.0, 46.0, "LAND / TARGET REACHED"),
]

# Cubic B-spline control points.
# These are control points, not individually collision-checked waypoints.
CONTROL_POINTS = np.array([
    [0.0, 0.0, 1.4],
    [0.0, 0.0, 2.4],
    [0.4, 0.0, 3.8],
    [1.0, 0.0, 3.8],
    [2.2, 0.0, 3.8],
    [3.2, 0.0, 3.8],
    [4.5, 0.0, 3.8],
    [6.0, 0.0, 3.8],
    [7.0, 0.0, 3.8],
    [8.2, 0.0, 3.6],
    [9.5, 0.0, 2.6],
    [9.5, 0.0, 1.0],
], dtype=float)

SPLINE_DEGREE = 3
SPLINE_SAMPLES = 1601


def sigmoid(value):
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def train_demo_logistic_regression():
    """Train a demonstration classifier using synthetic data."""
    rng = np.random.default_rng(7)
    count = 2500

    roll = rng.uniform(0.0, 35.0, count)
    pitch = rng.uniform(0.0, 35.0, count)
    rate = rng.uniform(0.0, 3.5, count)

    features = np.column_stack([
        np.ones(count),
        roll / 30.0,
        pitch / 30.0,
        rate / 2.0,
    ])

    labels = (
        (roll > 18.0)
        | (pitch > 18.0)
        | ((roll + pitch > 24.0) & (rate > 1.2))
    ).astype(float)

    weights = np.zeros(4)
    learning_rate = 0.25

    for _ in range(900):
        probabilities = sigmoid(features @ weights)
        gradient = features.T @ (probabilities - labels) / count
        regularization = 0.002 * np.r_[0.0, weights[1:]]
        weights -= learning_rate * (gradient + regularization)

    return weights


def predict_unsafe(weights, roll_deg, pitch_deg, angular_speed):
    features = np.array([
        1.0,
        abs(roll_deg) / 30.0,
        abs(pitch_deg) / 30.0,
        angular_speed / 2.0,
    ])
    return float(sigmoid(features @ weights))


def quat_to_euler(quaternion):
    w, x, y, z = quaternion

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    pitch = np.arcsin(
        np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    )

    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return np.array([roll, pitch, yaw])


def bspline_basis(i, degree, u, knots):
    """Cox-de Boor recursive B-spline basis function."""
    if degree == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0

        if u == knots[-1] and knots[i + 1] == knots[-1]:
            return 1.0

        return 0.0

    left = 0.0
    right = 0.0

    left_denominator = knots[i + degree] - knots[i]

    if left_denominator > 0.0:
        left = (
            (u - knots[i]) / left_denominator
        ) * bspline_basis(i, degree - 1, u, knots)

    right_denominator = knots[i + degree + 1] - knots[i + 1]

    if right_denominator > 0.0:
        right = (
            (knots[i + degree + 1] - u) / right_denominator
        ) * bspline_basis(i + 1, degree - 1, u, knots)

    return left + right


def generate_bspline(control_points, degree=3, sample_count=1601):
    """Generate a clamped B-spline using Cox-de Boor recursion."""
    points = np.asarray(control_points, dtype=float)
    count = len(points)

    if degree < 1 or count <= degree:
        raise ValueError(
            "Control-point count must exceed spline degree."
        )

    interior_count = count - degree - 1

    if interior_count > 0:
        interior = np.linspace(
            0.0, 1.0, interior_count + 2
        )[1:-1]
    else:
        interior = np.array([], dtype=float)

    knots = np.concatenate([
        np.zeros(degree + 1),
        interior,
        np.ones(degree + 1),
    ])

    parameters = np.linspace(0.0, 1.0, sample_count)
    path = np.zeros((sample_count, 3), dtype=float)

    for row, u in enumerate(parameters):
        for i in range(count):
            path[row] += (
                bspline_basis(i, degree, u, knots) * points[i]
            )

    # A clamped spline starts and ends at the first and last control points.
    path[0] = points[0]
    path[-1] = points[-1]

    return parameters, path


def phase_at(t):
    for start, end, name in PHASES:
        if start <= t < end:
            return name

    return PHASES[-1][2]


def target_for_time(t, path):
    """
    Follow the B-spline in stages.
    Hold position during wind and recovery.
    Reach the endpoint by 38 seconds, then hold it.
    """
    if t < 3.0:
        return np.array([0.0, 0.0, 1.4])

    if t < 5.0:
        return np.array([0.0, 0.0, 1.4])

    if t < 8.0:
        progress = (t - 5.0) / 3.0 * 0.20

    elif t < 12.0:
        progress = 0.20 + (t - 8.0) / 4.0 * 0.20

    elif t < 16.0:
        progress = 0.40 + (t - 12.0) / 4.0 * 0.20

    elif t < 21.0:
        progress = 0.60

    elif t < 27.0:
        progress = 0.60

    elif t < 34.0:
        progress = 0.60 + (t - 27.0) / 7.0 * 0.30

    elif t < 38.0:
        progress = 0.90 + (t - 34.0) / 4.0 * 0.10

    else:
        # Hold the final B-spline endpoint for the last 8 seconds.
        progress = 1.0

    index = int(round(
        np.clip(progress, 0.0, 1.0) * (len(path) - 1)
    ))

    return path[index].copy()


def compute_control(model, data, drone_id, t, target, safety_weights):
    """Compute position and attitude control forces and torques."""
    mass = float(model.body_mass[drone_id])
    position = data.xpos[drone_id].copy()

    # This demo assumes the drone's free joint is the first six qvel values.
    velocity = data.qvel[:3].copy()

    rotation = data.xmat[drone_id].reshape(3, 3).copy()
    euler = quat_to_euler(data.xquat[drone_id].copy())
    angular_velocity = data.qvel[3:6].copy()

    roll_deg, pitch_deg = np.degrees(euler[:2])
    angular_speed = float(np.linalg.norm(angular_velocity))

    probability = predict_unsafe(
        safety_weights, roll_deg, pitch_deg, angular_speed
    )

    # Scheduled recovery remains active from 21 to 27 seconds.
    recovering = (
        (probability >= SAFETY_THRESHOLD and 21.0 <= t < 27.0) or 21.0 <= t < 27.0
    )

    kp_attitude = (
        RECOVERY_ATTITUDE_GAIN
        if recovering else NORMAL_ATTITUDE_GAIN
    )
    kd_attitude = (
        RECOVERY_DAMPING_GAIN
        if recovering else NORMAL_DAMPING_GAIN
    )

    acceleration = (
        POSITION_KP * (target - position)
        - POSITION_KD * velocity
    )

    force = mass * (acceleration - model.opt.gravity)

    # Simulated wind disturbance.
    if 16.0 <= t < 21.0:
        force += np.array([0.45, 1.8, 0.0])

    force = np.clip(force, MIN_FORCE, MAX_FORCE)

    torque_local = (
        -kp_attitude * euler
        -kd_attitude * angular_velocity
    )

    # Deliberate short pitch disturbance to demonstrate recovery.
    if 21.0 <= t < 21.55:
        torque_local += np.array([0.0, 4.0, 0.0])

    torque_world = rotation @ torque_local
    torque_world = np.clip(
        torque_world, -MAX_TORQUE, MAX_TORQUE
    )

    data.xfrc_applied[drone_id, :3] = force
    data.xfrc_applied[drone_id, 3:6] = torque_world

    return probability, recovering, position, euler


def save_path(path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "bspline_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["x_m", "y_m", "z_m"])
        writer.writerows(path)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    drone_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone"
    )

    if drone_id < 0:
        raise RuntimeError("Could not find body named 'drone'.")

    _, path = generate_bspline(
        CONTROL_POINTS, SPLINE_DEGREE, SPLINE_SAMPLES
    )

    save_path(path)
    safety_weights = train_demo_logistic_regression()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "drone_flight_log.csv"

    max_tracking_error = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    min_height = float("inf")
    max_height = float("-inf")

    last_phase = ""
    recovery_reported = False
    sim_time = 0.0

    print("\n=== B-SPLINE AUTONOMOUS DRONE DEMO ===")
    print("Blue = drone | Red = obstacles")
    print("Yellow = intermediate targets | Green = final target")
    print("Cubic B-spline: Cox-de Boor basis recursion")
    print("Safety classifier: synthetic demonstration data")
    print("Controller: direct body force/torque, not four-rotor dynamics")
    print("Final target hold begins at 38 seconds.")
    print("Close the viewer window to stop.\n")

    with log_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "time_s", "phase",
            "x_m", "y_m", "z_m",
            "target_x_m", "target_y_m", "target_z_m",
            "tracking_error_m",
            "roll_deg", "pitch_deg",
            "unsafe_probability", "recovery_active",
        ])

        with mujoco.viewer.launch_passive(
            model, data
        ) as viewer:
            viewer.cam.lookat[:] = [4.7, 0.0, 1.8]
            viewer.cam.distance = 13.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -18

            while viewer.is_running() and sim_time < DURATION:
                wall_start = time.perf_counter()

                phase = phase_at(sim_time)
                target = target_for_time(sim_time, path)

                if phase != last_phase:
                    print(f"[{sim_time:5.2f}s] {phase}")
                    last_phase = phase

                probability, recovering, position, euler = (
                    compute_control(
                        model, data, drone_id,
                        sim_time, target, safety_weights
                    )
                )

                error = float(np.linalg.norm(target - position))
                roll_deg, pitch_deg = np.degrees(euler[:2])

                max_tracking_error = max(
                    max_tracking_error, error
                )
                max_roll = max(max_roll, abs(roll_deg))
                max_pitch = max(max_pitch, abs(pitch_deg))
                min_height = min(min_height, position[2])
                max_height = max(max_height, position[2])

                writer.writerow([
                    f"{sim_time:.4f}",
                    phase,
                    *[f"{v:.6f}" for v in position],
                    *[f"{v:.6f}" for v in target],
                    f"{error:.6f}",
                    f"{roll_deg:.4f}",
                    f"{pitch_deg:.4f}",
                    f"{probability:.6f}",
                    int(recovering),
                ])

                if recovering and not recovery_reported:
                    reason = (
                        "ML classifier"
                        if probability >= SAFETY_THRESHOLD
                        else "scheduled recovery phase"
                    )

                    print(
                        f"  Recovery active: {reason}; "
                        f"unsafe probability={probability:.3f}"
                    )
                    recovery_reported = True

                mujoco.mj_step(model, data)
                viewer.sync()
                sim_time = float(data.time)

                remaining = (
                    float(model.opt.timestep)
                    - (time.perf_counter() - wall_start)
                )

                if remaining > 0:
                    time.sleep(remaining)

    print("\n=== DEMO SUMMARY ===")
    print(f"Simulated duration: {sim_time:.2f} s")
    print(f"Maximum tracking error: {max_tracking_error:.3f} m")
    print(f"Maximum absolute roll: {max_roll:.3f} degrees")
    print(f"Maximum absolute pitch: {max_pitch:.3f} degrees")
    print(f"Minimum drone-center height: {min_height:.3f} m")
    print(f"Maximum drone-center height: {max_height:.3f} m")
    print(f"B-spline CSV: {RESULTS_DIR / 'bspline_trajectory.csv'}")
    print(f"Flight log CSV: {log_path}")
    print(
        "Note: B-spline generation does not by itself "
        "prove collision avoidance."
    )


if __name__ == "__main__":
    main()




