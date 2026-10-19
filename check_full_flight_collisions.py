
import csv
from pathlib import Path

import mujoco
import numpy as np

from full_drone_demo import (
    DURATION,
    MODEL_PATH,
    RESULTS_DIR,
    SAFETY_THRESHOLD,
    generate_bspline,
    CONTROL_POINTS,
    SPLINE_DEGREE,
    SPLINE_SAMPLES,
    phase_at,
    target_for_time,
    train_demo_logistic_regression,
    compute_control,
)


CONTACT_CSV = RESULTS_DIR / "drone_collision_events.csv"
SUMMARY_FILE = RESULTS_DIR / "flight_validation_summary.txt"


def geom_name(model, geom_id):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
    )
    return name if name else f"geom_{geom_id}"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    drone_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "drone"
    )
    if drone_id < 0:
        raise RuntimeError("Could not find body named 'drone'.")

    drone_geoms = {
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == drone_id
    }
    obstacle_geoms = {
        gid for gid in range(model.ngeom)
        if (geom_name(model, gid).startswith("obstacle_"))
    }
    floor_geoms = {
        gid for gid in range(model.ngeom)
        if geom_name(model, gid) == "floor"
    }

    dt = float(model.opt.timestep)
    steps = int(round(DURATION / dt))
    safety_weights = train_demo_logistic_regression()

    _, path = generate_bspline(
        CONTROL_POINTS, SPLINE_DEGREE, SPLINE_SAMPLES
    )

    # Check the sampled B-spline points against inflated obstacle boxes.
    # This is a center-point clearance test, not a full swept-volume proof.
    obstacle_boxes = []
    for gid in obstacle_geoms:
        center = model.geom_pos[gid].copy()
        half_size = model.geom_size[gid].copy()
        obstacle_boxes.append((center, half_size))

    clearance_margin = 0.45
    unsafe_path_points = []

    for index, point in enumerate(path):
        for center, half_size in obstacle_boxes:
            inside_inflated_box = np.all(
                np.abs(point - center) <= half_size + clearance_margin
            )
            if inside_inflated_box:
                unsafe_path_points.append((index, point.copy(), center.copy()))
                break

    contact_rows = []
    phase_stats = {}
    obstacle_contact_steps = set()
    floor_contact_steps = set()
    penetration_steps = set()
    unsafe_prediction_steps = set()
    recovery_steps = set()

    min_height = float("inf")
    max_height = float("-inf")
    max_tracking_error = 0.0
    max_roll = 0.0
    max_pitch = 0.0

    print("=" * 64)
    print("B-SPLINE FLIGHT COLLISION AND SAFETY VALIDATION")
    print("=" * 64)
    print(f"Model: {MODEL_PATH.name}")
    print(f"Time step: {dt:.4f} s")
    print(f"Duration: {DURATION:.2f} s")
    print(f"Simulation steps: {steps}")
    print(f"Drone geometries: {len(drone_geoms)}")
    print(f"Obstacle geometries: {len(obstacle_geoms)}")
    print(f"Sampled B-spline points: {len(path)}")
    print(f"Path points inside inflated obstacle boxes: {len(unsafe_path_points)}")
    print()

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    for step in range(steps):
        t = float(data.time)
        phase = phase_at(t)
        target = target_for_time(t, path)

        stats = phase_stats.setdefault(phase, {
            "steps": 0,
            "obstacle": set(),
            "floor": set(),
            "penetration": set(),
            "max_error": 0.0,
        })
        stats["steps"] += 1

        probability, recovering, position, euler = compute_control(
            model, data, drone_id, t, target, safety_weights
        )

        error = float(np.linalg.norm(target - position))
        roll_deg, pitch_deg = np.degrees(euler[:2])

        max_tracking_error = max(max_tracking_error, error)
        max_roll = max(max_roll, abs(float(roll_deg)))
        max_pitch = max(max_pitch, abs(float(pitch_deg)))
        min_height = min(min_height, float(position[2]))
        max_height = max(max_height, float(position[2]))
        stats["max_error"] = max(stats["max_error"], error)

        if probability >= SAFETY_THRESHOLD:
            unsafe_prediction_steps.add(step)
        if recovering:
            recovery_steps.add(step)

        mujoco.mj_step(model, data)

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 not in drone_geoms and geom2 not in drone_geoms:
                continue

            other = geom2 if geom1 in drone_geoms else geom1
            is_obstacle = other in obstacle_geoms
            is_floor = other in floor_geoms
            is_penetration = float(contact.dist) < 0.0

            if not (is_obstacle or is_floor):
                continue

            if is_obstacle:
                obstacle_contact_steps.add(step)
                stats["obstacle"].add(step)
            if is_floor:
                floor_contact_steps.add(step)
                stats["floor"].add(step)
            if is_penetration:
                penetration_steps.add(step)
                stats["penetration"].add(step)

            contact_rows.append([
                f"{data.time:.4f}", phase,
                geom_name(model, geom1),
                geom_name(model, geom2),
                f"{float(contact.dist):.6f}",
                int(is_obstacle), int(is_floor), int(is_penetration),
            ])

        data.xfrc_applied[drone_id, :] = 0.0

    with CONTACT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_s", "phase", "geom1", "geom2",
            "contact_distance_m", "obstacle_contact",
            "floor_contact", "penetration",
        ])
        writer.writerows(contact_rows)

    report = [
        "B-SPLINE FLIGHT COLLISION AND SAFETY VALIDATION",
        "=" * 56,
        f"Model: {MODEL_PATH.name}",
        f"Duration: {DURATION:.2f} s",
        f"Steps: {steps}",
        f"Time step: {dt:.4f} s",
        f"Drone geometry count: {len(drone_geoms)}",
        f"Obstacle geometry count: {len(obstacle_geoms)}",
        f"Sampled B-spline points: {len(path)}",
        f"Path points inside inflated obstacle boxes: {len(unsafe_path_points)}",
        f"Recorded contact events: {len(contact_rows)}",
        f"Obstacle-contact steps: {len(obstacle_contact_steps)}",
        f"Floor-contact steps: {len(floor_contact_steps)}",
        f"Penetration steps: {len(penetration_steps)}",
        f"Unsafe prediction steps: {len(unsafe_prediction_steps)}",
        f"Recovery-active steps: {len(recovery_steps)}",
        f"Minimum center height: {min_height:.4f} m",
        f"Maximum center height: {max_height:.4f} m",
        f"Maximum absolute roll: {max_roll:.3f} degrees",
        f"Maximum absolute pitch: {max_pitch:.3f} degrees",
        f"Maximum tracking error: {max_tracking_error:.4f} m",
        "",
        "PHASE RESULTS",
        "-" * 56,
    ]

    for name, stats in phase_stats.items():
        report.extend([
            name,
            f"  Steps: {stats['steps']}",
            f"  Obstacle-contact steps: {len(stats['obstacle'])}",
            f"  Floor-contact steps: {len(stats['floor'])}",
            f"  Penetration steps: {len(stats['penetration'])}",
            f"  Maximum tracking error: {stats['max_error']:.4f} m",
            "",
        ])

    report.extend([
        "CONTACT RESULT: "
        + ("Obstacle contacts detected." if obstacle_contact_steps
           else "No drone-obstacle contacts recorded."),
        "PATH CHECK: "
        + ("WARNING: sampled spline points enter inflated obstacle boxes."
           if unsafe_path_points
           else "No sampled spline points enter the inflated obstacle boxes."),
        "",
        "Limitations:",
        "- The path test samples points and does not prove swept-volume safety.",
        "- Contact validation uses this script's simulated controller replay.",
        "- The classifier is trained on synthetic demonstration data.",
        "- The controller applies direct forces and torques, not rotor dynamics.",
        f"Contact CSV: {CONTACT_CSV}",
    ])

    text = "\n".join(report)
    SUMMARY_FILE.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nContact CSV saved to: {CONTACT_CSV}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
