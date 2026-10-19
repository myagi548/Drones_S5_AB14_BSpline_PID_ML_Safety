
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the latest successful flight log
results_dir = "results"
csv_path = os.path.join(results_dir, "drone_flight_log.csv")
df = pd.read_csv(csv_path)

# Check required columns
required = [
    "time_s", "x_m", "y_m", "z_m",
    "target_x_m", "target_y_m", "target_z_m",
    "roll_deg", "pitch_deg"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

os.makedirs(results_dir, exist_ok=True)

# GRAPH 1: Actual trajectory vs desired trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    df["x_m"], df["y_m"], df["z_m"],
    label="Actual drone path", linewidth=2
)
ax.plot(
    df["target_x_m"], df["target_y_m"], df["target_z_m"],
    "--", label="Desired path", linewidth=2
)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("Autonomous Drone: Actual vs Desired Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_trajectory.png"), dpi=200)
plt.close()

# GRAPH 2: Roll and pitch during flight
plt.figure(figsize=(11, 5))
plt.plot(df["time_s"], df["roll_deg"], label="Roll")
plt.plot(df["time_s"], df["pitch_deg"], label="Pitch")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Drone Attitude Stabilization and Recovery")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_attitude.png"), dpi=200)
plt.close()

# GRAPH 3: ML safety prediction, if recorded in the CSV
if "unsafe_probability" in df.columns:
    plt.figure(figsize=(11, 5))
    plt.plot(
        df["time_s"], df["unsafe_probability"],
        label="Predicted unsafe probability"
    )
    plt.axhline(
        0.20, linestyle="--",
        label="Demo trigger threshold (0.20)"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("ML-Based Drone Safety Prediction")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "final_safety.png"), dpi=200)
    plt.close()

print("Final graphs created successfully:")
print(os.path.join(results_dir, "final_trajectory.png"))
print(os.path.join(results_dir, "final_attitude.png"))

if "unsafe_probability" in df.columns:
    print(os.path.join(results_dir, "final_safety.png"))
