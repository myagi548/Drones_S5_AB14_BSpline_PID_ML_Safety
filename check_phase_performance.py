
import csv
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path(__file__).resolve().parent / "results" / "drone_validation.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Validation CSV not found: {CSV_PATH}\n"
        "Run python validate_drone.py first."
    )

groups = defaultdict(list)

with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    required = {
        "phase",
        "target_error_m",
        "approx_clearance_m",
        "roll_deg",
        "pitch_deg",
    }

    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("The CSV does not contain all required columns.")

    for row in reader:
        groups[row["phase"]].append(row)

print("\n===== PERFORMANCE BY FLIGHT PHASE =====")

for phase, rows in groups.items():
    errors = [float(r["target_error_m"]) for r in rows]
    clearances = [float(r["approx_clearance_m"]) for r in rows]
    rolls = [abs(float(r["roll_deg"])) for r in rows]
    pitches = [abs(float(r["pitch_deg"])) for r in rows]

    print(f"\n{phase}")
    print(f"  Samples: {len(rows)}")
    print(f"  Mean target error: {sum(errors) / len(errors):.3f} m")
    print(f"  Maximum target error: {max(errors):.3f} m")
    print(f"  Minimum estimated clearance: {min(clearances):.3f} m")
    print(f"  Maximum absolute roll: {max(rolls):.2f} degrees")
    print(f"  Maximum absolute pitch: {max(pitches):.2f} degrees")

print("\nNote:")
print("- Target changes can temporarily increase tracking error.")
print("- Clearance is approximate, not an exact collision test.")
print("- These results describe this simulation run only.")
