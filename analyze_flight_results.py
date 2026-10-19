import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "results" / "drone_flight_log.csv"

def main():
    if not LOG_FILE.exists():
        print("ERROR: Flight log not found:", LOG_FILE)
        return

    phases = {}

    with LOG_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("ERROR: CSV header is missing.")
            return

        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        required = [
            "phase", "x_m", "y_m", "z_m",
            "target_x_m", "target_y_m", "target_z_m",
            "tracking_error_m"
        ]

        missing = [name for name in required if name not in reader.fieldnames]

        if missing:
            print("ERROR: Missing columns:", missing)
            print("Found columns:", reader.fieldnames)
            return

        for row in reader:
            try:
                phase = row["phase"].strip()
                record = {
                    "x": float(row["x_m"]),
                    "y": float(row["y_m"]),
                    "z": float(row["z_m"]),
                    "tx": float(row["target_x_m"]),
                    "ty": float(row["target_y_m"]),
                    "tz": float(row["target_z_m"]),
                    "error": float(row["tracking_error_m"])
                }
                if phase:
                    phases.setdefault(phase, []).append(record)
            except (ValueError, TypeError, KeyError):
                continue

    if not phases:
        print("ERROR: No valid flight records found.")
        return

    print("\nDRONE FLIGHT LOG ANALYSIS")
    print("=" * 65)

    for phase, rows in phases.items():
        last = rows[-1]
        max_error = max(r["error"] for r in rows)

        print(f"\nPhase: {phase}")
        print(f"Final position: X={last['x']:.3f}, Y={last['y']:.3f}, Z={last['z']:.3f} m")
        print(f"Final target:   X={last['tx']:.3f}, Y={last['ty']:.3f}, Z={last['tz']:.3f} m")
        print(f"Final tracking error: {last['error']:.3f} m")
        print(f"Maximum phase error:  {max_error:.3f} m")

    print("\nAnalysis completed. Simulation files were not changed.")

if __name__ == "__main__":
    main()
