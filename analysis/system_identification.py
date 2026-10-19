import csv, os
import numpy as np

def identify(path="results/drone_log.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run main.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError("Not enough rows to identify a model.")
    state_cols = ["x","y","z","vx","vy","vz"]
    input_cols = ["fx","fy","fz"]
    X = np.array([[float(r[c]) for c in state_cols] for r in rows])
    U = np.array([[float(r[c]) for c in input_cols] for r in rows])
    Z = np.hstack([X[:-1], U[:-1], np.ones((len(X)-1,1))])
    Y = X[1:]
    theta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    A, B, c = theta[:6].T, theta[6:9].T, theta[9]
    rmse = np.sqrt(np.mean((Y - Z @ theta)**2))
    print("Fitted model: x[k+1] = A x[k] + B u[k] + c")
    print("\nA =\n", np.round(A, 4))
    print("\nB =\n", np.round(B, 4))
    print("\nc =\n", np.round(c, 4))
    print(f"One-step training RMSE: {rmse:.6f}")
    print("This is a fitted linear approximation for the collected run, not a globally valid nonlinear model.")

if __name__ == "__main__":
    identify()
