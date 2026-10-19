# Drone MuJoCo Clean Start

This is a clean modular starting point. It does not overwrite your older project.

## Included in this first baseline
- MuJoCo 6-DoF free-body drone model
- Smooth waypoint trajectory with an elevated detour around a box obstacle
- Position PID and attitude PID
- Optional wind disturbance and noisy position measurements
- Constant-velocity position Kalman filter
- CSV logging and reference-vs-actual plots
- Least-squares linear state-space identification from the logged run

Important limitation: this baseline applies resultant world-frame force and torque directly to the drone body. It is a simplified control abstraction, not a rotor-level quadrotor model. A formal MPC controller and SINDy nonlinear identification are not implemented yet; we should add and validate them as separate next steps instead of claiming they already work.

## 1. Install
Open Anaconda Prompt:
```powershell
conda create -n drone_clean python=3.11 -y
conda activate drone_clean
python -m pip install -r requirements.txt
```

## 2. Run the baseline
Open PowerShell inside the extracted `Drone_MuJoCo_Clean_Start` folder:
```powershell
python main.py
```
To compare without wind:
```powershell
python main.py --no-wind
```
To compare without sensor noise:
```powershell
python main.py --no-noise
```

## 3. Outputs
The program creates:
- `results/drone_log.csv`
- `results/trajectory.png`
- `results/attitude.png`

## 4. Run system identification
After a successful run:
```powershell
python analysis/system_identification.py
```
It fits `x[k+1] = A x[k] + B u[k] + c` using least squares. This is a linear approximation fitted to this run, not a globally valid nonlinear model.

If an error occurs, stop and send the complete terminal error before changing anything.
