import numpy as np

class PositionKalmanFilter:
    # Constant-velocity state: [x,y,z,vx,vy,vz].
    def __init__(self, dt, process_var=0.05, measurement_var=0.04):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.F = np.block([[np.eye(3), dt*np.eye(3)], [np.zeros((3,3)), np.eye(3)]])
        self.H = np.block([np.eye(3), np.zeros((3,3))])
        self.Q = process_var*np.eye(6)
        self.R = measurement_var*np.eye(3)
        self.initialized = False

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(3)
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        return self.x[:3].copy(), self.x[3:].copy()
