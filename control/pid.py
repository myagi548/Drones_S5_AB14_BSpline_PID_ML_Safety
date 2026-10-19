
class PID:
    """Discrete PID with derivative-on-measurement and integral clamping."""

    def __init__(self, kp, ki, kd, integral_limit=5.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))

        self.integral = 0.0
        self.previous_measurement = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        error = float(setpoint) - float(measurement)
        measurement = float(measurement)

        # Integral term with clamping.
        self.integral += error * dt
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral)
        )

        # Derivative on measurement avoids a derivative kick
        # caused by a sudden setpoint change.
        if self.previous_measurement is None:
            derivative = 0.0
        else:
            derivative = (
                measurement - self.previous_measurement
            ) / dt

        self.previous_measurement = measurement

        return (
            self.kp * error
            + self.ki * self.integral
            - self.kd * derivative
        )
