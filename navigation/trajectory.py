
import numpy as np


def detour_reference(t, duration=16.0):
    """
    Generate a smooth reference trajectory for obstacle avoidance.

    Start:  (0, 0, 1.0)
    Climb:   rise gradually while moving forward
    Cruise:  pass above the obstacle
    Descend: return gradually to target altitude
    Goal:    (4, 0, 1.0)

    Returns:
        A NumPy array containing [x, y, z] in metres.
    """

    points = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.7, 0.0, 2.5],
        [2.8, 0.0, 2.5],
        [3.5, 0.0, 1.0],
        [4.0, 0.0, 1.0],
    ], dtype=float)

    # Time assigned to each waypoint, in seconds.
    times = np.array([
        0.0,
        3.0,
        5.0,
        10.0,
        13.0,
        duration,
    ], dtype=float)

    t = float(np.clip(t, 0.0, duration))

    # Identify the current segment.
    i = np.searchsorted(times, t, side="right") - 1
    i = int(np.clip(i, 0, len(times) - 2))

    # Normalised progress through this segment.
    segment_duration = times[i + 1] - times[i]
    u = (t - times[i]) / segment_duration
    u = float(np.clip(u, 0.0, 1.0))

    # Smoothstep interpolation: zero slope at segment endpoints.
    smooth_u = u * u * (3.0 - 2.0 * u)

    # Interpolate x, y and z together.
    reference = (
        points[i]
        + smooth_u * (points[i + 1] - points[i])
    )

    return reference
