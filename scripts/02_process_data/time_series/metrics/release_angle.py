import math

def compute_release_angle(trajectory, points_to_use=8):
    """
    Estimate release angle in degrees from the first upward movement in trajectory.
    Uses first N trajectory points.

    Parameters:
        trajectory (list of (x, y)): Full list of ball center points.
        points_to_use (int): How many points to use for estimating direction.

    Returns:
        float or None: Release angle in degrees, or None if insufficient data.
    """
    if len(trajectory) < 2:
        return None

    # Use only the first N points
    points = trajectory[:min(points_to_use, len(trajectory))]

    # Linear fit (least squares) to estimate slope
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    n = len(points)
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    numerator = sum((x_vals[i] - mean_x) * (y_vals[i] - mean_y) for i in range(n))
    denominator = sum((x_vals[i] - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return None  # Vertical line or bad data

    slope = numerator / denominator  # rise over run

    # Convert slope to angle in degrees
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 2)
