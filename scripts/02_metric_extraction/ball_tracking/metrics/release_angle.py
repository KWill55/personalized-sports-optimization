import math

def compute_release_angle(trajectory, release_frame, window=10):
    """
    Compute angle from first point after release_frame to a second point 'window' frames later.
    """
    if release_frame is None:
        raise ValueError("Invalid release frame")

    # Find pt1: first valid point where idx >= release_frame
    pt1 = None
    idx1 = None
    for (idx, pt) in trajectory:
        if idx >= release_frame and pt is not None:
            pt1 = pt
            idx1 = idx
            break

    if pt1 is None:
        raise ValueError("No valid trajectory point at or after release frame")

    # Find pt2: any next point 'window' frames later
    pt2 = None
    for (idx, pt) in trajectory:
        if idx1 is not None and pt is not None and idx >= idx1 + window:
            pt2 = pt
            break

    if pt2 is None:
        raise ValueError("No valid second trajectory point after release")

    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle_rad = math.atan2(-dy, dx)
    return math.degrees(angle_rad)
