"""
Purpose:
    Identify the windup, release, and follow-through phases in a free throw based on kinematic joint velocities.
    Specifically:
        - Windup starts when the average velocity of elbow and shoulder flexion exceeds a threshold.
        - Release is defined as the frame with minimum elbow flexion.
        - Follow-through ends 300 ms after release.

Prerequisites:
    - A .mot file from OpenCap containing joint angle data.

Output:
    - A dictionary with frame indices for 'windup_start', 'release_frame', and 'followthrough_end'
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ======================================== Parameters ========================================
freethrow_path = Path("../data/freethrows1/mot_files/Freethrow1.mot")  # Path to freethrow .mot file

# ======================================== Functions ========================================

def load_mot_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            start_idx = i
            break
    return pd.read_csv(filepath, sep=r'\s+', skiprows=start_idx)

def compute_velocity(df, column_name, dt):
    return df[column_name].diff() / dt

def detect_throw_phases(df, fps=30, threshold=10, window=3):
    dt = 1.0 / fps
    
    # Compute velocities
    df['elbow_vel'] = compute_velocity(df, 'elbow_flex_r', dt).abs()
    df['shoulder_vel'] = compute_velocity(df, 'arm_flex_r', dt).abs()
    df['avg_arm_vel'] = df[['elbow_vel', 'shoulder_vel']].mean(axis=1)

    # Find release frame
    release_frame = df['elbow_flex_r'].idxmin()

    # Look backwards to detect windup start
    start = max(0, release_frame - fps)  # Look up to 1 second back
    for i in range(release_frame, start, -1):
        recent_window = df['avg_arm_vel'].iloc[i - window + 1:i + 1] if i - window + 1 >= 0 else None
        if recent_window is not None and (recent_window > threshold).all():
            windup_start = i - window + 1
            break
    else:
        windup_start = start

    # Follow-through ends 300 ms after release
    followthrough_end = min(len(df) - 1, release_frame + int(0.3 * fps))

    return {
        'windup_start': windup_start,
        'release_frame': release_frame,
        'followthrough_end': followthrough_end
    }

# ======================================== Main Processing ========================================

df = load_mot_file(freethrow_path)
phases = detect_throw_phases(df, fps=30)
print(phases)
