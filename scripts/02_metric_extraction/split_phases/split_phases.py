"""
Purpose:
    Identify the windup, release, and follow-through phases in all free throws within a folder.
    Specifically:
        - Windup starts when average velocity of elbow and shoulder flexion exceeds a threshold.
        - Release is the frame with minimum elbow flexion.
        - Follow-through ends 300 ms after release.

Prerequisites:
    - Folder of .mot files from OpenCap
    - Each .mot file contains joint angle data with at least: time, elbow_flex_r, arm_flex_r

Output:
    - A CSV summarizing frame indices per phase for each file: windup_start, release_frame, followthrough_end
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ========================================
# Parameters
# ========================================

ATHLETE = "tests"
SESSION = "player_tracking_tests"  # Change this to switch sessions
FPS = 30  # Frames per second for velocity calculations

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[1]
session_dir = base_dir / "data" / ATHLETE / SESSION

mot_folder = session_dir / "player_tracking_1" / "01_record_data" / "mot_files"
output_csv = session_dir / "player_tracking_1" / "02_process_data" / "time_series" / "freethrow_phases.csv"

# ======================================== 
# Functions 
# ========================================

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

def detect_throw_phases(df, FPS, threshold=10, window=3):
    dt = 1.0 / FPS
    
    # Compute velocities
    df['elbow_vel'] = compute_velocity(df, 'elbow_flex_r', dt).abs()
    df['shoulder_vel'] = compute_velocity(df, 'arm_flex_r', dt).abs()
    df['avg_arm_vel'] = df[['elbow_vel', 'shoulder_vel']].mean(axis=1)

    # Find release frame (min elbow angle)
    release_frame = df['elbow_flex_r'].idxmin()

    # Look backwards to detect windup start
    start = max(0, release_frame - FPS)
    windup_start = start  # fallback
    for i in range(start, release_frame - window + 1):
        recent_window = df['avg_arm_vel'].iloc[i:i + window]
        if (recent_window > threshold).all():
            windup_start = i
            break


    # Follow-through ends 300ms after release
    followthrough_end = min(len(df) - 1, release_frame + int(0.3 * FPS))

    return {
        'windup_start': windup_start,
        'release_frame': release_frame,
        'followthrough_end': followthrough_end
    }

def extract_shot_number(filename):
    import re
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else float('inf')

# ======================================== 
# Main Processing
# ========================================

results = []

print(f"Looking for .mot files in: {mot_folder}")
print(f"Found files: {list(mot_folder.glob('*.mot'))}")


for mot_file in sorted(mot_folder.glob("*.mot"), key=extract_shot_number):
    df = load_mot_file(mot_file)
    phases = detect_throw_phases(df, FPS)
    phases['file'] = mot_file.name
    results.append(phases)

# Save to CSV
output_df = pd.DataFrame(results)
output_csv.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"Saved phase data for {len(results)} files to {output_csv}")
