"""
Title: plot_keypoint_velocity_time_series.py

Purpose:
    Generate and save time series plots of selected 2D keypoint velocities for each free throw.
    Velocities are computed from MediaPipe-tracked 2D positions and visualized between windup and follow-through phases.

Output:
    - PNG plots saved to data/<session>/03_visualize_data/time_series/<csv_filename>_velocity_plot.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================
# Parameters
# ========================================

ATHLETE = "tests"
SESSION = "player_tracking_tests"
CLIP_NAME = ""
KEYPOINTS_TO_PLOT = ["right_wrist", "right_elbow", "right_shoulder"]
FPS = 30

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION
player_tracking_metrics_dir = session_dir / "extracted_metrics" / "player_tracking_metrics"

#inputs 
csv_folder = player_tracking_metrics_dir / "timeseries" / "raw_3d" / f"{CLIP_NAME}_timeseries.csv"
phases_csv = player_tracking_metrics_dir / "phases" / f"{CLIP_NAME}_phases.csv"

#outputs 
output_dir = player_tracking_metrics_dir / "timeseries" / "plots" / "velocity" 

output_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Functions
# ========================================

def compute_velocity(df, column, dt):
    return df[column].diff() / dt

# ========================================
# Load Phase Info
# ========================================

phases_df = pd.read_csv(phases_csv)
phases_df['file'] = phases_df['file'].str.strip().str.lower()
phase_dict = phases_df.set_index('file').to_dict(orient='index')

# ========================================
# Main Loop
# ========================================

for csv_file in csv_folder.glob("*.csv"):
    filename = csv_file.name.strip().lower()
    if filename not in phase_dict:
        print(f"Skipping {filename}: no phase info found.")
        continue

    df = pd.read_csv(csv_file)
    dt = 1.0 / FPS

    # Compute velocities
    for name in KEYPOINTS_TO_PLOT:
        for axis in ["x", "y"]:
            col = f"{name}_{axis}"
            if col in df.columns:
                df[f"{col}_vel"] = compute_velocity(df, col, dt)

    # Get phase frame indices
    windup_start = phase_dict[filename]['windup_start']
    release_frame = phase_dict[filename]['release_frame']
    followthrough_end = phase_dict[filename]['followthrough_end']

    df_slice = df.iloc[windup_start:followthrough_end + 1]
    frame_times = df_slice['frame'] - df_slice['frame'].iloc[0]  # relative frame time

    # Plot velocities
    plt.figure(figsize=(12, 6))
    for name in KEYPOINTS_TO_PLOT:
        for axis in ["x", "y"]:
            col = f"{name}_{axis}_vel"
            if col in df_slice.columns:
                plt.plot(frame_times, df_slice[col], label=f"{name} {axis}-vel")

    # Add vertical lines for phases
    release_time = release_frame - windup_start
    followthrough_time = followthrough_end - windup_start
    plt.axvline(x=0, color='gray', linestyle='--', label='Windup Start')
    plt.axvline(x=release_time, color='red', linestyle='--', label='Release')
    plt.axvline(x=followthrough_time, color='blue', linestyle='--', label='Follow-Through End')

    plt.title(f"2D Keypoint Velocities (Windup to Follow-Through) - {csv_file.stem}")
    plt.xlabel("Frame (relative)")
    plt.ylabel("Velocity (normalized units/frame)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{csv_file.stem}_velocity_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved velocity plot: {output_path}")
