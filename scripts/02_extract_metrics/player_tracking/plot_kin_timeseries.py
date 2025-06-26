"""
Title: plot_keypoint_trajectories.py

Purpose:
    Generate and save 2D keypoint time series plots for selected landmarks from MediaPipe CSVs.
    Each plot shows x and y positions over time with windup, release, and follow-through annotations.

Output:
    - PNG plots saved to data/<session>/03_visualize_data/time_series/<csv_filename>_keypoints_plot.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================
# Parameters
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"
CLIP_NAME = ""
KEYPOINTS_TO_PLOT = ["right_wrist", "right_elbow", "right_shoulder"]

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
output_dir = player_tracking_metrics_dir / "timeseries" / "plots" / "kinematics" 

output_dir.mkdir(parents=True, exist_ok=True)

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

    # Get phase frame indices
    windup_start = phase_dict[filename]['windup_start']
    release_frame = phase_dict[filename]['release_frame']
    followthrough_end = phase_dict[filename]['followthrough_end']

    df_slice = df.iloc[windup_start:followthrough_end + 1]
    frame_times = df_slice['frame'] - df_slice['frame'].iloc[0]  # relative time

    # Plot
    plt.figure(figsize=(12, 6))

    for name in KEYPOINTS_TO_PLOT:
        if f"{name}_x" in df.columns and f"{name}_y" in df.columns:
            plt.plot(frame_times, df_slice[f"{name}_x"], label=f"{name} x", linestyle='-')
            plt.plot(frame_times, df_slice[f"{name}_y"], label=f"{name} y", linestyle='--')

    # Add vertical lines for phases (relative to sliced data)
    release_time = release_frame - windup_start
    followthrough_time = followthrough_end - windup_start
    plt.axvline(x=0, color='gray', linestyle='--', label='Windup Start')
    plt.axvline(x=release_time, color='red', linestyle='--', label='Release')
    plt.axvline(x=followthrough_time, color='blue', linestyle='--', label='Follow-Through End')

    plt.title(f"2D Keypoint Trajectories - {csv_file.stem}")
    plt.xlabel("Frame (relative)")
    plt.ylabel("Normalized Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{csv_file.stem}_keypoints_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")
