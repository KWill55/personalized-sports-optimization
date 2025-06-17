"""
Title: plot_time_series.py

Purpose:
    Generate and save time series plots of selected joint angles for each free throw.
    Each plot visualizes joint angles over time from OpenCap .mot files, starting at windup and ending after follow-through,
    with vertical lines marking windup start, release, and follow-through end.

Output:
    - PNG plots saved to data/<session>/03_visualize_data/time_series/<mot_filename>_timeseries_plot.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ========================================
# Parameters
# ========================================

ATHLETE = "tests"
SESSION = "player_tracking_tests"  # Change this to switch sessions
JOINTS_TO_PLOT = ['elbow_flex_r', 'arm_flex_r', 'hip_flexion_r', 'knee_angle_r', 'lumbar_extension']

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION

mot_folder = session_dir / "player_tracking_1" / "01_record_data" / "mot_files"
phases_csv = session_dir / "player_tracking_1" / "02_process_data" / "time_series" / "freethrow_phases.csv"
output_dir = session_dir / "player_tracking_1" / "03_visualize_data" / "time_series" / "kinematic_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Functions
# ========================================

def load_mot_file(filepath):
    """Reads a .mot file and returns a DataFrame starting at the time header."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            start_idx = i
            break
    return pd.read_csv(filepath, sep=r'\s+', skiprows=start_idx)

# ========================================
# Load Phase Info
# ========================================

phases_df = pd.read_csv(phases_csv)
phases_df['file'] = phases_df['file'].str.strip().str.lower()
phase_dict = phases_df.set_index('file').to_dict(orient='index')

# ========================================
# Main Loop
# ========================================

for mot_file in mot_folder.glob("*.mot"):
    filename = mot_file.name.strip().lower()
    if filename not in phase_dict:
        print(f"Skipping {filename}: no phase info found.")
        continue

    df = load_mot_file(mot_file)

    # Get phase frame indices
    windup_start = phase_dict[filename]['windup_start']
    release_frame = phase_dict[filename]['release_frame']
    followthrough_end = phase_dict[filename]['followthrough_end']

    # Get corresponding time values
    windup_time = df.at[windup_start, 'time']
    release_time = df.at[release_frame, 'time']
    followthrough_time = df.at[followthrough_end, 'time']

    # Slice data for plotting
    df_slice = df.iloc[windup_start:followthrough_end + 1]

       # Slice data for plotting
    df_slice = df.iloc[windup_start:followthrough_end + 1]

    # Get time range for plotting and annotations
    windup_time = df_slice['time'].iloc[0]
    release_time = df['time'].iloc[release_frame]
    followthrough_time = df_slice['time'].iloc[-1]

    # Plot
    plt.figure(figsize=(10, 6))
    for joint in JOINTS_TO_PLOT:
        if joint in df_slice.columns:
            plt.plot(df_slice['time'], df_slice[joint], label=joint)

    # Optional: Add text labels if overlapping is a concern
    # plt.text(windup_time, plt.ylim()[1], 'Windup', rotation=90, color='gray', va='top')
    # plt.text(release_time, plt.ylim()[1], 'Release', rotation=90, color='red', va='top')
    # plt.text(followthrough_time, plt.ylim()[1], 'Follow-Through', rotation=90, color='blue', va='top')


    # Add vertical lines for phases
    plt.axvline(x=windup_time, color='gray', linestyle='--', label='Windup Start')
    plt.axvline(x=release_time, color='red', linestyle='--', label='Release')
    plt.axvline(x=followthrough_time, color='blue', linestyle='--', label='Follow-Through End')

    plt.title(f"Joint Angles (Windup to Follow-Through) - {mot_file.stem}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{mot_file.stem}_timeseries_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")
