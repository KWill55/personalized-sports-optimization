"""
Title: plot_velocity_time_series.py

Purpose:
    Generate and save time series plots of selected joint angular velocities for each free throw.
    Velocities are calculated from joint angles in OpenCap .mot files and visualized between
    windup and follow-through phases with vertical markers.

Output:
    - PNG plots saved to data/<session>/03_visualize_data/time_series/<mot_filename>_velocity_plot.png
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
FPS = 30

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION

mot_folder = session_dir / "player_tracking_1" / "01_record_data" / "mot_files"
phases_csv = session_dir / "player_tracking_1" / "02_process_data" / "time_series" / "freethrow_phases.csv"
output_dir = session_dir / "player_tracking_1" / "03_visualize_data" / "time_series" / "velocity_plots"
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

for mot_file in mot_folder.glob("*.mot"):
    filename = mot_file.name.strip().lower()
    if filename not in phase_dict:
        print(f"Skipping {filename}: no phase info found.")
        continue

    df = load_mot_file(mot_file)
    dt = 1.0 / FPS

    # Compute velocities
    for joint in JOINTS_TO_PLOT:
        if joint in df.columns:
            df[f"{joint}_vel"] = compute_velocity(df, joint, dt)

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

    # Plot velocities
    plt.figure(figsize=(10, 6))
    for joint in JOINTS_TO_PLOT:
        vel_col = f"{joint}_vel"
        if vel_col in df_slice.columns:
            plt.plot(df_slice['time'], df_slice[vel_col], label=vel_col)

    # Add vertical lines for phases
    plt.axvline(x=windup_time, color='gray', linestyle='--', label='Windup Start')
    plt.axvline(x=release_time, color='red', linestyle='--', label='Release')
    plt.axvline(x=followthrough_time, color='blue', linestyle='--', label='Follow-Through End')

    plt.title(f"Joint Velocities (Windup to Follow-Through) - {mot_file.stem}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (deg/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{mot_file.stem}_velocity_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved velocity plot: {output_path}")
