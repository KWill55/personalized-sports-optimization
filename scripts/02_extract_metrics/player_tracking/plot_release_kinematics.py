"""
Purpose/Description:
    Visualizes MediaPipe-derived joint angles at the moment of release for each shot.
    Grouped into Core, Arms, and Legs.
    Background shading indicates shot outcome (green for made, red for miss).

Output:
    - Displays 3 figures:
        1. Core angles (pelvis proxy, hips, spine)
        2. Arm angles (Right and Left)
        3. Leg angles (Right and Left)
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

# ======================================== 
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION
player_tracking_metrics_dir = session_dir / "extracted_metrics" / "player_tracking_metrics"

release_path = player_tracking_metrics_dir / "release" / "raw_3d" / f"{CLIP_NAME}_release.csv"

# ======================================== 
# Read Release Summary CSV 
# ========================================

df = pd.read_csv(release_path)

# ======================================== 
# Define MediaPipe-Based Angle Groups 
# ========================================

core_angles = ['shoulder_lean', 'hip_tilt', 'torso_lean']
arms_r = ['elbow_angle_r', 'shoulder_angle_r', 'wrist_angle_r']
arms_l = ['elbow_angle_l', 'shoulder_angle_l', 'wrist_angle_l']
legs_r = ['knee_angle_r', 'hip_angle_r', 'ankle_angle_r']
legs_l = ['knee_angle_l', 'hip_angle_l', 'ankle_angle_l']

# ======================================== 
# Plotting Function 
# ========================================

def plot_group(df, cols, title, ax):
    for i, outcome in enumerate(df['outcome']):
        color = 'green' if outcome == 'made' else 'red'
        ax.axvspan(i - 0.3, i + 0.3, color=color, alpha=0.1)

    for col in cols:
        if col in df.columns:
            ax.plot(df['clip_name'], df[col], marker='o', label=col)

    ax.set_title(title)
    ax.set_ylabel("Angle (degrees)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True)

# ======================================== 
# Plot All Groups 
# ========================================

# Core
fig_core, axs_core = plt.subplots(1, 1, figsize=(14, 5))
plot_group(df, core_angles, "Core Angles (Spine, Pelvis, Torso)", axs_core)
fig_core.suptitle("Core Angles", fontsize=16)
fig_core.tight_layout(rect=[0, 0.03, 1, 0.95])

# Arms
fig_arms, axs_arms = plt.subplots(2, 1, figsize=(14, 10))
plot_group(df, arms_r, "Right Arm Angles", axs_arms[0])
plot_group(df, arms_l, "Left Arm Angles", axs_arms[1])
fig_arms.suptitle("Arm Angles", fontsize=16)
fig_arms.tight_layout(rect=[0, 0.03, 1, 0.95])

# Legs
fig_legs, axs_legs = plt.subplots(2, 1, figsize=(14, 10))
plot_group(df, legs_r, "Right Leg Angles", axs_legs[0])
plot_group(df, legs_l, "Left Leg Angles", axs_legs[1])
fig_legs.suptitle("Leg Angles", fontsize=16)
fig_legs.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
