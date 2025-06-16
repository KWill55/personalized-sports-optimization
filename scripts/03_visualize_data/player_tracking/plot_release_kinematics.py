"""
Purpose/Description:
    Visualizes joint angle kinematics at the moment of release for each shot.
    Grouped into Core (Pelvis, Hips, Back), Arms (Right & Left), and Legs (Right & Left).
    Background shading indicates shot outcome (green for made, red for miss).

Prerequisites:
    - release_summary.csv: A CSV file created by process_mot_to_csv.py with all kinematic data and outcomes

Output:
    - Displays 3 figures:
        1. Core angles (3 subplots: Pelvis, Hips, Back)
        2. Arm angles (2 subplots: Right and Left Arm)
        3. Leg angles (2 subplots: Right and Left Leg)

Usage:
    Run this script after generating release_summary.csv to visualize free throw kinematics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================== 
# Parameters 
# ========================================

session = "freethrows3"  # Change this to switch sessions

# ======================================== 
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
summary_path = base_dir / "data" / session / "02_process_data" /  "release" / "release_summary.csv"

# ======================================== 
# Read Release Summary CSV
# ========================================

release_data_df = pd.read_csv(summary_path)

# ======================================== 
# Define Joint Groups 
# ========================================

pelvis = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
hips_r = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r']
hips_l = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l']
lumbar = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
arms_r = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r']
arms_l = ['arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l']
legs_r = ['knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']
legs_l = ['knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']

# ======================================== 
# Plotting Function 
# ========================================

def plot_joint_group(data, columns, title, ax):
    for idx, outcome in enumerate(data['outcome']):
        color = 'green' if outcome == 'made' else 'red'
        ax.axvspan(idx - 0.3, idx + 0.3, color=color, alpha=0.1)
    for col in columns:
        if col in data.columns:
            ax.plot(data['clip_name'], data[col], marker='o', label=col)
    ax.set_title(title)
    # ax.set_xlabel("Clip Name")
    ax.set_ylabel("Angle (degrees)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True)


# ======================================== 
# Plotting Core, Arms, and Legs Angles
# ========================================

# ======= Core Plot (Pelvis, Hips, Back) =======
fig_core, axs_core = plt.subplots(3, 1, figsize=(14, 16))
plot_joint_group(release_data_df, pelvis, "Pelvis Angles", axs_core[0])
plot_joint_group(release_data_df, hips_r + hips_l, "Hip Angles (Right & Left)", axs_core[1])
plot_joint_group(release_data_df, lumbar, "Back (Lumbar) Angles", axs_core[2])
fig_core.suptitle("Core Angles (Pelvis, Hips, Back)", fontsize=16)
fig_core.tight_layout(rect=[0, 0.03, 1, 0.95])

# ======= Arm Plot (Right vs Left) =======
fig_arms, axs_arms = plt.subplots(2, 1, figsize=(14, 12))
plot_joint_group(release_data_df, arms_r, "Right Arm Angles", axs_arms[0])
plot_joint_group(release_data_df, arms_l, "Left Arm Angles", axs_arms[1])
fig_arms.suptitle("Arm Angles", fontsize=16)
fig_arms.tight_layout(rect=[0, 0.03, 1, 0.95])

# ======= Leg Plot (Right vs Left) =======
fig_legs, axs_legs = plt.subplots(2, 1, figsize=(14, 12))
plot_joint_group(release_data_df, legs_r, "Right Leg Angles", axs_legs[0])
plot_joint_group(release_data_df, legs_l, "Left Leg Angles", axs_legs[1])
fig_legs.suptitle("Leg Angles", fontsize=16)
fig_legs.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
