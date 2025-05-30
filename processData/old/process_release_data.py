import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ======================================== Parameters ========================================
 
# input files
mot_folder = "../data/freethrows1/mot_files"  # Directory containing the .mot files
outcomes_path = Path(mot_folder).parent / "outcomes.csv" # 

# output files
relesase_summary_path = Path(mot_folder).parent / "release_summary.csv"
avg_kinematics_path = Path(mot_folder).parent / "average_kinematics_by_outcome.csv"

# ======================================== Functions ========================================

def load_mot_file(filepath):
    """Loads a .mot file and returns it as a DataFrame."""
    
    # read lines to locate header start (in our case at "time")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            start_idx = i
            break
    
    # return the data after the header
    return pd.read_csv(filepath, sep=r'\s+', skiprows=start_idx)

def extract_release_frame(df):
    """Returns the frame index where elbow is maximally extended (min elbow_flex_r)."""
    
    # Returns when shooting arm is fully extended
    return df['elbow_flex_r'].idxmin() #returns index minimum of elbow_flex_r

def extract_kinematics_at_release(df, release_idx):
    """Extracts all kinematic values at the release frame."""
    return df.loc[release_idx].to_dict()

def extract_shot_number(filename):
    """Extracts a numeric value from the filename for sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def process_all_mot_files(directory):
    """Processes all .mot files and returns summary DataFrame with kinematics at release."""
    results = []
    directory_path = Path(directory)
    files = sorted(directory_path.glob("*.mot"), key=lambda f: extract_shot_number(f.name))
    
    for file in files:
        df = load_mot_file(file)
        release_idx = extract_release_frame(df)
        kinematics = extract_kinematics_at_release(df, release_idx)
        kinematics['file'] = file.name
        results.append(kinematics)

    return pd.DataFrame(results)

def plot_joint_group(data, columns, title, ax):
    """Plots a group of joint angles on the same axes."""
    
    # Shade background by outcome (made vs. missed)
    for idx, outcome in enumerate(release_data_df['outcome']):
        color = 'green' if outcome == 'made' else 'red'
        ax.axvspan(idx - 0.30, idx + 0.30, color=color, alpha=0.1)
    
    # Plot each column in the group
    for col in columns:
        if col in data.columns:
            ax.plot(data['clip_name'], data[col], marker='o', label=col)
    ax.set_title(title)
    ax.set_xlabel("Clip Name")
    ax.set_ylabel("Angle (degrees)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True)

# ======================================== Load Data and Process Kinematics at release ========================================

# Load data
release_data_df = process_all_mot_files(mot_folder) # DataFrame with kinematics at release
outcomes_df = pd.read_csv(outcomes_path) # DataFrame with shot success outcomes

# Generate clean labels
release_data_df = release_data_df.reset_index(drop=True) 
release_data_df['clip_name'] = [Path(f).stem for f in release_data_df['file']] # Extracting clip names from file names
# release_data_df['shot_number'] = ['Shot ' + str(i + 1) for i in range(len(release_data_df))] # Generate shot numbers

# Ensure outcomes_df has the same clip names
release_data_df['file'] = release_data_df['file'].str.strip()
outcomes_df['file'] = outcomes_df['file'].str.strip()
release_data_df['file'] = release_data_df['file'].str.lower()
outcomes_df['file'] = outcomes_df['file'].str.lower()

# Merge outcomes into main DataFrame
release_data_df = pd.merge(release_data_df, outcomes_df, on='file', how='left')

# Normalize case and strip whitespace for new 'outcome' column
release_data_df['outcome'] = release_data_df['outcome'].str.strip().str.lower()
release_data_df['outcome'] = release_data_df['outcome'].fillna('Unknown')

# Bring 'file' and 'clip_name' to the first two columns 
cols = ['file', 'clip_name'] + [col for col in release_data_df.columns if col not in ['file', 'clip_name']]
release_data_df = release_data_df[cols]

# ======================================== Plot Data in Groups ========================================

# Define joint groups
pelvis = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
hips_r = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r']
hips_l = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l']
lumbar = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']

arms_r = ['arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r']
arms_l = ['arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l']

legs_r = ['knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r']
legs_l = ['knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']

# ---- Plot Core: Pelvis, Hips, Back ----
fig_core, axs_core = plt.subplots(3, 1, figsize=(14, 16))
plot_joint_group(release_data_df, pelvis, "Pelvis Angles", axs_core[0])
plot_joint_group(release_data_df, hips_r + hips_l, "Hip Angles (Right & Left)", axs_core[1])
plot_joint_group(release_data_df, lumbar, "Back (Lumbar) Angles", axs_core[2])
fig_core.suptitle("Core Angles (Pelvis, Hips, Back)", fontsize=16)
fig_core.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---- Plot Arms: Right vs Left ----
fig_arms, axs_arms = plt.subplots(2, 1, figsize=(14, 12))
plot_joint_group(release_data_df, arms_r, "Right Arm Angles", axs_arms[0])
plot_joint_group(release_data_df, arms_l, "Left Arm Angles", axs_arms[1])
fig_arms.suptitle("Arm Angles", fontsize=16)
fig_arms.tight_layout(rect=[0, 0.03, 1, 0.95])

# ---- Plot Legs: Right vs Left ----
fig_legs, axs_legs = plt.subplots(2, 1, figsize=(14, 12))
plot_joint_group(release_data_df, legs_r, "Right Leg Angles", axs_legs[0])
plot_joint_group(release_data_df, legs_l, "Left Leg Angles", axs_legs[1])
fig_legs.suptitle("Leg Angles", fontsize=16)
fig_legs.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# ======================================== Analyze Averages by Outcome ========================================

# Filter out rows where outcome is unknown
filtered_df = release_data_df[release_data_df['outcome'].isin(['made', 'miss'])]

# Group by outcome and calculate mean of all kinematic columns (excluding non-numeric)
numeric_cols = filtered_df.select_dtypes(include='number').columns.difference(['time'])
average_kinematics = filtered_df.groupby('outcome')[numeric_cols].mean()

# ======================================== Print results and Save ========================================

# Save the release data for each group to CSV files
release_data_df.to_csv(relesase_summary_path, index=False)

#save the average kinematics for each group by outcome to CSV files
average_kinematics.to_csv(avg_kinematics_path, index=True)
