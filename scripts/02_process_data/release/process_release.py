"""
Title: process_release.py

Purpose:
    Extracts kinematic joint angle data at the moment of release (when the shooting elbow is fully extended) from OpenCap .mot files
    and consolidates this data into a single CSV file. Also merges outcome labels (made/miss) for each shot.

Prerequisites:
    - Folder of .mot files exported from OpenCap for each free throw (one per shot)
    - A CSV file named 'outcomes.csv' in the parent directory of the .mot files folder, with columns: ['file', 'outcome']

Output:
    - release_summary.csv: A single CSV file containing the extracted kinematic data for each shot at moment of release

Usage:
    Adjust the folder paths to match your own directories
"""

import os
import pandas as pd
from pathlib import Path
import re

# ======================================== 
# Parameters 
# ========================================

session = "freethrows3"  # Change this to switch sessions

# ======================================== 
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
session_dir = base_dir / "data" / session

mot_folder = session_dir / "01_record_data" / "mot_files"
outcomes_path = session_dir / "01_record_data" / "outcomes.csv"

release_dir = session_dir / "02_process_data" / "release"
release_summary_path = release_dir / "release_summary.csv"

# Ensure output folder exists
release_dir.mkdir(parents=True, exist_ok=True)


# Ensure output folder exists
release_dir.mkdir(parents=True, exist_ok=True)


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

def extract_release_frame(df):
    return df['elbow_flex_r'].idxmin()

def extract_kinematics_at_release(df, release_idx):
    return df.loc[release_idx].to_dict()

def extract_shot_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def process_all_mot_files(directory):
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

# ======================================== 
# Main Processing 
# ========================================

release_data_df = process_all_mot_files(mot_folder)
outcomes_df = pd.read_csv(outcomes_path)

release_data_df = release_data_df.reset_index(drop=True)
release_data_df['clip_name'] = [Path(f).stem for f in release_data_df['file']]

release_data_df['file'] = release_data_df['file'].str.strip().str.lower()
outcomes_df['file'] = outcomes_df['file'].str.strip().str.lower()

release_data_df = pd.merge(release_data_df, outcomes_df, on='file', how='left')
release_data_df['outcome'] = release_data_df['outcome'].str.strip().str.lower()
release_data_df['outcome'] = release_data_df['outcome'].fillna('unknown')

cols = ['file', 'clip_name'] + [col for col in release_data_df.columns if col not in ['file', 'clip_name']]
release_data_df = release_data_df[cols]

release_data_df.to_csv(release_summary_path, index=False)
print(f"Saved release summary to {release_summary_path}")



