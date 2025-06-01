"""
Title: analyze_average.py

Purpose:
    Analyzes average joint angles at the moment of release for made vs. missed free throws.
    This script loads the release_summary.csv (generated from process_mot_to_csv.py), filters out
    unknown outcomes, and computes average kinematic measurements grouped by outcome.

Prerequisites:
    - A CSV file named 'release_summary.csv' containing kinematic measurements and shot outcomes

Output:
    - average_kinematics_by_outcome.csv: A CSV file with mean joint angles for 'made' and 'missed' shots

Usage:
    Run this script after generating release_summary.csv. Make sure the file exists in the correct path.
"""

import pandas as pd
from pathlib import Path

# ======================================== 
# Parameters 
# ========================================

session = "freethrows1"  # Change this to switch sessions

# ======================================== 
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
data_dir = base_dir / "data" / session
release_dir = data_dir / "release"

input_path = release_dir / "release_summary.csv"
output_path = release_dir / "average_kinematics_by_outcome.csv"

# ======================================== 
# # Load release summary CSV
# ========================================

release_df = pd.read_csv(input_path)
filtered_df = release_df[release_df['outcome'].isin(['made', 'miss'])]

# ======================================== 
# Compute Average Kinematics
# ========================================

numeric_cols = filtered_df.select_dtypes(include='number').columns.difference(['time'])
average_kinematics = filtered_df.groupby('outcome')[numeric_cols].mean()

# ======================================== 
# Save Results
# ========================================

average_kinematics.to_csv(output_path, index=True)
print(f"Average kinematics saved to {output_path}")
