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

# ========== Parameters ==========
data_dir = Path("../data/freethrows1")
input_path = data_dir / "release_summary.csv"
output_path = data_dir / "average_kinematics_by_outcome.csv"

# ========== Load and Filter Data ==========
release_df = pd.read_csv(input_path)
filtered_df = release_df[release_df['outcome'].isin(['made', 'miss'])]

# ========== Compute Averages ==========
numeric_cols = filtered_df.select_dtypes(include='number').columns.difference(['time'])
average_kinematics = filtered_df.groupby('outcome')[numeric_cols].mean()

# ========== Save Output ==========
average_kinematics.to_csv(output_path, index=True)
print(f"Average kinematics saved to {output_path}")
