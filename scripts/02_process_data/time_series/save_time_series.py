"""
Title: save_timeseries_from_mot.py

Purpose:
    Save full kinematic angle data over time from all OpenCap .mot files
    to individual .csv files for further analysis.

Output:
    - data/<session>/02_process_data/time_series/<mot_filename>_timeseries.csv for each .mot file
"""

import pandas as pd
from pathlib import Path

# ========================================
# Parameters
# ========================================

session = "freethrows3"

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
session_dir = script_dir.parents[2] / "data" / session
mot_folder = session_dir / "01_record_data" / "mot_files"
output_dir = session_dir / "02_process_data" / "time_series"
output_dir.mkdir(parents=True, exist_ok=True)

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

# ========================================
# Main Loop
# ========================================

mot_files = mot_folder.glob("*.mot")

for mot_file in mot_files:
    df = load_mot_file(mot_file)
    output_path = output_dir / f"{mot_file.stem}_time_series.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved time series data to {output_path}")
