"""
Title: save_timeseries_from_mot.py

Purpose:
    Save full kinematic angle data over time from a single OpenCap .mot file
    to a .csv file for further analysis.

Output:
    - data/<session>/time_series/<mot_filename>_timeseries.csv
"""

import pandas as pd
from pathlib import Path

# ======================================== 
# Parameters  
# ========================================

session = "freethrows1"

# ======================================== 
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
session_dir = script_dir.parents[2] / "data" / session
mot_folder = session_dir / "mot_files"
mot_file = mot_folder / "Freethrow1.mot"
output_dir = session_dir / "time_series"
output_path = output_dir / f"{mot_file.stem}_timeseries.csv"
output_dir.mkdir(parents=True, exist_ok=True)

# ======================================== 
# Load and Save Data
# ========================================
def load_mot_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("time"):
            start_idx = i
            break
    return pd.read_csv(filepath, sep=r'\s+', skiprows=start_idx)

df = load_mot_file(mot_file)
df.to_csv(output_path, index=False)
print(f"Saved time series data to {output_path}")
