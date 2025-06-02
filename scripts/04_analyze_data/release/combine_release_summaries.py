
"""
Title: combine_release_summaries.py

Purpose:
    Combine release_summary.csv files from multiple sessions into a single dataset
    for unified analysis or machine learning.

Output:
    - combined_release_summary.csv saved to data/04_analyze_data/
"""

import pandas as pd
from pathlib import Path

# ========================================
# Parameters
# ========================================

sessions = ["freethrows1", "freethrows2", "freethrows3"]  # Add more as needed
output_title = "combined_release_summary.csv"

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
data_dir = base_dir / "data"
output_dir = data_dir / "summary" / "release"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / output_title

# ========================================
# Combine Data
# ========================================

dfs = []

for session in sessions:
    path = data_dir / session / "02_process_data" / "release" / "release_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        dfs.append(df)
    else:
        print(f"Warning: {path} not found")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined release summaries saved to {output_path}")
else:
    print("No release_summary.csv files found.")
