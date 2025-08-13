"""
Title: prepare_release_features.py

Purpose:
    Clean and prepare features/labels from the combined release summary
    for use in machine learning models. Includes downsampling of majority class (miss).

Output:
    - X_release.csv: Cleaned input features
    - y_release.csv: Corresponding labels
"""

import pandas as pd
from pathlib import Path

# ========================================
# Parameters
# ========================================

input_file = "combined_release_summary.csv"
features_out = "X_release.csv"
labels_out = "y_release.csv"

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
data_dir = base_dir / "data"
release_dir = data_dir / "summary" / "release"

input_path = release_dir / input_file
X_path = release_dir / features_out
y_path = release_dir / labels_out

# ========================================
# Load & Clean Data
# ========================================

df = pd.read_csv(input_path)

# Filter out unknown outcomes
df = df[df['outcome'].isin(['made', 'miss'])].copy()

# ========================================
# Downsample Majority Class
# ========================================

df_made = df[df['outcome'] == 'made']
df_miss = df[df['outcome'] == 'miss']

if len(df_miss) > len(df_made):
    df_miss_down = df_miss.sample(n=len(df_made), random_state=42)
    df = pd.concat([df_made, df_miss_down]).sample(frac=1, random_state=42)
    print(f"Downsampled: {len(df_made)} made, {len(df_miss_down)} miss (total: {len(df)})")
else:
    print("No downsampling needed.")

# ========================================
# Split and Save
# ========================================

y = df['outcome'].map({'made': 1, 'miss': 0})
y.to_csv(y_path, index=False)

X = df.drop(columns=["file", "clip_name", "time", "outcome"])
X.to_csv(X_path, index=False)

print(f"Saved features to {X_path}")
print(f"Saved labels to {y_path}")
