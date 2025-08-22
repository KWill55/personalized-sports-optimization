"""
split_features_labels.py

Purpose:
    Split the combined release summary CSV into:
    - A features CSV (for model inputs)
    - A labels CSV (for model targets)

Assumes that combine_release_summaries.py has already been run.
"""

import pandas as pd
import os

# ========================================
# CONFIGURATION PARAMTERS
# ========================================

# Path to the combined CSV
COMBINED_CSV_PATH = "../data/ATHLETE/SESSION/metrics/combined_metrics/combined_release_summary.csv"

# Where to save the split files
FEATURES_OUTPUT_PATH = "../data/ATHLETE/SESSION/metrics/ml_data/ml_features/features.csv"
LABELS_OUTPUT_PATH   = "../data/ATHLETE/SESSION/metrics/ml_data/ml_labels/labels.csv"

# Define which columns are features vs. labels

FEATURE_COLUMNS = [
    "release_angle_deg",
    "release_velocity_mps",
    "release_height_m",
    "backspin_rps",
    "elbow_angle_deg",
    "arc_peak_height_m"
]

LABEL_COLUMNS = [
    "made_shot",
]

# ========================================
# Main Logic 
# ========================================

def split_csv():
    if not os.path.exists(COMBINED_CSV_PATH):
        print(f"❌ Combined CSV not found at: {COMBINED_CSV_PATH}")
        return

    df = pd.read_csv(COMBINED_CSV_PATH)

    # Check that all selected columns exist
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
    
    if missing_features or missing_labels:
        print("❌ Missing columns in the input CSV:")
        if missing_features:
            print(f"  Features: {missing_features}")
        if missing_labels:
            print(f"  Labels: {missing_labels}")
        return

    features = df[FEATURE_COLUMNS]
    labels = df[LABEL_COLUMNS]

    features.to_csv(FEATURES_OUTPUT_PATH, index=False)
    labels.to_csv(LABELS_OUTPUT_PATH, index=False)

    print(f"✅ Saved features to: {FEATURES_OUTPUT_PATH}")
    print(f"✅ Saved labels to: {LABELS_OUTPUT_PATH}")

if __name__ == "__main__":
    split_csv()
