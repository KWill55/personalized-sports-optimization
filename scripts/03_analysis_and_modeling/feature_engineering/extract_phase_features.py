import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# -------------------------
# Config
# -------------------------
cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

BASE_DIR = Path(__file__).resolve().parents[3]
SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
ANGLES_DIR = SESSION_DIR / "metrics" / "3d_angles"
PHASES_CSV = SESSION_DIR / "metrics" / "freethrow_phases.csv"
OUTPUT_CSV = SESSION_DIR / "analysis" / "datasets" / "phase_features.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utility
# -------------------------
def compute_stats(series):
    """Return basic stats for a numeric pandas Series."""
    return {
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "std": series.std(),
        "range": series.max() - series.min(),
        "total_change": series.iloc[-1] - series.iloc[0]
    }

# -------------------------
# Main
# -------------------------
def main():
    # Load phase timing table
    phases_df = pd.read_csv(PHASES_CSV)

    feature_rows = []

    for _, row in phases_df.iterrows():
        throw_file = row["file"]  # your CSV uses 'file'
        windup_start = int(row["windup_start"])
        release_frame = int(row["release_frame"])
        followthrough_end = int(row["followthrough_end"])

        angle_file = ANGLES_DIR / throw_file
        if not angle_file.exists():
            print(f"⚠️ Skipping {throw_file}, angle file not found.")
            continue

        df_angles = pd.read_csv(angle_file)

        # Prepare feature dict
        throw_features = {"file": throw_file}

        # Phase durations
        throw_features["windup_duration"] = release_frame - windup_start + 1
        throw_features["follow_duration"] = followthrough_end - release_frame + 1
        throw_features["total_duration"] = followthrough_end - windup_start + 1

        for col in df_angles.columns:
            if col == "frame":
                continue

            # Windup stats
            windup_stats = compute_stats(df_angles[col].iloc[windup_start:release_frame + 1])
            throw_features.update({f"{col}_windup_{k}": v for k, v in windup_stats.items()})

            # Release frame value
            throw_features[f"{col}_release_value"] = df_angles[col].iloc[release_frame]

            # Follow-through stats
            follow_stats = compute_stats(df_angles[col].iloc[release_frame:followthrough_end + 1])
            throw_features.update({f"{col}_follow_{k}": v for k, v in follow_stats.items()})

        feature_rows.append(throw_features)

    # Save compiled features
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
