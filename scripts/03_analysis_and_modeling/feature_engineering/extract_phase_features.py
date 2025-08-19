import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.translate_config import load_paths

# -------------------------
# Project Config
# -------------------------

# Load paths (athlete/session come from project_config.yaml)
project_cfg_path = ROOT_DIR / "project_config.yaml"
paths = load_paths(project_cfg_path)

# Resolve key paths from YAML
ANGLES_DIR   = paths["angles"]
PHASES_CSV   = paths["phases"]
DATASETS_DIR = paths["datasets"]
MODELS_DIR   = paths["models"]
# PLOTS_DIR    = paths["plots"]
CALIB_DIR    = paths["calibration"]
VIDEOS_DIR   = paths["videos"]

# -------------------------
# Feature Config
# -------------------------
feature_cfg_path = ROOT_DIR / "feature_config.yaml"
with open(feature_cfg_path, "r") as f:
    feature_cfg = yaml.safe_load(f)

INCLUDE_3D_ANGLES        = feature_cfg["include_features"].get("angles", True)
INCLUDE_3D_VELOCITIES    = feature_cfg["include_features"].get("velocities", True)
INCLUDE_3D_ACCELERATIONS = feature_cfg["include_features"].get("accelerations", False)
SELECTED_3D_ANGLES       = feature_cfg.get("selected_columns", {}).get("angles", None)

VERSION = str(feature_cfg.get("version", "1.0"))  # fallback if not defined

# -------------------------
# Output paths
# -------------------------
OUTPUT_CSV = DATASETS_DIR / f"phase_features_{VERSION}.csv"
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

        if SELECTED_3D_ANGLES == "ALL":
            angle_columns = [c for c in df_angles.columns if c != "frame"]
        else:
            angle_columns = SELECTED_3D_ANGLES


        # Prepare feature dict
        throw_features = {"file": throw_file}

        # Phase durations (always keep these; they’re useful meta-features)
        throw_features["windup_duration"] = release_frame - windup_start + 1
        throw_features["follow_duration"] = followthrough_end - release_frame + 1
        throw_features["total_duration"] = followthrough_end - windup_start + 1

        # Precompute phase slices once
        def slc(series, a, b):  # inclusive of b
            # guard against any off-by-one or bounds issues in CSVs
            a0 = max(0, min(len(series) - 1, a))
            b0 = max(0, min(len(series) - 1, b))
            if b0 < a0:
                a0, b0 = b0, a0
            return series.iloc[a0:b0 + 1]

        for col in angle_columns:
            if col == "frame":
                continue

            s = df_angles[col]

            # ---------- ANGLES ----------
            if INCLUDE_3D_ANGLES:
                # Windup stats
                windup_stats = compute_stats(slc(s, windup_start, release_frame))
                throw_features.update({f"{col}_windup_{k}": v for k, v in windup_stats.items()})

                # Release frame value
                if 0 <= release_frame < len(s):
                    throw_features[f"{col}_release_value"] = s.iloc[release_frame]
                else:
                    throw_features[f"{col}_release_value"] = np.nan

                # Follow-through stats
                follow_stats = compute_stats(slc(s, release_frame, followthrough_end))
                throw_features.update({f"{col}_follow_{k}": v for k, v in follow_stats.items()})

            # ---------- VELOCITIES (first diff) ----------
            if INCLUDE_3D_VELOCITIES:
                v = s.diff().fillna(0.0)

                windup_v = compute_stats(slc(v, windup_start, release_frame))
                throw_features.update({f"{col}_vel_windup_{k}": v for k, v in windup_v.items()})

                # velocity at release is just diff at that frame (safe guard)
                throw_features[f"{col}_vel_release_value"] = v.iloc[release_frame] if 0 <= release_frame < len(v) else np.nan

                follow_v = compute_stats(slc(v, release_frame, followthrough_end))
                throw_features.update({f"{col}_vel_follow_{k}": v for k, v in follow_v.items()})

            # ---------- ACCELERATIONS (second diff) ----------
            if INCLUDE_3D_ACCELERATIONS:
                a = s.diff().diff().fillna(0.0)

                windup_a = compute_stats(slc(a, windup_start, release_frame))
                throw_features.update({f"{col}_acc_windup_{k}": v for k, v in windup_a.items()})

                throw_features[f"{col}_acc_release_value"] = a.iloc[release_frame] if 0 <= release_frame < len(a) else np.nan

                follow_a = compute_stats(slc(a, release_frame, followthrough_end))
                throw_features.update({f"{col}_acc_follow_{k}": v for k, v in follow_a.items()})

        feature_rows.append(throw_features)

    # Save compiled features
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(OUTPUT_CSV, index=False)

    # Save the config snapshot into the same folder
    snapshot_path = OUTPUT_CSV.with_suffix(".features_used.yaml")
    feature_cfg["snapshot_version"] = VERSION
    with open(snapshot_path, "w") as f:
        yaml.dump(feature_cfg, f)

    print(f"✅ Saved features to {OUTPUT_CSV}")
    print(f"📄 Snapshot of features used: {snapshot_path}")



if __name__ == "__main__":
    main()
