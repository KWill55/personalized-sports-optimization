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
project_cfg_path = ROOT_DIR / "project_config.yaml"
paths = load_paths(project_cfg_path)

ANGLES_DIR   = paths["angles"]
PHASES_CSV   = paths["phases"]
DATASETS_DIR = paths["datasets"]
MODELS_DIR   = paths["models"]
CALIB_DIR    = paths["calibration"]
VIDEOS_DIR   = paths["videos"]

# -------------------------
# Feature Config
# -------------------------
feature_cfg_path = ROOT_DIR / "feature_config.yaml"
with open(feature_cfg_path, "r") as f:
    feature_cfg = yaml.safe_load(f)

INCLUDE_3D_ANGLES        = feature_cfg["include_features"].get("angles", False)
INCLUDE_3D_VELOCITIES    = feature_cfg["include_features"].get("velocities", False)
INCLUDE_3D_ACCELERATIONS = feature_cfg["include_features"].get("accelerations", False)

# NEW: normalize selectors to support "ALL" or lists
def _normalize_selector(val):
    if isinstance(val, str):
        return "ALL" if val.strip().upper() == "ALL" else [val]
    if isinstance(val, list):
        return "ALL" if any(str(x).strip().upper() == "ALL" for x in val) else val
    return None

ANGLES_SELECTOR = _normalize_selector(feature_cfg.get("selected_columns", {}).get("angles", None))
STATS_SELECTOR  = _normalize_selector(feature_cfg.get("selected_columns", {}).get("summary_stats", None))

MODEL_TYPE      = str(feature_cfg.get("model_type", "unknown_model"))
FEATURE_VERSION = str(feature_cfg.get("feature_version", "000"))

# -------------------------
# Output paths
# -------------------------
OUTPUT_CSV = DATASETS_DIR / "features" / MODEL_TYPE / FEATURE_VERSION / f"phase_features_{FEATURE_VERSION}.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utility
# -------------------------
# canonical stat keys we know how to compute
BASE_STAT_KEYS = ["min", "max", "mean", "std", "range", "total_change"]
RELEASE_KEY    = "release_value"

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

def _stat_keys_allowed():
    """Return set of allowed stat names or 'ALL'."""
    if STATS_SELECTOR == "ALL" or STATS_SELECTOR is None:
        # default to ALL if not specified
        return "ALL"
    return set(str(x) for x in STATS_SELECTOR)

def _angle_cols(df_angles):
    """Resolve which angle columns to use (ALL or list)."""
    if ANGLES_SELECTOR == "ALL" or ANGLES_SELECTOR is None:
        return [c for c in df_angles.columns if c != "frame"]
    return [c for c in ANGLES_SELECTOR if c in df_angles.columns and c != "frame"]

def _filter_and_prefix(stats_dict, prefix, allowed):
    """
    From {'min':..., 'max':...} produce {f'{prefix}_min':..., ...}
    respecting allowed (set or 'ALL').
    """
    out = {}
    for k, v in stats_dict.items():
        if allowed == "ALL" or k in allowed:
            out[f"{prefix}_{k}"] = v
    return out

# -------------------------
# Main
# -------------------------
def main():
    phases_df = pd.read_csv(PHASES_CSV)
    feature_rows = []
    allowed_stats = _stat_keys_allowed()

    for _, row in phases_df.iterrows():
        throw_file = row["file"]
        windup_start = int(row["windup_start"])
        release_frame = int(row["release_frame"])
        followthrough_end = int(row["followthrough_end"])

        angle_file = ANGLES_DIR / throw_file
        if not angle_file.exists():
            print(f"⚠️ Skipping {throw_file}, angle file not found.")
            continue

        df_angles = pd.read_csv(angle_file)
        angle_columns = _angle_cols(df_angles)

        # Prepare feature dict
        throw_features = {"file": throw_file}

        # Phase durations (meta-features)
        throw_features["windup_duration"] = release_frame - windup_start + 1
        throw_features["follow_duration"] = followthrough_end - release_frame + 1
        throw_features["total_duration"]  = followthrough_end - windup_start + 1

        # robust slice helper
        def slc(series, a, b):
            a0 = max(0, min(len(series) - 1, a))
            b0 = max(0, min(len(series) - 1, b))
            if b0 < a0:
                a0, b0 = b0, a0
            return series.iloc[a0:b0 + 1]

        for col in angle_columns:
            s = df_angles[col]

            # ---------- ANGLES ----------
            if INCLUDE_3D_ANGLES:
                # windup stats
                w_stats = compute_stats(slc(s, windup_start, release_frame))
                throw_features.update(_filter_and_prefix(w_stats, f"{col}_windup", allowed_stats))

                # release value (treat as a "stat" named release_value)
                if 0 <= release_frame < len(s):
                    if allowed_stats == "ALL" or RELEASE_KEY in allowed_stats:
                        throw_features[f"{col}_{RELEASE_KEY}"] = s.iloc[release_frame]
                else:
                    if allowed_stats == "ALL" or RELEASE_KEY in allowed_stats:
                        throw_features[f"{col}_{RELEASE_KEY}"] = np.nan

                # follow-through stats
                f_stats = compute_stats(slc(s, release_frame, followthrough_end))
                throw_features.update(_filter_and_prefix(f_stats, f"{col}_follow", allowed_stats))

            # ---------- VELOCITIES ----------
            if INCLUDE_3D_VELOCITIES:
                v = s.diff().fillna(0.0)

                wv_stats = compute_stats(slc(v, windup_start, release_frame))
                throw_features.update(_filter_and_prefix(wv_stats, f"{col}_vel_windup", allowed_stats))

                if allowed_stats == "ALL" or RELEASE_KEY in allowed_stats:
                    throw_features[f"{col}_vel_{RELEASE_KEY}"] = v.iloc[release_frame] if 0 <= release_frame < len(v) else np.nan

                fv_stats = compute_stats(slc(v, release_frame, followthrough_end))
                throw_features.update(_filter_and_prefix(fv_stats, f"{col}_vel_follow", allowed_stats))

            # ---------- ACCELERATIONS ----------
            if INCLUDE_3D_ACCELERATIONS:
                a = s.diff().diff().fillna(0.0)

                wa_stats = compute_stats(slc(a, windup_start, release_frame))
                throw_features.update(_filter_and_prefix(wa_stats, f"{col}_acc_windup", allowed_stats))

                if allowed_stats == "ALL" or RELEASE_KEY in allowed_stats:
                    throw_features[f"{col}_acc_{RELEASE_KEY}"] = a.iloc[release_frame] if 0 <= release_frame < len(a) else np.nan

                fa_stats = compute_stats(slc(a, release_frame, followthrough_end))
                throw_features.update(_filter_and_prefix(fa_stats, f"{col}_acc_follow", allowed_stats))

        feature_rows.append(throw_features)

    features_df = pd.DataFrame(feature_rows)

    # -------------------------
    # Robust "derived" filter at the end
    # -------------------------
    derived_sel = feature_cfg.get("selected_features", {}).get("derived", [])
    use_allow = None
    if isinstance(derived_sel, str) and derived_sel.strip().upper() == "ALL":
        use_allow = None  # keep everything
    elif isinstance(derived_sel, list) and any(str(x).strip().upper() == "ALL" for x in derived_sel):
        use_allow = None
    else:
        # only filter if a non-empty list of exact column names is provided
        if isinstance(derived_sel, list) and len(derived_sel) > 0:
            use_allow = set(derived_sel)

    meta_cols = {"file", "windup_duration", "follow_duration", "total_duration"}
    if use_allow is not None:
        keep_cols = [c for c in features_df.columns if (c in use_allow or c in meta_cols)]
        features_df = features_df[keep_cols]

    if OUTPUT_CSV.exists():
        raise RuntimeError(f"{OUTPUT_CSV} already exists — bump version in YAML!")

    features_df.to_csv(OUTPUT_CSV, index=False)

    # Save the config snapshot into the same folder
    snapshot_path = OUTPUT_CSV.with_suffix(".features_used.yaml")
    feature_cfg["snapshot_version"] = FEATURE_VERSION
    with open(snapshot_path, "w") as f:
        yaml.dump(feature_cfg, f)

    print(f"✅ Saved features to {OUTPUT_CSV}")
    print(f"📄 Snapshot of features used: {snapshot_path}")

if __name__ == "__main__":
    main()
