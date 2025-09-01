#!/usr/bin/env python3
"""
extract_phase_features_simple.py
A minimal, config-driven extractor that only handles summary_stats on angles.
- Phases and stats come directly from feature_config.yaml
- No velocities/accelerations
- time_series mode prints WIP and exits
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# -------------------------
# Project paths
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.translate_config import load_paths  # noqa: E402

project_cfg_path = ROOT_DIR / "project_config.yaml"
paths = load_paths(project_cfg_path)

ANGLES_DIR   = paths["angles"]
PHASES_CSV   = paths["phases"]
DATASETS_DIR = paths["datasets"]

# -------------------------
# Load feature config
# -------------------------
feature_cfg_path = ROOT_DIR / "feature_config.yaml"
CFG = yaml.safe_load(open(feature_cfg_path, "r"))

MODEL_TYPE      = str(CFG.get("model_type", "summary_stats"))
FEATURE_VERSION = str(CFG.get("feature_version", "000"))

INCLUDE_ANGLES  = bool(CFG.get("include_features", {}).get("angles", False))
PHASES_CFG      = CFG.get("phases", {}) or {}
STATS_CFG       = CFG.get("stats", {}) or {}
ANGLES_CFG      = CFG.get("angles", {}) or {}  # optional per-angle toggles/overrides

# -------------------------
# Early exit for time_series
# -------------------------
if MODEL_TYPE.strip().lower() == "time_series":
    print("work in progress. time_series is not ready yet")
    sys.exit(0)

# -------------------------
# Output path
# -------------------------
OUT_DIR    = DATASETS_DIR / "features" / MODEL_TYPE / FEATURE_VERSION
OUTPUT_CSV = OUT_DIR / f"phase_features_{FEATURE_VERSION}.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
SUPPORTED_STATS = ("min", "max", "mean", "std", "range", "total_change")
RELEASE_KEY     = "release_value"

def compute_stats(series: pd.Series) -> dict:
    s = series.astype(float)
    if len(s) == 0:
        return {k: np.nan for k in SUPPORTED_STATS}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "range": float(s.max() - s.min()),
        "total_change": float(s.iloc[-1] - s.iloc[0]),
    }

def slc(series: pd.Series, a: int, b: int) -> pd.Series:
    if len(series) == 0:
        return series
    a0 = max(0, min(len(series) - 1, int(a)))
    b0 = max(0, min(len(series) - 1, int(b)))
    if b0 < a0:
        a0, b0 = b0, a0
    return series.iloc[a0:b0+1]

def angle_enabled(name: str) -> bool:
    cfg = ANGLES_CFG.get(name)
    # default include if angle not listed
    return True if cfg is None else bool(cfg.get("include", True))

def want_phase(phase: str) -> bool:
    return bool(PHASES_CFG.get(phase, False))

def enabled_stats_for_phase(phase: str) -> dict:
    """
    Returns dict of {stat_name: bool} for the given phase
    """
    return dict(STATS_CFG.get(phase, {}))

def warn(msg: str):
    print(f"⚠️ {msg}")

# -------------------------
# Main
# -------------------------
def main():
    if not INCLUDE_ANGLES:
        raise RuntimeError("include_features.angles is False; nothing to compute.")

    phases_df = pd.read_csv(PHASES_CSV)
    feature_rows = []

    for _, row in phases_df.iterrows():
        throw_file = row["file"]
        windup_start    = int(row["windup_start"])
        release_frame   = int(row["release_frame"])
        followthrough_end = int(row["followthrough_end"])

        angle_file = ANGLES_DIR / throw_file
        if not angle_file.exists():
            warn(f"Skipping {throw_file}: angle file not found at {angle_file}")
            continue

        df_angles = pd.read_csv(angle_file)

        # discover angle columns from CSV, then filter by ANGLES_CFG.include
        angle_cols = [c for c in df_angles.columns if c != "frame" and angle_enabled(c)]
        if not angle_cols:
            warn(f"No enabled angle columns for {throw_file}; skipping.")
            continue

        # base row
        out = {
            "file": throw_file,
            "windup_duration": release_frame - windup_start + 1,
            "follow_duration": followthrough_end - release_frame + 1,
            "total_duration":  followthrough_end - windup_start + 1,
        }

        for col in angle_cols:
            s = df_angles[col].astype(float)

            # ---- windup ----
            if want_phase("windup"):
                stat_flags = enabled_stats_for_phase("windup")
                seg = slc(s, windup_start, release_frame)
                stats = compute_stats(seg)
                for st, on in stat_flags.items():
                    if on:
                        if st in SUPPORTED_STATS:
                            out[f"{col}_windup_{st}"] = stats[st]
                        else:
                            warn(f"Ignoring unsupported windup stat '{st}' for {col}")

            # ---- release ----
            if want_phase("release"):
                stat_flags = enabled_stats_for_phase("release")
                # Only release_value is meaningful at a single frame
                if stat_flags.get(RELEASE_KEY, False):
                    if 0 <= release_frame < len(s):
                        out[f"{col}_{RELEASE_KEY}"] = float(s.iloc[release_frame])
                    else:
                        out[f"{col}_{RELEASE_KEY}"] = np.nan
                # warn on any other enabled stat under release
                for st, on in stat_flags.items():
                    if on and st != RELEASE_KEY:
                        warn(f"Ignoring unsupported release stat '{st}' for {col}; only '{RELEASE_KEY}' is used.")

            # ---- follow ----
            if want_phase("follow"):
                stat_flags = enabled_stats_for_phase("follow")
                seg = slc(s, release_frame, followthrough_end)
                stats = compute_stats(seg)
                for st, on in stat_flags.items():
                    if on:
                        if st in SUPPORTED_STATS:
                            out[f"{col}_follow_{st}"] = stats[st]
                        else:
                            warn(f"Ignoring unsupported follow stat '{st}' for {col}")

        feature_rows.append(out)

    features_df = pd.DataFrame(feature_rows)

    if OUTPUT_CSV.exists():
        raise RuntimeError(f"{OUTPUT_CSV} already exists — bump version in YAML!")

    features_df.to_csv(OUTPUT_CSV, index=False)

    # snapshot config
    snapshot_path = OUTPUT_CSV.with_suffix(".features_used.yaml")
    snap = dict(CFG)
    snap["snapshot_version"] = FEATURE_VERSION
    with open(snapshot_path, "w") as f:
        yaml.dump(snap, f)

    print(f"✅ Saved features to {OUTPUT_CSV}")
    print(f"🧩 Rows: {len(features_df)}, Cols: {features_df.shape[1]}")

if __name__ == "__main__":
    main()
