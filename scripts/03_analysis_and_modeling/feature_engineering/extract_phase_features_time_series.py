#!/usr/bin/env python3
"""
extract_phase_features_time_series.py

Create a fixed, seconds-based time-series window around the release frame and
export per-angle features as wide columns named like: angle@-15 ... angle@0 ... angle@+10.

Key ideas:
- Window is configured in **seconds** and converted to frames via FPS from project_config.yaml.
- Same number/order of columns for every row (throw). If a throw lacks frames on
  either side, we PAD with a configurable value and emit a MASK per offset.
- Angle inclusion can be toggled in feature_config.yaml -> angles: { angle_name: {include: true/false} }
- Output lives under DATASETS_DIR / "features" / model_type / feature_version

Config (feature_config.yaml) minimal example:

model_type: "time_series"
feature_version: "v2_release_window"
include_features:
  angles: true
window:
  pre_seconds: 0.75   # seconds before release
  post_seconds: 0.30  # seconds after release
padding:
  mode: "pad"         # "pad" or "drop" (drop rows lacking full window)
  pad_value: null      # null -> NaN
  emit_mask: true      # also emit mask@<offset> columns (1=valid, 0=padded)
angles:
  # optional per-angle toggles; omit to include all present columns
  elbow_flex_r: { include: true }
  shoulder_flex_r: { include: true }

"""

from __future__ import annotations
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

# -------------------------
# Project paths & imports
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.translate_config import load_paths  # noqa: E402

project_cfg_path = ROOT_DIR / "project_config.yaml"
paths = load_paths(project_cfg_path)

ANGLES_DIR   = Path(paths["angles"])         # folder of per-throw angles CSVs
PHASES_CSV   = Path(paths["phases"])         # CSV with file, windup_start, release_frame, followthrough_end
DATASETS_DIR = Path(paths["datasets"])       # base datasets dir

# Load project config to get FPS (fallbacks handled below)
with open(project_cfg_path, "r") as f:
    project_cfg = yaml.safe_load(f)

# -------------------------
# Load feature config
# -------------------------
feature_cfg_path = ROOT_DIR / "feature_config.yaml"
with open(feature_cfg_path, "r") as f:
    CFG = yaml.safe_load(f)

MODEL_TYPE      = str(CFG.get("model_type", "time_series"))
FEATURE_VERSION = str(CFG.get("feature_version", "000"))

INCLUDE_ANGLES  = bool(CFG.get("include_features", {}).get("angles", False))
ANGLES_CFG      = CFG.get("angles", {}) or {}

WIN_CFG         = CFG.get("window", {}) or {}
PRE_SECONDS     = float(WIN_CFG.get("pre_seconds", 0.75))
POST_SECONDS    = float(WIN_CFG.get("post_seconds", 0.30))

PAD_CFG         = CFG.get("padding", {}) or {}
PAD_MODE        = str(PAD_CFG.get("mode", "pad")).lower()   # "pad" or "drop"
PAD_VALUE_RAW   = PAD_CFG.get("pad_value", None)              # None -> NaN
EMIT_MASK       = bool(PAD_CFG.get("emit_mask", True))

# -------------------------
# Resolve FPS robustly
# -------------------------
# Preferred key in project_config.yaml (as used elsewhere)
FPS = project_cfg.get("player_tracking_fps")
# Fallbacks
if FPS is None:
    FPS = paths.get("fps", None)
if FPS is None:
    # last resort default; but better to set it in project_config.yaml
    FPS = 60
FPS = float(FPS)

# Convert seconds to integer frame counts (consistent across all throws)
K_PRE  = int(round(PRE_SECONDS  * FPS))
K_POST = int(round(POST_SECONDS * FPS))
if K_PRE < 0 or K_POST < 0:
    raise ValueError("pre_seconds/post_seconds must be non-negative")
if K_PRE + K_POST == 0:
    raise ValueError("Window cannot be empty; increase pre_seconds and/or post_seconds.")

# -------------------------
# Output path
# -------------------------
OUT_DIR    = DATASETS_DIR / "features" / MODEL_TYPE / FEATURE_VERSION
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUT_DIR / f"time_series_{FEATURE_VERSION}.csv"
SNAPSHOT_YAML = OUTPUT_CSV.with_suffix(".features_used.yaml")

# -------------------------
# Helpers
# -------------------------
SUPPORTED_PAD_MODES = {"pad", "drop"}
if PAD_MODE not in SUPPORTED_PAD_MODES:
    raise ValueError(f"Unsupported padding.mode '{PAD_MODE}'. Use one of {SUPPORTED_PAD_MODES}.")

# Interpret pad value
PAD_VALUE = np.nan if PAD_VALUE_RAW is None else float(PAD_VALUE_RAW)


def angle_enabled(name: str) -> bool:
    cfg = ANGLES_CFG.get(name)
    return True if cfg is None else bool(cfg.get("include", True))


def build_offset_axis() -> np.ndarray:
    """Return array of integer offsets relative to release frame, e.g.,
    [-K_PRE, ..., -1, 0, 1, ..., K_POST]."""
    neg = np.arange(-K_PRE, 0) if K_PRE > 0 else np.array([], dtype=int)
    pos = np.arange(1, K_POST + 1) if K_POST > 0 else np.array([], dtype=int)
    return np.concatenate([neg, np.array([0], dtype=int), pos])


OFFSETS = build_offset_axis()


def slice_with_padding(series: pd.Series, rel_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return values array aligned to OFFSETS and a validity mask (1 valid, 0 padded).

    series: 1D pandas Series for a single angle column
    rel_idx: integer index of release frame (row index in the CSV)
    """
    n = len(series)
    values = np.empty(len(OFFSETS), dtype=float)
    mask = np.zeros(len(OFFSETS), dtype=int)

    for i, off in enumerate(OFFSETS):
        idx = rel_idx + off
        if 0 <= idx < n:
            val = float(series.iloc[idx])
            # Some logs may contain inf or NaN; normalize
            if np.isfinite(val):
                values[i] = val
                mask[i] = 1
            else:
                values[i] = PAD_VALUE
                mask[i] = 0
        else:
            values[i] = PAD_VALUE
            mask[i] = 0
    return values, mask


# -------------------------
# Main
# -------------------------

def main() -> None:
    if MODEL_TYPE.strip().lower() != "time_series":
        raise RuntimeError("feature_config.yaml model_type must be 'time_series' for this script.")
    if not INCLUDE_ANGLES:
        raise RuntimeError("include_features.angles is False; nothing to compute.")

    if not PHASES_CSV.exists():
        raise FileNotFoundError(f"Phases CSV not found at {PHASES_CSV}")

    phases_df = pd.read_csv(PHASES_CSV)
    if phases_df.empty:
        raise RuntimeError("Phases CSV is empty.")

    # Validate required columns
    for col in ("file", "windup_start", "release_frame", "followthrough_end"):
        if col not in phases_df.columns:
            raise KeyError(f"Required column '{col}' missing from phases CSV.")

    feature_rows: list[dict] = []

    # Build column name templates once
    # We'll accumulate angle@offset columns dynamically from the first file
    first_row_columns: list[str] | None = None

    for _, row in phases_df.iterrows():
        throw_file = str(row["file"])  # filename relative to ANGLES_DIR
        windup_start = int(row["windup_start"])          # not used for slicing, but useful metadata
        release_frame = int(row["release_frame"])        # CENTER point
        follow_end = int(row["followthrough_end"])       # not used for slicing, but useful metadata

        angle_path = ANGLES_DIR / throw_file
        if not angle_path.exists():
            print(f"⚠️ Skipping {throw_file}: angle file not found at {angle_path}")
            continue

        df_angles = pd.read_csv(angle_path)
        if df_angles.empty:
            print(f"⚠️ Skipping {throw_file}: angles CSV is empty")
            continue

        # Identify angle columns (exclude 'frame' if present)
        angle_cols = [c for c in df_angles.columns if c != "frame" and angle_enabled(c)]
        if not angle_cols:
            print(f"⚠️ No enabled angle columns for {throw_file}; skipping.")
            continue

        # Check if we have a complete window; if PAD_MODE=="drop" and the throw cannot
        # satisfy the full window without padding, drop the row.
        need_start = release_frame - K_PRE
        need_end   = release_frame + K_POST
        can_cover = (need_start >= 0) and (need_end < len(df_angles))
        if PAD_MODE == "drop" and not can_cover:
            # skip this throw
            continue

        # Base metadata
        out: dict[str, float | int | str] = {
            "file": throw_file,
            "windup_duration": int(release_frame - windup_start + 1),
            "follow_duration": int(follow_end - release_frame + 1),
            "total_duration": int(follow_end - windup_start + 1),
            "fps": float(FPS),
            "pre_frames": int(K_PRE),
            "post_frames": int(K_POST),
        }

        # Emit mask@<offset> once (same for all angles); use the first included angle to compute
        # validity positions.
        if EMIT_MASK:
            ref_col = angle_cols[0]
            ref_series = df_angles[ref_col].astype(float)
            _, mask = slice_with_padding(ref_series, release_frame)
            for off, m in zip(OFFSETS, mask):
                out[f"mask@{off:+d}"] = int(m)

        # Emit angle@<offset> columns
        for col in angle_cols:
            s = df_angles[col].astype(float)
            vals, _ = slice_with_padding(s, release_frame)
            for off, v in zip(OFFSETS, vals):
                out[f"{col}@{off:+d}"] = float(v)

        feature_rows.append(out)

        # Capture ordered column names from the first successfully built row to enforce ordering
        if first_row_columns is None:
            first_row_columns = list(out.keys())

    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)

    if features_df.empty:
        raise RuntimeError("No rows were produced. Check your phases CSV, angle files, and config.")

    # Enforce consistent column ordering (metadata -> masks -> angles, by offset order)
    # If first_row_columns was captured, use it; otherwise rely on DataFrame's current order.
    if first_row_columns is not None:
        # It is possible later rows introduced new angle names; include any extras at the end
        missing = [c for c in features_df.columns if c not in first_row_columns]
        ordered_cols = first_row_columns + missing
        features_df = features_df.reindex(columns=ordered_cols)

    if OUTPUT_CSV.exists():
        raise RuntimeError(f"{OUTPUT_CSV} already exists — bump feature_version in YAML!")

    features_df.to_csv(OUTPUT_CSV, index=False)

    # snapshot config (augment with resolved params)
    snap = dict(CFG)
    snap["snapshot_version"] = FEATURE_VERSION
    snap["resolved"] = {
        "fps": FPS,
        "pre_seconds": PRE_SECONDS,
        "post_seconds": POST_SECONDS,
        "pre_frames": K_PRE,
        "post_frames": K_POST,
        "offsets": OFFSETS.tolist(),
        "pad_mode": PAD_MODE,
        "pad_value": None if np.isnan(PAD_VALUE) else float(PAD_VALUE),
        "emit_mask": EMIT_MASK,
    }
    with open(SNAPSHOT_YAML, "w") as f:
        yaml.dump(snap, f)

    print(f"✅ Saved time-series features to {OUTPUT_CSV}")
    print(f"🧩 Rows: {len(features_df)}, Cols: {features_df.shape[1]}")


if __name__ == "__main__":
    main()
