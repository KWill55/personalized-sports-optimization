"""
Purpose:
    Identify windup, release, and follow-through phases from precomputed angles CSVs.

Input:
    - Folder of angles CSVs (columns include: frame?, elbow_flex_r, arm_flex_r)

Output:
    - CSV summarizing: windup_start, release_frame, followthrough_end for each file
"""

#TODO add angle smoothing 
#TODO maybe fourth phase (splitting windup into windup_bend and windup_extend)

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import re

# ==============================
# Config / paths
# ==============================
cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
FPS     = cfg["player_tracking_fps"]

SHOULDER_THRESHOLD = 100 #degrees 
WINDUP_SHOULDER_MIN = 30 #degrees

base_dir    = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

# angles live here:
input_folder = session_dir / "metrics" / "3d_angles"
output_csv   = session_dir / "metrics" / "freethrow_phases.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

# ==============================
# Phase detection helpers
# ==============================
def compute_velocity(series_deg: pd.Series, dt: float) -> pd.Series:
    v = series_deg.diff() / dt
    v.iloc[0] = 0.0
    return v.fillna(0.0)

def detect_throw_phases_from_angles(df_angles: pd.DataFrame, fps: int,
                                    threshold=10.0, window=3):
    """
    df_angles must have columns: elbow_flex_r (deg), shoulder_flex_r (deg)
    threshold is deg/s for avg velocity of these two angles.
    """
    dt = 1.0 / float(fps)

    # Velocity for elbow and shoulder
    elbow_vel    = compute_velocity(df_angles["elbow_flex_r"], dt).abs()
    shoulder_vel = compute_velocity(df_angles["shoulder_flex_r"], dt).abs()
    avg_arm_vel  = pd.concat([elbow_vel, shoulder_vel], axis=1).mean(axis=1)

    # Release = max elbow angle while shoulder_flex_r >= SHOULDER_THRESHOLD
    release_frame = int(
        df_angles[df_angles["shoulder_flex_r"] >= SHOULDER_THRESHOLD]["elbow_flex_r"].idxmax()
    )

    # Windup start: last point before release where:
    #   - shoulder_flex_r >= WINDUP_SHOULDER_MIN
    #   - velocity sustained above threshold for `window` frames
    windup_start = 0
    for i in range(max(0, release_frame - fps), release_frame):
        if (df_angles["shoulder_flex_r"].iloc[i] >= WINDUP_SHOULDER_MIN and
            (avg_arm_vel.iloc[i:i+window] > threshold).all()):
            windup_start = i
            break

    # Follow-through end: first frame after release when shoulder drops below the min angle
    followthrough_end = len(df_angles) - 1
    for i in range(release_frame + 1, len(df_angles)):
        if df_angles["shoulder_flex_r"].iloc[i] < SHOULDER_THRESHOLD:
            followthrough_end = i
            break

    return {
        "windup_start": int(windup_start),
        "release_frame": int(release_frame),
        "followthrough_end": int(followthrough_end),
    }


# ==============================
# I/O
# ==============================
def extract_shot_number(path: Path):
    m = re.search(r'\d+', path.stem)
    return int(m.group()) if m else float('inf')

def load_angles_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # sort by frame if present
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)

    required = ["elbow_flex_r", "shoulder_flex_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    # basic NaN handling
    return df[["elbow_flex_r", "shoulder_flex_r"]].astype(float)\
        .interpolate(limit_direction="both").bfill().ffill()

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    results = []
    if not input_folder.exists():
        print(f"[ERROR] Angles folder does not exist: {input_folder}")
    files = sorted(input_folder.glob("*.csv"), key=extract_shot_number)
    if not files:
        print(f"[ERROR] No angle CSVs found in {input_folder}")
    else:
        print(f"[INFO] Found {len(files)} angle file(s) in {input_folder}")

    for f in files:
        try:
            df_angles = load_angles_csv(f)
            phases = detect_throw_phases_from_angles(df_angles, FPS, threshold=10.0, window=3)
            phases["file"] = f.name
            results.append(phases)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

    pd.DataFrame(results, columns=["file","windup_start","release_frame","followthrough_end"]).to_csv(output_csv, index=False)
    print(f"[INFO] Saved phase data for {len(results)} files to {output_csv}")
