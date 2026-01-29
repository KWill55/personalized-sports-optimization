#!/usr/bin/env python3
"""
compute_mp33_angles.py

Make per-frame joint angles (deg) from MediaPipe 33 3D keypoint CSVs.

Input  : data/<ATHLETE>/<SESSION>/metrics/3d_keypoints/*_3d.csv
Output : data/<ATHLETE>/<SESSION>/metrics/angles/<same>_angles.csv

Angles:
  - elbow_flex_l/r     = angle at elbow  (shoulder-elbow-wrist)
  - shoulder_flex_l/r  = angle at shoulder (hip-shoulder-elbow)
  - hip_flex_l/r       = angle at hip (shoulder-hip-knee)    [proxy]
  - knee_flex_l/r      = angle at knee (hip-knee-ankle)
  - ankle_flex_l/r     = angle at ankle (knee-ankle-foot_index)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import re

# ---------- config / paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
cfg_path = PROJECT_ROOT / "project_config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
paths_cfg = cfg.get("paths", {})

def cfg_path_resolve(key: str) -> Path:
    try:
        template = paths_cfg[key]
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' in project_config.yaml paths") from exc
    return PROJECT_ROOT / Path(template.format(athlete=ATHLETE, session=SESSION))

in_dir  = cfg_path_resolve("keypoints_3d")
out_dir = cfg_path_resolve("angles")
out_dir.mkdir(parents=True, exist_ok=True)

# optional light smoothing (set 0 to disable)
SMOOTH_WINDOW = int(cfg.get("angle_smooth_window", 0))  # e.g., 5 or 7 (odd)

# ---------- MediaPipe 33 names ----------
NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]
IDX = {n:i for i,n in enumerate(NAMES)}

# ---------- helpers ----------
def load_mp33_csv_to_array(path: Path):
    """
    Return frames array (T,33,3) and a frame index array (T,).
    """
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        frames = df["frame"].to_numpy()
    else:
        frames = np.arange(len(df), dtype=int)

    # build columns in the strict MP order
    cols = []
    for name in NAMES:
        for suf in ("_x","_y","_z"):
            col = f"{name}{suf}"
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path.name}")
            cols.append(col)

    arr = df[cols].to_numpy(float).reshape(len(df), len(NAMES), 3)  # (T,33,3)
    return arr, frames

def angle_series(frames, a, b, c):
    """
    Angle at vertex 'b' (deg) across all frames using points a,b,c by name.
    """
    A = frames[:, IDX[a], :]
    B = frames[:, IDX[b], :]
    C = frames[:, IDX[c], :]
    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2

    ok = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1) & (denom > 1e-8)
    ang = np.full(len(frames), np.nan, float)
    if np.any(ok):
        cosang = np.einsum('ij,ij->i', v1[ok], v2[ok]) / denom[ok]
        cosang = np.clip(cosang, -1.0, 1.0)
        ang[ok] = np.degrees(np.arccos(cosang))
    return ang

def compute_angles(frames):
    """
    Compute a dictionary of angle time series (each (T,) in degrees).
    """
    ang = {}
    # Arms
    ang["elbow_flex_l"]    = angle_series(frames, "left_shoulder",  "left_elbow",  "left_wrist")
    ang["elbow_flex_r"]    = angle_series(frames, "right_shoulder", "right_elbow", "right_wrist")
    ang["shoulder_flex_l"] = angle_series(frames, "left_hip",       "left_shoulder","left_elbow")
    ang["shoulder_flex_r"] = angle_series(frames, "right_hip",      "right_shoulder","right_elbow")
    # Legs
    ang["hip_flex_l"]      = angle_series(frames, "left_shoulder",  "left_hip",   "left_knee")
    ang["hip_flex_r"]      = angle_series(frames, "right_shoulder", "right_hip",  "right_knee")
    ang["knee_flex_l"]     = angle_series(frames, "left_hip",       "left_knee",  "left_ankle")
    ang["knee_flex_r"]     = angle_series(frames, "right_hip",      "right_knee", "right_ankle")
    ang["ankle_flex_l"]    = angle_series(frames, "left_knee",      "left_ankle", "left_foot_index")
    ang["ankle_flex_r"]    = angle_series(frames, "right_knee",     "right_ankle","right_foot_index")
    return ang

def maybe_smooth(s, window):
    if window and window >= 3 and window % 2 == 1:
        # simple centered median then mean to tame spikes
        s = pd.Series(s, dtype=float)
        s = s.rolling(window, center=True, min_periods=1).median()
        s = s.rolling(window, center=True, min_periods=1).mean()
        return s.to_numpy(float)
    return s

def base_num(path: Path):
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else float('inf')

# ---------- main ----------
if __name__ == "__main__":
    files = sorted(in_dir.glob("*_3d.csv"), key=base_num)
    if not files:
        print(f"[ERROR] No *_3d.csv files in {in_dir}")
        raise SystemExit(1)

    print(f"[INFO] Making angles for {len(files)} files from {in_dir}")
    for f in files:
        try:
            frames, frame_idx = load_mp33_csv_to_array(f)
            ang = compute_angles(frames)
            # fill/smooth each series
            out = {"frame": frame_idx}
            for k, v in ang.items():
                # fill gaps at ends and inside
                s = pd.Series(v, dtype=float).interpolate(limit_direction="both")
                s = s.fillna(method="bfill").fillna(method="ffill")
                out[k] = maybe_smooth(s.to_numpy(float), SMOOTH_WINDOW)

            df_out = pd.DataFrame(out)
            out_path = out_dir / f"{f.stem.replace('_3d','')}_angles.csv"
            df_out.to_csv(out_path, index=False)
            print(f"  ✔ {f.name} -> {out_path.name}  (T={len(df_out)})")
        except Exception as e:
            print(f"  ✖ {f.name}: {e}")

    print(f"[DONE] Wrote angles to {out_dir}")
