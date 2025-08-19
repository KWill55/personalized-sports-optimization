#!/usr/bin/env python3
"""
compute_3d_vel_acc.py

Per-frame 3D velocities and accelerations from MediaPipe 33 keypoint CSVs.

Input :
  data/<ATHLETE>/<SESSION>/metrics/3d_keypoints/*_3d.csv

Output:
  data/<ATHLETE>/<SESSION>/metrics/3d_velocities/<same>_3d_velocities.csv
  data/<ATHLETE>/<SESSION>/metrics/3d_accelerations/<same>_3d_accelerations.csv

For each landmark L:
  Velocities:  L_vx, L_vy, L_vz, L_speed
  Accelerations: L_ax, L_ay, L_az, L_accel
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import re

# ---------- config / paths ----------
cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
FPS = float(cfg.get("player_tracking_fps", cfg.get("ball_tracking_fps", 30.0)))

# optional light smoothing (set 0 to disable; use odd values like 5/7)
POS_SMOOTH_WINDOW = int(cfg.get("kin_pos_smooth_window", 0))  # pre-diff smoothing on x,y,z
VEL_SMOOTH_WINDOW = int(cfg.get("kin_vel_smooth_window", 0))  # post-diff smoothing on vx,vy,vz,speed
ACC_SMOOTH_WINDOW = int(cfg.get("kin_acc_smooth_window", 0))  # post-diff smoothing on ax,ay,az,accel

base_dir    = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
in_dir      = session_dir / "metrics" / "3d_keypoints"
vel_dir     = session_dir / "metrics" / "3d_velocities"
acc_dir     = session_dir / "metrics" / "3d_accelerations"
vel_dir.mkdir(parents=True, exist_ok=True)
acc_dir.mkdir(parents=True, exist_ok=True)

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
    Requires columns like 'left_wrist_x', 'left_wrist_y', 'left_wrist_z'.
    """
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        frames = df["frame"].to_numpy()
    else:
        frames = np.arange(len(df), dtype=int)

    cols = []
    for name in NAMES:
        for suf in ("_x","_y","_z"):
            col = f"{name}{suf}"
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path.name}")
            cols.append(col)

    arr = df[cols].to_numpy(float).reshape(len(df), len(NAMES), 3)  # (T,33,3)
    return arr, frames

def maybe_smooth_vec(x: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth a (T,) series with centered median+mean (like your angles script).
    """
    if window and window >= 3 and window % 2 == 1:
        s = pd.Series(x, dtype=float)
        s = s.rolling(window, center=True, min_periods=1).median()
        s = s.rolling(window, center=True, min_periods=1).mean()
        return s.to_numpy(float)
    return x

def interp_fill_vec(x: np.ndarray) -> np.ndarray:
    """
    Fill NaNs via linear interpolation with edge fill (1D).
    """
    s = pd.Series(x, dtype=float)
    s = s.interpolate(limit_direction="both")
    s = s.fillna(method="bfill").fillna(method="ffill")
    return s.to_numpy(float)

def central_diff_1d(x: np.ndarray, fps: float) -> np.ndarray:
    """
    Central difference derivative for a (T,) signal at sample rate 'fps'.
    Uses forward/backward at the ends.
    """
    T = len(x)
    if T == 0:
        return np.array([], dtype=float)
    if T == 1:
        return np.zeros(1, dtype=float)
    dx = np.zeros(T, dtype=float)
    # forward/backward at edges
    dx[0]  = (x[1] - x[0]) * fps
    dx[-1] = (x[-1] - x[-2]) * fps
    if T > 2:
        dx[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    return dx

def base_num(path: Path):
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else float('inf')

# ---------- core computation ----------
def compute_vel_acc(frames: np.ndarray, fps: float):
    """
    frames: (T,33,3) positions
    returns: (vel_dict, acc_dict) with columns for every landmark
    """
    T = frames.shape[0]
    vel_cols = {}
    acc_cols = {}

    # Optional: smooth/interp positions per joint/dim before diff
    pos = frames.copy()
    for j in range(len(NAMES)):
        for d in range(3):
            pos[:, j, d] = interp_fill_vec(pos[:, j, d])
            pos[:, j, d] = maybe_smooth_vec(pos[:, j, d], POS_SMOOTH_WINDOW)

    # Velocities
    vx = np.zeros((T, len(NAMES)), dtype=float)
    vy = np.zeros((T, len(NAMES)), dtype=float)
    vz = np.zeros((T, len(NAMES)), dtype=float)
    for j in range(len(NAMES)):
        vx[:, j] = central_diff_1d(pos[:, j, 0], fps)
        vy[:, j] = central_diff_1d(pos[:, j, 1], fps)
        vz[:, j] = central_diff_1d(pos[:, j, 2], fps)

    speed = np.sqrt(vx*vx + vy*vy + vz*vz)

    # Optional smoothing of velocity series
    for j, name in enumerate(NAMES):
        vxs = maybe_smooth_vec(vx[:, j], VEL_SMOOTH_WINDOW)
        vys = maybe_smooth_vec(vy[:, j], VEL_SMOOTH_WINDOW)
        vzs = maybe_smooth_vec(vz[:, j], VEL_SMOOTH_WINDOW)
        spd = maybe_smooth_vec(speed[:, j], VEL_SMOOTH_WINDOW)

        vel_cols[f"{name}_vx"] = vxs
        vel_cols[f"{name}_vy"] = vys
        vel_cols[f"{name}_vz"] = vzs
        vel_cols[f"{name}_speed"] = spd

    # Accelerations from (optionally smoothed) velocities
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    az = np.zeros_like(vz)
    for j in range(len(NAMES)):
        ax[:, j] = central_diff_1d(vel_cols[f"{NAMES[j]}_vx"], fps)
        ay[:, j] = central_diff_1d(vel_cols[f"{NAMES[j]}_vy"], fps)
        az[:, j] = central_diff_1d(vel_cols[f"{NAMES[j]}_vz"], fps)
    accel = np.sqrt(ax*ax + ay*ay + az*az)

    # Optional smoothing of accelerations
    for j, name in enumerate(NAMES):
        axs = maybe_smooth_vec(ax[:, j], ACC_SMOOTH_WINDOW)
        ays = maybe_smooth_vec(ay[:, j], ACC_SMOOTH_WINDOW)
        azs = maybe_smooth_vec(az[:, j], ACC_SMOOTH_WINDOW)
        acc = maybe_smooth_vec(accel[:, j], ACC_SMOOTH_WINDOW)

        acc_cols[f"{name}_ax"] = axs
        acc_cols[f"{name}_ay"] = ays
        acc_cols[f"{name}_az"] = azs
        acc_cols[f"{name}_accel"] = acc

    return vel_cols, acc_cols

# ---------- main ----------
if __name__ == "__main__":
    files = sorted(in_dir.glob("*_3d.csv"), key=base_num)
    if not files:
        print(f"[ERROR] No *_3d.csv files in {in_dir}")
        raise SystemExit(1)

    print(f"[INFO] Making velocities & accelerations for {len(files)} files from {in_dir}")
    print(f"       FPS={FPS}  pos_smooth={POS_SMOOTH_WINDOW}  vel_smooth={VEL_SMOOTH_WINDOW}  acc_smooth={ACC_SMOOTH_WINDOW}")

    for f in files:
        try:
            frames, frame_idx = load_mp33_csv_to_array(f)
            vel_cols, acc_cols = compute_vel_acc(frames, FPS)

            vel_out = {"frame": frame_idx}
            acc_out = {"frame": frame_idx}
            # attach all columns
            for k, v in vel_cols.items():
                vel_out[k] = v
            for k, v in acc_cols.items():
                acc_out[k] = v

            df_vel = pd.DataFrame(vel_out)
            df_acc = pd.DataFrame(acc_out)

            vel_path = vel_dir / f"{f.stem.replace('_3d','')}_3d_velocities.csv"
            acc_path = acc_dir / f"{f.stem.replace('_3d','')}_3d_accelerations.csv"
            df_vel.to_csv(vel_path, index=False)
            df_acc.to_csv(acc_path, index=False)

            print(f"  ✔ {f.name} -> {vel_path.name}  &  {acc_path.name}  (T={len(df_vel)})")
        except Exception as e:
            print(f"  ✖ {f.name}: {e}")

    print(f"[DONE] Wrote velocities to {vel_dir}")
    print(f"[DONE] Wrote accelerations to {acc_dir}")
