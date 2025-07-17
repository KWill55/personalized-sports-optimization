

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import yaml

# ========================================
# Config
# ========================================

config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# ======================================== 
# Paths 
# ========================================
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION

# Calibration path
calib_path = session_dir / "calibration" / "stereo_calib.npz"

# Input 2D keypoint CSVs
left_dir = session_dir / "videos" / "player_tracking" / "processed" / "left"
right_dir = session_dir / "videos" / "player_tracking" / "processed" / "right"

# Output directory
output_dir = session_dir / "02_process_data" / "triangulated"
output_dir.mkdir(parents=True, exist_ok=True)

# ======================================== 
# Load Calibration Parameters
# ========================================
calib = np.load(calib_path)
K1, D1 = calib["mtxL"], calib["distL"]
K2, D2 = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))

# ========================================
# MediaPipe landmark names
# ========================================
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# ========================================
# Triangulation Function
# ========================================
def triangulate_clip(left_csv, right_csv, output_path):
    df_left = pd.read_csv(left_csv)
    df_right = pd.read_csv(right_csv)
    triangulated_data = []

    for idx in range(len(df_left)):
        frame_data = [idx]
        for name in landmark_names:
            lx, ly = df_left.loc[idx, f"{name}_x"], df_left.loc[idx, f"{name}_y"]
            rx, ry = df_right.loc[idx, f"{name}_x"], df_right.loc[idx, f"{name}_y"]

            if -1 in (lx, ly, rx, ry):
                frame_data.extend([-1, -1, -1])
                continue

            pt_left = np.array([[[lx, ly]]], dtype=np.float32)
            pt_right = np.array([[[rx, ry]]], dtype=np.float32)
            undist_left = cv2.undistortPoints(pt_left, K1, D1, P=K1).reshape(2, 1)
            undist_right = cv2.undistortPoints(pt_right, K2, D2, P=K2).reshape(2, 1)
            point_4d = cv2.triangulatePoints(P1, P2, undist_left, undist_right)
            point_3d = point_4d[:3] / point_4d[3]
            frame_data.extend(point_3d.flatten())

        triangulated_data.append(frame_data)

    # Save output
    columns = ["frame"]
    for name in landmark_names:
        columns += [f"{name}_x", f"{name}_y", f"{name}_z"]
    df_out = pd.DataFrame(triangulated_data, columns=columns)
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved 3D keypoints to: {output_path.name}")

# ========================================
# Batch Process All CSVs
# ========================================
for left_file in sorted(left_dir.glob("*_left_2d.csv")):
    clip_base = left_file.stem.replace("_left_2d", "")
    right_file = right_dir / f"{clip_base}_right_2d.csv"

    if not right_file.exists():
        print(f"⚠️ Skipping {clip_base}: right file not found.")
        continue

    output_csv = output_dir / f"{clip_base}_3d.csv"
    triangulate_clip(left_file, right_file, output_csv)
