import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import pickle

ATHLETE = "tests" 
SESSION = "player_tracking_tests"
LEFT_CLIP_NAME = "freethrow1_sync_left_2d.csv"
RIGHT_CLIP_NAME = "freethrow1_sync_right_2d.csv"

# ======================================== 
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root

# input csv paths
left_path = base_dir / "data" / ATHLETE / SESSION / "player_tracking_1" / "02_process_data" / "left_2d" / LEFT_CLIP_NAME
right_path = base_dir / "data" / ATHLETE / SESSION / "player_tracking_1" / "02_process_data" / "right_2d" / RIGHT_CLIP_NAME

# output path
output_path = base_dir / "data" / ATHLETE / SESSION / "02_process_data" / "triangulated" / LEFT_CLIP_NAME.replace("_2d", "_3d")
output_path.parent.mkdir(parents=True, exist_ok=True)

# ======== Load calibration from pickle ========
with open("cameraIntrinsicsExtrinsics.pickle", "rb") as f:
    calib = pickle.load(f)

K = calib["intrinsicMat"]
R = calib["rotation"]
T = calib["translation"]

P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, T))

# ======== MediaPipe landmark names (0–32) ========
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

# ======== Load 2D CSVs ========

df_left = pd.read_csv(left_path)
df_right = pd.read_csv(right_path)

# ======== Triangulation ========
triangulated_data = []

for idx in range(len(df_left)):
    frame_data = [idx]
    for i, name in enumerate(landmark_names):
        lx, ly = df_left.iloc[idx][f"{name}_x"], df_left.iloc[idx][f"{name}_y"]
        rx, ry = df_right.iloc[idx][f"{name}_x"], df_right.iloc[idx][f"{name}_y"]

        if -1 in (lx, ly, rx, ry):
            frame_data.extend([-1, -1, -1])
            continue

        # Prepare homogeneous 2D points
        pt_left = np.array([[lx], [ly]], dtype=np.float64)
        pt_right = np.array([[rx], [ry]], dtype=np.float64)

        point_4d = cv2.triangulatePoints(P1, P2, pt_left, pt_right)
        point_3d = point_4d[:3] / point_4d[3]  # Convert from homogeneous

        frame_data.extend(point_3d.flatten())

    triangulated_data.append(frame_data)

# ======== Save as CSV ========
columns = ["frame"]
for name in landmark_names:
    columns += [f"{name}_x", f"{name}_y", f"{name}_z"]

df_out = pd.DataFrame(triangulated_data, columns=columns)
df_out.to_csv(output_path, index=False)

print(f"Saved triangulated 3D keypoints to: {output_path}")
