import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import pickle

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
clip_base = "freethrow2_sync"
left_path = Path(f"../../data/freethrows1/02_process_data/left_2d/{clip_base}_left_2d.csv")
right_path = Path(f"../../data/freethrows1/02_process_data/right_2d/{clip_base}_right_2d.csv")

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
output_path = Path(f"../../data/freethrows1/02_process_data/triangulated/{clip_base}_3d.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"Saved triangulated 3D keypoints to: {output_path}")
