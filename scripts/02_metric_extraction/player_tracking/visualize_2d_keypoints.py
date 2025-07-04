import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# ========================================
# Config
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"
CLIP_NAME = "freethrow1_sync" 
VIEW = "right" 

frame_to_draw = 100  # <<< change this to the frame you want to display

# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / ATHLETE / SESSION

csv_path = session_dir / "extracted_metrics" / "player_tracking_metrics" / "raw_metrics" / f"{CLIP_NAME}_{VIEW}_2d.csv"
video_path = session_dir / "videos" / "player_tracking" / "processed" / VIEW / f"{CLIP_NAME}.mp4"


# ========================================
# MediaPipe COCO-style landmark connections
# ========================================

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23), (12, 24),  # torso
    (0, 11), (0, 12),    # head to shoulders
]

# ========================================
# Load data
# ========================================

df = pd.read_csv(csv_path)
frame_data = df[df["frame"] == frame_to_draw].values[0][1:]  # skip frame index

# Reshape into (33, 3) → (x, y, visibility)
keypoints = np.array(frame_data).reshape(-1, 3)

# ========================================
# Load video frame
# ========================================

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_draw)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError(f"Could not read frame {frame_to_draw} from {video_path}")

frame_h, frame_w = frame.shape[:2]

# ========================================
# Draw keypoints and lines
# ========================================

for i, (x, y, v) in enumerate(keypoints):
    if v > 0.3:  # visibility threshold
        cx, cy = int(x * frame_w), int(y * frame_h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

for a, b in POSE_CONNECTIONS:
    if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
        ax, ay = int(keypoints[a][0] * frame_w), int(keypoints[a][1] * frame_h)
        bx, by = int(keypoints[b][0] * frame_w), int(keypoints[b][1] * frame_h)
        cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

# ========================================
# Show image
# ========================================

cv2.imshow(f"{CLIP_NAME} - Frame {frame_to_draw}", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
