import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# ========================================
# Config
# ========================================

ATHLETE = "kenny"
SESSION = "session_001"
VIEW = "right"  # "left" or "right"
DRAW_DELAY_MS = 33  # ~30 FPS playback


# ========================================
# Paths
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / ATHLETE / SESSION

csv_folder = session_dir / "metrics" / "player_tracking_metrics" / "time_series" / "2d_csvs"
video_folder = session_dir / "videos" / "player_tracking" / "synchronized" / VIEW

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
# Process each CSV/video pair
# ========================================

csv_files = sorted(csv_folder.glob(f"*_{VIEW}_2d.csv"))

for csv_path in csv_files:
    clip_name = csv_path.name.replace(f"_{VIEW}_2d.csv", "")
    video_path = video_folder / f"{clip_name}.mp4"

    if not video_path.exists():
        print(f"[WARNING] Skipping {clip_name}: video not found.")
        continue

    print(f"[INFO] Playing: {clip_name}")

    # Load CSV
    df = pd.read_csv(csv_path)
    keypoints_all = df.drop(columns=["frame"]).values.reshape((-1, 33, 3))  # (num_frames, 33, 3)

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(keypoints_all):
            break

        keypoints = keypoints_all[frame_idx]
        frame_h, frame_w = frame.shape[:2]

        # Draw keypoints
        for i, (x, y, v) in enumerate(keypoints):
            if v > 0.3:
                cx, cy = int(x * frame_w), int(y * frame_h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw connections
        for a, b in POSE_CONNECTIONS:
            if keypoints[a][2] > 0.3 and keypoints[b][2] > 0.3:
                ax, ay = int(keypoints[a][0] * frame_w), int(keypoints[a][1] * frame_h)
                bx, by = int(keypoints[b][0] * frame_w), int(keypoints[b][1] * frame_h)
                cv2.line(frame, (ax, ay), (bx, by), (255, 0, 0), 2)

        # Show frame
        cv2.imshow(f"{clip_name} - {VIEW}", frame)
        if cv2.waitKey(DRAW_DELAY_MS) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()