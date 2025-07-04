import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# ======================================== 
# Configuration Parameters 
# ========================================

ATHLETE = "tests"
SESSION = "player_tracking_tests"

# ======================================== 
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]  # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION 
videos_dir = session_dir / "videos"
extracted_dir = session_dir / "extracted_metrics"

# input video paths
left_video_dir = videos_dir / "player_tracking" / "processed" / "left"
right_video_dir = videos_dir / "player_tracking" / "processed" / "right" 

# Define output directories
left_out_dir = extracted_dir / "player_tracking_metrics" / "timeseries" / "raw_2d" / "raw" / "left"
right_out_dir = extracted_dir / "player_tracking_metrics" / "timeseries" / "raw_2d" / "raw" / "right"


# ======================================== 
# Functions
# ========================================

def extract_2d_keypoints(video_path, output_csv):
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    
    all_frames = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.visibility])
        else:
            # Fill with -1 for missing landmarks
            keypoints = [-1] * (33 * 3)

        all_frames.append([frame_idx] + keypoints)
        frame_idx += 1

    cap.release()
    pose.close()

   # MediaPipe landmark names (0–32)
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

    columns = ['frame']
    for name in landmark_names:
        columns += [f'{name}_x', f'{name}_y', f'{name}_v']


    df = pd.DataFrame(all_frames, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved 2D keypoints to {output_csv}")

for left_video in sorted(left_video_dir.glob("*.mp4")):
    clip_base = left_video.stem  # e.g., 'freethrow1_sync'
    right_video = right_video_dir / f"{clip_base}.mp4"

    if not right_video.exists():
        print(f"Skipping {clip_base}: no matching right video.")
        continue

    # Create output directories if they don't exist
    left_out_dir.mkdir(parents=True, exist_ok=True)
    right_out_dir.mkdir(parents=True, exist_ok=True)

    # Construct output file paths
    left_out = left_out_dir / f"{clip_base}_left_2d.csv"
    right_out = right_out_dir / f"{clip_base}_right_2d.csv"

    # Extract and save keypoints
    extract_2d_keypoints(left_video, left_out)
    extract_2d_keypoints(right_video, right_out)


