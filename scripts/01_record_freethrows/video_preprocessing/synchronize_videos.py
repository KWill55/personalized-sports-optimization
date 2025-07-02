
"""
Title: synchronize_videos.py

Purpose: synchronize downsampled videos 
"""

import cv2
from pathlib import Path
import numpy as np

# ========================================
# Constants 
# ========================================

ATHLETE = "kenny"
SESSION = "session_001"

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

downsampled_left_dir = session_dir / "videos" / "player_tracking" / "downsampled" / "left"
downsampled_right_dir = session_dir / "videos" / "player_tracking" / "downsampled" / "right"
downsampled_ball_dir = session_dir / "videos" / "ball_tracking" / "downsampled"

synchronized_left_dir = session_dir / "videos" / "player_tracking" / "synchronized" / "left"
synchronized_right_dir = session_dir / "videos" / "player_tracking" / "synchronized" / "right"
synchronized_ball_dir = session_dir / "videos" / "ball_tracking" / "synchronized"

# Make sure output dirs exist
for dir_path in [synchronized_left_dir, synchronized_right_dir, synchronized_ball_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ========================================
# Functions 
# ========================================

def detect_flash_index(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_brightness = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        frame_brightness.append(brightness)

    cap.release()

    # Detect flash by finding the frame with the sharpest brightness increase
    brightness_diff = np.diff(frame_brightness)
    flash_index = int(np.argmax(brightness_diff)) + 1  # +1 since diff is one shorter

    return flash_index, len(frame_brightness)

def synchronize_videos(input_dir: Path, output_dir: Path):
    for video_path in input_dir.glob("*.mp4"):
        print(f"Processing {video_path.name}...")
        flash_index, total_frames = detect_flash_index(video_path)
        print(f"Detected flash at frame {flash_index} out of {total_frames}")

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = output_dir / video_path.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Skip to flash frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, flash_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print(f"Synchronized video saved to {out_path.name}\n")


# ========================================
# Run Downsampling on All Cameras
# ========================================

synchronize_videos(downsampled_left_dir, synchronized_left_dir)
synchronize_videos(downsampled_right_dir, synchronized_right_dir)
synchronize_videos(downsampled_ball_dir, synchronized_ball_dir)