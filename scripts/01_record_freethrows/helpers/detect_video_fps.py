"""
Title: detect_fps_from_videos.py

Description:
    This script loads video files and reports:
    - The FPS encoded in the video
    - The actual read speed (approx FPS)

Usage:
    Place video files in the specified folder and run the script.
"""

import cv2
import time
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

ATHLETE = "Kenny"  # Change this to your athlete's name
SESSION = "session_001"
ANGLE = "player_tracking" 
SIDE = "left"  # Change to "right" if needed or "" empty if ball tracking

SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov"]
FRAME_COUNT_TEST = 100  # Number of frames to use for actual FPS test


# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3] # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION

video_folder = session_dir / "videos" / ANGLE / "raw" / SIDE

# ========================================
# Process Each Video
# ========================================

video_files = [f for f in video_folder.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

if not video_files:
    print("[ERROR] No video files found.")
    exit()

for video_path in video_files:
    print(f"\n[INFO] Processing: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    # Get reported FPS
    encoded_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Encoded FPS: {encoded_fps:.2f}")

    # Read frames and time it
    frames_read = 0
    start = time.time()

    while frames_read < FRAME_COUNT_TEST:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Video ended at frame {frames_read}")
            break
        frames_read += 1

    end = time.time()
    cap.release()

    actual_fps = frames_read / (end - start)
    print(f"[RESULT] Actual read FPS: {actual_fps:.2f}")
