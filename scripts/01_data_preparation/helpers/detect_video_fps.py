"""
Title: detect_fps_from_videos.py

Description:
    This script loads video files and reports:
    - The FPS encoded in the video
    - The actual read speed (approx FPS)

Usage:
    Place video files in the specified folder and run the script.

Notes:
    - if FPS is lower than expected, try using external USB driver for one camera. 
"""

import cv2
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"
ANGLE = "ball_tracking" # "player_tracking", "ball_tracking"
SIDE = "" #"left", "right", "" (for ball tracking)

SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov"]

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3]  # Adjust if needed
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

    encoded_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / encoded_fps if encoded_fps else 0

    print(f"[INFO] Encoded FPS       : {encoded_fps:.2f}")
    print(f"[INFO] Frame count       : {frame_count}")
    print(f"[INFO] Duration (seconds): {duration_sec:.2f}")

    # Optional: Calculate real average FPS from metadata
    avg_fps = frame_count / duration_sec if duration_sec else 0
    print(f"[RESULT] Estimated FPS   : {avg_fps:.2f}")

    cap.release()
