"""
Title: auto_trim_by_flash.py

Purpose:
    automate the process of trimming video files through the use
    of LED flashes 

Output
    - trimmed video files 

Usage:
    - just run the script

"""

import os
import cv2
import numpy as np
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

ATHLETE = "kenny"
SESSION = "session_test"

VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mov', '.hevc']
RESIZE_DIMENSIONS = (640, 480)

START_HSV_LOWER = np.array([0, 0, 200])      # white flash
START_HSV_UPPER = np.array([180, 30, 255])

STOP_HSV_LOWER = np.array([0, 0, 200])      # white flash again (red no work)
STOP_HSV_UPPER = np.array([180, 30, 255])


BRIGHTNESS_THRESHOLD = 200
PIXEL_RATIO_THRESHOLD = 0.02
MIN_FRAME_SEPARATION = 30  # Minimum frames between start and stop (adjust based on your FPS and flash duration)


# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

raw_left_dir = session_dir / "videos" / "player_tracking" / "raw" / "left"
raw_right_dir = session_dir / "videos" / "player_tracking" / "raw" / "right"
raw_ball_dir = session_dir / "videos" / "ball_tracking" / "raw"

trimmed_left_dir = session_dir / "videos" / "player_tracking" / "trimmed" / "left"
trimmed_right_dir = session_dir / "videos" / "player_tracking" / "trimmed" / "right"
trimmed_ball_dir = session_dir / "videos" / "ball_tracking" / "trimmed"

# ========================================
# Flash Detection Function
# ========================================

def is_flash_frame(hsv_frame, lower, upper):
    mask = cv2.inRange(hsv_frame, lower, upper)
    ratio = np.sum(mask > 0) / mask.size
    return ratio > PIXEL_RATIO_THRESHOLD

# ========================================
# Trim Video by Flash
# ========================================

def trim_video_by_flash(input_path, output_path):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = None
    stop_frame = None
    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if start_frame is None and is_flash_frame(hsv, START_HSV_LOWER, START_HSV_UPPER):
            print(f"[INFO] Start flash detected at frame {frame_index}")
            start_frame = frame_index

        elif (
            start_frame is not None and
            stop_frame is None and
            frame_index - start_frame > MIN_FRAME_SEPARATION and
            is_flash_frame(hsv, STOP_HSV_LOWER, STOP_HSV_UPPER)
        ):
            print(f"[INFO] Stop flash detected at frame {frame_index}")
            stop_frame = frame_index
            break

        frames.append(frame)
        frame_index += 1

    cap.release()

    if start_frame is None or stop_frame is None or start_frame >= stop_frame:
        print(f"[ERROR] Flash detection failed for {input_path.name}")
        return

    print(f"[INFO] Trimming {input_path.name} from frame {start_frame} to {stop_frame}...")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frames[start_frame:stop_frame + 1]:
        out.write(f)
    out.release()
    print(f"[SUCCESS] Saved: {output_path.name}")


# ========================================
# Batch Process All Videos
# ========================================

video_sets = [
    {"input": raw_left_dir,  "output": trimmed_left_dir},
    {"input": raw_right_dir, "output": trimmed_right_dir},
    {"input": raw_ball_dir,  "output": trimmed_ball_dir},
]

for set_info in video_sets:
    input_dir = set_info["input"]
    output_dir = set_info["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_name = input_dir.parts[-1]  # Get folder name like "left", "right", or "raw"

    for filename in os.listdir(input_dir):
        if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        input_path = input_dir / filename
        output_path = output_dir / filename.replace(".mp4", "_trimmed.mp4").replace(".mov", "_trimmed.mp4")

        print(f"\n📹 Processing {camera_name.upper()} - {filename}")
        trim_video_by_flash(input_path, output_path)

