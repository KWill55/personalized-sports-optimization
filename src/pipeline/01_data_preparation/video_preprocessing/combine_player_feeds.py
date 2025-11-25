"""
Title: combine_player_feeds.py

Description: 
    This module's purpose is to combine player tracking feeds from two cameras into a single video feed.
    It combines two 640x640 videos into a single 1280x640 video for player tracking. 
    The combined video is saved in a structurured directory. 

Inputs:
    - Left and right player tracking videos (each 640x640)

Usage:
    - Running the script combines the two player feeds into a single video feed. 

Outputs
    - 1280x640 videos for player tracking:
        - side by side stereo feeds (each 640x640)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import re
import yaml

# =========================
# Config
# =========================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# =========================
# Paths and Directories 
# =========================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

# Input video directories 
input_video_dirs = {
    "left": session_dir / "videos" / "player_tracking" / "raw" / "left",
    "right": session_dir / "videos" / "player_tracking" / "raw" / "right",
}

# Output video directory 
output_video_dir = session_dir / "videos" / "player_tracking" / "synchronized"
output_video_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Helper: Get matching freethrow filenames
# =========================
def get_matching_video_pairs(left_dir, right_dir):
    
    # Match freethrow files like freethrow1.avi, freethrow2.avi, etc.
    pattern = re.compile(r"freethrow(\d+)\.avi")
    left_files = sorted(left_dir.glob("freethrow*.avi"))

    matches = []
    for lf in left_files:
        match = pattern.match(lf.name)
        if match:
            num = match.group(1)
            rf = right_dir / f"freethrow{num}.avi"
            if rf.exists():
                matches.append((lf, rf))
    return matches


# =========================
# Main Combining Logic
# =========================
def combine_videos():

    # Match left/right video pairs
    pairs = get_matching_video_pairs(input_video_dirs["left"], input_video_dirs["right"])
    print(f"Found {len(pairs)} matching left/right video pairs.")
    
    for left_path, right_path in pairs:
        print(f"Combining {left_path.name} and {right_path.name}...")

        # Open video readers
        left_cap = cv.VideoCapture(str(left_path))
        right_cap = cv.VideoCapture(str(right_path))

        # Get video properties from left
        fps = left_cap.get(cv.CAP_PROP_FPS)

        left_width = int(left_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        left_height = int(left_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        right_width = int(right_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        right_height = int(right_cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Ensure both videos are 640x640
        if (left_width, left_height) != (640, 640):
            print(f"❌ ERROR: {left_path.name} is {left_width}x{left_height}, expected 640x640")
            left_cap.release()
            right_cap.release()
            continue

        if (right_width, right_height) != (640, 640):
            print(f"❌ ERROR: {right_path.name} is {right_width}x{right_height}, expected 640x640")
            left_cap.release()
            right_cap.release()
            continue

        # If valid, proceed with 1280x640 output
        combined_width = 1280
        combined_height = 640


        # Setup output path
        output_name = left_path.name  # e.g. freethrow1.avi
        output_path = output_video_dir / output_name
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter(str(output_path), fourcc, fps, (combined_width, combined_height))

        total_frames = int(left_cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while True:
            ret_left, frame_left = left_cap.read()
            ret_right, frame_right = right_cap.read()

            if not ret_left or not ret_right:
                break

            # Combine side-by-side
            combined_frame = np.hstack((frame_left, frame_right))
            out.write(combined_frame)

            # Progress update
            frame_count += 1
            if frame_count % 50 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"   ➤ Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')


        left_cap.release()
        right_cap.release()
        out.release()
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    combine_videos()