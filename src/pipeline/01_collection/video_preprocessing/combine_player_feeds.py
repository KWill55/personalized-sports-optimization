"""
Title: combine_player_feeds.py

Description: 
    Combine the cropped player-tracking feeds from the left/right cameras into
    a single side-by-side video. Output dimensions are driven by the cropped
    stereo resolution from project_config.yaml.

Inputs:
    - Left and right player tracking videos (cropped stereo resolution)

Usage:
    - Running the script combines the two player feeds into a single video feed. 

Outputs
    - Side-by-side stereo feeds of size (2 * cropped_width, cropped_height)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import re
import yaml

# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
config_path = PROJECT_ROOT / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
NAME_PREFIX = cfg.get("freethrow_name_prefix", "freethrow")
PATHS_CFG = cfg.get("paths", {})

STEREO_CROP_RES = tuple(int(v) for v in cfg.get("cropped_stereo_resolution", cfg["uncropped_stereo_resolution"]))
CROP_WIDTH, CROP_HEIGHT = STEREO_CROP_RES
COMBINED_WIDTH = CROP_WIDTH * 2
COMBINED_HEIGHT = CROP_HEIGHT


def cfg_path(key: str) -> Path:
    """Resolve a path from project_config.yaml paths section."""
    try:
        template = PATHS_CFG[key]
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' in project_config.yaml paths") from exc
    return PROJECT_ROOT / Path(template.format(athlete=ATHLETE, session=SESSION))


# =========================
# Paths and Directories 
# =========================
input_video_dirs = {
    "left": cfg_path("player_tracking_left"),
    "right": cfg_path("player_tracking_right"),
}

output_video_dir = cfg_path("player_tracking_sync")
output_video_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Helper: Get matching freethrow filenames
# =========================
def get_matching_video_pairs(left_dir, right_dir):
    
    # Match files like {NAME_PREFIX}001.avi, {NAME_PREFIX}002.avi, etc.
    pattern = re.compile(rf"^{re.escape(NAME_PREFIX)}(\d+)$")
    left_files = sorted(left_dir.glob(f"{NAME_PREFIX}*.avi"))

    matches = []
    for lf in left_files:
        match = pattern.match(lf.stem)
        if match:
            num = match.group(1)  # Preserve leading zeros
            rf = right_dir / f"{NAME_PREFIX}{num}.avi"
            if rf.exists():
                matches.append((lf, rf))
    return matches


# =========================
# Main Combining Logic
# =========================
def combine_videos():

    left_dir = input_video_dirs["left"]
    right_dir = input_video_dirs["right"]

    if not left_dir.exists() or not right_dir.exists():
        print(f"[ERROR] Input directories missing:\n  left:  {left_dir}\n  right: {right_dir}")
        print("Ensure record_freethrows.py has saved the raw feeds before combining.")
        return

    # Match left/right video pairs
    pairs = get_matching_video_pairs(left_dir, right_dir)
    print(f"Found {len(pairs)} matching left/right video pairs.")
    if not pairs:
        print(f"[WARNING] No files starting with '{NAME_PREFIX}' were found in {left_dir}.")
        return
    
    for left_path, right_path in pairs:
        print(f"Combining {left_path.name} and {right_path.name}...")

        # Open video readers
        left_cap = cv.VideoCapture(str(left_path))
        right_cap = cv.VideoCapture(str(right_path))

        # Get video properties from left
        fps = left_cap.get(cv.CAP_PROP_FPS)
        if fps <= 0:
            fps = float(cfg.get("player_tracking_fps", 60))

        left_width = int(left_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        left_height = int(left_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        right_width = int(right_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        right_height = int(right_cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        expected_size = (CROP_WIDTH, CROP_HEIGHT)

        # Ensure both videos match configured crop size
        if (left_width, left_height) != expected_size:
            print(f"❌ ERROR: {left_path.name} is {left_width}x{left_height}, expected {expected_size}")
            left_cap.release()
            right_cap.release()
            continue

        if (right_width, right_height) != expected_size:
            print(f"❌ ERROR: {right_path.name} is {right_width}x{right_height}, expected {expected_size}")
            left_cap.release()
            right_cap.release()
            continue

        # If valid, proceed with combined output
        combined_width = COMBINED_WIDTH
        combined_height = COMBINED_HEIGHT


        # Setup output path
        output_name = left_path.name  # e.g. freethrow001.avi
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
