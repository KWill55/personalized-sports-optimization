"""
Title: 05_record_freethrows 

Purpose:
    - Records stereo video pairs for each basketball free throw,
    saving each throw as a separate video file using stereo calibration parameters.

Prerequisites:
    - Ensure stereo cameras are calibrated and the calibration file 'stereo_calib.npz' exists. (use 03_calibrate_stereo.py)

Usage:
    - Press 's' to start recording a throw (3 seconds by default)
    - Press 'q' to quit at any time

Important Notes:
    - For now this script only records player biomechanics data, not ball tracking.
    Ball tracking will be done manually for now. 
    - i actually havne't gotten this to work yet
"""

import cv2 as cv
import numpy as np
import os
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1

ATHLETE = "Kenny"
SESSION = "session_001"

# === Settings ===
CLIP_LENGTH = 3  # seconds
FPS = 30

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[2]
session_dir = base_dir / "data" / ATHLETE / SESSION

# Calibration file
calib_path = session_dir / "calibration" / "stereo_calibration" / "stereo_calib.npz"

# Output directories for recorded throws
left_videos_dir = session_dir / "videos" / "player_tracking" / "raw" / "left"
right_videos_dir = session_dir / "videos" / "player_tracking" / "raw" / "right"

# Create directories if they don't exist
left_videos_dir.mkdir(parents=True, exist_ok=True)
right_videos_dir.mkdir(parents=True, exist_ok=True)


# ========================================
# Load Calibration Parameters 
# ========================================

calib = np.load(calib_path)
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

# ========================================
# Open Camera Streams
# ========================================

capL = cv.VideoCapture(CAMERA_LEFT_INDEX)
capR = cv.VideoCapture(CAMERA_RIGHT_INDEX)

retL, frameL = capL.read()
retR, frameR = capR.read()
if not retL or not retR:
    raise RuntimeError("Could not read from both cameras.")

h, w = frameL.shape[:2]
image_size = (w, h)

print("Press 's' to record a throw. Press 'q' to quit.")

# ========================================
# Stereo Rectification
# ========================================

# === Rectification Maps ===
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, flags=cv.CALIB_ZERO_DISPARITY)
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv.CV_16SC2)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv.CV_16SC2)

# ========================================
# Main Recording Loop
# ========================================

throw_index = 1
frame_count = CLIP_LENGTH * FPS

while True:
    # Always show live preview
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Camera read error.")
        break

    previewL = cv.remap(frameL, map1L, map2L, cv.INTER_LINEAR)
    previewR = cv.remap(frameR, map1R, map2R, cv.INTER_LINEAR)
    combined_preview = np.hstack((previewL, previewR))
    cv.imshow("Live Preview (press 's' to start recording)", combined_preview)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print(f"🎬 Recording throw {throw_index}...")

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        outL = cv.VideoWriter(str(left_videos_dir / f"throw_{throw_index:03}.avi"), fourcc, FPS, image_size)
        outR = cv.VideoWriter(str(right_videos_dir / f"throw_{throw_index:03}.avi"), fourcc, FPS, image_size)

        for _ in range(frame_count):
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            if not retL or not retR:
                print("Camera read error.")
                break

            rectL = cv.remap(frameL, map1L, map2L, cv.INTER_LINEAR)
            rectR = cv.remap(frameR, map1R, map2R, cv.INTER_LINEAR)

            outL.write(rectL)
            outR.write(rectR)

            combined = np.hstack((rectL, rectR))
            cv.imshow("Recording (press 'q' to stop)", combined)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        outL.release()
        outR.release()
        print(f"Throw {throw_index} saved.")
        throw_index += 1

    elif key == ord('q'):
        print("Exiting.")
        break

# ========================================
# Cleanup
# ========================================

capL.release()
capR.release()
cv.destroyAllWindows()
