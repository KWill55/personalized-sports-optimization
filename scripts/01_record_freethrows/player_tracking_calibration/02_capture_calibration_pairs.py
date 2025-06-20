"""
Title: 02_capture_calibration_pairs.py: Stereo Image Pair Capture Script

Purpose:
    Captures synchronized image pairs from two USB webcams for calibration purposes.
    Allows the user to save pairs of images by pressing SPACE, and exit the script with ESC.

Prerequisites:
    - Two USB webcams connected to the system.
    - Ensure the cameras are positioned to capture the same scene.
    - Adjust camera settings (focus, zoom) as needed before starting the script. (tune_intrinsics.py)

Output:
    - Saves images into calib_images/left and calib_images/right directories.
    - Each pair of images is saved with a sequential filename format (left_00.jpg, right_00.jpg, etc.).
"""

import cv2 as cv
import os
from pathlib import Path


# ========================================
# Configuration Constants 
# ========================================

CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 2 

# path parameters
ATHLETE = "Kenny" 
SESSION = "session_001"

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3] # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION
left_calib_dir = session_dir / "videos" / "calib_images" / "left"
right_calib_dir = session_dir / "videos" / "calib_images" / "right"

os.makedirs(right_calib_dir, exist_ok=True)
os.makedirs(left_calib_dir, exist_ok=True)

# ========================================
# Setup
# ========================================

# open video capture for both cameras
capL = cv.VideoCapture(CAMERA_LEFT_INDEX)
capR = cv.VideoCapture(CAMERA_RIGHT_INDEX)

if not capL.isOpened() or not capR.isOpened():
    print("[ERROR] Could not open one or both cameras.")
    exit()

frame_id = 0
print("[INFO] Press SPACE to capture image pair. Press ESC to quit.")

# ========================================
# Main Loop
# ========================================

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("[WARNING] Failed to grab frames.")
        continue

    # Display live preview
    cv.imshow("Left Camera", frameL)
    cv.imshow("Right Camera", frameR)

    key = cv.waitKey(1)
    if key % 256 == 27:  # ESC key
        print("[INFO] Exiting.")
        break
    elif key % 256 == 32:  # SPACE key
        fnameL = os.path.join(left_calib_dir, f"left_{frame_id:02}.jpg")
        fnameR = os.path.join(right_calib_dir, f"right_{frame_id:02}.jpg")
        cv.waitKey(300)  # Wait 300 ms after saving (debounce)
        cv.imwrite(fnameL, frameL)
        cv.imwrite(fnameR, frameR)
        print(f"[INFO] Saved pair #{frame_id} → {fnameL}, {fnameR}")
        frame_id += 1

# ========================================
# Cleanup
# ========================================

capL.release()
capR.release()
cv.destroyAllWindows()
