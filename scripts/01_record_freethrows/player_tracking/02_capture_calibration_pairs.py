# ========================================
# Title: Stereo Image Pair Capture Script
# ========================================
# Purpose: Captures synchronized image pairs from two USB webcams.
# 
# Usage: Press SPACE to save a pair. Press ESC to quit.
# 
# Output: Saves images into calib_images/left and calib_images/right.
# ========================================

import cv2 as cv
import os
from pathlib import Path


# ========================================
# Configuration
# ========================================

CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 2 

# path parameters
SESSION = "test_own_cameras"
BASE_DIR = Path(__file__).resolve().parents[2]
SAVE_DIR_LEFT = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images" / "left"
SAVE_DIR_RIGHT = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images" / "right"

# ========================================
# Setup
# ========================================
os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

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
        fnameL = os.path.join(SAVE_DIR_LEFT, f"left_{frame_id:02}.jpg")
        fnameR = os.path.join(SAVE_DIR_RIGHT, f"right_{frame_id:02}.jpg")
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
