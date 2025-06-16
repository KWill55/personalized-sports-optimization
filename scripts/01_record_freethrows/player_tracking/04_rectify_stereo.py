"""
Stereo Camera Rectifying Script

Purpose:
    - Rectifies stereo camera images using pre-calibrated parameters to align the left and right camera feeds.
    Useful for generating disparity maps and 3D reconstructions.

Prerequisites:
    - Run '03_calibrate_stereo.py' to generate 'stereo_calib.npz'.
    - Ensure the same stereo camera setup is used when rectifying.

Output:
    - Displays live rectified stereo image pairs with alignment lines.
    - Press 'q' to exit.
"""

import cv2 as cv
import numpy as np
from pathlib import Path

# ========================================
# Parameters
# ========================================

CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 2

SESSION = "test_own_cameras"
BASE_DIR = Path(__file__).resolve().parents[2]
CALIB_PATH = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images" / "stereo_calib.npz"

# ========================================
# Load stereo calibration parameters
# ========================================
if not CALIB_PATH.exists():
    raise FileNotFoundError(f"[ERROR] Calibration file not found: {CALIB_PATH}")

calib = np.load(CALIB_PATH)
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

# ========================================
# Open camera streams
# ========================================
capL = cv.VideoCapture(CAMERA_LEFT_INDEX)
capR = cv.VideoCapture(CAMERA_RIGHT_INDEX)

retL, frameL = capL.read()
retR, frameR = capR.read()

if not retL or not retR:
    raise RuntimeError("[ERROR] Could not read from both cameras. Check connections and indexes.")

h, w = frameL.shape[:2]
image_size = (w, h)
print(f"[INFO] Resolution: {image_size}")

# ========================================
# Compute rectification transforms and maps
# ========================================
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, flags=cv.CALIB_ZERO_DISPARITY
)
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv.CV_16SC2)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv.CV_16SC2)

print("[INFO] Rectification maps computed. Press 'q' to quit.")

# ========================================
# Display loop
# ========================================
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    capL.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capL.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    capR.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capR.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.imshow("Raw Left", frameL)
    cv.imshow("Raw Right", frameR)


    if not retL or not retR:
        print("[WARNING] Failed to grab frames.")
        break

    rectL = cv.remap(frameL, map1L, map2L, cv.INTER_LINEAR)
    rectR = cv.remap(frameR, map1R, map2R, cv.INTER_LINEAR)

    # Stack and draw horizontal lines for verification
    combined = np.hstack((rectL, rectR))
    for y in range(0, combined.shape[0], 40):
        cv.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv.imshow("Rectified Stereo Pair", combined)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ========================================
# Cleanup
# ========================================
capL.release()
capR.release()
cv.destroyAllWindows()
print("[INFO] Exited rectification viewer.")
