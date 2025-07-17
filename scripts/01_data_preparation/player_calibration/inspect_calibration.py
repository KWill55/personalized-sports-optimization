"""
Title: inspect_calibration.py

Purpose:
    Display calibration parameters from stereo_calib.npz and validate visually:
    - Undistortion preview
    - Rectification with epipolar lines
    - Optional disparity map for depth check

Inputs:
    - stereo_calib.npz (calibration results)
    - Combined checkerboard images (pair_XX.png)

Usage:
    - Run script to print parameters
    - GUI shows undistortion, rectification, and optional disparity map
"""

#K1, K2: Intrinsic camera matrices for both cameras.
# dist1, dist2: Lens distortion coefficients.
# R: Rotation between camera 1 and camera 2.
# T: Translation vector (baseline distance between cameras).
# P1, P2: Projection matrices for rectified stereo pair.
# E, F: Essential and Fundamental matrices (used for epipolar geometry).

import cv2 as cv
import numpy as np
from pathlib import Path
import glob

# ========================================
# Config
# ========================================
ATHLETE = "kenny"
SESSION = "session_test"
SHOW_DISPARITY = True  # Set to False to skip disparity preview
NUM_PREVIEW_IMAGES = 3  # How many image pairs to visualize

# ========================================
# Paths
# ========================================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
calib_file = session_dir / "calibration" / "stereo_calibration" / "stereo_calib.npz"
image_dir = session_dir / "calibration" / "calib_images"

# ========================================
# Load Calibration Parameters
# ========================================
calib = np.load(calib_file)
K1, K2 = calib["K1"], calib["K2"]
dist1, dist2 = calib["dist1"], calib["dist2"]
R, T = calib["R"], calib["T"]
P1, P2 = calib["P1"], calib["P2"]
E, F = calib["E"], calib["F"]

np.set_printoptions(precision=4, suppress=True)

# Print header and summary
print("=" * 60)
print(f"Stereo Calibration Data: {calib_file.name}")
print("=" * 60)

print("\nSummary of Calibration Arrays:")
print(f"{'Key':<20}{'Shape':<15}{'Dtype'}")
print("-" * 60)
for key in calib.files:
    arr = calib[key]
    print(f"{key:<20}{str(arr.shape):<15}{arr.dtype}")

print("\nDetailed Matrices:")
for key in calib.files:
    print(f"\n--- {key} ---\n{calib[key]}")

# ========================================
# Load Sample Images
# ========================================
combined_images = sorted(glob.glob(str(image_dir / "pair_*.png")))[:NUM_PREVIEW_IMAGES]

if not combined_images:
    print("[ERROR] No calibration images found for preview.")
    exit()

print(f"\n[INFO] Showing visual validation for {len(combined_images)} pairs...")

# Compute rectification transforms
image_size = (640, 640)  # From our capture scripts
R1, R2, P1_rect, P2_rect, Q, _, _ = cv.stereoRectify(K1, dist1, K2, dist2, image_size, R, T)

# Precompute undistortion/rectification maps
map1_L, map2_L = cv.initUndistortRectifyMap(K1, dist1, R1, P1_rect, image_size, cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(K2, dist2, R2, P2_rect, image_size, cv.CV_16SC2)

# ========================================
# Preview Loop
# ========================================
for img_path in combined_images:
    combined = cv.imread(img_path)
    if combined is None or combined.shape[1] < 1280:
        continue

    # Split into left and right
    imgL = combined[:, 0:640]
    imgR = combined[:, 640:1280]

    # Undistort
    undistL = cv.undistort(imgL, K1, dist1)
    undistR = cv.undistort(imgR, K2, dist2)

    # Rectify
    rectL = cv.remap(imgL, map1_L, map2_L, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, map1_R, map2_R, cv.INTER_LINEAR)

    # Draw epipolar lines on rectified pair
    for y in range(0, rectL.shape[0], 50):
        cv.line(rectL, (0, y), (rectL.shape[1], y), (0, 255, 0), 1)
        cv.line(rectR, (0, y), (rectR.shape[1], y), (0, 255, 0), 1)

    # Display windows
    cv.imshow("Original Left | Undistorted Left", np.hstack([imgL, undistL]))
    cv.imshow("Original Right | Undistorted Right", np.hstack([imgR, undistR]))
    cv.imshow("Rectified Pair", np.hstack([rectL, rectR]))

    if SHOW_DISPARITY:
        # Compute disparity (quick preview)
        grayL = cv.cvtColor(rectL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)
        stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(grayL, grayR)
        disp_norm = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        cv.imshow("Disparity Map", disp_norm)

    key = cv.waitKey(0)
    if key == 27:  # ESC to break
        break

cv.destroyAllWindows()
print("[INFO] Visual validation complete.")
