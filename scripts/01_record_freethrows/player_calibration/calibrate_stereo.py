"""
Title: calibrate_stereo.py

Purpose:
    Perform stereo camera calibration using checkerboard image pairs,
    and generate the projection matrices needed for 3D reconstruction.

Output: 
    - Intrinsic parameters (K1, K2): focal lengths, principal points
    - Extrinsic parameters (R, T): rotation and translation between cameras
    - Distortion coefficients (dist1, dist2): lens distortion per camera
    - Projection matrices (P1, P2): used for triangulating 3D points

Prerequisites:
    - Use 'capture_cb_pairs' to save matching left/right images of a checkerboard.
    - Ensure the checkerboard size and square dimensions are set correctly.
    - Ensure image pair directories match LEFT_IMAGES_PATH and RIGHT_IMAGES_PATH.
"""

import cv2 as cv
import numpy as np
import glob
from pathlib import Path

# ======================================== 
# Configuration Constants
# ========================================

CHECKERBOARD_SIZE = (5,4) # Checkerboard size (columns, rows)
SQUARE_SIZE = 2.5 # size of one square in centimeters 

ATHLETE = "Kenny"
SESSION = "session_001"


# ======================================== 
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

left_calib_dir = session_dir / "calibration" / "calib_images" / "left"
right_calib_dir = session_dir / "calibration" / "calib_images" / "right"

output_dir = session_dir / "calibration" / "stereo_calibration"
output_file = output_dir / "stereo_calib.npz"

output_dir.mkdir(parents=True, exist_ok=True) 

# ======================================== 
# Prepare Object Points
# ========================================

# creates 3D grid of points corresponding to the checkerboard pattern
objp = np.zeros((CHECKERBOARD_SIZE[0]*CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store points
imgpointsL = []    # 2D points in left image
imgpointsR = []    # 2D points in right image
objpoints = []     # 3D points in real world

# ========================================
# Image Loading and Checkerboard Detection
# ========================================

# Load image files 
left_images = sorted(glob.glob(str(left_calib_dir / "*.jpg")))
right_images = sorted(glob.glob(str(right_calib_dir / "*.jpg")))

for left, right in zip(left_images, right_images):
    imgL = cv.imread(left)
    imgR = cv.imread(right)

    if imgL is None or imgR is None:
        print(f"[WARNING] Could not load image pair:\n  Left: {left}\n  Right: {right}")
        continue

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD_SIZE, None)
    retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD_SIZE, None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

if len(objpoints) == 0:
    print("[ERROR] No valid checkerboard detections found. Check your image pairs.")
    exit()
else:
    print(f"[INFO] Chessboard found in pair: {left}, {right}")

# ========================================
# Camera Calibration
# ========================================

# Calibrate each camera
retL, mtxL, distL, _, _ = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# ========================================
# Stereo Calibration
# ========================================

flags = cv.CALIB_FIX_INTRINSIC
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=flags
)

# ========================================
# Build Projection Matrices
# ========================================

P1 = mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K1 [I | 0]
P2 = mtxR @ np.hstack((R, T))                         # P2 = K2 [R | T]

# ========================================
# Save All Parameters
# ========================================

np.savez(
    output_file,
    K1=mtxL, dist1=distL,
    K2=mtxR, dist2=distR,
    R=R, T=T,
    P1=P1, P2=P2,
    E=E, F=F
)

print("[INFO] Stereo calibration complete.")
print(f"[INFO] Saved to: {output_file}")
print("[INFO] Saved parameters:")
print(" - K1, K2 (intrinsics)")
print(" - dist1, dist2 (distortion coefficients)")
print(" - R, T (extrinsics)")
print(" - P1, P2 (projection matrices)")
print(" - E, F (essential & fundamental matrices)")
print("[INFO] RMS re-projection error from stereo calibration:", retval)

