"""
Title: 03_calibrate_stereo.py Stereo Camera Calibration Script

Purpose:
    Calibrates a stereo camera setup using images of a checkerboard pattern to compute 
    intrinsic parameters (camera matrix, distortion) for each camera and extrinsic 
    parameters (rotation and translation between them).

Prerequisites:
    - Use 'captureImagePairs.py' to save matching left/right images of a checkerboard.
    - Ensure the checkerboard size and square dimensions are set correctly.
    - Ensure image pair directories match LEFT_IMAGES_PATH and RIGHT_IMAGES_PATH.

Output:
    - Saves stereo calibration parameters to 'stereo_calib.npz':
        - mtxL, distL: Intrinsic matrix and distortion for left camera
        - mtxR, distR: Intrinsic matrix and distortion for right camera
        - R, T: Rotation and translation from left to right camera

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

# Stereo calibration
flags = cv.CALIB_FIX_INTRINSIC
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria, flags=flags)

# Save for later use
np.savez(output_file, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print(f"Stereo calibration complete and saved to {output_file}")
