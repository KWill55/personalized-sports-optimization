"""
Stereo Camera Calibration Script

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

These values can be used for stereo rectification, disparity map generation,
and 3D reconstruction using OpenCV.
"""

import cv2 as cv
import numpy as np
import glob
from pathlib import Path


# Parameters 
CHECKERBOARD_SIZE = (4,5) # Checkerboard size (columns, rows)
SQUARE_SIZE = 2.5 # size of one square in centimeters 

# path parameters
BASE_DIR = Path(__file__).resolve().parents[2]
SESSION = "test_own_cameras"
LEFT_IMAGES_PATH = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images" / "left"
RIGHT_IMAGES_PATH = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images" / "right"

OUTPUT_DIR = BASE_DIR / "data" / SESSION / "01_record_data" / "calib_images"
OUTPUT_FILE = OUTPUT_DIR / "stereo_calib.npz"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Prepare 3D object points like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD_SIZE[0]*CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store points
imgpointsL = []    # 2D points in left image
imgpointsR = []    # 2D points in right image
objpoints = []     # 3D points in real world

# Load image files 
left_images = sorted(glob.glob(str(LEFT_IMAGES_PATH / "*.jpg")))
right_images = sorted(glob.glob(str(RIGHT_IMAGES_PATH / "*.jpg")))

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
np.savez(OUTPUT_FILE, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T)
print(f"Stereo calibration complete and saved to {OUTPUT_FILE}")
