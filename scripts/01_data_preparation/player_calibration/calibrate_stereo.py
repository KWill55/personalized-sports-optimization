"""
Title: calibrate_stereo.py

Purpose:
    Perform stereo camera calibration using checkerboard image pairs
    from combined images captured with 'capture_cb_pairs_gui.py'.

Input:
    - Combined images where left and right cameras are stitched side-by-side (1280x640).
    - Checkerboard dimensions and square size must match the capture script.

Output:
    - Intrinsic parameters (K1, K2)
    - Distortion coefficients (dist1, dist2)
    - Extrinsic parameters (R, T)
    - Projection matrices (P1, P2)
    - Essential (E) and Fundamental (F) matrices
    - Saves all results to stereo_calib.npz
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import glob

# ========================================
# Configuration
# ========================================
CHECKERBOARD_SIZE = (5, 4)  # (columns, rows)
SQUARE_SIZE = 2.5           # cm per square

ATHLETE = "kenny"
SESSION = "session_test"

# ========================================
# Paths
# ========================================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
calib_images_dir = session_dir / "calibration" / "calib_images"
output_dir = session_dir / "calibration" / "stereo_calibration"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "stereo_calib.npz"

# ========================================
# Prepare Object Points
# ========================================
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []    # 3D points
imgpointsL = []   # 2D points in left
imgpointsR = []   # 2D points in right

# ========================================
# Load and Process Images
# ========================================
combined_images = sorted(glob.glob(str(calib_images_dir / "pair_*.png")))

print(f"[INFO] Found {len(combined_images)} combined images.")
if len(combined_images) < 10:
    print("[WARNING] Less than 10 image pairs may reduce calibration accuracy.")

for img_path in combined_images:
    combined = cv.imread(img_path)
    if combined is None or combined.shape[1] < 1280:
        print(f"[ERROR] Invalid combined image: {img_path}")
        continue

    # Split into left and right halves
    frameL = combined[:, 0:640]
    frameR = combined[:, 640:1280]

    grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

    # Detect checkerboard
    retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD_SIZE, None)
    retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD_SIZE, None)

    if retL and retR:
        # Refine corner positions for accuracy
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
    else:
        print(f"[WARNING] Checkerboard not detected in {img_path}")

if len(objpoints) == 0:
    print("[ERROR] No valid checkerboard detections found. Check your image pairs.")
    exit()

print(f"[INFO] Using {len(objpoints)} valid pairs for calibration.")

# ========================================
# Calibrate Each Camera
# ========================================
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
# Projection Matrices
# ========================================
P1 = mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K1 [I|0]
P2 = mtxR @ np.hstack((R, T))                         # P2 = K2 [R|T]

# ========================================
# Save Parameters
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
print("[INFO] RMS re-projection error:", retval)
