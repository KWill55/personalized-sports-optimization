"""
Title: 01_tune_intrinsics.py: Real-Time Intrinsic Calibration Script

Purpose:
    Allows real-time tuning of camera intrinsics by displaying the camera matrix 
    and distortion coefficients as the user adjusts the zoom or focus on the lens.
    Helps align two stereo cameras to have similar focal lengths and minimal distortion
    before capturing calibration pairs.

Prerequisites:
    - display a checkerboard pattern in view of the camera.
    - Set the correct checkerboard dimensions (number of internal corners) and square size.
    - Adjust CAMERA_INDEX to select the camera you want to tune.

Output:
    - Displays calibration results (camera matrix and distortion coefficients) every N frames.
    - No files are saved; this script is for live feedback and physical tuning only.

Usage:
    - Run the script and adjust the camera lens.
    - Press ESC to exit the script.
"""

import cv2 as cv
import numpy as np

# ======================================== 
# Configuration Constants 
# ========================================

CAMERA_INDEX = 1 # use identify_cameras.py to find the correct index
CHECKERBOARD = (5,4)  # (cols, rows) — numberof internal corners
SQUARE_SIZE = 2.5  # cm 
CALIBRATE_EVERY = 40  # frames between calibrations
MAX_CALIBRATION_SAMPLES = 100  # max number of samples to keep for calibration

SKIP_FRAMES = True  # skip frames for smoother operation

# ======================================== 
# Prepare Object Points
# ========================================

# creates 3D grid of points corresponding to the checkerboard pattern
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) 
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 
objp *= SQUARE_SIZE 

# ======================================== 
# Initialize Camera 
# ========================================

cap = cv.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Could not open camera index {CAMERA_INDEX}")
    exit()

print("[INFO] Press ESC to quit. Adjust your camera lens while this runs.")

# ======================================== 
# Calibration Loop
# ========================================

objpoints = []
imgpoints = []
frame_count = 0
toggle = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed.")
        break

    # Skip frames for smoother operation
    toggle = not toggle
    if SKIP_FRAMES and not toggle:
        continue

    frame = cv.resize(frame, (640, 480))

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret_cb, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret_cb:
        cv.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_cb)

        # Always use .copy() for objp to avoid modifying all entries later
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        frame_count += 1

        # Limit number of samples to avoid lag
        if len(objpoints) > MAX_CALIBRATION_SAMPLES:
            objpoints = objpoints[-MAX_CALIBRATION_SAMPLES:]
            imgpoints = imgpoints[-MAX_CALIBRATION_SAMPLES:]

        # calibrate every N frames AND only if we have enough valid samples
        if frame_count % CALIBRATE_EVERY == 0 and len(objpoints) >= 10:
            ret_calib, mtx, dist, _, _ = cv.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            print(f"\n[FRAME {frame_count}] Intrinsic Matrix:")
            print(np.round(mtx, 2))
            print("Distortion Coefficients:")
            print(np.round(dist.ravel(), 4))



    # Overlay info on frame
    # cv.putText(frame, f"Valid Frames: {frame_count}", (20, 40),
    #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Tune Intrinsics", frame)
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

# ======================================== 
# Cleanup 
# ========================================

cap.release()
cv.destroyWindow("Tune Intrinsics")
cv.waitKey(100)  # Give time for the window to actually close
cv.destroyAllWindows()

