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
# Config
# ========================================

CAMERA_RANGE = range(0, 5)  # Indices to try
CHECKERBOARD = (5, 4)  # (internal corners)
SQUARE_SIZE = 2.5  # in cm
CALIBRATE_EVERY = 40
MAX_CALIBRATION_SAMPLES = 100
SKIP_FRAMES = True

# ========================================
# Object Points Setup
# ========================================

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# ========================================
# Loop Through All Cameras
# ========================================

for cam_index in CAMERA_RANGE:
    print(f"\n[INFO] Trying camera index {cam_index}...")
    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[WARNING] Skipping index {cam_index} (not available).")
        continue

    print(f"[INFO] Tuning camera index {cam_index}. Press SPACE to move to next. Press ESC to quit.")

    objpoints, imgpoints = [], []
    frame_count = 0
    toggle = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        toggle = not toggle
        if SKIP_FRAMES and not toggle:
            continue

        frame = cv.resize(frame, (640, 480))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        ret_cb, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_cb:
            cv.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_cb)
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            frame_count += 1

            if len(objpoints) > MAX_CALIBRATION_SAMPLES:
                objpoints = objpoints[-MAX_CALIBRATION_SAMPLES:]
                imgpoints = imgpoints[-MAX_CALIBRATION_SAMPLES:]

            if frame_count % CALIBRATE_EVERY == 0 and len(objpoints) >= 10:
                ret_calib, mtx, dist, _, _ = cv.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                print(f"\n[CAM {cam_index} - FRAME {frame_count}] Intrinsic Matrix:")
                print(np.round(mtx, 2))
                print("Distortion Coefficients:")
                print(np.round(dist.ravel(), 4))

        # Draw overlay
        cv.putText(frame, f"Cam {cam_index} | Valid Frames: {len(objpoints)}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow("Tune Intrinsics", frame)
        key = cv.waitKey(1)

        if key == 27:  # ESC to quit all
            cap.release()
            cv.destroyAllWindows()
            exit()
        elif key == 32:  # SPACE to move to next camera
            cap.release()
            cv.destroyAllWindows()
            break

    print(f"[INFO] Finished tuning cam index {cam_index}.")

print("[INFO] Done with all available cameras.")
