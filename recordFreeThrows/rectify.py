"""
Stereo Camera Rectifying Script

Purpose:
    - Rectifies stereo camera images using pre-calibrated parameters to align the left and right camera feeds.
    This is useful for generating disparity maps and 3D reconstructions.

Prerequisites:
    - Ensure stereo calibration has been performed and parameters are saved in 'stereo_calib.npz'.
    - The cameras should be set up to capture images of the same scene simultaneously.
    - Run this script after capturing images and calibrating the stereo camera setup.

Output:
    - Displays rectified stereo image pairs side by side.
    - Draws horizontal lines to verify alignment of the rectified images.

Usage:
    - Press 'q' to exit the display window.
"""


import cv2 as cv
import numpy as np

# === Load stereo calibration ===
calib = np.load("stereo_calib.npz")
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

# === Open camera streams ===
capL = cv.VideoCapture(0)
capR = cv.VideoCapture(1)

# === Get image size from initial frame ===
retL, frameL = capL.read()
retR, frameR = capR.read()
if not retL or not retR:
    raise RuntimeError("Failed to read from both cameras.")

h, w = frameL.shape[:2]
image_size = (w, h)

# === Compute stereo rectification ===
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, flags=cv.CALIB_ZERO_DISPARITY
)
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv.CV_16SC2)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv.CV_16SC2)

# === Show rectified camera feeds ===
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Camera read failed.")
        break

    rectL = cv.remap(frameL, map1L, map2L, cv.INTER_LINEAR)
    rectR = cv.remap(frameR, map1R, map2R, cv.INTER_LINEAR)

    # Combine and draw lines to verify alignment
    combined = np.hstack((rectL, rectR))
    for y in range(0, combined.shape[0], 40):
        cv.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv.imshow("Rectified Stereo Pair", combined)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv.destroyAllWindows()

