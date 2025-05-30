"""
Stereo Free Throw Recording Script

Records stereo video pairs (rectified) for each basketball free throw,
saving each throw as a separate video file using stereo calibration parameters.

Usage:
    - Press 's' to start recording a throw (3 seconds by default)
    - Press 'q' to quit at any time
"""

import cv2 as cv
import numpy as np
import os

# === Load Calibration ===
calib = np.load("stereo_calib.npz")
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

# === Open Cameras ===
capL = cv.VideoCapture(0)
capR = cv.VideoCapture(1)

retL, frameL = capL.read()
retR, frameR = capR.read()
if not retL or not retR:
    raise RuntimeError("Could not read from both cameras.")

h, w = frameL.shape[:2]
image_size = (w, h)

# === Rectification Maps ===
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, flags=cv.CALIB_ZERO_DISPARITY)
map1L, map2L = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv.CV_16SC2)
map1R, map2R = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv.CV_16SC2)

# === Prepare Output Folders ===
os.makedirs("throws/left", exist_ok=True)
os.makedirs("throws/right", exist_ok=True)

# === Settings ===
throw_index = 1
clip_length = 3  # seconds
fps = 30
frame_count = clip_length * fps

print("Press 's' to record a throw. Press 'q' to quit.")

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print(f"🎬 Recording throw {throw_index}...")

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        outL = cv.VideoWriter(f"throws/left/throw_{throw_index:03}.avi", fourcc, fps, image_size)
        outR = cv.VideoWriter(f"throws/right/throw_{throw_index:03}.avi", fourcc, fps, image_size)

        for _ in range(frame_count):
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            if not retL or not retR:
                print("⚠️ Camera read error.")
                break

            rectL = cv.remap(frameL, map1L, map2L, cv.INTER_LINEAR)
            rectR = cv.remap(frameR, map1R, map2R, cv.INTER_LINEAR)

            outL.write(rectL)
            outR.write(rectR)

            combined = np.hstack((rectL, rectR))
            cv.imshow("Recording (press 'q' to stop)", combined)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        outL.release()
        outR.release()
        print(f"✅ Throw {throw_index} saved.")
        throw_index += 1

    elif key == ord('q'):
        print("👋 Exiting.")
        break

capL.release()
capR.release()
cv.destroyAllWindows()
