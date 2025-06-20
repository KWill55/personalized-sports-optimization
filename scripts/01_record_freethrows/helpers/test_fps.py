"""
Title: detect_fps.py

Description: This script tests multiple requested FPS settings
and prints the actual achieved FPS for each. It reports whether
the requested FPS was successfully applied by the camera.
"""

import cv2
import time

# ========================================
# Configuration Constants
# ========================================

CAMERA_INDEX = 0
TEST_FPS_VALUES = [15, 30, 60, 120]  # Modify this list as needed
RESOLUTION = (320, 240)


# ========================================
# Run FPS Tests
# ========================================

for fps_requested in TEST_FPS_VALUES:
    print(f"\n[INFO] Testing FPS setting: {fps_requested}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, fps_requested)

    # Check what the camera reports
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera reports FPS: {reported_fps:.2f}")

    # Warm up the camera
    time.sleep(0.5)
    frames = 0
    start = time.time()

    while frames < 100:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break
        frames += 1

    end = time.time()
    cap.release()

    fps_actual = frames / (end - start)
    print(f"[RESULT] Actual FPS achieved: {fps_actual:.2f}")

    # Determine success
    if abs(fps_actual - fps_requested) <= 2:
        print("[SUCCESS] Requested FPS is working well.\n")
    else:
        print("[WARNING] Requested FPS not achieved accurately.\n")
