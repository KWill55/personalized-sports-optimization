# ========================================
# Live Camera Identifier Script
# ========================================
# This script shows a live video feed from each connected camera
# and overlays the camera index directly on the frame.
# Press any key to go to the next camera. Press ESC to quit.
# ========================================

import cv2 as cv

MAX_CAMERAS = 10  # Change if needed

print("[INFO] Starting live camera identification...")
print("[INFO] Press any key to move to the next camera. Press ESC to quit.")

for i in range(MAX_CAMERAS):
    print(f"[INFO] Trying camera index {i}...")
    cap = cv.VideoCapture(i)

    if not cap.isOpened():
        print(f"[WARNING] Camera index {i} not available.")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Failed to read from camera {i}.")
            break

        # Draw camera index on frame
        label = f"Camera {i}"
        cv.putText(frame, label, (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Show the frame
        cv.imshow("Camera Preview", frame)
        key = cv.waitKey(1)

        if key == 27:  # ESC key
            print("[INFO] Exiting.")
            cap.release()
            cv.destroyAllWindows()
            exit()
        elif key != -1:  # Any key to go to next camera
            break

    cap.release()
    cv.destroyAllWindows()
