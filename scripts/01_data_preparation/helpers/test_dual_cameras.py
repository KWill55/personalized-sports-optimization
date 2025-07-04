import cv2 as cv
import numpy as np
import time


# Set camera indices (adjust if needed)
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1

# Optional: Set lower resolution for stability
WIDTH = 320
HEIGHT = 240

# Open cameras
capL = cv.VideoCapture(CAMERA_LEFT_INDEX)
capR = cv.VideoCapture(CAMERA_RIGHT_INDEX)

capL.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
capR.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
capL.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
capL.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
capR.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
capR.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

time.sleep(2.0)  # Let cameras warm up

if not capL.isOpened() or not capR.isOpened():
    print("[ERROR] One or both cameras failed to open.")
    exit(1)

print("[INFO] Cameras opened successfully. Press 'q' to quit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or frameL is None:
        print("[WARNING] Left camera failed to grab frame.")
        frameL = cv.putText(
            np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            "Left Camera Failed",
            (30, HEIGHT // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    if not retR or frameR is None:
        print("[WARNING] Right camera failed to grab frame.")
        frameR = cv.putText(
            np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            "Right Camera Failed",
            (30, HEIGHT // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Label the frames
    cv.putText(frameL, "Camera 0", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(frameR, "Camera 1", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Combine frames side-by-side
    combined = cv.hconcat([frameL, frameR])
    cv.imshow("Dual Camera Preview", combined)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv.destroyAllWindows()
