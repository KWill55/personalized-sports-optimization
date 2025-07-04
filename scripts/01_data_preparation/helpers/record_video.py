import cv2
from pathlib import Path

# ========== Config ==========
CAMERA_INDEX = 0  
OUTPUT_FILENAME = "recorded_video.mp4"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30

# ========== Setup ==========
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

if not cap.isOpened():
    print(f"[ERROR] Cannot open camera with index {CAMERA_INDEX}")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

print("[INFO] Recording started. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    out.write(frame)
    cv2.imshow("Recording (press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Recording stopped by user.")
        break

# ========== Cleanup ==========
cap.release()
out.release()
cv2.destroyAllWindows()
