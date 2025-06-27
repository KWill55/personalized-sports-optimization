"""
TODO
- visuals saying if im recording or stopped 
- say what free throw im on
- use tkinter or something to organize the free throws and buttons 
- make sure i can use a flash to trim my freethrows 
- check fps of all the freethrows
- check that i can synchronize them 
- my external webcam didn't detect the flashes... more leds?
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import datetime

# =========================
# Config
# =========================
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
CAMERA_THIRD_INDEX = 2

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ATHLETE = "kenny"
SESSION = "session_001"

# =========================
# Paths
# =========================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

video_dirs = {
    "left": session_dir / "videos" / "player_tracking" / "raw" / "left",
    "right": session_dir / "videos" / "player_tracking" / "raw" / "right",
    "third": session_dir / "videos" / "ball_tracking" / "raw"
}

for path in video_dirs.values():
    path.mkdir(parents=True, exist_ok=True)

# =========================
# Open Cameras
# =========================
caps = {
    "left": cv.VideoCapture(CAMERA_LEFT_INDEX),
    "right": cv.VideoCapture(CAMERA_RIGHT_INDEX),
    "third": cv.VideoCapture(CAMERA_THIRD_INDEX)
}

# Set resolution (optional)
for cap in caps.values():
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Check if opened and get actual FPS
actual_fps = {}
for name, cap in caps.items():
    if not cap.isOpened():
        print(f"❌ Could not open {name} camera.")
        exit()
    fps = cap.get(cv.CAP_PROP_FPS)
    actual_fps[name] = fps if fps > 1 else 30
    print(f"{name} camera FPS: {actual_fps[name]}")

print("✅ All cameras opened. Press 's' to start/stop recording. Press 'q' to quit.")

# =========================
# Recording Loop
# =========================
recording = False
writers = {}

# Window layout
window_positions = {
    "left": (0, 0),
    "right": (FRAME_WIDTH + 20, 0),
    "third": (0, FRAME_HEIGHT + 60)
}

while True:
    frames = {}

    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read from {name}")
            continue

        # Ensure consistent frame size
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        frames[name] = frame
        cv.imshow(name, frame)

        if name in window_positions:
            x, y = window_positions[name]
            cv.moveWindow(name, x, y)

    key = cv.waitKey(1) & 0xFF

    if key == ord('s'):
        recording = not recording
        if recording:
            def get_next_freethrow_number():
                max_count = 0
                for path in video_dirs.values():
                    count = len(list(path.glob("freethrow*.mp4")))
                    max_count = max(max_count, count)
                return max_count + 1

            throw_num = get_next_freethrow_number()
            print(f"🔴 Started recording freethrow{throw_num}")

            writers = {}
            for name, frame in frames.items():
                h, w = frame.shape[:2]
                filename = f"freethrow{throw_num}.mp4"
                filepath = video_dirs[name] / filename
                print(f"💾 Saving {name} feed to: {filepath}")
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                writers[name] = cv.VideoWriter(str(filepath), fourcc, actual_fps[name], (w, h))
        else:
            print("🛑 Stopped recording.")
            for writer in writers.values():
                writer.release()
            writers = {}

    elif key == ord('q'):
        print("👋 Quitting.")
        break

    # Write frames if recording
    if recording:
        for name in writers:
            if name in frames:
                writers[name].write(frames[name])

# Cleanup
for cap in caps.values():
    cap.release()
for writer in writers.values():
    writer.release()
cv.destroyAllWindows()
