"""
Title: capture_cb_pairs_gui.py

Purpose:
    GUI tool to capture synchronized checkerboard images from two webcams.

Output:
    - Saves image pairs to left_calib_dir and right_calib_dir
    - Images named: left_00.jpg, right_00.jpg, etc.

Usage:
    - GUI displays both camera feeds.
    - Button click captures synchronized image pair.
    - Image pair number displayed.
"""

import cv2 as cv
import os
from pathlib import Path
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# =========================
# Config
# =========================
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ATHLETE = "kenny"
SESSION = "session_001"

# =========================
# Paths
# =========================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
left_calib_dir = session_dir / "calibration" / "calib_images" / "left"
right_calib_dir = session_dir / "calibration" / "calib_images" / "right"

for d in [left_calib_dir, right_calib_dir]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Open Cameras
# =========================
caps = {
    "left": cv.VideoCapture(CAMERA_LEFT_INDEX),
    "right": cv.VideoCapture(CAMERA_RIGHT_INDEX)
}

for cap in caps.values():
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

for name, cap in caps.items():
    if not cap.isOpened():
        print(f"[ERROR] Could not open {name} camera.")
        exit()

# =========================
# Tkinter GUI Setup
# =========================
root = tk.Tk()
root.title("Calibration Image Capture")

labels = {}
images = {}
frame_id = 0

status_text = tk.StringVar()
status_text.set("Ready to capture image pairs")

# =========================
# Helper Functions
# =========================
def get_next_image_pair_number():
    left_images = list(left_calib_dir.glob("left_*.jpg"))
    return len(left_images)

def capture_image_pair():
    global frame_id
    retL, frameL = caps["left"].read()
    retR, frameR = caps["right"].read()

    if not retL or not retR:
        print("[ERROR] Failed to grab one or both frames.")
        return

    fnameL = left_calib_dir / f"left_{frame_id:02}.jpg"
    fnameR = right_calib_dir / f"right_{frame_id:02}.jpg"
    cv.imwrite(str(fnameL), frameL)
    cv.imwrite(str(fnameR), frameR)
    print(f"[INFO] Saved pair #{frame_id}: {fnameL.name}, {fnameR.name}")
    frame_id += 1
    status_text.set(f"Captured pair #{frame_id}")

def update_frames():
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.resize(frame, (426, 240))
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        images[name] = img
        labels[name].configure(image=img)

    root.after(15, update_frames)

# =========================
# Create GUI Components
# =========================
frame_top = tk.Frame(root)
frame_top.pack()

for name in ["left", "right"]:
    labels[name] = Label(frame_top)
    labels[name].pack(side=tk.LEFT, padx=5)

Button(root, text="Capture Image Pair", command=capture_image_pair, height=2, width=30).pack(pady=10)
Label(root, textvariable=status_text, font=("Helvetica", 14)).pack()

# =========================
# Launch
# =========================
frame_id = get_next_image_pair_number()
update_frames()
root.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()

# =========================
# Cleanup
# =========================
for cap in caps.values():
    cap.release()
cv.destroyAllWindows()
