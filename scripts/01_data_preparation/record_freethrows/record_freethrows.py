"""
Title: record_freethrows.py 

Purpose
    Record free throws 

Output
    - Combined video from both cameras
    - Video for 

Usage 
    - GUI has a record and stop recording button
    - GUI displays free throw attempt number 
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# promised settings on amazon: 
# 30fps@1080P and 60fps@720p and 100fps@480P
#Label	Resolution (W × H)	Aspect Ratio
#30FPS --> 1080p	1920 × 1080	16:9	
#60FPS --> 720p	1280 × 720	16:9
#100FPS --> 480p	640 × 480	4:3	

# =========================
# Config
# =========================
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
CAMERA_THIRD_INDEX = 3

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ATHLETE = "kenny"
SESSION = "session_test"

FPS = 30

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

for name, cap in caps.items():
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH) # set frame width
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) # set frame height
    cap.set(cv.CAP_PROP_FPS, FPS) # set FPS
    
    # test if camera has correct FPS 
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    print(f"[{name.upper()}] Requested {FPS} FPS, camera reports: {actual_fps:.2f}")


# =========================
# Tkinter GUI Setup
# =========================
root = tk.Tk()
root.title("Free Throw Recorder")

labels = {}
images = {}
recording = False
writers = {}
throw_num = 0

status_text = tk.StringVar()
status_text.set("Status: Idle")

def get_next_freethrow_number():
    max_count = 0
    for path in video_dirs.values():
        count = len(list(path.glob("freethrow*.avi")))
        max_count = max(max_count, count)
    return max_count + 1

def toggle_recording():
    global recording, writers, throw_num
    recording = not recording

    if recording:
        throw_num = get_next_freethrow_number()
        status_text.set(f"🎯 Recording freethrow{throw_num}")
        writers = {}
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            filename = f"freethrow{throw_num}.avi"
            filepath = video_dirs[name] / filename
            print(f"💾 Saving {name} feed to: {filepath}")
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            writers[name] = cv.VideoWriter(str(filepath), fourcc, FPS, (w, h))

    else:
        print("Stopped recording.")
        status_text.set("Status: Idle")
        for writer in writers.values():
            writer.release()
        writers = {}

def update_frames():
    for name, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.resize(frame, (426, 240))  # Resize for now to fit in UI
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        images[name] = img
        labels[name].configure(image=img)

        if recording and name in writers:
            writers[name].write(cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))  # save full res

    root.after(15, update_frames)

# =========================
# Create GUI Components
# =========================
frame_top = tk.Frame(root)
frame_top.pack()

for name in ["left", "right", "third"]:
    labels[name] = Label(frame_top)
    labels[name].pack(side=tk.LEFT, padx=5)

Button(root, text="Start/Stop Recording", command=toggle_recording, height=2, width=30).pack(pady=10)
Label(root, textvariable=status_text, font=("Helvetica", 14)).pack()

update_frames()
root.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()

# Cleanup
for cap in caps.values():
    cap.release()
for writer in writers.values():
    writer.release()
cv.destroyAllWindows()
