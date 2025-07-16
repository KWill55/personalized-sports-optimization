"""
Title: record_freethrows.py 

Description:
    This module's purpose is to record free throw attempts from three cameras. 
    Two cameras are using stereo vision to track the player, and one camera is used to track the ball. 
    The module provides a GUI to start and stop recording and displays the current free throw attempt number. 
    This script saves the video files in a stuctured directory. 

Inputs
    - Three cameras (two for player tracking, one for ball tracking)

Usage 
    - GUI has "Record" and "Stop Recording" buttons
    
Outputs
    - 640x640 videos for player tracking (left and right cameras)
    - 1080p videos for ball tracking

Last Updated: 16 July 2025
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import Label, Button
import time
import threading


# Label | Res (W×H) | A Ratio | FPS    | Notes
#-----------------------------------------------------
# 1080p | 1920×1080 | 16:9    | 30FPS  | works well 	
# 720p	| 1280×720  | 16:9    | 60FPS  | curently used
# 480p	| 640×480   | 4:3	  | 100FPS | has problems 

### TODO ###
# i should display the video resolution before and after the submatrix 
# make iphone 60FPS instead of 30FPS
# keep 1080p for ball tracking 
# find a good way to toggle some comments on/off with a flag or something 
# make the iphone camera stay at 1920x1080 resolution instead of dropping to 720p like the other two 
# find distance to place tripods to make freethrows visible. (harder now that we're cropping the feed)
# maybe add a way of sensing how far from the tripods the athlete is currently standing instead of having use measuring tape? 
    # maybe do this in calibration steps
# switch to C++ later if I want to do things real time 
# enable MJPG later 

import cv2 as cv
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps

# =========================
# Config
# =========================
CAMERA_LEFT_INDEX = 1
CAMERA_RIGHT_INDEX = 2
CAMERA_THIRD_INDEX = 4

ATHLETE = "kenny"
SESSION = "session_test"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720   
FPS_LEFT_RIGHT = 60  # Stereo cameras
FPS_THIRD = 30       # Ball tracking camera
GUI_REFRESH_MS = 30  # ~30 FPS for GUI

BORDER_COLORS = {"left": "red", "right": "blue", "third": "green"} 
BORDER_THICKNESS = 5

# =========================
# Paths and Directories 
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
# Shared Resources
# =========================
frames = {"left": None, "right": None, "third": None}
frame_locks = {k: threading.Lock() for k in frames}
recording = False
writers = {}
throw_count = 0
frame_counters = {"left": 0, "right": 0, "third": 0}
start_time = None


# =========================
# Camera Capture Thread
# =========================
def capture_camera(name, index, crop=False):
    """
    Continuously captures frames from the specified camera in a separate thread.

    Args:
        name (str): Identifier for the camera ('left', 'right', 'third').
        index (int): Index of the camera for cv2.VideoCapture.
        crop (bool): If True, crop frames to 640x640 (used for left/right cameras).
    """
    cap = cv.VideoCapture(index)
    fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_THIRD #picks correct FPS per which camera 
    
    #set camera properties 
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, fps)

    print(f"[{name.upper()}] Initialized: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}, FPS: {cap.get(cv.CAP_PROP_FPS)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if crop:
            frame = frame[40:680, 320:960]  # Center crop to 640x640

        with frame_locks[name]:
            frames[name] = frame

        if recording:
            frame_counters[name] += 1

        time.sleep(0.001)

# =========================
# Video Writer Thread (FPS Throttling included)
# =========================
def write_frames():
    """
    Writes frames to disk at the correct FPS using throttling, so the output video duration matches real time.
    """
    # Compute intervals for each camera
    intervals = {
        "left": 1.0 / FPS_LEFT_RIGHT,
        "right": 1.0 / FPS_LEFT_RIGHT,
        "third": 1.0 / FPS_THIRD
    }
    last_write_time = {name: 0 for name in frames}

    while True:
        if recording:
            now = time.time()
            for name, writer in writers.items():
                if now - last_write_time[name] >= intervals[name]:
                    with frame_locks[name]:
                        if frames[name] is not None:
                            writer.write(frames[name])
                            last_write_time[name] = now
        time.sleep(0.001)



# =========================
# Recording Functions
# =========================
def get_next_throw_number():
    """
    Scans all video directories for existing 'freethrow*.avi' files and returns
    the next available throw number.

    Returns:
        int: Next throw number (max existing + 1).
    """
    max_count = 0
    for path in video_dirs.values():
        for file in path.glob("freethrow*.avi"):
            try:
                num = int(file.stem.replace("freethrow", ""))
                max_count = max(max_count, num)
            except ValueError:
                continue
    return max_count + 1


def start_recording(dims):
    """
    Starts a new recording session by initializing VideoWriter objects.

    Args:
        dims (dict): Dictionary mapping camera names to their frame dimensions.
    """
    global writers, recording, throw_count, start_time, frame_counters
    throw_count = get_next_throw_number()
    print(f"🟢 Starting freethrow{throw_count}")

    fourcc = cv.VideoWriter_fourcc(*'MJPG')

    # Create writers for each camera
    for name, size in dims.items():
        filepath = video_dirs[name] / f"freethrow{throw_count}.avi"
        fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_THIRD
        writers[name] = cv.VideoWriter(str(filepath), fourcc, fps, size)
        print(f"[INFO] Writing {name} to {filepath} @ {fps} FPS")

    recording = True
    start_time = time.time()
    frame_counters = {k: 0 for k in frame_counters}


def stop_recording():
    """
    Stops the current recording session, calculates FPS, and releases video writers.
    """
    global writers, recording
    duration = time.time() - start_time
    print(f"🛑 Stopping recording after {duration:.1f}s")

    # Compute FPS for each camera
    for name, count in frame_counters.items():
        actual_fps = count / duration if duration > 0 else 0
        print(f"[RESULT] {name.upper()} Actual FPS: {actual_fps:.1f}")
        if name in ["left", "right"] and actual_fps < FPS_LEFT_RIGHT * 0.8:
            print(f"[WARNING] {name.upper()} is below target FPS ({actual_fps:.1f} vs {FPS_LEFT_RIGHT})")
        if name == "third" and actual_fps < FPS_THIRD * 0.8:
            print(f"[WARNING] THIRD is below expected FPS ({actual_fps:.1f})")

    # Release writers
    for w in writers.values():
        w.release()
    writers.clear()
    recording = False
    print("[INFO] Writers closed.")


# =========================
# GUI App
# =========================
class FreeThrowRecorderApp:
    """
    Tkinter-based GUI application for displaying camera feeds and managing recordings.
    """

    def __init__(self, root):
        """
        Initializes the GUI and starts periodic frame updates.

        Args:
            root (Tk): Tkinter root window.
        """
        self.root = root
        self.root.title("Free Throw Recorder")
        self.root.geometry("1800x1200")

        self.status_text = tk.StringVar(value="Status: Idle")
        self.labels = {}
        self.images = {}

        self.setup_gui()
        self.update_gui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_gui(self):
        """
        Creates and arranges all GUI components including buttons, legend, and camera feeds.
        """
        frame_top = tk.Frame(self.root)
        frame_top.pack()
        self.labels["left"] = Label(frame_top)
        self.labels["left"].pack(side=tk.LEFT, padx=5)
        self.labels["right"] = Label(frame_top)
        self.labels["right"].pack(side=tk.LEFT, padx=5)

        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(pady=10)

        button_frame = tk.Frame(frame_bottom)
        button_frame.pack(side=tk.LEFT, padx=10)
        Button(button_frame, text="Start/Stop Recording", command=self.toggle_recording, height=2, width=20).pack()
        Label(button_frame, textvariable=self.status_text, font=("Helvetica", 14)).pack(pady=10)

        legend_frame = tk.Frame(frame_bottom)
        legend_frame.pack(side=tk.LEFT, padx=20)
        for name, color in BORDER_COLORS.items():
            lf = tk.Frame(legend_frame)
            lf.pack(anchor="w", pady=5)
            tk.Label(lf, width=2, height=1, bg=color).pack(side=tk.LEFT)
            text = f"{name.capitalize()} Camera"
            tk.Label(lf, text=text).pack(side=tk.LEFT, padx=5)

        self.labels["third"] = Label(frame_bottom)
        self.labels["third"].pack(side=tk.LEFT, padx=10)

    def toggle_recording(self):
        """
        Starts or stops recording based on the current state.
        """
        global recording
        if not recording:
            dims = {"left": (640, 640), "right": (640, 640), "third": (FRAME_WIDTH, FRAME_HEIGHT)}
            start_recording(dims)
            self.status_text.set(f"Recording freethrow{throw_count}")
        else:
            stop_recording()
            self.status_text.set("Status: Idle")

    def update_gui(self):
        """
        Updates the GUI with the latest frames from each camera every GUI_REFRESH_MS milliseconds.
        """
        for name in frames:
            with frame_locks[name]:
                frame = frames[name]
            if frame is not None:
                if name in ["left", "right"]:
                    frame_display = cv.resize(frame, (640, 640))
                else:
                    frame_display = cv.resize(frame, (760, 427))

                frame_rgb = cv.cvtColor(frame_display, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img = ImageOps.expand(pil_img, border=BORDER_THICKNESS, fill=BORDER_COLORS[name])
                img = ImageTk.PhotoImage(pil_img)
                self.images[name] = img
                self.labels[name].configure(image=img)

        self.root.after(GUI_REFRESH_MS, self.update_gui)

    def on_close(self):
        """
        Handles window close event by stopping recording and shutting down the GUI safely.
        """
        if recording:
            stop_recording()
        self.root.destroy()


# =========================
# Main
# =========================
if __name__ == "__main__":
    threading.Thread(target=capture_camera, args=("left", CAMERA_LEFT_INDEX, True), daemon=True).start()
    threading.Thread(target=capture_camera, args=("right", CAMERA_RIGHT_INDEX, True), daemon=True).start()
    threading.Thread(target=capture_camera, args=("third", CAMERA_THIRD_INDEX, False), daemon=True).start()

    threading.Thread(target=write_frames, daemon=True).start()

    root = tk.Tk()
    app = FreeThrowRecorderApp(root)
    root.mainloop() 