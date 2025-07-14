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

Last Updated: 14 July 2025
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


# Label | Res (W×H) | A Ratio | FPS    | Notes
#-----------------------------------------------------
# 1080p | 1920×1080 | 16:9    | 30FPS  | works well 	
# 720p	| 1280×720  | 16:9    | 60FPS  | curently used
# 480p	| 640×480   | 4:3	  | 100FPS | has problems 

### TODO ###
# i should display the video resolution before and after the submatrix 
# make iphone 60FPS 
# keep 1080p for ball tracking 
# find a good way to toggle some comments on/off with a flag or something 

# =========================
# Config
# =========================
CAMERA_LEFT_INDEX = 0
CAMERA_RIGHT_INDEX = 1
CAMERA_THIRD_INDEX = 3

ATHLETE = "kenny"
SESSION = "session_test"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 60

BORDER_COLORS = {
    "left": "red",
    "right": "blue",
    "third": "green"
}
BORDER_THICKNESS = 5

# =========================
# Paths and Directories 
# =========================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

# Output video directories 
video_dirs = {
    "left": session_dir / "videos" / "player_tracking" / "raw" / "left",
    "right": session_dir / "videos" / "player_tracking" / "raw" / "right",
    "third": session_dir / "videos" / "ball_tracking" / "raw"
}

# =========================
# Camera Manager Class
# =========================
class CameraManager:
    
    # CameraManager initialization: Open video streams and set properties
    def __init__(self):
        self.caps = {
            "left": cv.VideoCapture(CAMERA_LEFT_INDEX),
            "right": cv.VideoCapture(CAMERA_RIGHT_INDEX),
            "third": cv.VideoCapture(CAMERA_THIRD_INDEX)
        }

        # Check if cameras are opened successfully 
        for name, cap in self.caps.items():
            if not cap.isOpened():
                print(f"[ERROR] Could not open camera '{name}' (index {cap})")

        self.set_properties()

    # Set camera properties for each camera
    def set_properties(self):
        for name, cap in self.caps.items():
            # display original resolution and FPS 
            actual_fps = cap.get(cv.CAP_PROP_FPS)
            print(f"[{name.upper()}] Original camera properties: \
                  {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}, FPS: {actual_fps:.2f}")
            
            cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv.CAP_PROP_FPS, FPS)
            print(f"Setting camera {name} to {FRAME_WIDTH}x{FRAME_HEIGHT} at {FPS} FPS...")

            # display original resolution and FPS 
            actual_fps = cap.get(cv.CAP_PROP_FPS)
            print(f"[{name.upper()}] Adjusted camera properties: \
                  {cap.get(cv.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT)}, FPS: {actual_fps:.2f}")

    # Read frames from all cameras
    def read_frames(self):
        return {name: cap.read()[1] for name, cap in self.caps.items()}
    
    # Release all camera resources 
    def release(self):
        for cap in self.caps.values():
            cap.release()

# =========================
# Video Recorder Class
# =========================
class VideoRecorder: 

    # VideoRecorder initialization: Create video writers for each camera
    #TODO what are video writers?
    def __init__(self, session_dir):
        self.writers = {}
        self.recording = False
        self.throw_num = 0
        self.session_dir = session_dir
        self.video_dirs = video_dirs
        for path in self.video_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    # Get the next throw number based on existing video files 
    def get_next_throw_number(self):
        max_count = 0
        for path in self.video_dirs.values():
            count = len(list(path.glob("freethrow*.avi")))
            max_count = max(max_count, count)
        return max_count + 1
    
    # Start recording: 
    def start(self, sample_frame_dims):
        self.recording = True
        self.throw_num = self.get_next_throw_number()
        print(f"🟢 Starting freethrow{self.throw_num}")
        fourcc = cv.VideoWriter_fourcc(*'MJPG')

        for name, dims in sample_frame_dims.items():
            filename = f"freethrow{self.throw_num}.avi"
            filepath = self.video_dirs[name] / filename
            print(f"💾 Saving {name} feed to: {filepath} at resolution bracketTODObracketxTODO")
            self.writers[name] = cv.VideoWriter(str(filepath), fourcc, FPS, dims)

    # Write frames to video files
    def write(self, frames):
        for name, frame in frames.items():
            if name in self.writers:
                self.writers[name].write(frame)

    # Stop recording: Release all video writers
    def stop(self):
        print("🛑 Stopping recording.")
        for writer in self.writers.values():
            writer.release()
        self.writers = {}
        self.recording = False

    def is_recording(self):
        return self.recording
    

# =========================
# Free Throw Recorder App 
# ========================= 
class FreeThrowRecorderApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Free Throw Recorder")
        self.root.resizable(True, True)
        self.root.geometry("1800x1200")  # adjust as needed
        self.camera_manager = CameraManager()
        self.recorder = VideoRecorder(session_dir)

        self.status_text = tk.StringVar(value="Status: Idle")
        self.labels = {}
        self.images = {}

        self.setup_gui()
        self.update_frames()

    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10)

        # ---- TOP ROW: Left and Right Feeds ----
        frame_top = tk.Frame(main_frame)
        frame_top.pack()

        self.labels["left"] = Label(frame_top)
        self.labels["left"].pack(side=tk.LEFT, padx=5)

        self.labels["right"] = Label(frame_top)
        self.labels["right"].pack(side=tk.LEFT, padx=5)

        # ---- BOTTOM ROW: Button + Legend + Ball Feed ----
        frame_bottom = tk.Frame(main_frame)
        frame_bottom.pack(pady=10)

        # Button on the LEFT
        button_frame = tk.Frame(frame_bottom)
        button_frame.pack(side=tk.LEFT, padx=10)
        Button(button_frame, text="Start/Stop Recording",
            command=self.toggle_recording, height=2, width=20).pack()

        # Legend in the MIDDLE
        legend_frame = tk.Frame(frame_bottom)
        legend_frame.pack(side=tk.LEFT, padx=20)

        for name, color in BORDER_COLORS.items():
            key_frame = tk.Frame(legend_frame)
            key_frame.pack(anchor="w", pady=5)

            # Colored square
            color_label = tk.Label(key_frame, width=2, height=1, bg=color)
            color_label.pack(side=tk.LEFT)

            # Text label
            if name == "left":
                text = "Left Camera (640x640)"
            elif name == "right":
                text = "Right Camera (640x640)"
            else:
                text = "Ball Camera (1920×1080)"

            tk.Label(key_frame, text=text).pack(side=tk.LEFT, padx=5)

        # Ball feed on the RIGHT
        self.labels["third"] = Label(frame_bottom)
        self.labels["third"].pack(side=tk.LEFT, padx=10)

        # Status text UNDER the button
        Label(button_frame, textvariable=self.status_text, font=("Helvetica", 14)).pack(pady=10)


        self.root.protocol("WM_DELETE_WINDOW", self.on_close)



    def toggle_recording(self):
        if not self.recorder.is_recording():
            # Get frame dimensions by reading sample frame
            frames = self.camera_manager.read_frames()
            dims = {
                "left": (640, 640),
                "right": (640, 640),
                "third": (FRAME_WIDTH, FRAME_HEIGHT)
            }
            self.recorder.start(dims)
            self.status_text.set(f"Recording freethrow{self.recorder.throw_num}")
        else:
            self.recorder.stop()
            self.status_text.set("Status: Idle")

    def update_frames(self):
        frames = self.camera_manager.read_frames()

        for name, frame in frames.items():
            if frame is None:
                continue

            # Crop left/right to 640x640 from 1280x720
            if name in ["left", "right"]:
                frame = frame[40:680, 320:960]  # crop center 640x640

            # Resize for display (preserve correct aspect ratio for third)
            if name in ["left", "right"]:
                frame_display = cv.resize(frame, (640, 640))

            else:  # third camera (shrink slightly but keep aspect ratio)
                frame_display = cv.resize(frame, (760, 427)) # 16:9 aspect ratio

            # Convert to RGB and add border
            frame_rgb = cv.cvtColor(frame_display, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = ImageOps.expand(pil_img, border=BORDER_THICKNESS, fill=BORDER_COLORS[name])

            img = ImageTk.PhotoImage(pil_img)
            self.images[name] = img
            self.labels[name].configure(image=img)

            # Save cropped or original frame depending on camera
            frames[name] = frame

        if self.recorder.is_recording():
            self.recorder.write(frames)

        self.root.after(15, self.update_frames)

    def on_close(self):
        self.camera_manager.release()
        self.recorder.stop()
        cv.destroyAllWindows()
        self.root.quit()


# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = FreeThrowRecorderApp(root)
    root.mainloop()