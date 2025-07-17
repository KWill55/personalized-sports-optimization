"""
Title: Video Player for AVI Files

Description:
    The purpose of this module is to provide a GUI-based video player for AVI files.
    It allows users to load a folder containing AVI files, navigate through them, and display video.
    Information such as resolution, duration, FPS, and frame count are displayed.

Inputs:
    - Folder containing video files

Usage: 
    - Load a folder conttaining video files
    - Navigate through the videos using buttons

Outputs
    - Displays video files in GUI 
    - Displays video information such as resolution, duration, FPS, and frame count

Last Updated: 15 July 2025
"""

### TODO ###:
# make this reusable for 2D pose estimation videos and other video types besides .avi (such as .mp4)
# maybe switch this file to be in shared if it is used by multiple scripts (I think it will be) 

import cv2
import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from pathlib import Path
import yaml

# =========================
# Config
# =========================

# Load YAML Config
config_path = Path(__file__).resolve().parents[1] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# =========================
# Paths and Directories 
# =========================
base_dir = Path(__file__).resolve().parents[1]
session_dir = base_dir / "data" / ATHLETE / SESSION

# =========================
# Class for Video Player 
# =========================

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AVI Video Player")

        self.video_files = []
        self.current_index = 0
        self.cap = None
        self.playing = False

        # UI Elements
        Label(root, text="AVI Video Player", font=("Helvetica", 24, "bold")).pack(pady=(10, 2))

        # Create a frame to act as a border for video display
        video_frame_container = tk.Frame(root, bg="black", padx=5, pady=5)
        video_frame_container.pack(pady=10)

        # Place the video display label inside the frame
        self.label = Label(video_frame_container)
        self.label.pack()


        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        Button(control_frame, text="Load Folder", command=self.load_folder).grid(row=0, column=0, padx=5)
        Button(control_frame, text="Previous Clip", command=self.prev_video).grid(row=0, column=1, padx=5)
        Button(control_frame, text="Play/Pause", command=self.toggle_play).grid(row=0, column=2, padx=5)
        Button(control_frame, text="Next Clip", command=self.next_video).grid(row=0, column=3, padx=5)

        frame_control = tk.Frame(root)
        frame_control.pack(pady=5)
        Button(frame_control, text="<< Frame", command=self.prev_frame).grid(row=0, column=0, padx=5)
        Button(frame_control, text="Frame >>", command=self.next_frame).grid(row=0, column=1, padx=5)

        # Create a container frame for the info box
        info_container = tk.Frame(root, highlightbackground="black", highlightthickness=5, bd=0, padx=10, pady=10)
        info_container.pack(pady=15)

        # Add a title inside the container
        Label(info_container, text="Video Clip Info", font=("Helvetica", 18, "bold")).pack(pady=(5, 10))

        # Static video info
        self.video_info = tk.StringVar(value="No videos loaded")
        Label(info_container, textvariable=self.video_info, font=("Helvetica", 15)).pack(pady=2)

        # Dynamic frame info
        self.status = tk.StringVar(value="Press 'Load Folder' to select videos")
        Label(info_container, textvariable=self.status, font=("Helvetica", 15)).pack(pady=0)


        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_folder(self):
        folder = filedialog.askdirectory(initialdir=session_dir, title="Select Folder with AVI Files")
        if not folder:
            return
        self.video_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".avi")]
        self.video_files.sort()
        if self.video_files:
            self.current_index = 0
            self.open_video(self.video_files[self.current_index])
        self.status.set(f"Loaded {len(self.video_files)} videos")

    def open_video(self, filepath):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(filepath)
        self.playing = True
        self.show_frame()

        # Calculate video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        filename = os.path.basename(filepath)

        # Update video info label 
        self.video_info.set(
            f"Filename: {filename}\n"
            f"Resolution: {width}x{height} | Total Frames: {total_frames} | Duration: {duration:.2f}s | FPS: {fps:.2f}"
        )

    def show_frame(self):
        if self.cap and self.playing:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            # Display the frame 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.label.configure(image=img)
            self.label.image = img

            # Update frame info
            self.update_frame_info()

            # Schedule next frame
            delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))  # dynamically get video FPS
            self.root.after(delay, self.show_frame)

    def update_frame_info(self):
        if self.cap:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Calculate elapsed time
            elapsed_time = current_frame / fps if fps > 0 else 0
            total_time = total_frames / fps if fps > 0 else 0

            self.status.set(
                f"Current Frame: {current_frame}/{total_frames}  |  Elapsed Time: {elapsed_time:.2f}s/{total_time:.2f}s"
            )


    def toggle_play(self):
        if not self.cap:
            return

        # If at the end of the video, restart from beginning
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if current_frame >= total_frames:  # Video finished
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Toggle play/pause
        self.playing = not self.playing
        if self.playing:
            self.show_frame()


    def next_video(self):
        if self.video_files:
            self.current_index = (self.current_index + 1) % len(self.video_files)
            self.open_video(self.video_files[self.current_index])

    def prev_video(self):
        if self.video_files:
            self.current_index = (self.current_index - 1) % len(self.video_files)
            self.open_video(self.video_files[self.current_index])

    def next_frame(self):
        if self.cap:
            self.playing = False
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.label.configure(image=img)
                self.label.image = img
                self.update_frame_info()

    def prev_frame(self):
        if self.cap:
            self.playing = False
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))  # Go back two frames
            self.next_frame()

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()
