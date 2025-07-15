"""
Title: Video Player for AVI Files

Status bar label is used to display changing things like current frame, 
Video info label is used to display static information about the current video clip

"""

### TODO ###:
# make this reusable for 2D pose estimation videos and other video types besides .avi 
# make borders around the video frames 
# make the video player more visually appealing

### Done ###
# make load folder start at session directory 
# make sure videos are displayed in correct resolution 
# make going to next or previous frame update the displayed frame 
# make video info label to display static inforatmino about current video clip
# make pressing the play button repeat the clip from the beginning 

### IDeas ###
# for the 2d pose estimation visualization, see both visualizations side by side connected again. during processing just split the image and then
# sew them back together for visualization
# show which frame is currently displayed for each side out of how many total frames each side has 

import cv2
import os
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from pathlib import Path

# =========================
# Constant Parameters 
# =========================

ATHLETE = "kenny"
SESSION = "session_test"

# =========================
# Paths and Directories 
# =========================
base_dir = Path(__file__).resolve().parents[3]
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
        self.label = Label(root)
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

        # label to display video clip title 
        Label(root, text="Video Clip Info", font=("Helvetica", 16, "bold")).pack(pady=(10, 2))

        # label to display static video information
        self.video_info = tk.StringVar()
        self.video_info.set("No video loaded")
        Label(root, textvariable=self.video_info, font=("Helvetica", 12)).pack(pady=5)

        # label to display video frame info 
        self.status = tk.StringVar()
        self.status.set("No folder loaded")
        Label(root, textvariable=self.status, font=("Helvetica", 12)).pack(pady=10)

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
            filename = os.path.basename(self.video_files[self.current_index])
            self.status.set(f"Current Frame: {current_frame}/{total_frames}")



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
