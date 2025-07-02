"""
IMPORTANT:This script is currently broken and need to be fixed.
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.hevc']  # Supported formats
RESIZE_DIMENSIONS = (640, 480)     # Display size in GUI

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3] # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION

left_calib_dir = session_dir / "calibration" / "calib_images" / "left"
right_calib_dir = session_dir / "calibration" / "calib_images" / "right"

# raw video directories 
raw_left_dir = session_dir / "videos" / "player_tracking" / "raw" / "left"
raw_right_dir = session_dir / "videos" / "player_tracking" / "raw" / "right"
raw_ball_dir = session_dir / "videos" / "ball_tracking" / "raw"

# processed video directories
processed_left_dir = session_dir / "videos" / "player_tracking" / "processed" / "left"
processed_right_dir = session_dir / "videos" / "player_tracking" / "processed" / "right"
processed_ball_dir = session_dir / "videos" / "ball_tracking" / "processed"

# ========================================
# Video Trimmer Class
# ========================================

class VideoTrimmer:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Trimmer")

        # Initialize video tracking variables
        self.cap = None
        self.current_frame = 0
        self.start_frame = None
        self.end_frame = None
        self.frames = []
        self.fps = 30
        self.video_files = []
        self.video_index = 0
        self.setting_start = True  # Track whether we are setting start or end

        self.load_video_folder()  # Load videos from folder

        # ========================================
        # GUI Layout
        # ========================================

        # Create a canvas for displaying frames
        self.canvas = tk.Canvas(master, width=RESIZE_DIMENSIONS[0], height=RESIZE_DIMENSIONS[1])
        self.canvas.pack()

        # Label to show frame info
        self.info_label = tk.Label(master, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Frame to hold control buttons
        controls = tk.Frame(master)
        controls.pack()

        # Button: go to previous frame
        self.prev_btn = tk.Button(controls, text="<< Prev", command=self.prev_frame)
        self.prev_btn.pack(side=tk.LEFT)

        # Button: go to next frame
        self.next_btn = tk.Button(controls, text="Next >>", command=self.next_frame)
        self.next_btn.pack(side=tk.LEFT)

        # Button: mark current frame as start of trimmed video
        self.set_start_btn = tk.Button(controls, text="Set Start", command=self.set_start)
        self.set_start_btn.pack(side=tk.LEFT)

        # Button: mark current frame as end of trimmed video
        self.set_end_btn = tk.Button(controls, text="Set End", command=self.set_end)
        self.set_end_btn.pack(side=tk.LEFT)

        # Button: save trimmed clip using start/end
        self.save_btn = tk.Button(controls, text="Save Trim", command=self.save_trim)
        self.save_btn.pack(side=tk.LEFT)

        # Button: skip current video without saving
        self.skip_btn = tk.Button(controls, text="Skip Video", command=self.next_video)
        self.skip_btn.pack(side=tk.LEFT)

        # Basic keys
        self.master.bind("<Right>", lambda e: self.jump_frames(1))
        self.master.bind("<Left>", lambda e: self.jump_frames(-1))
        self.master.bind("<Return>", self.key_handler)
        self.master.bind("<Home>", self.key_handler)
        self.master.bind("<End>", self.key_handler)
        self.master.bind("<Next>", self.key_handler)   # PageDown
        self.master.bind("<Prior>", self.key_handler)  # PageUp

        # Shifted arrow keys
        self.master.bind("<Shift-Right>", lambda e: self.jump_frames(10))
        self.master.bind("<Shift-Left>", lambda e: self.jump_frames(-10))


        self.load_video()  # Load the first video

    # ========================================
    # Load Videos from Folder
    # ========================================

    def load_video_folder(self):
        input_dir = Path(SESSION_FOLDER) / INPUT_SUBDIR
        output_dir = Path(SESSION_FOLDER) / OUTPUT_SUBDIR

        if not input_dir.exists():
            print(f"Input folder not found: {input_dir}")
            self.master.quit()
            return

        # Load all valid videos
        self.video_files = [
            str(input_dir / f)
            for f in os.listdir(input_dir)
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        ]

        if not self.video_files:
            print("No videos found.")
            self.master.quit()
            return

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def load_video(self):
        # If all videos are processed, exit GUI
        if self.video_index >= len(self.video_files):
            print("All videos done!")
            self.master.quit()
            return

        # Load video frames into memory
        video_path = self.video_files[self.video_index]
        print(f"Loading: {video_path}")
        self.cap = cv2.VideoCapture(video_path)

        # === Automatically detect FPS ===
        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = detected_fps if detected_fps > 0 else 30  # fallback if 0
        print(f"Detected FPS: {self.fps:.2f}")

        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        self.cap.release()

        # Reset state for current video
        self.current_frame = 0
        self.start_frame = None
        self.end_frame = None
        self.setting_start = True  # Always reset to set start first
        self.show_frame()

    # ========================================
    # Frame Navigation and Display
    # ========================================

    def show_frame(self):
        # Display the current frame on the canvas
        if 0 <= self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            frame = cv2.resize(frame, RESIZE_DIMENSIONS)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Fixed color
            img = tk.PhotoImage(master=self.canvas, data=cv2.imencode(".ppm", rgb)[1].tobytes())
            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.update_info_label()

    def update_info_label(self):
        info = f"Current Frame: {self.current_frame}"
        if self.start_frame is not None:
            info += f" | Start: {self.start_frame}"
        if self.end_frame is not None:
            info += f" | End: {self.end_frame}"
        self.info_label.config(text=info)

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()

    def next_frame(self):
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.show_frame()
    
    def jump_frames(self, amount):
        self.current_frame = max(0, min(self.current_frame + amount, len(self.frames) - 1))
        self.show_frame()


    # ========================================
    # Frame Selection
    # ========================================

    def set_start(self):
        self.start_frame = self.current_frame
        print(f"Start frame set: {self.start_frame}")
        self.update_info_label()

    def set_end(self):
        self.end_frame = self.current_frame
        print(f"End frame set: {self.end_frame}")
        self.update_info_label()

    # ========================================
    # Save Trimmed Video
    # ========================================

    def save_trim(self):
        if self.start_frame is None or self.end_frame is None:
            print("Please set both start and end frames!")
            return

        start = min(self.start_frame, self.end_frame)
        end = max(self.start_frame, self.end_frame)

        filename = f"freethrow{self.video_index + 1:03d}_trimmed.mp4"
        output_path = self.output_dir / filename

        height, width = self.frames[0].shape[:2]
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))

        for f in self.frames[start:end+1]:
            out.write(f)
        out.release()

        print(f"Saved trimmed video: {output_path}")
        self.next_video()

    def next_video(self):
        self.video_index += 1
        self.load_video()

    # ========================================
    # Keyboard Shortcuts Handler
    # ========================================

    def key_handler(self, event):
        key = event.keysym
        if key == 'Return':
            if self.setting_start:
                self.set_start()
                print("Start frame marked. Press Enter again to mark End frame.")
                self.setting_start = False
            else:
                self.set_end()
                self.save_trim()
                self.setting_start = True
        elif key == 'Home':
            self.current_frame = 0
            self.show_frame()
        elif key == 'End':
            self.current_frame = len(self.frames) - 1
            self.show_frame()
        elif key == 'Next':  # PageDown
            self.jump_frames(60)
        elif key == 'Prior':  # PageUp
            self.jump_frames(-60)



# ========================================
# Main Entry Point
# ========================================

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmer(root)
    root.mainloop()
