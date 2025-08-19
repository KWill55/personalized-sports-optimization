"""
Title: record_freethrows.py 

Description:
    Record free throw attempts from three cameras: two (stereo) for player tracking
    and one for ball tracking. Provides a Tkinter GUI to start/stop recording and
    displays the current attempt number. Videos are saved in a structured directory.

Inputs
    - Three cameras (two for player tracking, one for ball tracking)

Usage 
    - GUI has "Start/Stop Recording" button
    
Outputs
    - 640x640 videos for player tracking (left and right cameras)
    - 1080p (or configured) video for ball tracking

Last Updated: 08 August 2025
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps
import time
import threading
import yaml

# =========================
# Config (from YAML)
# =========================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Camera Indices
CAMERA_LEFT_INDEX = cfg["left_cam_index"]
CAMERA_RIGHT_INDEX = cfg["right_cam_index"]
CAMERA_THIRD_INDEX = cfg["third_cam_index"]

# Session Info
ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# Video Settings
FRAME_WIDTH = int(cfg["original_frame_width"])
FRAME_HEIGHT = int(cfg["original_frame_height"])
FPS_LEFT_RIGHT = float(cfg["player_tracking_fps"])
FPS_THIRD = float(cfg["ball_tracking_fps"])
GUI_REFRESH_MS = 30

# Visual Settings
BORDER_COLORS = {"left": "red", "right": "blue", "third": "green"}
BORDER_THICKNESS = 5  # Can move to YAML if desired

PAD_WIDTH = int(cfg.get("throw_number_width", 3))  # zero-pad to 3 digits
NAME_PREFIX = "freethrow"


# =========================
# Paths and Directories
# =========================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
video_dirs = {
    "left": session_dir / "videos" / "player_tracking" / "raw" / "left",
    "right": session_dir / "videos" / "player_tracking" / "raw" / "right",
    "third": session_dir / "videos" / "ball_tracking" / "raw",
}
for path in video_dirs.values():
    path.mkdir(parents=True, exist_ok=True)

# =========================
# Shared Resources
# =========================
frames = {"left": None, "right": None, "third": None}
frame_locks = {k: threading.Lock() for k in frames}

# control flags & synchronization
recording = False
writers = {}
writers_lock = threading.Lock()
stop_event = threading.Event()

throw_count = 0
frame_counters = {"left": 0, "right": 0, "third": 0}
start_time = None


# =========================
# Utilities
# =========================
def _print_cam_init(name: str, cap: cv.VideoCapture):
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"[{name.upper()}] Initialized: {w}x{h}, FPS: {fps:.5f}")


# =========================
# Camera Capture Thread
# =========================
def capture_camera(name, index, crop=False):
    """
    Continuously captures frames from the specified camera in a separate thread.

    Args:
        name (str): 'left' | 'right' | 'third'
        index (int): cv2.VideoCapture index
        crop (bool): If True, crop frames to 640x640 (used for left/right)
    """
    cap = cv.VideoCapture(index)

    # Request camera properties (some drivers will ignore these)
    target_fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_THIRD
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, target_fps)

    # Reduce latency/drop by shrinking internal buffer
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Try requesting MJPG from the camera to lower USB bandwidth (best-effort)
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    _print_cam_init(name, cap)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # avoid busy spin; camera hiccup
            time.sleep(0.002)
            continue

        if crop:
            # Center-crop to 640x640 from 1280x720
            # (y: 40->680, x: 320->960)
            # Ensure bounds are safe even if the camera drops to a different res
            h, w = frame.shape[:2]
            y0, y1 = 40, min(680, h)
            x0, x1 = 320, min(960, w)
            frame = frame[y0:y1, x0:x1]
            # If dimensions not exactly 640x640 due to driver quirks, resize
            if frame.shape[0] != 640 or frame.shape[1] != 640:
                frame = cv.resize(frame, (640, 640))

        with frame_locks[name]:
            frames[name] = frame

        if recording:
            # Increment under capture loop to reflect actual captured frame rate
            # (writer is throttled separately)
            try:
                frame_counters[name] += 1
            except Exception:
                # Extremely rare race if counters reset during stop/start
                pass

        # tiny sleep to yield; do not oversleep or you'll throttle capture unintentionally
        time.sleep(0.001)

    cap.release()


# =========================
# Video Writer Thread (FPS Throttling)
# =========================
def write_frames():
    """
    Writes frames to disk at the correct FPS using throttling, so the output
    video duration matches real time. Access to `writers` is synchronized.
    """
    intervals = {"left": 1.0 / FPS_LEFT_RIGHT, "right": 1.0 / FPS_LEFT_RIGHT, "third": 1.0 / FPS_THIRD}
    last_write_time = {name: 0.0 for name in frames}

    while not stop_event.is_set():
        if recording:
            now = time.time()
            # Snapshot writers under lock to avoid iterator invalidation
            with writers_lock:
                local_writers = list(writers.items())

            for name, writer in local_writers:
                if now - last_write_time.get(name, 0.0) >= intervals[name]:
                    with frame_locks[name]:
                        frame = frames[name]
                    if frame is not None:
                        try:
                            writer.write(frame)
                        except Exception as e:
                            print(f"[ERROR] write({name}) failed: {e}")
                        last_write_time[name] = now

        time.sleep(0.001)


# =========================
# Recording Functions
# =========================
def get_next_throw_number():
    """
    Scans all video directories for existing 'freethrow*.avi' files and returns
    the next available throw number.
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
        dims (dict): { 'left': (w,h), 'right': (w,h), 'third': (w,h) }
    """
    global writers, recording, throw_count, start_time, frame_counters

    throw_count = get_next_throw_number()
    label = f"{NAME_PREFIX}{throw_count:0{PAD_WIDTH}d}"
    print(f"🟢 Starting {label}")

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    with writers_lock:
        for name, size in dims.items():
            filepath = video_dirs[name] / f"{label}.avi"   # <— padded!
            fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_THIRD
            writers[name] = cv.VideoWriter(str(filepath), fourcc, fps, size)
            print(f"[INFO] Writing {name} to {filepath} @ {fps} FPS")

    # reset counters AFTER writers are ready
    frame_counters = {k: 0 for k in frame_counters}
    start_time = time.time()
    # flip recording last so writer thread sees consistent state
    recording = True


def stop_recording():
    """
    Stops the current recording session, calculates FPS, and releases video writers.
    """
    global writers, recording

    duration = max(0.0, time.time() - (start_time or time.time()))
    print(f"🛑 Stopping recording after {duration:.1f}s")

    # Compute FPS stats (counters are safe enough here)
    for name, count in frame_counters.items():
        actual_fps = (count / duration) if duration > 0 else 0.0
        print(f"[RESULT] {name.upper()} Actual FPS: {actual_fps:.1f}")
        if name in ["left", "right"] and actual_fps < FPS_LEFT_RIGHT * 0.8:
            print(f"[WARNING] {name.upper()} is below target FPS ({actual_fps:.1f} vs {FPS_LEFT_RIGHT})")
        if name == "third" and actual_fps < FPS_THIRD * 0.8:
            print(f"[WARNING] THIRD is below expected FPS ({actual_fps:.1f})")

    # 1) Stop writer loop first so it won't touch writers while we release them
    recording = False
    # let write_frames loop observe the flag
    time.sleep(0.02)

    # 2) Release writers under lock
    with writers_lock:
        for w in writers.values():
            try:
                w.release()
            except Exception as e:
                print(f"[ERROR] release() failed: {e}")
        writers.clear()

    print("[INFO] Writers closed.")


# =========================
# GUI App
# =========================
class FreeThrowRecorderApp:
    """
    Tkinter-based GUI application for displaying camera feeds and managing recordings.
    """

    def __init__(self, root):
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
        global recording
        if not recording:
            dims = {"left": (640, 640), "right": (640, 640), "third": (FRAME_WIDTH, FRAME_HEIGHT)}
            start_recording(dims)
            label = f"{NAME_PREFIX}{get_next_throw_number()-1:0{PAD_WIDTH}d}"
            self.status_text.set(f"Recording {label}")
        else:
            stop_recording()
            self.status_text.set("Status: Idle")

    def update_gui(self):
        # Update displayed frames
        for name in frames:
            with frame_locks[name]:
                frame = frames[name]

            if frame is not None:
                if name in ["left", "right"]:
                    frame_display = cv.resize(frame, (640, 640))
                else:
                    # maintain a reasonable preview size for the third camera
                    # (approx 16:9 preview box)
                    frame_display = cv.resize(frame, (760, 427))

                frame_rgb = cv.cvtColor(frame_display, cv.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img = ImageOps.expand(pil_img, border=BORDER_THICKNESS, fill=BORDER_COLORS[name])
                img = ImageTk.PhotoImage(pil_img)
                self.images[name] = img
                self.labels[name].configure(image=img)

        if not stop_event.is_set():
            self.root.after(GUI_REFRESH_MS, self.update_gui)

    def on_close(self):
        # Stop any active recording first
        if recording:
            stop_recording()

        # Signal threads to exit and close GUI
        stop_event.set()
        # small delay to let threads unwind
        time.sleep(0.03)
        self.root.destroy()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Capture threads
    threading.Thread(target=capture_camera, args=("left", CAMERA_LEFT_INDEX, True), daemon=True).start()
    threading.Thread(target=capture_camera, args=("right", CAMERA_RIGHT_INDEX, True), daemon=True).start    ()
    threading.Thread(target=capture_camera, args=("third", CAMERA_THIRD_INDEX, False), daemon=True).start()

    # Writer thread
    threading.Thread(target=write_frames, daemon=True).start()

    # GUI
    root = tk.Tk()
    app = FreeThrowRecorderApp(root)
    root.mainloop()
