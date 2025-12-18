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
    - two videos for player tracking and one for ball tracking

Last Updated: 17 December 2025
"""

import cv2 as cv
from pathlib import Path
import sys
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps
import time
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from utils.io_utils import load_config

# =========================
# Config (from YAML)
# =========================
cfg = load_config("project_config.yaml")

# Camera Indices
CAMERA_LEFT_INDEX = cfg["left_cam_index"]
CAMERA_RIGHT_INDEX = cfg["right_cam_index"]
CAMERA_BALL_INDEX = cfg["ball_cam_index"]

# Session Info
ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# Video Settings
STEREO_UNCROPPED_RES = tuple(int(v) for v in cfg["uncropped_stereo_resolution"])
STEREO_CROP_RES = tuple(int(v) for v in cfg.get("cropped_stereo_resolution", STEREO_UNCROPPED_RES))
BALL_UNCROPPED_RES = tuple(int(v) for v in cfg["uncropped_ball_resolution"])
FPS_LEFT_RIGHT = float(cfg["player_tracking_fps"])
FPS_BALL = float(cfg["ball_tracking_fps"])
GUI_REFRESH_MS = 30

# Visual Settings
BORDER_COLORS = {"left": "red", "right": "blue", "ball": "green"}
BORDER_THICKNESS = 5

PAD_WIDTH = int(cfg.get("throw_number_width", 3))  # zero-pad to 3 digits
NAME_PREFIX = "batch"

try:
    CROP_W, CROP_H = map(int, STEREO_CROP_RES)
except Exception:
    CROP_W, CROP_H = 640, 640

CROP_SIZE = (CROP_W, CROP_H)  # (width, height)

# These are your "intent" values for recording / labels; GUI will dynamically resize feeds.
DISPLAY_RES = {"left": CROP_SIZE, "right": CROP_SIZE, "ball": BALL_UNCROPPED_RES}
TARGET_FPS = {"left": FPS_LEFT_RIGHT, "right": FPS_LEFT_RIGHT, "ball": FPS_BALL}


# =========================
# Paths and Directories
# =========================
session_dir = PROJECT_ROOT / "data" / ATHLETE / SESSION
video_dirs = {
    "left": session_dir / "videos" / "player_tracking" / "raw" / "left",
    "right": session_dir / "videos" / "player_tracking" / "raw" / "right",
    "ball": session_dir / "videos" / "ball_tracking" / "raw",
}
for path in video_dirs.values():
    path.mkdir(parents=True, exist_ok=True)

# =========================
# Shared Resources
# =========================
frames = {"left": None, "right": None, "ball": None}
frame_locks = {k: threading.Lock() for k in frames}

# control flags & synchronization
recording = False
writers = {}
writers_lock = threading.Lock()
stop_event = threading.Event()

throw_count = 0
frame_counters = {"left": 0, "right": 0, "ball": 0}
start_time = None


# =========================
# Utilities
# =========================
def _print_cam_init(name: str, cap: cv.VideoCapture):
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"[{name.upper()}] Initialized: {w}x{h}, FPS: {fps:.5f}")


def resize_to_fit(frame, max_w: int, max_h: int):
    """
    Resize frame to fit within (max_w, max_h) preserving aspect ratio.
    Returns resized frame and (new_w, new_h).

    Note: We do NOT upscale by default (keeps CPU lower and avoids blurry enlargement).
    """
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame, (w, h)

    scale = min(max_w / w, max_h / h)
    scale = min(scale, 1.0)  # don't upscale

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if (new_w, new_h) == (w, h):
        return frame, (w, h)

    resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized, (new_w, new_h)


# =========================
# Camera Capture Thread
# =========================
def capture_camera(name, index, crop=False):
    """
    Continuously captures frames from the specified camera in a separate thread.

    Args:
        name (str): 'left' | 'right' | 'ball'
        index (int): cv2.VideoCapture index
        crop (bool): If True, crop frames for left/right cameras
    """
    cap = cv.VideoCapture(index)

    # Request camera properties (some drivers will ignore these)
    target_fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_BALL
    if name in ["left", "right"]:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, STEREO_UNCROPPED_RES[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, STEREO_UNCROPPED_RES[1])
    else:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, BALL_UNCROPPED_RES[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, BALL_UNCROPPED_RES[1])
    cap.set(cv.CAP_PROP_FPS, target_fps)

    # Reduce latency/drop by shrinking internal buffer
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Try requesting MJPG from the camera to lower USB bandwidth
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    _print_cam_init(name, cap)

    crop_x0 = crop_x1 = crop_y0 = crop_y1 = 0
    crop_resize_needed = False
    crop_resize_interp = cv.INTER_LINEAR
    crop_last_hw = None
    if crop:
        crop_w, crop_h = CROP_SIZE

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.002)
            continue

        if crop:
            if crop_last_hw is None:
                h, w = frame.shape[:2]
                crop_w_eff = crop_w if crop_w <= w else w
                crop_h_eff = crop_h if crop_h <= h else h
                crop_x0 = (w - crop_w_eff) // 2
                crop_y0 = (h - crop_h_eff) // 2
                crop_x1 = crop_x0 + crop_w_eff
                crop_y1 = crop_y0 + crop_h_eff
                crop_resize_needed = (crop_w_eff != crop_w) or (crop_h_eff != crop_h)
                crop_resize_interp = cv.INTER_LINEAR
                crop_last_hw = (h, w)

            frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            if crop_resize_needed:
                frame = cv.resize(frame, (crop_w, crop_h), interpolation=crop_resize_interp)

        with frame_locks[name]:
            frames[name] = frame

        if recording:
            try:
                frame_counters[name] += 1
            except Exception:
                pass

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
    intervals = {
        "left": 1.0 / FPS_LEFT_RIGHT,
        "right": 1.0 / FPS_LEFT_RIGHT,
        "ball": 1.0 / FPS_BALL,
    }
    last_write_time = {name: 0.0 for name in frames}

    while not stop_event.is_set():
        if recording:
            now = time.time()
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
    Scans all video directories for existing files with NAME_PREFIX and returns
    the next available throw number.
    """
    max_count = 0
    for path in video_dirs.values():
        pattern = f"{NAME_PREFIX}*.avi"
        prefix_len = len(NAME_PREFIX)
        for file in path.glob(pattern):
            try:
                num = int(file.stem[prefix_len:])
                max_count = max(max_count, num)
            except ValueError:
                continue
    return max_count + 1


def start_recording(dims):
    """
    Starts a new recording session by initializing VideoWriter objects.

    Args:
        dims (dict): { 'left': (w,h), 'right': (w,h), 'ball': (w,h) }
    """
    global writers, recording, throw_count, start_time, frame_counters

    throw_count = get_next_throw_number()
    label = f"{NAME_PREFIX}{throw_count:0{PAD_WIDTH}d}"
    print(f"🟢 Starting {label}")

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    with writers_lock:
        for name, size in dims.items():
            filepath = video_dirs[name] / f"{label}.avi"
            fps = FPS_LEFT_RIGHT if name in ["left", "right"] else FPS_BALL
            writers[name] = cv.VideoWriter(str(filepath), fourcc, fps, size)
            print(f"[INFO] Writing {name} to {filepath} @ {fps} FPS")

    frame_counters = {k: 0 for k in frame_counters}
    start_time = time.time()
    recording = True


def stop_recording():
    """
    Stops the current recording session, calculates FPS, and releases video writers.
    """
    global writers, recording

    duration = max(0.0, time.time() - (start_time or time.time()))
    print(f"🛑 Stopping recording after {duration:.1f}s")

    for name, count in frame_counters.items():
        actual_fps = (count / duration) if duration > 0 else 0.0
        print(f"[RESULT] {name.upper()} Actual FPS: {actual_fps:.1f}")
        if name in ["left", "right"] and actual_fps < FPS_LEFT_RIGHT * 0.8:
            print(f"[WARNING] {name.upper()} is below target FPS ({actual_fps:.1f} vs {FPS_LEFT_RIGHT})")
        if name == "ball" and actual_fps < FPS_BALL * 0.8:
            print(f"[WARNING] ball is below expected FPS ({actual_fps:.1f})")

    recording = False
    time.sleep(0.02)

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
    Feeds are dynamically resized to fit the window while preserving aspect ratio.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Free Throw Recorder")
        self.root.geometry("1800x1200")
        self.root.minsize(1200, 800)

        # make resizing feel nicer
        self.root.update_idletasks()

        self.status_text = tk.StringVar(value="Status: Idle")
        self.labels = {}
        self.images = {}
        self.info_labels = {}

        self.setup_gui()
        self.update_gui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_gui(self):
        frame_top = tk.Frame(self.root)
        frame_top.pack(fill="both", expand=True)

        self._create_camera_view(frame_top, "left", side=tk.LEFT, padx=5, pady=5)
        self._create_camera_view(frame_top, "right", side=tk.LEFT, padx=5, pady=5)

        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(fill="both", expand=True, pady=10)

        button_frame = tk.Frame(frame_bottom)
        button_frame.pack(side=tk.LEFT, padx=10, pady=5)

        Button(
            button_frame,
            text="Start/Stop Recording",
            command=self.toggle_recording,
            height=2,
            width=20,
        ).pack()
        Label(button_frame, textvariable=self.status_text, font=("Helvetica", 14)).pack(pady=10)

        legend_frame = tk.Frame(frame_bottom)
        legend_frame.pack(side=tk.LEFT, padx=20, pady=5)

        for name, color in BORDER_COLORS.items():
            lf = tk.Frame(legend_frame)
            lf.pack(anchor="w", pady=5)
            tk.Label(lf, width=2, height=1, bg=color).pack(side=tk.LEFT)
            text = f"{name.capitalize()} Camera"
            tk.Label(lf, text=text).pack(side=tk.LEFT, padx=5)

        self._create_camera_view(frame_bottom, "ball", side=tk.LEFT, padx=10, pady=5)

    def _create_camera_view(self, parent, name, **pack_kwargs):
        wrapper = tk.Frame(parent)
        info_label = tk.Label(wrapper, text=self._camera_info_text(name), font=("Helvetica", 12))
        info_label.pack()

        img_label = Label(wrapper)
        img_label.pack()

        wrapper.pack(fill="both", expand=True, **pack_kwargs)
        self.labels[name] = img_label
        self.info_labels[name] = info_label

    def _camera_info_text(self, name):
        w, h = DISPLAY_RES[name]
        fps = TARGET_FPS[name]
        return f"{name.capitalize()} Camera — {w}x{h} @ {fps:.0f} FPS"

    def _get_display_boxes(self):
        """
        Compute max display boxes (max_w, max_h) for each feed based on current window size.
        Layout assumption:
          - Left & Right on top row
          - Controls + legend + Ball on bottom row
        """
        W = max(800, self.root.winfo_width())
        H = max(600, self.root.winfo_height())

        # Reserve space for labels/borders/padding.
        pad_w = 120
        pad_h_top = 140
        pad_h_bottom = 240

        top_box_w = max(200, (W - pad_w) // 2)
        top_box_h = max(200, (H - pad_h_top) // 2)

        bottom_box_h = max(200, (H - pad_h_bottom) // 2)

        # Bottom row has button+legend taking up left space;
        # give ball something like ~55% of width.
        ball_box_w = max(260, int(W * 0.55) - 60)

        return {
            "left": (top_box_w, top_box_h),
            "right": (top_box_w, top_box_h),
            "ball": (ball_box_w, bottom_box_h),
        }

    def toggle_recording(self):
        global recording, throw_count
        if not recording:
            dims = {"left": CROP_SIZE, "right": CROP_SIZE, "ball": BALL_UNCROPPED_RES}
            start_recording(dims)
            label = f"{NAME_PREFIX}{throw_count:0{PAD_WIDTH}d}"
            self.status_text.set(f"Recording {label}")
        else:
            stop_recording()
            self.status_text.set("Status: Idle")

    def update_gui(self):
        boxes = self._get_display_boxes()

        for name in frames:
            with frame_locks[name]:
                frame = frames[name]

            if frame is None:
                continue

            max_w, max_h = boxes[name]
            frame_display, _ = resize_to_fit(frame, max_w, max_h)

            frame_rgb = cv.cvtColor(frame_display, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = ImageOps.expand(pil_img, border=BORDER_THICKNESS, fill=BORDER_COLORS[name])
            img = ImageTk.PhotoImage(pil_img)

            self.images[name] = img
            self.labels[name].configure(image=img)

        if not stop_event.is_set():
            self.root.after(GUI_REFRESH_MS, self.update_gui)

    def on_close(self):
        if recording:
            stop_recording()

        stop_event.set()
        time.sleep(0.03)
        self.root.destroy()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Capture threads
    threading.Thread(target=capture_camera, args=("left", CAMERA_LEFT_INDEX, True), daemon=True).start()
    threading.Thread(target=capture_camera, args=("right", CAMERA_RIGHT_INDEX, True), daemon=True).start()
    threading.Thread(target=capture_camera, args=("ball", CAMERA_BALL_INDEX, False), daemon=True).start()

    # Writer thread
    threading.Thread(target=write_frames, daemon=True).start()

    # GUI
    root = tk.Tk()
    app = FreeThrowRecorderApp(root)
    root.mainloop()
