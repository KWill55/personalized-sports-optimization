"""
Title: capture_cb_mono.py  (single-camera checkerboard capture)

Purpose:
    Capture checkerboard images from ONE camera at a time for mono intrinsics.
    - Crops center 640x640 from the selected camera (LEFT or RIGHT)
    - Displays a single 640x640 window
    - Saves into separate folders: mono_left/ and mono_right/
    - Requires checkerboard detection AND average square size >= MIN_SQUARE_PX

Controls:
    - TAB       : switch active camera (LEFT <-> RIGHT)
    - SPACE     : capture (if detection + size gate pass)
    - ESC       : quit
"""

import cv2 as cv
import os
from pathlib import Path
import threading
import time
import yaml
import numpy as np

# ========================================
# Config (from project YAML)
# ========================================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Session info
ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# Camera Parameters
CAM_LEFT_INDEX = cfg["left_cam_index"]
CAM_RIGHT_INDEX = cfg["right_cam_index"]
CAM_RESOLUTION = (cfg["original_frame_width"], cfg["original_frame_height"])  # (1280, 720)
CROP_RESOLUTION = tuple(cfg["crop_size"])  # (640, 640)
PLAYER_TRACKING_FPS = cfg["player_tracking_fps"]

# Calibration Parameters
CHECKERBOARD = tuple(cfg["inner_corners"])  # (columns, rows)
MIN_SQUARE_PX = float(cfg.get("min_square_px", 40.0))  # default 40px

# ========================================
# Paths and Directories
# ========================================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

mono_left_dir  = session_dir / "calibration" / "calib_images" / "mono_left"
mono_right_dir = session_dir / "calibration" / "calib_images" / "mono_right"
mono_left_dir.mkdir(parents=True, exist_ok=True)
mono_right_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Camera Thread
# ========================================
class CameraThread(threading.Thread):
    def __init__(self, index, name):
        super().__init__()
        self.index = index
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  CAM_RESOLUTION[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        self.cap.set(cv.CAP_PROP_FPS, PLAYER_TRACKING_FPS)
        self.name = name
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
        self.cap.release()

    def stop(self):
        self.running = False

# ========================================
# Helpers
# ========================================
def crop_center_640(frame):
    # center crop to 640x640 from 1280x720 (same as before)
    return frame[40:680, 320:960]

def quick_square_px(pts, cols, rows):
    """Estimate pixels-per-square using corner spans; conservative min of width/height."""
    pts = pts.reshape(-1, 2)
    tl = pts[0]
    tr = pts[cols - 1]
    bl = pts[(rows - 1) * cols]
    w = np.linalg.norm(tr - tl) / (cols - 1)
    h = np.linalg.norm(bl - tl) / (rows - 1)
    return float(min(w, h))

def next_id(out_dir: Path, prefix: str) -> int:
    existing = sorted(out_dir.glob(f"{prefix}_*.png"))
    return len(existing) + 1

# ========================================
# Mono Capture GUI
# ========================================
class MonoCaptureGUI:
    def __init__(self, left_cam: CameraThread, right_cam: CameraThread):
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.active = "LEFT"  # or "RIGHT"
        self.id_left  = next_id(mono_left_dir,  "left")
        self.id_right = next_id(mono_right_dir, "right")
        self.status_text = ""
        self.status_color = (255, 255, 255)
        self.status_time = 0

    def show_status(self, text, color):
        self.status_text = text
        self.status_color = color
        self.status_time = time.time()

    def capture_one(self, frame, cam_side: str):
        # Detect checkerboard
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flags = cv.CALIB_CB_EXHAUSTIVE
        ret, pts = cv.findChessboardCornersSB(gray, CHECKERBOARD, flags=flags)
        if not ret:
            msg = "Checkerboard NOT detected."
            print(f"[WARNING] {cam_side}: {msg}")
            self.show_status(msg, (0, 0, 255))
            return

        # Refine + size gate
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

        cols, rows = CHECKERBOARD
        sq = quick_square_px(pts, cols, rows)
        if sq < MIN_SQUARE_PX:
            msg = f"Board too small ({sq:.1f}px). Move closer."
            print(f"[WARNING] {cam_side}: {msg}")
            self.show_status(msg, (0, 0, 255))
            return

        # Save
        if cam_side == "LEFT":
            fname = mono_left_dir / f"left_{self.id_left:02}.png"
            self.id_left += 1
        else:
            fname = mono_right_dir / f"right_{self.id_right:02}.png"
            self.id_right += 1

        ok = cv.imwrite(str(fname), frame)
        if ok:
            msg = f"Saved {fname.name} (square ~ {sq:.1f}px)"
            print(f"[INFO] {cam_side}: {msg}")
            self.show_status(msg, (0, 255, 0))
        else:
            msg = "Failed to save image."
            print(f"[ERROR] {cam_side}: {msg}")
            self.show_status(msg, (0, 0, 255))

    def run(self):
        print("[INFO] Controls: TAB = switch camera, SPACE = capture, ESC = quit")
        print("[INFO] Saving to:")
        print(f"       LEFT  -> {mono_left_dir}")
        print(f"       RIGHT -> {mono_right_dir}")

        while True:
            # Pick active frame
            frame_src = self.left_cam if self.active == "LEFT" else self.right_cam
            frame = frame_src.frame
            if frame is None:
                cv.waitKey(1)
                continue

            view = crop_center_640(frame)

            # UI overlay
            header = f"Active: {self.active} | MIN_SQUARE_PX={MIN_SQUARE_PX:.0f} | TAB: switch | SPACE: capture | ESC: quit"
            cv.putText(view, header, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if time.time() - self.status_time < 1.5:
                cv.putText(view, self.status_text, (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, self.status_color, 2)

            cv.imshow("Mono Checkerboard Capture", view)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 9:  # TAB
                self.active = "RIGHT" if self.active == "LEFT" else "LEFT"
                self.show_status(f"Switched to {self.active}", (0, 255, 255))
            elif key == 32:  # SPACE
                # capture from the currently active camera
                self.capture_one(view, self.active)

        print("[INFO] Closing window.")

# ========================================
# Main
# ========================================
def main():
    left_cam = CameraThread(CAM_LEFT_INDEX, "Left")
    right_cam = CameraThread(CAM_RIGHT_INDEX, "Right")
    left_cam.start()
    right_cam.start()

    gui = MonoCaptureGUI(left_cam, right_cam)

    try:
        gui.run()
    finally:
        left_cam.stop();  right_cam.stop()
        left_cam.join();  right_cam.join()
        cv.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
