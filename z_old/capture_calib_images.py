#!/usr/bin/env python3

import cv2 as cv
import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

# --- GUI ---
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ========= Config =========
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

CAM_LEFT_INDEX = cfg["left_cam_index"]
CAM_RIGHT_INDEX = cfg["right_cam_index"]
CAM_RESOLUTION = (cfg["original_frame_width"], cfg["original_frame_height"])  # e.g., (1280, 720)
CROP_RESOLUTION = tuple(cfg["crop_size"])  # e.g., (640, 640)
PLAYER_TRACKING_FPS = cfg["player_tracking_fps"]

CHECKERBOARD = tuple(cfg["inner_corners"])  # (columns, rows)
MIN_SQUARE_PX = float(cfg.get("min_square_px", 40.0))

# ========= Paths =========
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
mono_left_dir  = session_dir / "calibration" / "calib_images" / "mono_left"
mono_right_dir = session_dir / "calibration" / "calib_images" / "mono_right"
mono_left_dir.mkdir(parents=True, exist_ok=True)
mono_right_dir.mkdir(parents=True, exist_ok=True)

# ========= Helpers =========
def crop_center_640(frame):
    # center crop to 640x640 from 1280x720 
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

def bgr_to_tk(bgr):
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)

def hstack_same_height(imgL, imgR):
    if imgL is None and imgR is None:
        return None
    if imgL is None:
        return imgR
    if imgR is None:
        return imgL
    h = min(imgL.shape[0], imgR.shape[0])
    def rsz(img):
        scale = h / img.shape[0]
        return cv.resize(img, (int(img.shape[1] * scale), h))
    Lr = rsz(imgL); Rr = rsz(imgR)
    return np.hstack([Lr, Rr])

# ========= Camera Thread =========
class CameraThread(threading.Thread):
    def __init__(self, index, name):
        super().__init__(daemon=True)
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

# ========= Main App =========
MODES = ("LEFT", "RIGHT", "COMBINED")

@dataclass
class ProgBars:
    X: float = 0.0
    Y: float = 0.0
    SIZE: float = 0.0
    SKEW: float = 0.0

@dataclass
class CategoryState:
    count: int = 0
    bars: ProgBars = field(default_factory=ProgBars)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration Capture (Tk)")
        self.mode_idx = 0  # 0: LEFT, 1: RIGHT, 2: COMBINED

        # cameras
        self.left_cam = CameraThread(CAM_LEFT_INDEX, "Left")
        self.right_cam = CameraThread(CAM_RIGHT_INDEX, "Right")
        self.left_cam.start()
        self.right_cam.start()

        # next ids
        self.next_left_id  = next_id(mono_left_dir,  "left")
        self.next_right_id = next_id(mono_right_dir, "right")

        # per-category state
        self.states: Dict[str, CategoryState] = {
            "LEFT": CategoryState(count=len(list(mono_left_dir.glob("left_*.png")))),
            "RIGHT": CategoryState(count=len(list(mono_right_dir.glob("right_*.png")))),
            "COMBINED": CategoryState(count=0)  # logical combined count if both saved; not incremented in this stub
        }

        # ---------- Layout ----------
        # Top title & instructions
        top = ttk.Frame(root, padding=(8,8,8,4))
        top.pack(side=tk.TOP, fill=tk.X)
        self.title_lbl = ttk.Label(top, text=f"Calibration Capture — Athlete: {ATHLETE} | Session: {SESSION}", font=("Helvetica", 14, "bold"))
        self.title_lbl.pack(side=tk.TOP, anchor="w")
        self.subtitle_lbl = ttk.Label(top, text=f"Min square px = {MIN_SQUARE_PX:.0f} | TAB: switch mode  SPACE: capture  ESC: quit", font=("Helvetica", 10))
        self.subtitle_lbl.pack(side=tk.TOP, anchor="w", pady=(2,0))

        # Mode + filenames + counts
        info = ttk.Frame(root, padding=(8,0,8,6))
        info.pack(side=tk.TOP, fill=tk.X)
        self.mode_var = tk.StringVar(value=f"Mode: {MODES[self.mode_idx]}")
        self.next_var = tk.StringVar(value=self._next_names())
        self.counts_var = tk.StringVar(value=self._counts_text())
        ttk.Label(info, textvariable=self.mode_var,  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(0,12))
        ttk.Label(info, textvariable=self.next_var,  font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0,12))
        ttk.Label(info, textvariable=self.counts_var, font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0,12))

        # Progress bars (Notebook with tabs per category)
        nb = ttk.Notebook(root)
        nb.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,6))
        self.pb_vars: Dict[str, Dict[str, tk.DoubleVar]] = {}
        for cat in MODES:
            frame = ttk.Frame(nb, padding=8)
            nb.add(frame, text=cat)
            vars_cat = {}
            for label in ("X", "Y", "SIZE", "SKEW"):
                row = ttk.Frame(frame)
                row.pack(side=tk.TOP, fill=tk.X, pady=2)
                ttk.Label(row, text=f"{label:>4}", width=6).pack(side=tk.LEFT)
                v = tk.DoubleVar(value=0.0)
                pb = ttk.Progressbar(row, variable=v, maximum=1.0)
                pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
                vars_cat[label] = v
            self.pb_vars[cat] = vars_cat

        # Video area
        video_frame = ttk.Frame(root, padding=(8,0,8,8))
        video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.video_lbl = ttk.Label(video_frame)
        self.video_lbl.pack(side=tk.TOP, anchor="center")

        # Status line
        status = ttk.Frame(root, padding=(8,0,8,8))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

        # key bindings
        root.bind("<Escape>", lambda e: self.on_quit())
        root.bind("<space>",  lambda e: self.on_capture())
        root.bind("<Tab>",    lambda e: self.on_switch_mode())

        # periodic update
        self._update_video()

    # ---------- UI helpers ----------
    def _next_names(self):
        m = MODES[self.mode_idx]
        if m == "LEFT":
            return f"Next: left_{self.next_left_id:02}.png"
        elif m == "RIGHT":
            return f"Next: right_{self.next_right_id:02}.png"
        else:
            return f"Next: left_{self.next_left_id:02}.png | right_{self.next_right_id:02}.png"

    def _counts_text(self):
        return f"Counts — LEFT: {self.states['LEFT'].count}  RIGHT: {self.states['RIGHT'].count}"

    # ---------- Video update ----------
    def _update_video(self):
        m = MODES[self.mode_idx]
        frame = None
        L = self.left_cam.frame
        R = self.right_cam.frame

        if m == "LEFT":
            frame = crop_center_640(L) if L is not None else None
        elif m == "RIGHT":
            frame = crop_center_640(R) if R is not None else None
        else:  # COMBINED
            LV = crop_center_640(L) if L is not None else None
            RV = crop_center_640(R) if R is not None else None
            frame = hstack_same_height(LV, RV) if LV is not None or RV is not None else None

        if frame is not None:
            image_tk = bgr_to_tk(frame)
            self.video_lbl.configure(image=image_tk)
            self.video_lbl.image = image_tk  # keep ref
        else:
            self.video_lbl.configure(text="Waiting for camera frames...", image="")
            self.video_lbl.image = None

        # schedule next update
        self.root.after(20, self._update_video)  # ~50 FPS max

    # ---------- Actions ----------
    def on_switch_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(MODES)
        self.mode_var.set(f"Mode: {MODES[self.mode_idx]}")
        self.next_var.set(self._next_names())
        self.status_var.set(f"Switched to {MODES[self.mode_idx]}")

    def _detect_and_gate(self, frame, side_name: str):
        if frame is None:
            self.status_var.set(f"{side_name}: No frame")
            return False, None
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flags = cv.CALIB_CB_EXHAUSTIVE
        ret, pts = cv.findChessboardCornersSB(gray, CHECKERBOARD, flags=flags)
        if not ret:
            self.status_var.set(f"{side_name}: Checkerboard NOT detected")
            return False, None
        # refine
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)
        cols, rows = CHECKERBOARD
        sq = quick_square_px(pts, cols, rows)
        if sq < MIN_SQUARE_PX:
            self.status_var.set(f"{side_name}: Board too small ({sq:.1f}px). Move closer.")
            return False, None
        return True, sq

    def on_capture(self):
        m = MODES[self.mode_idx]
        Lf = crop_center_640(self.left_cam.frame)  if self.left_cam.frame  is not None else None
        Rf = crop_center_640(self.right_cam.frame) if self.right_cam.frame is not None else None

        if m == "LEFT":
            ok, sq = self._detect_and_gate(Lf, "LEFT")
            if ok:
                fname = mono_left_dir / f"left_{self.next_left_id:02}.png"
                if cv.imwrite(str(fname), Lf):
                    self.states["LEFT"].count += 1
                    self.next_left_id += 1
                    self.status_var.set(f"Saved {fname.name} (square~{sq:.1f}px)")
                self._update_counts_and_bars("LEFT")

        elif m == "RIGHT":
            ok, sq = self._detect_and_gate(Rf, "RIGHT")
            if ok:
                fname = mono_right_dir / f"right_{self.next_right_id:02}.png"
                if cv.imwrite(str(fname), Rf):
                    self.states["RIGHT"].count += 1
                    self.next_right_id += 1
                    self.status_var.set(f"Saved {fname.name} (square~{sq:.1f}px)")
                self._update_counts_and_bars("RIGHT")

        else:  # COMBINED
            saved_any = False
            if Lf is not None:
                okL, sqL = self._detect_and_gate(Lf, "LEFT")
                if okL:
                    fnameL = mono_left_dir / f"left_{self.next_left_id:02}.png"
                    if cv.imwrite(str(fnameL), Lf):
                        self.states["LEFT"].count += 1
                        self.next_left_id += 1
                        saved_any = True
            if Rf is not None:
                okR, sqR = self._detect_and_gate(Rf, "RIGHT")
                if okR:
                    fnameR = mono_right_dir / f"right_{self.next_right_id:02}.png"
                    if cv.imwrite(str(fnameR), Rf):
                        self.states["RIGHT"].count += 1
                        self.next_right_id += 1
                        saved_any = True
            if saved_any:
                self.status_var.set("Saved combined (left/right) frames")
                self._update_counts_and_bars("LEFT")
                self._update_counts_and_bars("RIGHT")
            else:
                self.status_var.set("Combined capture failed checks")

        # update header fields
        self.next_var.set(self._next_names())
        self.counts_var.set(self._counts_text())

    def _update_counts_and_bars(self, cat: str):
        # TODO: Plug in diversity metrics here to update progress bars.
        # For now we just leave values as-is (placeholders). Example:
        # self.pb_vars[cat]["X"].set(new_x_value_0_to_1)
        pass

    def on_quit(self):
        self.left_cam.stop(); self.right_cam.stop()
        self.root.quit()

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_quit)
    root.mainloop()

if __name__ == "__main__":
    main()
