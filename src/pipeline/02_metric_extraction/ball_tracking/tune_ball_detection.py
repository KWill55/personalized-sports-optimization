#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background-subtraction BallDetector tuner
- Open a folder of videos
- Scrub frames, visualize detection + short trajectory
- Tune: varThreshold, learningRate (-1=auto), morph (open/close, kernel),
        pre-blur, area/circularity/fill gates, MOG2 vs KNN, shadows on/off
- Overlay either raw or cleaned foreground mask

Requires: opencv-python, pyyaml, tkinter (std), numpy
"""

from __future__ import annotations
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import yaml
import math

# =========================================================
# Config: Load from project/session YAMLs (same layout you use)
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Adjust parents[..] if your repo depth differs
BASE_DIR = SCRIPT_DIR.parents[2]

PROJECT_PATH = BASE_DIR / "project_config.yaml"
SESSION_PATH = BASE_DIR / "session_config.yaml"

with open(PROJECT_PATH, "r") as f:
    CONFIG1 = yaml.safe_load(f)
with open(SESSION_PATH, "r") as f:
    CONFIG2 = yaml.safe_load(f)

ATHLETE = str(CONFIG1["athlete"])
SESSION = str(CONFIG1["session"])
FRAME_WIDTH  = int(CONFIG1["original_frame_width"])
FRAME_HEIGHT = int(CONFIG1["original_frame_height"])

SESSION_INFO = CONFIG2["athletes"][ATHLETE][SESSION]
UPPER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["upper"]))   # ((x1,y1),(x2,y2))
LOWER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["lower"]))

# Reasonable shape defaults (won't crash if YAML lacks them)
AREA_MIN_DEFAULT  = float(SESSION_INFO.get("ball_area_px", {}).get("min", 30))
AREA_MAX_DEFAULT  = float(SESSION_INFO.get("ball_area_px", {}).get("max", 2000))
CIRC_MIN_DEFAULT  = float(SESSION_INFO.get("circularity_min", 0.55))
FILL_MIN_DEFAULT  = float(SESSION_INFO.get("fill_ratio_min", 0.55))

SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
DEFAULT_VIDEO_DIR = SESSION_DIR / "videos" / "ball_tracking" / "raw"
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

# =========================================================
# BallDetector (BG-sub + mask cleaning + geometry + Kalman)
# =========================================================
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class Candidate:
    cnt: np.ndarray
    center: Tuple[float,float]
    radius: float
    area: float
    perimeter: float
    circularity: float  # 4πA/P^2
    fill: float         # A/(πr^2)

class BallDetector:
    def __init__(
        self,
        use_mog2: bool = True,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,      # -1 -> OpenCV auto
        use_color: bool = False,          # False: gray
        blur_ksize: int = 5,
        blur_sigma: float = 0.0,
        open_iters: int = 1,
        close_iters: int = 1,
        morph_kernel: Tuple[int,int] = (5,5),
        area_min: float = 50.0,
        area_max: float = 5000.0,
        circ_min: float = 0.6,
        fill_min: float = 0.6,
        max_jump: float = 50.0,
        ema: float = 0.3,
        score_area_log: bool = True,
        proximity_gain: float = 1.0,
        use_kalman: bool = True,
        dt: float = 1/30.0,
        process_var_pos: float = 1.0,
        process_var_vel: float = 50.0,
        meas_var_pos: float = 10.0,
    ):
        self.use_color = use_color
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.open_iters = int(open_iters)
        self.close_iters = int(close_iters)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

        self.area_min = float(area_min)
        self.area_max = float(area_max)
        self.circ_min = float(circ_min)
        self.fill_min = float(fill_min)
        self.max_jump = float(max_jump)

        self.ema = float(ema)
        self.score_area_log = bool(score_area_log)
        self.proximity_gain = float(proximity_gain)

        self.prev_center: Optional[Tuple[float,float]] = None
        self.smooth: Optional[Tuple[float,float,float]] = None  # (x,y,r)

        # BG subtractor
        if use_mog2:
            self.bg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                         varThreshold=var_threshold,
                                                         detectShadows=detect_shadows)
        else:
            self.bg = cv2.createBackgroundSubtractorKNN(history=history,
                                                        dist2Threshold=var_threshold,
                                                        detectShadows=detect_shadows)
        self.learning_rate = float(learning_rate)

        # KF (constant velocity)
        self.use_kalman = bool(use_kalman)
        if self.use_kalman:
            self.kf = cv2.KalmanFilter(4, 2, 0)
            self.kf.transitionMatrix = np.array([
                [1,0,dt,0],
                [0,1,0,dt],
                [0,0,1, 0],
                [0,0,0, 1]], dtype=np.float32)
            self.kf.measurementMatrix = np.array([
                [1,0,0,0],
                [0,1,0,0]], dtype=np.float32)
            self.kf.processNoiseCov = np.diag([process_var_pos, process_var_pos,
                                               process_var_vel, process_var_vel]).astype(np.float32)
            self.kf.measurementNoiseCov = np.array([[meas_var_pos, 0],
                                                    [0, meas_var_pos]], dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e3
            self.kf.statePost = np.zeros((4,1), dtype=np.float32)

    # ---------- pipeline steps ----------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = frame if self.use_color else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize % 2 == 1 and self.blur_ksize > 1:
            img = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), self.blur_sigma)
        return img

    def foreground_mask(self, img: np.ndarray) -> np.ndarray:
        mask = self.bg.apply(img, learningRate=self.learning_rate)
        # Shadows come as 127; binarize to 0/255
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)
        return mask

    def find_candidates(self, mask: np.ndarray) -> List[Candidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Candidate] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if not (self.area_min < area < self.area_max):
                continue
            per = float(cv2.arcLength(cnt, True))
            if per <= 0:
                continue
            circ = 4.0 * math.pi * area / (per * per)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r <= 0:
                continue
            fill = area / (math.pi * r * r)
            out.append(Candidate(cnt, (float(x), float(y)), float(r), area, per, circ, fill))
        return out

    def score_candidate(self, c: Candidate) -> float:
        if c.circularity < self.circ_min or c.fill < self.fill_min:
            return -1.0
        geom = c.circularity * c.fill
        geom *= (math.log1p(c.area) if self.score_area_log else max(c.area, 1.0))
        if self.prev_center is None:
            prox = 1.0
        else:
            jump = math.hypot(c.center[0]-self.prev_center[0], c.center[1]-self.prev_center[1])
            prox = 1.0 / (1.0 + self.proximity_gain * jump)
        return geom * prox

    def select_best(self, cands: List[Candidate]) -> Optional[Candidate]:
        best = None
        best_score = -1.0
        for c in cands:
            if self.prev_center is not None:
                jump = math.hypot(c.center[0]-self.prev_center[0], c.center[1]-self.prev_center[1])
                if jump > self.max_jump:
                    continue
            s = self.score_candidate(c)
            if s > best_score:
                best_score, best = s, c
        return best

    def kalman_predict(self) -> Optional[Tuple[float,float]]:
        if not self.use_kalman:
            return None
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def kalman_update(self, meas_xy: Optional[Tuple[float,float]]) -> Tuple[float,float]:
        if not self.use_kalman:
            return meas_xy if meas_xy is not None else (0.0, 0.0)
        if meas_xy is None:
            post = self.kf.statePost
            return float(post[0]), float(post[1])
        z = np.array([[meas_xy[0]], [meas_xy[1]]], dtype=np.float32)
        est = self.kf.correct(z)
        return float(est[0]), float(est[1])

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int,int]], Optional[int], Dict[str,Any]]:
        debug: Dict[str,Any] = {}
        img = self.preprocess(frame)
        mask_raw = self.foreground_mask(img)
        mask_clean = self.clean_mask(mask_raw)
        debug["mask_raw"] = mask_raw
        debug["mask_clean"] = mask_clean

        cands = self.find_candidates(mask_clean)
        debug["candidates"] = [
            dict(center=c.center, radius=c.radius, area=c.area, circ=c.circularity, fill=c.fill) for c in cands
        ]

        best = self.select_best(cands)

        meas_xy = None
        meas_r = None
        if best is not None:
            x, y, r = best.center[0], best.center[1], best.radius
            if self.smooth is None:
                self.smooth = (x, y, r)
            else:
                sx, sy, sr = self.smooth
                self.smooth = (sx + self.ema*(x-sx), sy + self.ema*(y-sy), sr + self.ema*(r-sr))
            meas_xy = (self.smooth[0], self.smooth[1])
            meas_r = self.smooth[2]

        # init KF on first hit
        if self.use_kalman and self.prev_center is None and meas_xy is not None:
            self.kf.statePost = np.array([[meas_xy[0]],[meas_xy[1]],[0.0],[0.0]], dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32)*10.0

        if self.use_kalman:
            _ = self.kalman_predict()
            kx, ky = self.kalman_update(meas_xy)
            center_out = (int(round(kx)), int(round(ky)))
            radius_out = int(round(meas_r)) if meas_r is not None else None
        else:
            center_out = (int(round(meas_xy[0])), int(round(meas_xy[1]))) if meas_xy else None
            radius_out = int(round(meas_r)) if meas_r is not None else None

        if center_out is not None:
            self.prev_center = (float(center_out[0]), float(center_out[1]))

        debug["chosen"] = dict(center=center_out, radius=radius_out) if center_out else None
        return center_out, radius_out, debug

# =========================================================
# GUI / Tuner
# =========================================================
# Global video state
video_files: list[Path] = []
current_vid_idx = -1
cap = None
frame_count = 1
fps = 30.0
cur_idx = [0]
frame_resized = None

# Detector instance
detector: BallDetector | None = None

# Short trajectory
traj_points: list[tuple[int,int] | None] = []

# ---------------- Tk App ----------------
root = tk.Tk()
root.title("Ball Tracking Tuner (Background Subtraction)")

style = ttk.Style(root)
try:
    style.theme_use("clam")
except tk.TclError:
    pass

# --- Tk variables / controls ---
# Background subtractor
DETECT_ALGO      = tk.StringVar(value="MOG2")  # "MOG2" | "KNN"
DETECT_SHADOWS   = tk.BooleanVar(value=True)
VAR_THRESH       = tk.DoubleVar(value=12.0)    # sensitivity
LEARN_RATE       = tk.DoubleVar(value=0.001)   # -1 for auto

# Morphology + blur
OPEN_ITERS       = tk.IntVar(value=1)
CLOSE_ITERS      = tk.IntVar(value=1)
MORPH_KSIZE      = tk.IntVar(value=7)          # odd
PREBLUR_KSIZE    = tk.IntVar(value=5)          # odd, 1=off

# Geometry gates
AREA_MIN         = tk.IntVar(value=int(AREA_MIN_DEFAULT))
AREA_MAX         = tk.IntVar(value=int(AREA_MAX_DEFAULT))
CIRC_MIN         = tk.DoubleVar(value=float(CIRC_MIN_DEFAULT))
FILL_MIN         = tk.DoubleVar(value=float(FILL_MIN_DEFAULT))

# Toggles
SHOW_MASK        = tk.BooleanVar(value=True)
MASK_MODE        = tk.StringVar(value="CLEAN")   # "RAW" | "CLEAN"
SHOW_TRAJ        = tk.BooleanVar(value=True)

# Active slider machinery (keyboard nudging)
active_name = [None]
label_vars: dict[str, tk.StringVar] = {}
slider_vars: dict[str, tk.Variable] = {}
slider_ranges: dict[str, tuple[float, float, bool]] = {}

metrics_var = tk.StringVar(value="area: —  circ: —  fill: —  r: —")
clip_label_var = tk.StringVar(value="No folder loaded")
frame_label_var = tk.StringVar(value=f"0 / {frame_count-1}")

def odd(x: int) -> int:
    x = int(x)
    return x if x % 2 == 1 else x + 1

def list_videos_in(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
                  key=lambda x: x.name.lower())

def close_cap():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def rebuild_detector():
    """Create/update detector from current UI values."""
    global detector
    use_mog2 = (DETECT_ALGO.get() == "MOG2")
    k_morph  = odd(MORPH_KSIZE.get())
    k_blur   = odd(PREBLUR_KSIZE.get())

    # Always recreate on algo/shadow change to reset BG model
    detector = BallDetector(
        use_mog2=use_mog2,
        var_threshold=float(VAR_THRESH.get()),
        detect_shadows=bool(DETECT_SHADOWS.get()),
        learning_rate=float(LEARN_RATE.get()),
        history=300,
        use_color=False,
        blur_ksize=k_blur,
        open_iters=int(OPEN_ITERS.get()),
        close_iters=int(CLOSE_ITERS.get()),
        morph_kernel=(k_morph, k_morph),
        area_min=float(AREA_MIN.get()),
        area_max=float(AREA_MAX.get()),
        circ_min=float(CIRC_MIN.get()),
        fill_min=float(FILL_MIN.get()),
        use_kalman=True,
        dt=(1.0 / max(fps, 1.0)),
    )

def warmup_bg_model(n=15):
    """Feed a few frames so BG model stabilizes a bit."""
    if detector is None or cap is None:
        return
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for _ in range(max(0, n)):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT))
        _ = detector.foreground_mask(detector.preprocess(f))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

def open_video(idx: int):
    """Open nth video; reset state; warmup detector."""
    global cap, frame_count, fps, cur_idx, frame_resized, traj_points, current_vid_idx
    if not video_files:
        return False
    idx = max(0, min(idx, len(video_files)-1))
    current_vid_idx = idx

    close_cap()
    path = video_files[idx]
    cap_local = cv2.VideoCapture(str(path))
    if not cap_local.isOpened():
        messagebox.showwarning("OpenCV", f"Could not open: {path.name}")
        return False
    cap = cap_local

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    fps_local = cap.get(cv2.CAP_PROP_FPS)
    if not fps_local or fps_local <= 0:
        raise ValueError(f"Invalid FPS ({fps_local}) reported for video: {path.name}")
    fps = float(fps_local)
    cur_idx[0] = 0
    frame_resized = None
    traj_points = []

    # Detector build + warmup
    rebuild_detector()
    warmup_bg_model(n=15)

    # Update UI labels/seek
    seek_slider.configure(to=max(0, frame_count-1))
    clip_label_var.set(f"Clip {current_vid_idx+1}/{len(video_files)} — {path.name}")

    # load first frame
    ok = load_frame(0)
    if ok:
        update_image()
    return ok

def load_frame(idx: int) -> bool:
    """Seek to frame idx; set frame_resized."""
    global frame_resized
    if cap is None:
        return False
    idx = max(0, min(idx, frame_count-1))
    cur_idx[0] = idx
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        # rare codecs: brute from 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(idx+1):
            ret, frame = cap.read()
            if not ret:
                break
    if not ret:
        print(f"❌ Could not load frame {idx} from: {video_files[current_vid_idx].name}")
        return False
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    return True

def run_detector(frame_bgr):
    """Thin wrapper to fetch center, raw/clean masks, and metrics of chosen cand."""
    if detector is None:
        return None, None, None, None
    center, radius, dbg = detector.detect(frame_bgr)
    raw_mask   = dbg.get("mask_raw")   if isinstance(dbg, dict) else None
    clean_mask = dbg.get("mask_clean") if isinstance(dbg, dict) else None

    metrics = None
    if center is not None and isinstance(dbg, dict):
        cx, cy = center
        for c in dbg.get("candidates", []):
            cc = c["center"]
            if int(round(cc[0])) == cx and int(round(cc[1])) == cy:
                metrics = {
                    "area": c["area"],
                    "circ": c["circ"],
                    "fill": c["fill"],
                    "radius": float(radius) if radius is not None else float(c.get("radius", 0.0))
                }
                break
    return center, raw_mask, clean_mask, metrics

def update_image():
    if frame_resized is None:
        return
    center, raw_mask, clean_mask, metrics = run_detector(frame_resized)

    annotated = frame_resized.copy()

    # Mask overlay
    if SHOW_MASK.get():
        use = raw_mask if MASK_MODE.get() == "RAW" else clean_mask
        if use is not None:
            overlay = np.zeros_like(annotated); overlay[:] = (0,255,255)
            annotated = np.where(use[..., None] > 0,
                                 cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0),
                                 annotated)

    # Hoop boxes
    cv2.rectangle(annotated, UPPER_HOOP_REGION[0], UPPER_HOOP_REGION[1], (255,0,0), 2)
    cv2.rectangle(annotated, LOWER_HOOP_REGION[0], LOWER_HOOP_REGION[1], (0,0,255), 2)

    # Detection + short trajectory
    if center is not None:
        cv2.circle(annotated, center, 6, (0,255,0), -1)
        if SHOW_TRAJ.get():
            traj_points.append(center)
            if len(traj_points) > 200:
                del traj_points[:len(traj_points)-200]
    else:
        traj_points.append(None)

    if SHOW_TRAJ.get() and len(traj_points) > 1:
        for pt in traj_points:
            if pt is not None:
                cv2.circle(annotated, pt, 2, (0,200,255), -1)

    # HUD
    t = cur_idx[0] / max(fps, 1.0)
    cv2.putText(annotated, f"Frame {cur_idx[0]} / {frame_count-1}  ({t:.2f}s)",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Ball Tracking Preview", annotated)
    cv2.waitKey(1)

    if metrics:
        metrics_var.set(f"area: {metrics['area']:.1f}   circ: {metrics['circ']:.3f}   "
                        f"fill: {metrics['fill']:.3f}   r: {metrics['radius']:.1f}")
    else:
        metrics_var.set("area: —   circ: —   fill: —   r: —")

# ----------------- UI plumbing -----------------
def build_slider(parent, name, var, from_, to_, is_float=False):
    slider_vars[name] = var
    slider_ranges[name] = (from_, to_, is_float)

    row = ttk.Frame(parent); row.pack(fill="x", pady=2)
    label_var = tk.StringVar(); label_vars[name] = label_var
    lab = ttk.Label(row, textvariable=label_var, width=22); lab.pack(side="left")

    def fmt():
        v = var.get()
        label_var.set(f"{name}: {float(v):.3f}" if is_float else f"{name}: {int(v)}")
    fmt()

    def on_move(_=None):
        fmt()
        # knobs that require detector refresh
        if name in {"varThreshold", "learningRate",
                    "Open iters", "Close iters", "Morph ksize",
                    "Pre-blur ksize",
                    "Area Min (px)", "Area Max (px)", "Circularity Min", "Fill Ratio Min"}:
            rebuild_detector()
        update_image()

    scale = ttk.Scale(row, from_=from_, to=to_, orient="horizontal",
                      variable=var, command=on_move)
    scale.pack(side="left", expand=True, fill="x", padx=6)

    def set_active(_evt=None, n=name): active_name[0] = n
    lab.bind("<Button-1>", set_active)

def on_key_adjust(event):
    name = active_name[0]
    if not name: return
    var = slider_vars[name]; lo, hi, is_float = slider_ranges[name]
    if is_float:
        step = 0.01
        if event.keysym == "Left":
            var.set(max(float(lo), float(var.get()) - step))
        elif event.keysym == "Right":
            var.set(min(float(hi), float(var.get()) + step))
    else:
        if event.keysym == "Left":
            var.set(max(int(lo), int(var.get()) - 1))
        elif event.keysym == "Right":
            var.set(min(int(hi), int(var.get()) + 1))
    # reflect & apply
    if name in label_vars:
        v = var.get()
        label_vars[name].set(f"{name}: {float(v):.3f}" if is_float else f"{name}: {int(v)}")
    if name in {"varThreshold", "learningRate",
                "Open iters", "Close iters", "Morph ksize",
                "Pre-blur ksize",
                "Area Min (px)", "Area Max (px)", "Circularity Min", "Fill Ratio Min"}:
        rebuild_detector()
    update_image()

def on_seek(val):
    idx = int(float(val))
    if load_frame(idx):
        update_image()

def step_frames(delta: int):
    new_idx = max(0, min(cur_idx[0] + delta, frame_count - 1))
    if load_frame(new_idx):
        seek_slider.set(new_idx)
        update_image()

def on_video_keys(event):
    if event.keysym in ("comma",):       step_frames(-1)
    elif event.keysym in ("period",):    step_frames(+1)
    elif event.keysym in ("j","J"):      step_frames(-int(fps) if fps>=1 else -10)
    elif event.keysym in ("k","K"):      step_frames(+int(fps) if fps>=1 else +10)
    elif event.keysym in ("n","N"):      next_clip()
    elif event.keysym in ("p","P"):      prev_clip()

def next_clip():
    if not video_files: return
    idx = (current_vid_idx + 1) % len(video_files)
    if open_video(idx):
        seek_slider.set(0); update_image()

def prev_clip():
    if not video_files: return
    idx = (current_vid_idx - 1) % len(video_files)
    if open_video(idx):
        seek_slider.set(0); update_image()

def load_folder():
    initial = str(DEFAULT_VIDEO_DIR if DEFAULT_VIDEO_DIR.exists() else Path.home())
    chosen = filedialog.askdirectory(initialdir=initial, title="Select Folder with Videos")
    if not chosen: return
    folder = Path(chosen)
    files = list_videos_in(folder)
    if not files:
        messagebox.showerror("No videos", "No .mp4/.mov/.avi files found in that folder.")
        return
    global video_files
    video_files = files
    if open_video(0):
        seek_slider.set(0); update_image()

# ---------------- Build GUI ----------------
top_row = ttk.Frame(root); top_row.pack(fill="x", pady=6, padx=6)
ttk.Button(top_row, text="Load Folder", command=load_folder).pack(side="left")
ttk.Button(top_row, text="Prev Clip (P)", command=prev_clip).pack(side="left", padx=(8,4))
ttk.Button(top_row, text="Next Clip (N)", command=next_clip).pack(side="left", padx=4)
ttk.Label(top_row, textvariable=clip_label_var).pack(side="left", padx=12)

seek_frame = ttk.Frame(root); seek_frame.pack(fill="x", pady=6, padx=6)
ttk.Label(seek_frame, text="Frame").pack(side="left")
seek_slider = ttk.Scale(seek_frame, from_=0, to=max(0, frame_count-1),
                        orient="horizontal", command=on_seek)
seek_slider.pack(side="left", expand=True, fill="x", padx=6)
ttk.Label(seek_frame, textvariable=frame_label_var, width=12).pack(side="right")

wrap = ttk.Frame(root); wrap.pack(fill="x", padx=6, pady=6)

# Background subtractor controls
ttk.Label(wrap, text="Background Subtractor", font=("Helvetica", 12, "bold")).pack(anchor="w")
algo_row = ttk.Frame(wrap); algo_row.pack(fill="x", pady=2)
ttk.Radiobutton(algo_row, text="MOG2", value="MOG2", variable=DETECT_ALGO,
                command=rebuild_detector).pack(side="left")
ttk.Radiobutton(algo_row, text="KNN",  value="KNN",  variable=DETECT_ALGO,
                command=rebuild_detector).pack(side="left", padx=6)
ttk.Checkbutton(algo_row, text="Detect shadows", variable=DETECT_SHADOWS,
                command=rebuild_detector).pack(side="left", padx=12)
build_slider(wrap, "varThreshold", VAR_THRESH, 1.0, 64.0, is_float=True)
build_slider(wrap, "learningRate", LEARN_RATE, -1.0, 0.05, is_float=True)

# Morphology / blur
ttk.Label(wrap, text="Mask Cleaning / Blur", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8,2))
build_slider(wrap, "Open iters",  OPEN_ITERS,  0, 6, is_float=False)
build_slider(wrap, "Close iters", CLOSE_ITERS, 0, 6, is_float=False)
build_slider(wrap, "Morph ksize", MORPH_KSIZE, 3, 15, is_float=False)
build_slider(wrap, "Pre-blur ksize", PREBLUR_KSIZE, 1, 21, is_float=False)

# Geometry gates
ttk.Label(wrap, text="Geometry Gates", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8,2))
build_slider(wrap, "Area Min (px)", AREA_MIN, 0, 10000, is_float=False)
build_slider(wrap, "Area Max (px)", AREA_MAX, 1, 20000, is_float=False)
build_slider(wrap, "Circularity Min", CIRC_MIN, 0.0, 1.0, is_float=True)
build_slider(wrap, "Fill Ratio Min", FILL_MIN, 0.0, 1.0, is_float=True)

# Toggles + metrics
tog_row = ttk.Frame(root); tog_row.pack(fill="x", padx=6, pady=6)
ttk.Checkbutton(tog_row, text="Show mask", variable=SHOW_MASK, command=update_image).pack(side="left", padx=6)
ttk.Checkbutton(tog_row, text="Show trajectory", variable=SHOW_TRAJ, command=update_image).pack(side="left", padx=6)
ttk.Radiobutton(tog_row, text="Raw FG mask",  value="RAW",   variable=MASK_MODE, command=update_image).pack(side="left", padx=6)
ttk.Radiobutton(tog_row, text="Cleaned mask", value="CLEAN", variable=MASK_MODE, command=update_image).pack(side="left", padx=6)
ttk.Label(tog_row, textvariable=metrics_var, width=48).pack(side="right")

# Keybinds
root.bind("<Left>",  on_key_adjust)
root.bind("<Right>", on_key_adjust)
root.bind(",", on_video_keys)
root.bind(".", on_video_keys)
root.bind("j", on_video_keys); root.bind("J", on_video_keys)
root.bind("k", on_video_keys); root.bind("K", on_video_keys)
root.bind("n", on_video_keys); root.bind("N", on_video_keys)
root.bind("p", on_video_keys); root.bind("P", on_video_keys)

# Keep seek label synced
def on_tick():
    if frame_count > 0:
        frame_label_var.set(f"{int(seek_slider.get())} / {frame_count-1}")
    root.after(100, on_tick)
root.after(100, on_tick)

# Initialize with default folder if available
if DEFAULT_VIDEO_DIR.exists():
    files = list_videos_in(DEFAULT_VIDEO_DIR)
    if files:
        video_files = files
        open_video(0)
        seek_slider.set(0)

root.protocol("WM_DELETE_WINDOW", lambda: (close_cap(), root.destroy()))
root.mainloop()
close_cap()
cv2.destroyAllWindows()
