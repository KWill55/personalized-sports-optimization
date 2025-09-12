from __future__ import annotations
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from pathlib import Path
from PIL import Image, ImageTk
import yaml
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# script configuration
SHOW_TRAJ = True
SHOW_MASK = False

# =========================
# Config loaders (project + session)
# =========================
def load_configs():
    # Expect these YAMLs next to where you run the script (like your detect_makes.py)
    proj_path = Path("project_config.yaml")
    sess_path = Path("session_config.yaml")
    if not proj_path.exists():
        raise FileNotFoundError("project_config.yaml not found")
    if not sess_path.exists():
        raise FileNotFoundError("session_config.yaml not found")

    with open(proj_path, "r") as f:
        config1 = yaml.safe_load(f)
    with open(sess_path, "r") as f:
        config2 = yaml.safe_load(f)

    ATHLETE = str(config1["athlete"])
    SESSION = str(config1["session"])
    FRAME_WIDTH = int(config1["original_frame_width"])
    FRAME_HEIGHT = int(config1["original_frame_height"])
    CROP_SIZE = tuple(config1["crop_size"])

    SESSION_INFO = config2["athletes"][ATHLETE][SESSION]
    UPPER = SESSION_INFO["hoop_regions"]["upper"]          # ((x1,y1),(x2,y2))
    LOWER = SESSION_INFO["hoop_regions"]["lower"]
    HSV_LOWER = np.array(SESSION_INFO["hsv_ranges"]["lower"], dtype=np.uint8)
    HSV_UPPER = np.array(SESSION_INFO["hsv_ranges"]["upper"], dtype=np.uint8)
    AREA_MIN = float(SESSION_INFO["ball_area_px"]["min"])
    AREA_MAX = float(SESSION_INFO["ball_area_px"]["max"])
    CIRC_MIN = float(SESSION_INFO["circularity_min"])
    FILL_MIN = float(SESSION_INFO["fill_ratio_min"])

    # Repo-style paths (like your script)
    try:
        BASE_DIR = Path(__file__).resolve().parents[3]
    except Exception:
        BASE_DIR = Path.cwd()

    SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
    INPUT_FOLDER = SESSION_DIR / "videos" / "ball_tracking" / "raw"
    OUTPUT_PATH = SESSION_DIR / "analysis" / "outcomes.csv"

    return {
        "ATHLETE": ATHLETE,
        "SESSION": SESSION,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT,
        "CROP_SIZE": CROP_SIZE,
        "UPPER_HOOP_REGION": tuple(map(tuple, UPPER)),
        "LOWER_HOOP_REGION": tuple(map(tuple, LOWER)),
        "HSV_LOWER": HSV_LOWER,
        "HSV_UPPER": HSV_UPPER,
        "AREA_MIN": AREA_MIN,
        "AREA_MAX": AREA_MAX,
        "CIRC_MIN": CIRC_MIN,
        "FILL_MIN": FILL_MIN,
        "INPUT_FOLDER": INPUT_FOLDER,
        "OUTPUT_PATH": OUTPUT_PATH,
        "BASE_DIR": BASE_DIR,
        "SESSION_DIR": SESSION_DIR,
    }


# =========================
# Geometry helpers
# =========================
def is_inside_rect(pt, rect_tl, rect_br):
    if pt is None:
        return False
    x, y = pt
    x1, y1 = rect_tl
    x2, y2 = rect_br
    return min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2)

# TODO eventually make this more robust 
def is_make(trajectory, upper_box, lower_box):
    in_upper = False
    waiting_for_next = False
    for i in range(1, len(trajectory)):
        prev, curr = trajectory[i-1], trajectory[i]
        if curr is None:
            continue
        # Step 1: Entered upper box from above (descending)
        if not in_upper and is_inside_rect(curr, *upper_box):
            if prev and curr[1] > prev[1]:
                in_upper = True
                waiting_for_next = True
            continue
        # Step 2: after upper entry, next visible point check
        if in_upper and waiting_for_next:
            if is_inside_rect(curr, *lower_box):
                return True
            elif not is_inside_rect(curr, *upper_box):
                return False
    return False

@dataclass
class Candidate:
    cnt: np.ndarray # contour points
    center: Tuple[float,float]
    radius: float
    area: float
    perimeter: float
    circularity: float # 4πA / P^2
    fill: float # A / (πr^2)

# =========================
# Detection (with “prev_center” smoothing)
# =========================
class BallDetector:
    def __init__(
        self,
        # Background subtractor options
        use_mog2: bool = True,
        history: int = 500,               # frames remembered by BG model
        var_threshold: float = 16.0,      # MOG2 sensitivity to change
        detect_shadows: bool = True,
        learning_rate: float = -1.0,      # -1 lets OpenCV auto-tune per frame
        
        # Preprocess
        use_color: bool = False,          # False: grayscale; True: keep color to help with shadows
        blur_ksize: int = 5,              # Gaussian blur kernel (odd)
        blur_sigma: float = 0.0,

        # Mask cleaning
        open_iters: int = 1,
        close_iters: int = 1,
        morph_kernel: Tuple[int,int] = (5,5),

        # Geometry gates
        area_min: float = 50.0,
        area_max: float = 5_000.0,
        circ_min: float = 0.6,            # 1.0 is perfect circle; allow blur/occlusion
        fill_min: float = 0.6,
        max_jump: float = 50.0,           # reject blobs that teleport far (pixels)

        # Scoring weights
        ema: float = 0.3,                 # smoothing for visual center/radius
        score_area_log: bool = True,      # use log(1+area) in score
        proximity_gain: float = 1.0,      # weight of proximity term 1/(1+jump)

        # Kalman filter (dt ~ 1/frame_rate)
        use_kalman: bool = True,
        dt: float = 1/30.0,               # assume 30 FPS; tune if known
        process_var_pos: float = 1.0,     # process noise for position
        process_var_vel: float = 50.0,    # process noise for velocity
        meas_var_pos: float = 10.0,       # measurement noise (pixels^2)
    ) -> None:
        """Initialize detector parameters, background model, and Kalman filter."""
        self.use_color = use_color
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma

        # Background subtractor: MOG2 (more robust to shadows) or KNN (lighter)
        if use_mog2:
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )
        else:
            self.bg = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=var_threshold,
                detectShadows=detect_shadows,
            )
        self.learning_rate = learning_rate

        # Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        self.open_iters = open_iters
        self.close_iters = close_iters

        # Geometry gates
        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.fill_min = fill_min
        self.max_jump = max_jump

        # Scoring
        self.ema = ema
        self.score_area_log = score_area_log
        self.proximity_gain = proximity_gain

        # Track state
        self.prev_center: Optional[Tuple[float,float]] = None
        self.smooth: Optional[Tuple[float,float,float]] = None  # (x,y,r)

        # Kalman filter setup (constant-velocity model)
        self.use_kalman = use_kalman
        if use_kalman:
            self.kf = cv2.KalmanFilter(4, 2, 0)  # state: [x,y,vx,vy], measurement: [x,y]
            # State transition (F)
            self.kf.transitionMatrix = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1],
            ], dtype=np.float32)
            # Measurement model (H) maps [x,y,vx,vy] -> [x,y]
            self.kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ], dtype=np.float32)
            # Process noise (Q)
            q = np.array([
                [self._q_from(process_var_pos), 0, 0, 0],
                [0, self._q_from(process_var_pos), 0, 0],
                [0, 0, self._q_from(process_var_vel), 0],
                [0, 0, 0, self._q_from(process_var_vel)],
            ], dtype=np.float32)
            self.kf.processNoiseCov = q
            # Measurement noise (R)
            self.kf.measurementNoiseCov = np.array([
                [meas_var_pos, 0],
                [0, meas_var_pos],
            ], dtype=np.float32)
            # Error covariance (P) initial
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e3
            # Initial state (will be set on first detection)
            self.kf.statePost = np.zeros((4,1), dtype=np.float32)

    @staticmethod
    def _q_from(v: float) -> float:
        """Helper to ensure non-negative process variance."""
        return max(float(v), 1e-6)

    # ---------------- Stage 1: Preprocess -----------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert to chosen color space and blur to reduce pixel noise.
        Returns an image ready for background subtraction (either gray or color)."""
        if self.use_color:
            img = frame.copy()
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize % 2 == 1:
            img = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), self.blur_sigma)
        return img

    # ------------- Stage 2: Background Subtraction --------
    def foreground_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply the background model to obtain a raw foreground mask (uint8 0/255)."""
        mask = self.bg.apply(img, learningRate=self.learning_rate)
        # If detectShadows=True, shadows often appear as 127 gray; binarize to 0/255
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask

    # ------------- Stage 3: Mask Cleaning -----------------
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Morphological opening (remove specks) then closing (fill holes)."""
        if self.open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        if self.close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)
        return mask

    # ------------- Stage 4: Candidate Extraction ----------
    def find_candidates(self, mask: np.ndarray) -> List[Candidate]:
        """Find contours in the mask and compute geometry metrics for each candidate."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands: List[Candidate] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.area_min or area > self.area_max:
                continue
            per = float(cv2.arcLength(cnt, True))
            if per <= 0:
                continue
            circ = 4.0 * math.pi * area / (per * per)  # roundness in [0,1]
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if r <= 0:
                continue
            fill = area / (math.pi * r * r)           # how full the circle is
            c = Candidate(cnt=cnt, center=(float(x), float(y)), radius=float(r),
                          area=area, perimeter=per, circularity=circ, fill=fill)
            cands.append(c)
        return cands

    # ------------- Stage 5: Candidate Scoring -------------
    def score_candidate(self, c: Candidate) -> float:
        """Combine geometry and motion proximity into a scalar score.
        Higher is better. Reject teleporting blobs via max_jump gate in select_best."""
        # Geometry score: prefer round, solid, reasonably sized
        if c.circularity < self.circ_min or c.fill < self.fill_min:
            return -1.0
        geom = c.circularity * c.fill
        if self.score_area_log:
            geom *= math.log1p(c.area)  # gentle preference for larger, stable blobs
        else:
            geom *= max(c.area, 1.0)
        # Proximity term: favor candidates near last known center
        if self.prev_center is None:
            prox = 1.0
        else:
            jump = math.hypot(c.center[0] - self.prev_center[0], c.center[1] - self.prev_center[1])
            prox = 1.0 / (1.0 + self.proximity_gain * jump)
        return geom * prox

    # ------------- Stage 6: Selection & Motion Gate -------
    def select_best(self, cands: List[Candidate]) -> Optional[Candidate]:
        """Pick best-scoring candidate, rejecting excessive jumps relative to prev_center."""
        best: Optional[Candidate] = None
        best_score = -1.0
        for c in cands:
            # Motion gate: if we have history, reject huge teleports
            if self.prev_center is not None:
                jump = math.hypot(c.center[0] - self.prev_center[0], c.center[1] - self.prev_center[1])
                if jump > self.max_jump:
                    continue
            s = self.score_candidate(c)
            if s > best_score:
                best_score, best = s, c
        return best

    # ------------- Stage 7/8: Kalman Predict/Update -------
    def kalman_predict(self) -> Optional[Tuple[float,float]]:
        if not self.use_kalman:
            return None
        pred = self.kf.predict()  # shape (4,1)
        return float(pred[0]), float(pred[1])

    def kalman_update(self, meas_xy: Optional[Tuple[float,float]]) -> Tuple[float,float]:
        """Update KF with measurement (x,y); if None, return prediction."""
        if not self.use_kalman:
            # No Kalman: just return measurement or keep previous
            if meas_xy is not None:
                return meas_xy
            return self.prev_center if self.prev_center is not None else (0.0, 0.0)

        if meas_xy is None:
            # No measurement: rely on prediction from previous call
            pred = self.kf.statePost  # last corrected state
            return float(pred[0]), float(pred[1])
        else:
            z = np.array([[meas_xy[0]], [meas_xy[1]]], dtype=np.float32)
            est = self.kf.correct(z)
            return float(est[0]), float(est[1])

    # ---------------- Stage 9: Orchestration ---------------
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int,int]], Optional[int], dict]:
        """Run the full pipeline on one frame.
        Returns:
          - center: (x,y) int or None
          - radius: int or None
          - debug: dict with intermediate artifacts (mask_raw, mask_clean, candidates, chosen)
        """
        debug = {}
        img = self.preprocess(frame)
        mask_raw = self.foreground_mask(img)
        mask = self.clean_mask(mask_raw)
        debug['mask_raw'] = mask_raw
        debug['mask_clean'] = mask

        cands = self.find_candidates(mask)
        debug['candidates'] = [
            dict(center=c.center, radius=c.radius, area=c.area, circ=c.circularity, fill=c.fill)
            for c in cands
        ]

        best = self.select_best(cands)

        # Smooth visual output with EMA on top of detection (optional)
        meas_xy: Optional[Tuple[float,float]] = None
        meas_r: Optional[float] = None
        if best is not None:
            x, y = best.center
            r = best.radius
            # EMA smoothing for display
            if self.smooth is None:
                self.smooth = (x, y, r)
            else:
                sx, sy, sr = self.smooth
                self.smooth = (
                    sx + self.ema * (x - sx),
                    sy + self.ema * (y - sy),
                    sr + self.ema * (r - sr),
                )
            meas_xy = (self.smooth[0], self.smooth[1])
            meas_r = self.smooth[2]
        
        # Initialize Kalman on first measurement
        if self.use_kalman and self.prev_center is None and meas_xy is not None:
            self.kf.statePost = np.array([[meas_xy[0]], [meas_xy[1]], [0.0], [0.0]], dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        # Kalman predict/update
        if self.use_kalman:
            _ = self.kalman_predict()
            kx, ky = self.kalman_update(meas_xy)
            center_out = (int(round(kx)), int(round(ky)))
            radius_out = int(round(meas_r)) if meas_r is not None else None
        else:
            center_out = (int(round(meas_xy[0])), int(round(meas_xy[1]))) if meas_xy is not None else None
            radius_out = int(round(meas_r)) if meas_r is not None else None

        # Update prev_center for next-frame proximity / motion gate
        if center_out is not None:
            self.prev_center = (float(center_out[0]), float(center_out[1]))

        debug['chosen'] = dict(center=center_out, radius=radius_out) if center_out is not None else None
        return center_out, radius_out, debug
       

# =========================
# GUI App
# =========================
class FreeThrowReviewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Free Throw Reviewer (Detect Makes)")

        # Configuration
        self.cfg = load_configs()
        self.auto_label = None
        self.manual_label = None
        self.current_index = 0
        self.results = {} 
        self.trajectory = [] 
        self.video_files = []
        self.show_traj = SHOW_TRAJ
        self.show_mask = SHOW_MASK

        # Playback state
        self.cap = None
        self.playing = False
        self.total_frames = 0
        self.width = 0
        self.height = 0
        
        #initialize BallDetector class
        self.detector = BallDetector(
            use_mog2=True,                    # True: mog2;  False: KNN
            area_min=self.cfg["AREA_MIN"],
            area_max=self.cfg["AREA_MAX"],
            circ_min=self.cfg["CIRC_MIN"],
            fill_min=self.cfg["FILL_MIN"],
            # optional tuners:
            # detect_shadows=True,
            # learning_rate=-1.0,
            # open_iters=1, close_iters=1,
            # max_jump=50.0, ema=0.3,
            # use_kalman=True, dt=1/self.fps if self.fps else 1/30
        )

        # ===== UI =====

        Label(root, text="Detect Makes — Player Viewer", font=("Helvetica", 22, "bold")).pack(pady=(8, 0))

        # create panel to place video 
        video_panel = tk.Frame(root, bg="black", padx=4, pady=4)
        video_panel.pack(pady=8)

        # place video label into panel
        self.video_label = Label(video_panel)
        self.video_label.pack()

        # Controls 1
        c1 = tk.Frame(root)
        c1.pack(pady=6)
        Button(c1, text="Load Folder", command=self.load_folder).grid(row=0, column=0, padx=5)
        Button(c1, text="Previous Clip", command=self.prev_video, 
            bg="orange", activebackground="darkorange").grid(row=0, column=1, padx=5)
        Button(c1, text="Play/Pause", command=self.toggle_play).grid(row=0, column=2, padx=5)
        Button(c1, text="Next Clip", command=self.next_video, 
            bg="orange", activebackground="darkorange").grid(row=0, column=3, padx=5)
        Button(c1, text="Restart Clip", command=self.restart_clip).grid(row=0, column=4, padx=5)

        # Controls 2
        c2 = tk.Frame(root)
        c2.pack(pady=6)
        Button(c2, text="<< Frame", command=self.prev_frame).grid(row=0, column=0, padx=5)
        Button(c2, text="Frame >>", command=self.next_frame).grid(row=0, column=1, padx=5)
        Button(c2, text="Reload YAML (R)", command=self.reload_yaml).grid(row=0, column=4, padx=12)

        # Labels/Status
        info = tk.Frame(root)
        info.pack(pady=8)
        self.video_info = tk.StringVar(value="No videos loaded")
        Label(info, textvariable=self.video_info, font=("Helvetica", 13)).pack()

        verdict_box = tk.Frame(root)
        verdict_box.pack(pady=6)
        self.verdict_info = tk.StringVar(value="Auto: —    Label: —")
        Label(verdict_box, textvariable=self.verdict_info, font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,6))
        Button(verdict_box, text="Mark MAKE (G)", command=lambda: self.set_label("made")).grid(...)
        Button(verdict_box, text="Mark MISS (B)", command=lambda: self.set_label("miss")).grid(...)
        Button(verdict_box, text="Mark UNKNOWN (U)", command=lambda: self.set_label("unknown")).grid(...)


        self.status = tk.StringVar(value="Press ‘Load Folder’ (defaults to session's ball_tracking/raw). Space=Play/Pause.")
        Label(root, textvariable=self.status, font=("Helvetica", 12)).pack(pady=(4, 10))

        # Key bindings
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Right>", lambda e: self.jump_frames(1))
        root.bind("<Left>",  lambda e: self.jump_frames(-1))
        root.bind("<Shift-Right>", lambda e: self.jump_frames(10))
        root.bind("<Shift-Left>",  lambda e: self.jump_frames(-10))
        root.bind("<Next>",  lambda e: self.jump_frames(60))   # PageDown
        root.bind("<Prior>", lambda e: self.jump_frames(-60))  # PageUp
        root.bind("<Home>",  lambda e: self.seek_frame(0))
        root.bind("<End>",   lambda e: self.seek_frame(self.total_frames - 1))

        root.bind("<g>",  lambda e: self.set_label("made"))
        root.bind("<b>",  lambda e: self.set_label("miss"))
        root.bind("<u>",  lambda e: self.set_label("unknown"))

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Prompt on launch
        self.load_folder(initial=self.cfg["INPUT_FOLDER"])

    # ---------- Folder / navigation ----------
    def load_folder(self, initial=None):
        init_dir = str(initial) if (initial and Path(initial).exists()) else str(Path.home())
        folder = filedialog.askdirectory(initialdir=init_dir, title="Select Folder with Videos")
        if not folder:
            return
        p = Path(folder)
        vids = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".mp4",".avi",".mov"}]
        self.video_files = sorted(vids, key=lambda q: q.name.lower())
        if not self.video_files:
            messagebox.showerror("No videos", "No .mp4/.mov/.avi files found.")
            return
        self.current_index = 0
        self.open_video(self.video_files[self.current_index])
        self.status.set(f"Loaded {len(self.video_files)} videos from: {p}")

    def open_video(self, path: Path):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showwarning("OpenCV", f"Could not open: {path.name}")
            self.next_video()
            return
     
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) reported for video: {path.name}")
        self.fps = float(fps)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # reset analysis state
        self.trajectory = []
        self.detector.prev_center = None
        self.auto_label = None
        
        # prefill manual label if previously set 
        self.manual_label = self.results.get(path.name, None)

        self.detector.smooth = None
        if self.detector.use_kalman:
            self.detector.kf.statePost[:] = 0
            self.detector.kf.errorCovPost[:] = np.eye(4, dtype=np.float32) * 1e3

        if self.detector.use_kalman and self.fps > 0:
            dt = 1.0 / self.fps
            self.detector.kf.transitionMatrix[:] = np.array(
                [[1,0,dt,0],
                [0,1,0,dt],
                [0,0,1, 0],
                [0,0,0, 1]], dtype=np.float32
            )



        # show first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._read_process_show()

        idx = self.current_index + 1
        total_videos = len(self.video_files)
        duration = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        self.video_info.set(
            f"Filename: {path.name}  (Clip {idx}/{total_videos})  |  "
            f"{self.width}x{self.height}, {self.total_frames} frames, {self.fps:.2f} FPS, {duration:.2f}s"
        )
        self._update_verdict_label()

    def next_video(self):
        if not self.video_files: return
        self._save_current_result()
        self.current_index = (self.current_index + 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def prev_video(self):
        if not self.video_files: return
        self._save_current_result()
        self.current_index = (self.current_index - 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def restart_clip(self):
        if not self.cap: return
        self.playing = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.trajectory = []
        self.detector.prev_center = None
        self.auto_label = None
        
        self.detector.smooth = None
        if self.detector.use_kalman:
            self.detector.kf.statePost[:] = 0
            self.detector.kf.errorCovPost[:] = np.eye(4, dtype=np.float32) * 1e3
        
        self._read_process_show()
        self._update_verdict_label()

    # ---------- Playback / stepping ----------
    def _play_loop(self):
        if not self.playing:
            return
        # if at end, compute auto-verdict once and stop
        if self._current_frame_index() >= self.total_frames - 1:
            self._compute_auto_label_if_needed()
            self.playing = False
            return
        self._read_process_show()
        delay = int(1000 / (self.fps if self.fps > 0 else 30.0))
        self.root.after(delay, self._play_loop)

    def toggle_play(self):
        if not self.cap: return
        if self._current_frame_index() >= self.total_frames - 1:
            self.seek_frame(0)
        self.playing = not self.playing
        if self.playing:
            self._play_loop()

    def next_frame(self):
        if not self.cap: return
        self.playing = False
        self._read_process_show()

    def prev_frame(self):
        if not self.cap: return
        self.playing = False
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
        self._read_process_show()

    def jump_frames(self, n):
        if not self.cap: return
        self.playing = False
        target = max(0, min(self.total_frames - 1, self._current_frame_index() + n))
        self.seek_frame(target)

    def seek_frame(self, idx):
        if not self.cap: return
        self.playing = False
        idx = max(0, min(self.total_frames - 1, int(idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self._read_process_show(update_trajectory=False)

    def _current_frame_index(self):
        if not self.cap: return 0
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, min(self.total_frames - 1, pos - 1))

    # ---------- Analysis / draw ----------
    def _read_process_show(self, update_trajectory=True):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._compute_auto_label_if_needed()
            return

        # Detect + optional append to trajectory (append None when no detection)
        center, radius, debug = self.detector.detect(frame)
        if update_trajectory:
            self.trajectory.append(center if center is not None else None)

        # Overlays: hoop regions, detection, trajectory, optional HSV mask
        out = frame.copy()

        # Optional mask overlay
        if self.show_mask and isinstance(debug, dict):
            fg = debug.get("mask_clean", None)
            if fg is not None:
                overlay = out.copy()
                tint = np.zeros_like(out); tint[:] = (0, 255, 255)  # BGR yellow
                # put the tint where the mask is >0
                mask3 = fg[..., None]  # HxWx1
                overlay = np.where(mask3 > 0, tint, out)
                out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

        # draw trajectory circles
        if center and radius:
            cv2.circle(out, center, max(radius, 2), (0, 255, 0), 2)
            cv2.circle(out, center, 2, (0, 0, 255), -1)

        # Draw hoop regions
        cv2.rectangle(out, self.cfg["UPPER_HOOP_REGION"][0], self.cfg["UPPER_HOOP_REGION"][1], (255, 0, 0), 2)
        cv2.rectangle(out, self.cfg["LOWER_HOOP_REGION"][0], self.cfg["LOWER_HOOP_REGION"][1], (0, 0, 255), 2)

     

        # Draw trajectory
        if self.show_traj and len(self.trajectory) > 1:
            for pt in self.trajectory:
                if pt is not None:
                    cv2.circle(out, pt, 2, (0, 200, 255), -1)

        # Header/status
        idx = self.current_index + 1
        total = len(self.video_files)
        name = self.video_files[self.current_index].name
        header = f"Clip {idx}/{total} — {name}"
        cv2.putText(out, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Convert to Tk image
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=img)
        self.video_label.image = img

        # Status line
        i = self._current_frame_index()
        t = (i / self.fps) if self.fps > 0 else 0.0
        self.status.set(f"Frame {i}/{self.total_frames-1}  |  t={t:.2f}s  |  "
                        f"Traj points: {sum(1 for p in self.trajectory if p is not None)}")
        self._update_verdict_label()

    def _compute_auto_label_if_needed(self):
        if self.auto_label is None and len(self.trajectory) > 1:
            self.auto_label = "made" if is_make(self.trajectory, self.cfg["UPPER_HOOP_REGION"], self.cfg["LOWER_HOOP_REGION"]) else "miss"
            self._update_verdict_label()

    def _update_verdict_label(self):
        auto_txt = self.auto_label if self.auto_label is not None else "—"
        label_txt = self.manual_label if self.manual_label is not None else "—"
        self.verdict_info.set(f"Auto: {auto_txt}    Label: {label_txt}")

    # ---------- Toggles / actions ----------

    def reload_yaml(self):
        try:
            new = load_configs()
            self.cfg.update(new)
            det = self.detector
            det.area_min = self.cfg["AREA_MIN"]
            det.area_max = self.cfg["AREA_MAX"]
            det.circ_min = self.cfg["CIRC_MIN"]
            det.fill_min = self.cfg["FILL_MIN"]
            # optional: tweak learning rate / morphology on reload too:
            # det.learning_rate = 0.001
            # det.open_iters = 1; det.close_iters = 1
        except Exception as e:
            messagebox.showerror("YAML reload error", str(e))

    def set_label(self, label):
        # "made" | "miss" | "unknown"
        self.manual_label = label
        self._update_verdict_label()

    # ---------- Save results ----------
    def _save_current_result(self):
        if not self.video_files:
            return
        name = self.video_files[self.current_index].name
        # Priority: manual, otherwise auto, otherwise unknown
        label = self.manual_label or self.auto_label or "unknown"
        self.results[name] = label

        # Write/append to CSV in the same schema your analysis expects
        out_path = self.cfg["OUTPUT_PATH"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build full rows: map video name -> feature_key + normalized label
        base = Path(name).stem
        core = base.split("_")[0]  # e.g., "freethrow001"
        feature_key = f"{core}_angles.csv"
        normalized = {"made":"made", "miss":"miss"}.get(label, None)
        if normalized is None:
            # skip unknown to keep dataset clean
            return
        
        existing = {}
        if out_path.exists():
            with open(out_path, "r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if len(row) >= 2:
                        existing[row[0]] = row[1]

        existing[feature_key] = normalized
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "outcome"])
            for k,v in sorted(existing.items()):
                w.writerow([k, v])

    # ---------- Cleanup ----------
    def on_close(self):
        # Save the current clip result before exiting
        try:
            self._save_current_result()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        self.root.quit()


def draw_detection(frame: np.ndarray, center: Optional[Tuple[int,int]], radius: Optional[int]) -> np.ndarray:
    """Utility to draw a circle and center on the frame for visualization."""
    out = frame.copy()
    if center is not None and radius is not None:
        cv2.circle(out, center, max(radius, 2), (0, 255, 0), 2)
        cv2.circle(out, center, 2, (0, 0, 255), -1)
    return out

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = FreeThrowReviewerApp(root)
    root.mainloop()
