"""
Title: capture_cb_mono.py (single-camera checkerboard capture with 5-fingers auto-capture)

Auto-captures after 2s of exactly FIVE fingers up.
Draws MediaPipe hand landmarks, shows finger count,
and displays which side/filename will be saved next.

Controls:
  - TAB   : switch active camera (LEFT <-> RIGHT)
  - SPACE : manual capture (if detection + size gate pass)
  - ESC   : quit
"""

import cv2 as cv
import os
from pathlib import Path
import threading
import time
import yaml
import numpy as np

# ========= Simple hand detector =========
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("mediapipe required. Install with: pip install mediapipe")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ========================================
# Config (from project YAML)
# ========================================
config_path = Path(__file__).resolve().parents[4] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Session info
ATHLETE = cfg["athlete"]
SESSION = cfg["session"]

# Camera Parameters
CAM_LEFT_INDEX = cfg["left_cam_index"]
CAM_RIGHT_INDEX = cfg["right_cam_index"]
CAM_RESOLUTION = tuple(cfg["uncropped_stereo_resolution"])  # e.g., (1280, 720)
CROP_RESOLUTION = tuple(cfg["cropped_stereo_resolution"])  # e.g., (720, 720)
PLAYER_TRACKING_FPS = cfg["player_tracking_fps"]

# Calibration Parameters
CHECKERBOARD = tuple(cfg["inner_corners"])  # (columns, rows)
# MIN_SQUARE_PX = float(cfg.get("min_square_px", 40.0))  # default 40px
MIN_SQUARE_PX = 5

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
def crop_center(frame, crop_size):
    """Center-crop frame to the configured crop_size (width, height)."""
    crop_w, crop_h = crop_size
    h, w = frame.shape[:2]
    if crop_w >= w and crop_h >= h:
        return frame

    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return frame[y1:y2, x1:x2]

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
# 5-fingers detector (count-based)
# ========================================
class FiveFingersDetector:
    """
    Counts fingers using a simple heuristic:
      - Index/Middle/Ring/Pinky: tip.y < pip.y -> "up" (image coords: smaller y = higher)
      - Thumb: tip farther from wrist than IP -> "extended" (orientation-agnostic-ish)
    Triggers when count == 5, held for `hold_seconds`.
    """
    def __init__(self, hold_seconds=2.0):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.hold_seconds = hold_seconds
        self.gesture_start_t = None
        self.last_count = 0

        # Thumb margin to avoid tiny jitter: normalized distance units
        self.THUMB_MARGIN = 0.02

    @staticmethod
    def _dist(a, b):
        dx, dy = a.x - b.x, a.y - b.y
        return (dx*dx + dy*dy) ** 0.5

    def _count_fingers(self, lm):
        # Indices
        IDX_TIP, IDX_PIP = 8, 6
        MID_TIP, MID_PIP = 12, 10
        RNG_TIP, RNG_PIP = 16, 14
        PNK_TIP, PNK_PIP = 20, 18
        TH_TIP, TH_IP    = 4, 3
        WRIST            = 0

        fingers_up = 0
        # index/middle/ring/pinky up if tip higher (smaller y) than PIP
        for tip, pip in [(IDX_TIP, IDX_PIP), (MID_TIP, MID_PIP), (RNG_TIP, RNG_PIP), (PNK_TIP, PNK_PIP)]:
            if lm[tip].y < lm[pip].y:
                fingers_up += 1

        # thumb "extended" if tip farther from wrist than IP (robust to left/right)
        d_tip_wr = self._dist(lm[TH_TIP], lm[WRIST])
        d_ip_wr  = self._dist(lm[TH_IP],  lm[WRIST])
        if d_tip_wr > d_ip_wr + self.THUMB_MARGIN:
            fingers_up += 1

        return fingers_up

    def process(self, bgr_640):
        """Return (count, is_target_now, seconds_held, annotated_frame)."""
        image = bgr_640
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        count = 0
        target_ok = False

        if res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                image, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
            count = self._count_fingers(hand_lms.landmark)
            target_ok = (count == 5)

        # temporal hold logic
        t = time.time()
        if target_ok:
            if self.gesture_start_t is None:
                self.gesture_start_t = t
        else:
            self.gesture_start_t = None

        held = 0.0 if self.gesture_start_t is None else (t - self.gesture_start_t)
        self.last_count = count
        return count, target_ok, held, image

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

        # five-finger detector
        self.detector = FiveFingersDetector(hold_seconds=2.0)
        self.cooldown_t = 0.0
        self.COOLDOWN_SEC = 1.0

    def show_status(self, text, color):
        self.status_text = text
        self.status_color = color
        self.status_time = time.time()

    def _next_filename(self):
        if self.active == "LEFT":
            return f"left_{self.id_left:02}.png"
        else:
            return f"right_{self.id_right:02}.png"

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
        print("[INFO] Controls: TAB = switch camera, SPACE = manual capture, ESC = quit")
        print("[INFO] Auto-capture: hold FIVE fingers up for 2 seconds")
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

            view = crop_center(frame, CROP_RESOLUTION)

            # === Hand detection & finger count ===
            count, ok, held, view = self.detector.process(view)

            # UI overlay
            header = (
                f"Active: {self.active} | Next: {self._next_filename()} | "
                f"MIN_SQUARE_PX={MIN_SQUARE_PX:.0f} | TAB switch | SPACE capture | ESC quit"
            )
            cv.putText(view, header, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Status text (recent events)
            if time.time() - self.status_time < 1.5:
                cv.putText(view, self.status_text, (10, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, self.status_color, 2)

            # Finger info
            cv.putText(view, f"Fingers Up: {count}",
                       (10, 95), cv.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2)

            # Auto-capture overlay
            if ok:
                remaining = max(0.0, self.detector.hold_seconds - held)
                color = (0, 220, 0) if remaining < 1.0 else (0, 200, 200)
                cv.putText(view, f"5 fingers detected: capturing in {remaining:0.1f}s",
                           (10, 125), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Auto-capture if held long enough (with a short cooldown)
            now = time.time()
            if ok and held >= self.detector.hold_seconds and (now - self.cooldown_t) > self.COOLDOWN_SEC:
                self.capture_one(view, self.active)
                self.cooldown_t = now  # debounce

            cv.imshow("Mono Checkerboard Capture", view)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 9:  # TAB
                self.active = "RIGHT" if self.active == "LEFT" else "LEFT"
                self.show_status(f"Switched to {self.active}", (0, 255, 255))
                self.detector.gesture_start_t = None  # reset hold timer on switch
            elif key == 32:  # SPACE (manual fallback)
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
