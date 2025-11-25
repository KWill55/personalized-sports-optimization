"""
Title: setup_cameras.py

Purpose: 
    - Provide real-time feedback to help position cameras for optimal 3D data acquisition

Prerequisites:
    - Ensure that cameras are level (yaw: z axis)
    - Ensure that cameras are equally spaced apart and are both ~45 degrees from athlete (use basic geometry to find angles)
    - Place correct checkerboard size in front of cameras 
    
Description:
    - Displays all three camera angles (Left, Right, Ball), each with their own thread
    - Visual aids for stereo cameras:
        - Plumb line: central vertical line to check roll (y axis)
        - Horizon grid: horizontal lines at 25%, 50%, 75% to check pitch (x axis)
    - Displays checkerboard detection and rolling detection success rate for stereo cameras
    - Ball camera shows raw feed (no checkerboard and no guides)

Usage:
    - Place a checkerboard in view of the stereo cameras
    - Adjust lenses/focus and mounts until differences are minimal
    - Press ESC to quit
"""

import cv2 as cv
import numpy as np
import threading
from collections import deque
from pathlib import Path
import yaml

# ========================================
# Config
# ========================================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    project_cfg = yaml.safe_load(f)

# Camera Indices
LEFT_CAM_INDEX = project_cfg["left_cam_index"]
RIGHT_CAM_INDEX = project_cfg["right_cam_index"]
BALL_CAM_INDEX  = project_cfg["third_cam_index"]

# Calibration Parameters
CHECKERBOARD   = tuple(project_cfg["inner_corners"])    # (columns, rows)
SQUARE_SIZE    = project_cfg["square_size_cm"]          
WINDOW_SIZE    = project_cfg["success_window"]

# Video Parameters
CAM_RESOLUTION = (project_cfg["original_frame_width"], project_cfg["original_frame_height"])
CROP_SIZE      = tuple(project_cfg["crop_size"])        
PLAYER_TRACKING_FPS = project_cfg["player_tracking_fps"]

# Detection Thresholds
THRESHOLD_DETECT = project_cfg["threshold_detect"]

#Constants 
HORIZ_COLOR = (0, 255, 255) 
PLUMB_COLOR = (255, 0, 255)

# ========================================
# Helpers
# ========================================
def center_crop(frame, crop_size):
    """Center-crop to (w, h). If frame smaller, returns original."""
    h, w = frame.shape[:2]
    cw, ch = crop_size
    if cw <= 0 or ch <= 0 or cw > w or ch > h:
        return frame
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    return frame[y1:y1+ch, x1:x1+cw]

# ========================================
# Camera Thread
# ========================================
class CameraThread(threading.Thread):
    def __init__(self, index, name):
        super().__init__(daemon=True)
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  CAM_RESOLUTION[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        self.cap.set(cv.CAP_PROP_FPS,          PLAYER_TRACKING_FPS)
        try: self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except: pass

        self.name = name
        self.frame = None
        self.lock = threading.Lock()   # NEW
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:        # NEW
                    self.frame = frame.copy()  # snapshot

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ========================================
# Stereo Tuning GUI
# ========================================
class StereoTuningGUI:
    def __init__(self, left_cam, right_cam, ball_cam):
        self.left_cam  = left_cam
        self.right_cam = right_cam
        self.ball_cam  = ball_cam

        # Detection stats (rolling window)
        self.detections_left  = deque(maxlen=WINDOW_SIZE)
        self.detections_right = deque(maxlen=WINDOW_SIZE)

        # Checkerboard object points
        self.objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        self.frame_count = 0

    def detect_and_overlay(self, frame, label):
        """Detects checkerboard for stereo cameras only."""
        if frame is None:
            return None, False

        view = center_crop(frame, CROP_SIZE) if CROP_SIZE != (0, 0) else frame
        gray = cv.cvtColor(view, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            cv.drawChessboardCorners(view, CHECKERBOARD, corners, ret)

        cv.putText(view, label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return view, ret

    def overlay_success_bars(self, canvas, left_ok, right_ok):
        """Show rolling success counts and GOOD/LOW status for stereo cameras only."""
        successL = sum(self.detections_left)
        successR = sum(self.detections_right)

        nL = len(self.detections_left) or 1
        nR = len(self.detections_right) or 1

        healthyL = (successL / nL) >= THRESHOLD_DETECT
        healthyR = (successR / nR) >= THRESHOLD_DETECT

        text = (
            f"Frame: {self.frame_count} | "
            f"L: {successL}/{nL} {'GOOD' if healthyL else 'LOW'} | "
            f"R: {successR}/{nR} {'GOOD' if healthyR else 'LOW'}"
        )
        cv.putText(canvas, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def draw_guides_per_view(self, canvas, left_w, right_w, top_h):
        """Draw alignment guides for stereo cameras only."""
        # Left view
        self._draw_view_guides(canvas, 0, 0, left_w, top_h)
        # Right view
        self._draw_view_guides(canvas, left_w, 0, right_w, top_h)

    @staticmethod
    def _draw_view_guides(img, x0, y0, w, h):
        """
        img: entire canvas
        x0, y0, w, h: sub-rectangle (view) to draw guides
        """
        
        # draw horizontal lines
        for frac in (0, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6):
            y = y0 + int(h * frac)
            cv.line(img, (x0, y), (x0 + w - 1, y), HORIZ_COLOR, 3)

        # draw vertical center line
        x_mid = x0 + w // 2
        cv.line(img, (x_mid, y0), (x_mid, y0 + h - 1), PLUMB_COLOR, 3)

    def compose_canvas(self, viewL, viewR, viewB):
        """Top row: stereo; Bottom row: Ball (raw)."""
        hL = viewL.shape[0]
        hR = viewR.shape[0]
        if hL != hR:
            print("[WARNING] Stereo pair resolutions are not the same")
            new_w = int(viewR.shape[1] * (hL / hR))
            viewR = cv.resize(viewR, (new_w, hL))

        top_row = cv.hconcat([viewL, viewR])
        top_w   = top_row.shape[1]

        # Keep Ball native size, pad/crop width if needed
        ball_h, ball_w = viewB.shape[:2]
        if ball_w < top_w:
            pad_total = top_w - ball_w
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            viewB = cv.copyMakeBorder(viewB, 0, 0, pad_left, pad_right, cv.BORDER_CONSTANT, (0,0,0))
        elif ball_w > top_w:
            start_x = (ball_w - top_w) // 2
            viewB = viewB[:, start_x:start_x+top_w]

        return cv.vconcat([top_row, viewB])

    def process_frame(self):
        L = self.left_cam.get_frame()
        R = self.right_cam.get_frame()
        B = self.ball_cam.get_frame()
        if L is None or R is None or B is None:
            return None

        viewL, retL = self.detect_and_overlay(L, "Left")
        viewR, retR = self.detect_and_overlay(R, "Right")

        viewB = B  # already a copy from get_frame
        cv.putText(viewB, "Ball", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Update stats
        self.detections_left.append(1 if retL else 0)
        self.detections_right.append(1 if retR else 0)
        self.frame_count += 1

        # Combine views
        canvas = self.compose_canvas(viewL, viewR, viewB)

        # Overlays
        self.overlay_success_bars(canvas, retL, retR)
        self.draw_guides_per_view(canvas, viewL.shape[1], viewR.shape[1], viewL.shape[0])

        return canvas

    def run(self):
        print("[INFO] Press ESC to exit.")
        while True:
            canvas = self.process_frame()
            if canvas is None:
                cv.waitKey(1)
                continue
            cv.imshow("Stereo + Ball Tuning", canvas)
            if cv.waitKey(1) == 27:
                break

# ========================================
# Main
# ========================================
def main():
    left_cam  = CameraThread(LEFT_CAM_INDEX, "Left")
    right_cam = CameraThread(RIGHT_CAM_INDEX, "Right")
    ball_cam  = CameraThread(BALL_CAM_INDEX, "Ball")

    left_cam.start()
    right_cam.start()
    ball_cam.start()

    gui = StereoTuningGUI(left_cam, right_cam, ball_cam)
    try:
        gui.run()
    finally:
        for cam in (left_cam, right_cam, ball_cam):
            cam.stop()
            cam.join()
        cv.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
