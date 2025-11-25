"""
Title: capture_cb_pairs.py

Description
    - Purpose: Capture synchronized stereo checkerboard image pairs for calibration.
    - Crops center 640x640 from both cameras
    - Displays combined 1280x640 window

Prerequisites:
    - calibration grid printed
    - ensure cameras can detect grid (check_cb_detection.py)

Usage:
    - capture at least 10 image pairs of the calibration grid (20-25 is best)
    - capture the calibration grid at different tilts, depths, corners of visiblity, etc
    - press 'space' to save combined image
    - Press 'escape' to exit

Outputs:
    - Displays combined view and saves cropped, combined 640x640 stereo images.


"""

import cv2 as cv
import os
from pathlib import Path
import threading
import time
import yaml
from pathlib import Path
import numpy as np


# ========================================
# Config (from project YAML)
# ========================================

# Load YAML Config
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    project_cfg = yaml.safe_load(f)

#session info 
ATHLETE = project_cfg["athlete"]
SESSION = project_cfg["session"]

# Camera Parameters
CAM_LEFT_INDEX = project_cfg["left_cam_index"]
CAM_RIGHT_INDEX = project_cfg["right_cam_index"]
CAM_RESOLUTION = (project_cfg["original_frame_width"], project_cfg["original_frame_height"])  # (1280, 720)
CROP_RESOLUTION = tuple(project_cfg["crop_size"])  # (640, 640)
PLAYER_TRACKING_FPS = project_cfg["player_tracking_fps"]

# Calibration Parameters
CHECKERBOARD = tuple(project_cfg["inner_corners"])  # (columns, rows)
# MIN_SQUARE_PX = cfg["min_square_px"] # usually 40px
MIN_SQUARE_PX = 5 # yes I know this is way too small. 

# ========================================
# Paths and Directories
# ========================================
# calib_dir = project_cfg["paths"]["calib_pairs"]
calib_dir = Path(f"data/{ATHLETE}/{SESSION}/calibration/calib_images/pairs")
calib_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Camera Thread
# ========================================
class CameraThread(threading.Thread):
    def __init__(self, index, name):
        super().__init__()
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
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
# Stereo Capture GUI
# ========================================
class StereoCaptureGUI:
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam
        self.pair_id = self.get_next_pair_id()
        self.status_text = ""
        self.status_color = (255, 255, 255)
        self.status_time = 0

    def get_next_pair_id(self):
        existing = list(calib_dir.glob("pair_*.png"))
        return len(existing) + 1

    def crop_center(self, frame):
        return frame[40:680, 320:960]  # center crop to 640x640

    def show_status(self, text, color):
        self.status_text = text
        self.status_color = color
        self.status_time = time.time()

    def run(self):
        print("[INFO] Press SPACE to capture a pair (only saves if checkerboard detected). ESC to exit.")

        while True:
            if self.left_cam.frame is None or self.right_cam.frame is None:
                continue

            # Crop and combine the two images 
            frameL = self.crop_center(self.left_cam.frame)
            frameR = self.crop_center(self.right_cam.frame)
            combined = cv.hconcat([frameL, frameR])

            # Overlay text
            cv.putText(combined, f"Pair #{self.pair_id}", (20, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv.putText(combined, "SPACE: Capture | ESC: Quit",
                       (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show last status message for 1.5 sec
            if time.time() - self.status_time < 1.5:
                cv.putText(combined, self.status_text, (20, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, self.status_color, 2)

            # Show combined feed
            cv.imshow("Capture Calibration Pairs", combined)

            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.capture_pair(frameL, frameR, combined)

        print("[INFO] Closing capture window.")

    @staticmethod
    def quick_square_px(pts, cols, rows):
        """
        Rough estimate of pixels-per-square using corner lengths.
        Uses TL->TR for width and TL->BL for height, then returns the conservative min.
        """
        pts = pts.reshape(-1, 2)
        tl = pts[0]                          # top-left
        tr = pts[cols - 1]                   # top-right (same row)
        bl = pts[(rows - 1) * cols]          # bottom-left (same col)
        w = np.linalg.norm(tr - tl) / (cols - 1)
        h = np.linalg.norm(bl - tl) / (rows - 1)
        return min(w, h)


    def capture_pair(self, frameL, frameR, combined):
        # 1) Detect checkerboard in both
        grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

        det_flags = cv.CALIB_CB_EXHAUSTIVE
        retL, ptsL = cv.findChessboardCornersSB(grayL, CHECKERBOARD, flags=det_flags)
        retR, ptsR = cv.findChessboardCornersSB(grayR, CHECKERBOARD, flags=det_flags)

        if not (retL and retR):
            msg = "Checkerboard NOT detected in both."
            print(f"[WARNING] {msg}")
            self.show_status(msg, (0, 0, 255))
            return

        # 2) Refine corners (helps the size estimate slightly)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(grayL, ptsL, (11, 11), (-1, -1), criteria)
        cv.cornerSubPix(grayR, ptsR, (11, 11), (-1, -1), criteria)

        # 3) Fast per-square pixel size using grid spans
        cols, rows = CHECKERBOARD  # (inner corners)
        sqL = self.quick_square_px(ptsL, cols, rows)
        sqR = self.quick_square_px(ptsR, cols, rows)

        if (sqL < MIN_SQUARE_PX) or (sqR < MIN_SQUARE_PX):
            msg = f"Board too small (L={sqL:.1f}px, R={sqR:.1f}px). Move closer."
            print(f"[WARNING] {msg}")
            self.show_status(msg, (0, 0, 255))
            return

        # 4) Passed both gates → save
        fname = calib_dir / f"pair_{self.pair_id:02}.png"
        cv.imwrite(str(fname), combined)
        print(f"[INFO] Saved {fname.name}  (square px L={sqL:.1f}, R={sqR:.1f})")
        self.show_status(f"Saved pair #{self.pair_id}", (0, 255, 0))
        self.pair_id += 1



# ========================================
# Main
# ========================================
def main():
    left_cam = CameraThread(CAM_LEFT_INDEX, "Left")
    right_cam = CameraThread(CAM_RIGHT_INDEX, "Right")
    left_cam.start()
    right_cam.start()

    gui = StereoCaptureGUI(left_cam, right_cam)

    try:
        gui.run()
    finally:
        left_cam.stop()
        right_cam.stop()
        left_cam.join()
        right_cam.join()
        cv.destroyAllWindows()
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()

