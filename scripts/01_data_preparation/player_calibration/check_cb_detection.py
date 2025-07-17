"""
Title: tune_stereo_intrinsics.py - Real-Time Stereo Intrinsic Tuning

Purpose:
    Display side-by-side stereo camera feeds (cropped to 640x640 each),
    detect checkerboard patterns, and show detection consistency stats.
    Print camera intrinsics and distortion coefficients every N frames.

Usage:
    - Place a checkerboard in view.
    - Press ESC to quit.
"""

"""
Title: tune_stereo_intrinsics.py - Real-Time Stereo Tuning

Purpose:
    Provide real-time feedback for physically tuning two cameras before stereo calibration.
    Displays detection consistency, focal length difference, principal point difference,
    and horizontal alignment guides.

Features:
    - Rolling detection success rate with GOOD/LOW status
    - Focal length difference (fx, fy) and principal point alignment (cx, cy)
    - Horizontal lines to help align stereo baseline
    - Color-coded overlays for visual clarity
    - Lightweight intrinsics computation (feedback only)

Usage:
    - Place a checkerboard in view of both cameras
    - Adjust lenses/focus until differences are minimal
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
import yaml
from pathlib import Path

config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Camera Indices
LEFT_CAM_INDEX = cfg["left_cam_index"]
RIGHT_CAM_INDEX = cfg["right_cam_index"]

# Calibration Parameters
CHECKERBOARD = tuple(cfg["inner_corners"])    # (columns, rows)
SQUARE_SIZE = cfg["square_size_cm"]           # cm
CALIBRATE_EVERY = cfg["calibrate_every"]
WINDOW_SIZE = cfg["success_window"]

# Video Parameters
CAM_RESOLUTION = (cfg["frame_width"], cfg["frame_height"])
CROP_SIZE = tuple(cfg["crop_size"])  # (width, height)

# Detection Thresholds
THRESHOLD_DETECT = cfg["threshold_detect"]
THRESHOLD_FL = cfg["threshold_fl"]  # px
THRESHOLD_PP = cfg["threshold_pp"]  # px


# ========================================
# Camera Thread
# ========================================
class CameraThread(threading.Thread):
    def __init__(self, index, name):
        super().__init__()
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        self.cap.set(cv.CAP_PROP_FPS, 30)
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
# Stereo Tuning GUI
# ========================================
class StereoTuningGUI:
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        # Detection stats
        self.detections_left = deque(maxlen=WINDOW_SIZE)
        self.detections_right = deque(maxlen=WINDOW_SIZE)

        # Calibration samples
        self.objpoints_left, self.imgpoints_left = [], []
        self.objpoints_right, self.imgpoints_right = [], []

        # Checkerboard object points
        self.objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        self.frame_count = 0
        self.fl_diff_text = ""
        self.pp_diff_text = ""
        self.fl_color = (255, 255, 255)
        self.pp_color = (255, 255, 255)

    def process_frame(self):
        if self.left_cam.frame is None or self.right_cam.frame is None:
            return None

        # Crop frames
        frameL = self.left_cam.frame[40:680, 320:960]
        frameR = self.right_cam.frame[40:680, 320:960]

        # Convert to gray
        grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

        # Detect checkerboard
        retL, cornersL = cv.findChessboardCorners(grayL, CHECKERBOARD, None)
        retR, cornersR = cv.findChessboardCorners(grayR, CHECKERBOARD, None)

        self.detections_left.append(1 if retL else 0)
        self.detections_right.append(1 if retR else 0)

        if retL:
            cv.drawChessboardCorners(frameL, CHECKERBOARD, cornersL, retL)
            self.objpoints_left.append(self.objp.copy())
            self.imgpoints_left.append(cornersL)
        if retR:
            cv.drawChessboardCorners(frameR, CHECKERBOARD, cornersR, retR)
            self.objpoints_right.append(self.objp.copy())
            self.imgpoints_right.append(cornersR)

        # Limit sample storage
        self.objpoints_left = self.objpoints_left[-50:]
        self.imgpoints_left = self.imgpoints_left[-50:]
        self.objpoints_right = self.objpoints_right[-50:]
        self.imgpoints_right = self.imgpoints_right[-50:]

        self.frame_count += 1

        # Quick calibration feedback every N frames
        if self.frame_count % CALIBRATE_EVERY == 0:

            self.update_intrinsics(grayL.shape[::-1])

        # Combine frames
        combined = cv.hconcat([frameL, frameR])

        # Draw overlays
        self.overlay_stats(combined)
        self.draw_alignment_lines(combined)

        return combined

    def update_intrinsics(self, img_size):
        # Use only the last 10 samples for speed
        recent_objpoints_left = self.objpoints_left[-10:]
        recent_imgpoints_left = self.imgpoints_left[-10:]
        recent_objpoints_right = self.objpoints_right[-10:]
        recent_imgpoints_right = self.imgpoints_right[-10:]

        # Ensure enough points exist for calibration
        if len(recent_objpoints_left) >= 5 and len(recent_objpoints_right) >= 5:
            # Lightweight calibration with recent samples
            _, mtxL, _, _, _ = cv.calibrateCamera(recent_objpoints_left, recent_imgpoints_left, img_size, None, None)
            _, mtxR, _, _, _ = cv.calibrateCamera(recent_objpoints_right, recent_imgpoints_right, img_size, None, None)

            # Compute differences
            fx_diff = abs(mtxL[0, 0] - mtxR[0, 0])
            fy_diff = abs(mtxL[1, 1] - mtxR[1, 1])
            cx_diff = abs(mtxL[0, 2] - mtxR[0, 2])
            cy_diff = abs(mtxL[1, 2] - mtxR[1, 2])

            # Update overlay text
            self.fl_diff_text = f"FL Diff: fx={fx_diff:.1f}px fy={fy_diff:.1f}px"
            self.pp_diff_text = f"PP Diff: cx={cx_diff:.1f}px cy={cy_diff:.1f}px"

            # Color code for GOOD/BAD status
            self.fl_color = (0, 255, 0) if fx_diff < THRESHOLD_FL and fy_diff < THRESHOLD_FL else (0, 0, 255)
            self.pp_color = (0, 255, 0) if cx_diff < THRESHOLD_PP and cy_diff < THRESHOLD_PP else (0, 0, 255)

    def overlay_stats(self, combined):
        successL = sum(self.detections_left)
        successR = sum(self.detections_right)

        healthyL = (successL / len(self.detections_left)) >= THRESHOLD_DETECT if len(self.detections_left) else False
        healthyR = (successR / len(self.detections_right)) >= THRESHOLD_DETECT if len(self.detections_right) else False

        statusL = "GOOD" if healthyL else "LOW"
        statusR = "GOOD" if healthyR else "LOW"

        colorL = (0, 255, 0) if healthyL else (0, 0, 255)
        colorR = (0, 255, 0) if healthyR else (0, 0, 255)

        # Detection status
        detect_text = f"Frame: {self.frame_count} | L: {successL}/{len(self.detections_left)} {statusL} | R: {successR}/{len(self.detections_right)} {statusR}"
        cv.putText(combined, detect_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Status under each view
        cv.putText(combined, statusL, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, colorL, 2)
        cv.putText(combined, statusR, (640 + 20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, colorR, 2)

        # Intrinsics feedback
        cv.putText(combined, self.fl_diff_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, self.fl_color, 2)
        cv.putText(combined, self.pp_diff_text, (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, self.pp_color, 2)

    def draw_alignment_lines(self, combined):
        # Horizontal guide lines for baseline alignment
        height = combined.shape[0]
        for y in [height // 4, height // 2, (3 * height) // 4]:
            cv.line(combined, (0, y), (combined.shape[1], y), (0, 255, 255), 1)

    def run(self):
        print("[INFO] Press ESC to exit.")
        while True:
            combined = self.process_frame()
            if combined is None:
                continue

            cv.imshow("Stereo Tuning", combined)
            key = cv.waitKey(1)
            if key == 27:
                break


# ========================================
# Main
# ========================================
def main():
    left_cam = CameraThread(LEFT_CAM_INDEX, "Left")
    right_cam = CameraThread(RIGHT_CAM_INDEX, "Right")
    left_cam.start()
    right_cam.start()

    gui = StereoTuningGUI(left_cam, right_cam)
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
