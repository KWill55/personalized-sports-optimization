"""
Title: capture_cb_pairs.py

Description
    - Purpose: Capture synchronized stereo checkerboard image pairs for calibration.
    - Crops center 640x640 from both cameras
    - Displays combined 1280x640 window

Prerequisites:
    - TODO 

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

# ========================================
# Config
# ========================================
CAM_LEFT_INDEX = 1
CAM_RIGHT_INDEX = 2
CAM_RESOLUTION = (1280, 720) # 720p
CROP_SIZE = (640, 640) 
CHECKERBOARD = (5, 4)  # internal corners
ATHLETE = "kenny"
SESSION = "session_test"

FPS = 60 

# Paths
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION
calib_dir = session_dir / "calibration" / "calib_images"
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
        self.cap.set(cv.CAP_PROP_FPS, FPS)
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

            # Crop and combine
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

    def capture_pair(self, frameL, frameR, combined):
        # Check checkerboard detection in both cameras
        grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

        retL, _ = cv.findChessboardCorners(grayL, CHECKERBOARD, None)
        retR, _ = cv.findChessboardCorners(grayR, CHECKERBOARD, None)

        if retL and retR:
            fname = calib_dir / f"pair_{self.pair_id:02}.png"
            cv.imwrite(str(fname), combined)
            print(f"[INFO] Saved {fname.name}")
            self.show_status(f"Saved pair #{self.pair_id}", (0, 255, 0))
            self.pair_id += 1
        else:
            print("[WARNING] Checkerboard not detected in both cameras. Try again.")
            self.show_status("Checkerboard NOT detected!", (0, 0, 255))


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

