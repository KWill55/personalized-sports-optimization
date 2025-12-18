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
import time
from pathlib import Path
import yaml

# ========================================
# Config
# ========================================
config_path = Path(__file__).resolve().parents[4] / "project_config.yaml"
with open(config_path, "r") as f:
    project_cfg = yaml.safe_load(f)

# Camera Indices
LEFT_CAM_INDEX = project_cfg["left_cam_index"]
RIGHT_CAM_INDEX = project_cfg["right_cam_index"]
BALL_CAM_INDEX  = project_cfg["ball_cam_index"]

# Calibration Parameters
CHECKERBOARD   = tuple(project_cfg["inner_corners"])    # (columns, rows)
SQUARE_SIZE    = project_cfg["square_size_in"]          

# Video Parameters
STEREO_UNCROPPED_RES = tuple(project_cfg["uncropped_stereo_resolution"])
STEREO_CROP_SIZE     = tuple(project_cfg.get("cropped_stereo_resolution", (0, 0)))
PLAYER_TRACKING_FPS  = project_cfg["player_tracking_fps"]

BALL_UNCROPPED_RES = tuple(project_cfg["uncropped_ball_resolution"])
BALL_TRACKING_FPS  = project_cfg["ball_tracking_fps"]

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
    def __init__(self, index, name, resolution, fps):
        super().__init__(daemon=True)
        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,  resolution[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv.CAP_PROP_FPS,          fps)
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

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ========================================
# Stereo Tuning GUI
# ========================================
class StereoTuningGUI:
    def __init__(self, left_cam, right_cam, ball_cam):
        self.left_cam  = left_cam
        self.right_cam = right_cam
        self.ball_cam  = ball_cam

        # Checkerboard object points
        self.objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        self.modes = ["stereo", "ball", "left", "right"]
        self.mode_index = 0
        self.display_mode = self.modes[self.mode_index]
        self.window_title = "Camera Setup"
        self.last_canvas_shape = None
        self.prev_time = None
        self.curr_fps = 0.0
        self.exit_requested = False
        self.exit_button_rect = None

    def detect_and_overlay(self, frame):
        """Detects checkerboard corners for stereo cameras."""
        if frame is None:
            return None, False

        view = center_crop(frame, STEREO_CROP_SIZE) if STEREO_CROP_SIZE != (0, 0) else frame
        gray = cv.cvtColor(view, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            cv.drawChessboardCorners(view, CHECKERBOARD, corners, ret)

        return view, ret

    def draw_guides_per_view(self, canvas, view_info):
        """Draw alignment guides for all displayed views."""
        for info in view_info:
            x, y, w, h = info["roi"]
            self._draw_view_guides(canvas, x, y, w, h)

    def draw_view_info(self, canvas, view_info):
        """Overlay resolution/FPS text for each displayed view."""
        for info in view_info:
            x, y, _, _ = info["roi"]
            label = info["label"]
            src_w, src_h = info["shape"]
            fps = info["fps"]
            text = f"{label}: {src_w}x{src_h} @ {fps} FPS"
            cv.putText(canvas, text, (x + 10, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_exit_button(self, canvas):
        """Render an Exit button and store its hitbox."""
        h, w = canvas.shape[:2]
        btn_w, btn_h = 140, 40
        margin = 20
        x0 = w - btn_w - margin
        y0 = margin
        self.exit_button_rect = (x0, y0, btn_w, btn_h)
        cv.rectangle(canvas, (x0, y0), (x0 + btn_w, y0 + btn_h), (50, 50, 50), -1)
        cv.rectangle(canvas, (x0, y0), (x0 + btn_w, y0 + btn_h), (200, 200, 200), 2)
        text = "EXIT"
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x0 + (btn_w - text_size[0]) // 2
        text_y = y0 + (btn_h + text_size[1]) // 2
        cv.putText(canvas, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    def compose_stereo_canvas(self, viewL, viewR, viewB):
        """Place stereo feeds side-by-side with ball feed below."""
        target_h = max(viewL.shape[0], viewR.shape[0])
        left  = self._resize_to_height(viewL, target_h)
        right = self._resize_to_height(viewR, target_h)

        col_w = max(left.shape[1], right.shape[1])
        canvas = np.zeros((target_h, col_w * 2, 3), dtype=np.uint8)

        left_roi = self._place_centered(canvas, left, 0, col_w)
        right_roi = self._place_centered(canvas, right, col_w, col_w)

        total_width = canvas.shape[1]
        ball = viewB.copy()
        if ball.shape[1] < total_width:
            pad_total = total_width - ball.shape[1]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            ball = cv.copyMakeBorder(ball, 0, 0, pad_left, pad_right, cv.BORDER_CONSTANT, (0, 0, 0))
        elif ball.shape[1] > total_width:
            start_x = (ball.shape[1] - total_width) // 2
            ball = ball[:, start_x:start_x + total_width]
        combined = np.zeros((target_h + ball.shape[0], total_width, 3), dtype=np.uint8)
        combined[0:target_h, :] = canvas
        combined[target_h:target_h + ball.shape[0], :] = ball

        ball_roi = (0, target_h, ball.shape[1], ball.shape[0])

        self.last_canvas_shape = combined.shape[:2]
        left_shape = (viewL.shape[1], viewL.shape[0])
        right_shape = (viewR.shape[1], viewR.shape[0])
        ball_shape = (viewB.shape[1], viewB.shape[0])
        view_info = [
            {"label": "Left",  "roi": left_roi,  "shape": left_shape,  "fps": PLAYER_TRACKING_FPS},
            {"label": "Right", "roi": right_roi, "shape": right_shape, "fps": PLAYER_TRACKING_FPS},
            {"label": "Ball",  "roi": ball_roi,  "shape": ball_shape,  "fps": BALL_TRACKING_FPS},
        ]
        return combined, view_info

    def _resize_to_height(self, image, target_h):
        h, w = image.shape[:2]
        if h == target_h:
            return image
        scale = target_h / float(h)
        new_w = max(1, int(w * scale))
        return cv.resize(image, (new_w, target_h), interpolation=cv.INTER_LINEAR)

    def _place_centered(self, canvas, image, x_start, width):
        """Paste image centered within [x_start, x_start+width)."""
        offset_x = x_start + max(0, (width - image.shape[1]) // 2)
        canvas[0:image.shape[0], offset_x:offset_x + image.shape[1]] = image
        return (offset_x, 0, image.shape[1], image.shape[0])

    def compose_single_canvas(self, view, label, fps):
        target_shape = self.last_canvas_shape or view.shape[:2]
        canvas, roi = self._letterbox_to_shape(view, target_shape)
        self.last_canvas_shape = canvas.shape[:2]
        src_shape = (view.shape[1], view.shape[0])
        return canvas, [{"label": label, "roi": roi, "shape": src_shape, "fps": fps}]

    @staticmethod
    def _letterbox_to_shape(image, target_shape):
        target_h, target_w = target_shape
        h, w = image.shape[:2]
        if h == target_h and w == target_w:
            return image, (0, 0, w, h)

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        canvas = cv.copyMakeBorder(resized, top, bottom, left, right, cv.BORDER_CONSTANT, (0, 0, 0))
        return canvas, (left, top, new_w, new_h)

    def process_frame(self):
        L = self.left_cam.get_frame()
        R = self.right_cam.get_frame()
        B = self.ball_cam.get_frame()
        if L is None or R is None or B is None:
            return None

        viewL, _ = self.detect_and_overlay(L)
        viewR, _ = self.detect_and_overlay(R)

        now = time.time()
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                fps = 1.0 / dt
                if self.curr_fps == 0.0:
                    self.curr_fps = fps
                else:
                    self.curr_fps = (self.curr_fps * 0.8) + (fps * 0.2)
        self.prev_time = now

        return viewL, viewR, B

    def render_canvas(self, views):
        viewL, viewR, viewB = views
        mode = self.display_mode
        if mode == "stereo":
            canvas, view_info = self.compose_stereo_canvas(viewL, viewR, viewB)
        elif mode == "ball":
            canvas, view_info = self.compose_single_canvas(viewB, "Ball", BALL_TRACKING_FPS)
        elif mode == "left":
            canvas, view_info = self.compose_single_canvas(viewL, "Left", PLAYER_TRACKING_FPS)
        else:  # right
            canvas, view_info = self.compose_single_canvas(viewR, "Right", PLAYER_TRACKING_FPS)

        self.draw_guides_per_view(canvas, view_info)
        self.draw_view_info(canvas, view_info)
        self._draw_footer(canvas)
        return canvas

    def _draw_footer(self, canvas):
        mode_names = {
            "stereo": "Stereo",
            "ball": "Ball",
            "left": "Left Only",
            "right": "Right Only",
        }
        h, w = canvas.shape[:2]
        text = (
            f"Mode: {mode_names[self.display_mode]} | "
            f"Canvas: {w}x{h} @ {self.curr_fps:.1f} display FPS | Press TAB to switch"
        )
        cv.putText(canvas, text, (20, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        self.draw_exit_button(canvas)

    def cycle_mode(self):
        self.mode_index = (self.mode_index + 1) % len(self.modes)
        self.display_mode = self.modes[self.mode_index]

    def handle_mouse(self, event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if not self.exit_button_rect:
            return
        x0, y0, w, h = self.exit_button_rect
        if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
            self.exit_requested = True

    def run(self):
        print("[INFO] Press ESC to exit. Press TAB to switch views.")
        cv.namedWindow(self.window_title, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.window_title, self.handle_mouse)
        while True:
            views = self.process_frame()
            if views is not None:
                canvas = self.render_canvas(views)
                cv.imshow(self.window_title, canvas)

            key = cv.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (9, ord('t'), ord('T')):
                self.cycle_mode()
            if self.exit_requested:
                break

# ========================================
# Main
# ========================================
def main():
    left_cam  = CameraThread(LEFT_CAM_INDEX, "Left", STEREO_UNCROPPED_RES, PLAYER_TRACKING_FPS)
    right_cam = CameraThread(RIGHT_CAM_INDEX, "Right", STEREO_UNCROPPED_RES, PLAYER_TRACKING_FPS)
    ball_cam  = CameraThread(BALL_CAM_INDEX, "Ball", BALL_UNCROPPED_RES, BALL_TRACKING_FPS)

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
