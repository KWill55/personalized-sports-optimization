from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

import cv2 as cv
import numpy as np

from utils.io_utils import PROJECT_ROOT, load_config


"""
Responsible for capturing mono/stereo checkerboard images for calibration 
"""

def crop_center(frame: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
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


def quick_square_px(pts: np.ndarray, cols: int, rows: int) -> float:
    pts = pts.reshape(-1, 2)
    tl = pts[0]
    tr = pts[cols - 1]
    bl = pts[(rows - 1) * cols]
    w = np.linalg.norm(tr - tl) / (cols - 1)
    h = np.linalg.norm(bl - tl) / (rows - 1)
    return float(min(w, h))


class CameraThread(threading.Thread):
    def __init__(self, index: int, name: str, resolution: tuple[int, int], fps: int) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.running = True
        self.frame: np.ndarray | None = None

        self.cap = cv.VideoCapture(index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv.CAP_PROP_FPS, fps)

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
        self.cap.release()

    def stop(self) -> None:
        self.running = False

class CalibrationCaptureGui:
    """
    Modes:
      - run_mono_capture(): mono_left / mono_right captures with TAB switching
      - run_stereo_pair_capture(): side-by-side pair captures
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

        self.cam_resolution = tuple(self.cfg["uncropped_stereo_resolution"])
        self.crop_resolution = tuple(self.cfg["cropped_stereo_resolution"])
        self.player_fps = int(self.cfg["player_tracking_fps"])

        self.left_cam_index = int(self.cfg["left_cam_index"])
        self.right_cam_index = int(self.cfg["right_cam_index"])

        self.checkerboard = tuple(self.cfg["inner_corners"])
        self.min_square_px = float(self.cfg.get("min_square_px", 5.0))

        session_dir = PROJECT_ROOT / "data" / self.cfg["athlete"] / self.cfg["session"]
        self.mono_left_dir = session_dir / "calibration" / "calib_images" / "mono_left"
        self.mono_right_dir = session_dir / "calibration" / "calib_images" / "mono_right"
        self.pairs_dir = session_dir / "calibration" / "calib_images" / "pairs"

        self.mono_left_dir.mkdir(parents=True, exist_ok=True)
        self.mono_right_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _next_id(out_dir: Path, prefix: str) -> int:
        existing = sorted(out_dir.glob(f"{prefix}_*.png"))
        return len(existing) + 1

    def _start_cameras(self) -> tuple[CameraThread, CameraThread]:
        left_cam = CameraThread(self.left_cam_index, "left", self.cam_resolution, self.player_fps)
        right_cam = CameraThread(self.right_cam_index, "right", self.cam_resolution, self.player_fps)
        left_cam.start()
        right_cam.start()
        return left_cam, right_cam

    @staticmethod
    def _stop_cameras(left_cam: CameraThread, right_cam: CameraThread) -> None:
        left_cam.stop()
        right_cam.stop()
        left_cam.join()
        right_cam.join()
        cv.destroyAllWindows()

    def run_mono_capture(self) -> None:
        left_cam, right_cam = self._start_cameras()
        active = "LEFT"
        id_left = self._next_id(self.mono_left_dir, "left")
        id_right = self._next_id(self.mono_right_dir, "right")
        status_text = ""
        status_color = (255, 255, 255)
        status_time = 0.0

        def show_status(text: str, color: tuple[int, int, int]) -> None:
            nonlocal status_text, status_color, status_time
            status_text = text
            status_color = color
            status_time = time.time()

        print("[INFO] Controls: TAB = switch camera, SPACE = manual capture, ESC = quit")
        print("[INFO] Saving to:")
        print(f"       LEFT  -> {self.mono_left_dir}")
        print(f"       RIGHT -> {self.mono_right_dir}")

        try:
            while True:
                src = left_cam if active == "LEFT" else right_cam
                if src.frame is None:
                    cv.waitKey(1)
                    continue

                view = crop_center(src.frame, self.crop_resolution)

                next_name = f"left_{id_left:02}.png" if active == "LEFT" else f"right_{id_right:02}.png"
                header = (
                    f"Active: {active} | Next: {next_name} | "
                    f"MIN_SQUARE_PX={self.min_square_px:.0f} | TAB switch | SPACE capture | ESC quit"
                )
                cv.putText(view, header, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                if time.time() - status_time < 1.5:
                    cv.putText(view, status_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv.imshow("Mono Checkerboard Capture", view)
                key = cv.waitKey(1) & 0xFF

                if key == 27:
                    break
                if key == 9:
                    active = "RIGHT" if active == "LEFT" else "LEFT"
                    show_status(f"Switched to {active}", (0, 255, 255))
                elif key == 32:
                    self._capture_mono_frame(view, active, id_left, id_right, show_status)
                    if active == "LEFT":
                        id_left += 1
                    else:
                        id_right += 1
        finally:
            self._stop_cameras(left_cam, right_cam)

    def _capture_mono_frame(
        self,
        frame: np.ndarray,
        active_side: str,
        id_left: int,
        id_right: int,
        show_status,
    ) -> None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, pts = cv.findChessboardCornersSB(gray, self.checkerboard, flags=cv.CALIB_CB_EXHAUSTIVE)

        if not ret:
            msg = "Checkerboard NOT detected."
            print(f"[WARNING] {active_side}: {msg}")
            show_status(msg, (0, 0, 255))
            return

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(gray, pts, (11, 11), (-1, -1), criteria)

        cols, rows = self.checkerboard
        square_px = quick_square_px(pts, cols, rows)
        if square_px < self.min_square_px:
            msg = f"Board too small ({square_px:.1f}px). Move closer."
            print(f"[WARNING] {active_side}: {msg}")
            show_status(msg, (0, 0, 255))
            return

        if active_side == "LEFT":
            fname = self.mono_left_dir / f"left_{id_left:02}.png"
        else:
            fname = self.mono_right_dir / f"right_{id_right:02}.png"

        ok = cv.imwrite(str(fname), frame)
        if ok:
            msg = f"Saved {fname.name} (square ~ {square_px:.1f}px)"
            print(f"[INFO] {active_side}: {msg}")
            show_status(msg, (0, 255, 0))
        else:
            msg = "Failed to save image."
            print(f"[ERROR] {active_side}: {msg}")
            show_status(msg, (0, 0, 255))

    def run_stereo_pair_capture(self) -> None:
        left_cam, right_cam = self._start_cameras()
        pair_id = len(list(self.pairs_dir.glob("pair_*.png"))) + 1
        status_text = ""
        status_color = (255, 255, 255)
        status_time = 0.0

        def show_status(text: str, color: tuple[int, int, int]) -> None:
            nonlocal status_text, status_color, status_time
            status_text = text
            status_color = color
            status_time = time.time()

        print("[INFO] Press SPACE to capture a pair (only saves if checkerboard detected). ESC to exit.")

        try:
            while True:
                if left_cam.frame is None or right_cam.frame is None:
                    cv.waitKey(1)
                    continue

                frame_left = crop_center(left_cam.frame, self.crop_resolution)
                frame_right = crop_center(right_cam.frame, self.crop_resolution)
                combined = cv.hconcat([frame_left, frame_right])

                cv.putText(combined, f"Pair #{pair_id}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv.putText(combined, "SPACE: Capture | ESC: Quit", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if time.time() - status_time < 1.5:
                    cv.putText(combined, status_text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv.imshow("Capture Calibration Pairs", combined)
                key = cv.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == 32:
                    saved = self._capture_pair(frame_left, frame_right, combined, pair_id, show_status)
                    if saved:
                        pair_id += 1
        finally:
            self._stop_cameras(left_cam, right_cam)

    def _capture_pair(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        combined: np.ndarray,
        pair_id: int,
        show_status,
    ) -> bool:
        gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

        ret_left, pts_left = cv.findChessboardCornersSB(gray_left, self.checkerboard, flags=cv.CALIB_CB_EXHAUSTIVE)
        ret_right, pts_right = cv.findChessboardCornersSB(gray_right, self.checkerboard, flags=cv.CALIB_CB_EXHAUSTIVE)

        if not (ret_left and ret_right):
            msg = "Checkerboard NOT detected in both."
            print(f"[WARNING] {msg}")
            show_status(msg, (0, 0, 255))
            return False

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(gray_left, pts_left, (11, 11), (-1, -1), criteria)
        cv.cornerSubPix(gray_right, pts_right, (11, 11), (-1, -1), criteria)

        cols, rows = self.checkerboard
        sq_left = quick_square_px(pts_left, cols, rows)
        sq_right = quick_square_px(pts_right, cols, rows)

        if sq_left < self.min_square_px or sq_right < self.min_square_px:
            msg = f"Board too small (L={sq_left:.1f}px, R={sq_right:.1f}px). Move closer."
            print(f"[WARNING] {msg}")
            show_status(msg, (0, 0, 255))
            return False

        fname = self.pairs_dir / f"pair_{pair_id:02}.png"
        ok = cv.imwrite(str(fname), combined)
        if ok:
            print(f"[INFO] Saved {fname.name} (square px L={sq_left:.1f}, R={sq_right:.1f})")
            show_status(f"Saved pair #{pair_id}", (0, 255, 0))
            return True

        show_status("Failed to save image.", (0, 0, 255))
        return False


if __name__ == "__main__":
    gui = CalibrationCaptureGui()
    gui.run_stereo_pair_capture()
