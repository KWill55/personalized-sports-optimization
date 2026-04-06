from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import Button, Label
from typing import Any, Dict

import cv2 as cv
from PIL import Image, ImageOps, ImageTk

from utils.io_utils import PROJECT_ROOT, load_config


class RecordingCaptureGui:
    """
    Class-based recording GUI for free throw capture.

    Captures 3 feeds:
      - left (player tracking)
      - right (player tracking)
      - ball (ball tracking)
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

        # camera indices
        self.camera_left_index = int(self.cfg["left_cam_index"])
        self.camera_right_index = int(self.cfg["right_cam_index"])
        self.camera_ball_index = int(self.cfg["ball_cam_index"])

        # session info
        self.athlete = self.cfg["athlete"]
        self.session = self.cfg["session"]

        # video settings
        self.stereo_uncropped_res = tuple(int(v) for v in self.cfg["uncropped_stereo_resolution"])
        self.stereo_crop_res = tuple(int(v) for v in self.cfg.get("cropped_stereo_resolution", self.stereo_uncropped_res))
        self.ball_uncropped_res = tuple(int(v) for v in self.cfg["uncropped_ball_resolution"])

        self.fps_left_right = float(self.cfg["player_tracking_fps"])
        self.fps_ball = float(self.cfg["ball_tracking_fps"])
        self.gui_refresh_ms = 30

        # visual settings
        self.border_colors = {"left": "red", "right": "blue", "ball": "green"}
        self.border_thickness = 5

        # naming
        self.pad_width = int(self.cfg.get("throw_number_width", 3))
        self.name_prefix = self.cfg.get("freethrow_name_prefix", "freethrow")

        crop_w, crop_h = map(int, self.stereo_crop_res)
        self.crop_size = (crop_w, crop_h)

        self.display_res = {
            "left": self.crop_size,
            "right": self.crop_size,
            "ball": self.ball_uncropped_res,
        }
        self.target_fps = {
            "left": self.fps_left_right,
            "right": self.fps_left_right,
            "ball": self.fps_ball,
        }

        # paths
        self.session_dir = PROJECT_ROOT / "data" / self.athlete / self.session
        self.video_dirs = {
            "left": self.session_dir / "videos" / "player_tracking" / "raw" / "left",
            "right": self.session_dir / "videos" / "player_tracking" / "raw" / "right",
            "ball": self.session_dir / "videos" / "ball_tracking" / "raw",
        }
        for path in self.video_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        # shared state
        self.frames = {"left": None, "right": None, "ball": None}
        self.frame_locks = {k: threading.Lock() for k in self.frames}

        self.recording = False
        self.writers: Dict[str, cv.VideoWriter] = {}
        self.writers_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.throw_count = 0
        self.recorded_total = 0
        self.frame_counters = {"left": 0, "right": 0, "ball": 0}
        self.start_time: float | None = None
        self.refresh_recorded_total()

    @staticmethod
    def _print_cam_init(name: str, cap: cv.VideoCapture) -> None:
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        print(f"[{name.upper()}] Initialized: {w}x{h}, FPS: {fps:.5f}")

    @staticmethod
    def _resize_to_fit(frame, max_w: int, max_h: int):
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame, (w, h)

        scale = min(max_w / w, max_h / h)
        scale = min(scale, 1.0)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if (new_w, new_h) == (w, h):
            return frame, (w, h)

        resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
        return resized, (new_w, new_h)

    def _capture_camera(self, name: str, index: int, crop: bool = False) -> None:
        cap = cv.VideoCapture(index)

        target_fps = self.fps_left_right if name in ["left", "right"] else self.fps_ball
        if name in ["left", "right"]:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.stereo_uncropped_res[0])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.stereo_uncropped_res[1])
        else:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.ball_uncropped_res[0])
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.ball_uncropped_res[1])

        cap.set(cv.CAP_PROP_FPS, target_fps)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        try:
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        self._print_cam_init(name, cap)

        crop_x0 = crop_x1 = crop_y0 = crop_y1 = 0
        crop_resize_needed = False
        crop_last_hw = None
        crop_w, crop_h = self.crop_size

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.002)
                continue

            if crop:
                if crop_last_hw is None:
                    h, w = frame.shape[:2]
                    crop_w_eff = crop_w if crop_w <= w else w
                    crop_h_eff = crop_h if crop_h <= h else h
                    crop_x0 = (w - crop_w_eff) // 2
                    crop_y0 = (h - crop_h_eff) // 2
                    crop_x1 = crop_x0 + crop_w_eff
                    crop_y1 = crop_y0 + crop_h_eff
                    crop_resize_needed = (crop_w_eff != crop_w) or (crop_h_eff != crop_h)
                    crop_last_hw = (h, w)

                frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]
                if crop_resize_needed:
                    frame = cv.resize(frame, (crop_w, crop_h), interpolation=cv.INTER_LINEAR)

            with self.frame_locks[name]:
                self.frames[name] = frame

            if self.recording:
                self.frame_counters[name] += 1

            time.sleep(0.001)

        cap.release()

    def _write_frames(self) -> None:
        intervals = {
            "left": 1.0 / self.fps_left_right,
            "right": 1.0 / self.fps_left_right,
            "ball": 1.0 / self.fps_ball,
        }
        last_write_time = {name: 0.0 for name in self.frames}

        while not self.stop_event.is_set():
            if self.recording:
                now = time.time()
                with self.writers_lock:
                    local_writers = list(self.writers.items())

                for name, writer in local_writers:
                    if now - last_write_time.get(name, 0.0) >= intervals[name]:
                        with self.frame_locks[name]:
                            frame = self.frames[name]
                        if frame is not None:
                            try:
                                writer.write(frame)
                            except Exception as exc:
                                print(f"[ERROR] write({name}) failed: {exc}")
                            last_write_time[name] = now

            time.sleep(0.001)

    def _get_next_throw_number(self) -> int:
        max_count = 0
        for path in self.video_dirs.values():
            pattern = f"{self.name_prefix}*.avi"
            prefix_len = len(self.name_prefix)
            for file in path.glob(pattern):
                try:
                    num = int(file.stem[prefix_len:])
                    max_count = max(max_count, num)
                except ValueError:
                    continue
        return max_count + 1

    def refresh_recorded_total(self) -> int:
        self.recorded_total = max(0, self._get_next_throw_number() - 1)
        return self.recorded_total

    def start_recording(self, dims: Dict[str, tuple[int, int]]) -> None:
        self.refresh_recorded_total()
        self.throw_count = self._get_next_throw_number()
        label = f"{self.name_prefix}{self.throw_count:0{self.pad_width}d}"
        print(f"[INFO] Starting {label}")

        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        with self.writers_lock:
            for name, size in dims.items():
                filepath = self.video_dirs[name] / f"{label}.avi"
                fps = self.fps_left_right if name in ["left", "right"] else self.fps_ball
                self.writers[name] = cv.VideoWriter(str(filepath), fourcc, fps, size)
                print(f"[INFO] Writing {name} to {filepath} @ {fps} FPS")

        self.frame_counters = {k: 0 for k in self.frame_counters}
        self.start_time = time.time()
        self.recording = True

    def stop_recording(self) -> None:
        duration = max(0.0, time.time() - (self.start_time or time.time()))
        print(f"[INFO] Stopping recording after {duration:.1f}s")

        for name, count in self.frame_counters.items():
            actual_fps = (count / duration) if duration > 0 else 0.0
            print(f"[RESULT] {name.upper()} Actual FPS: {actual_fps:.1f}")
            if name in ["left", "right"] and actual_fps < self.fps_left_right * 0.8:
                print(f"[WARNING] {name.upper()} is below target FPS ({actual_fps:.1f} vs {self.fps_left_right})")
            if name == "ball" and actual_fps < self.fps_ball * 0.8:
                print(f"[WARNING] BALL is below expected FPS ({actual_fps:.1f} vs {self.fps_ball})")

        self.recording = False
        time.sleep(0.02)

        with self.writers_lock:
            for writer in self.writers.values():
                try:
                    writer.release()
                except Exception as exc:
                    print(f"[ERROR] release() failed: {exc}")
            self.writers.clear()

        print("[INFO] Writers closed.")
        total = self.refresh_recorded_total()
        print(f"[INFO] Total recorded free throws: {total}")

    def _camera_info_text(self, name: str) -> str:
        w, h = self.display_res[name]
        fps = self.target_fps[name]
        return f"{name.capitalize()} Camera - {w}x{h} @ {fps:.0f} FPS"

    def _get_display_boxes(self, root: tk.Tk) -> Dict[str, tuple[int, int]]:
        width = max(800, root.winfo_width())
        height = max(600, root.winfo_height())

        pad_w = 120
        pad_h_top = 140
        pad_h_bottom = 240

        top_box_w = max(200, (width - pad_w) // 2)
        top_box_h = max(200, (height - pad_h_top) // 2)
        bottom_box_h = max(200, (height - pad_h_bottom) // 2)
        ball_box_w = max(260, int(width * 0.55) - 60)

        return {
            "left": (top_box_w, top_box_h),
            "right": (top_box_w, top_box_h),
            "ball": (ball_box_w, bottom_box_h),
        }

    def run(self) -> None:
        threading.Thread(target=self._capture_camera, args=("left", self.camera_left_index, True), daemon=True).start()
        threading.Thread(target=self._capture_camera, args=("right", self.camera_right_index, True), daemon=True).start()
        threading.Thread(target=self._capture_camera, args=("ball", self.camera_ball_index, False), daemon=True).start()
        threading.Thread(target=self._write_frames, daemon=True).start()

        root = tk.Tk()
        app = _FreeThrowRecorderApp(root, self)
        root.mainloop()


class _FreeThrowRecorderApp:
    def __init__(self, root: tk.Tk, recorder: RecordingCaptureGui) -> None:
        self.root = root
        self.recorder = recorder

        self.root.title("Free Throw Recorder")
        self.root.geometry("1800x1200")
        self.root.minsize(1200, 800)

        self.status_text = tk.StringVar(value="Status: Idle")
        self.count_text = tk.StringVar(value="")
        self.labels: Dict[str, Label] = {}
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        self._setup_gui()
        self._refresh_count_text()
        self._update_gui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_gui(self) -> None:
        frame_top = tk.Frame(self.root)
        frame_top.pack(fill="both", expand=True)

        self._create_camera_view(frame_top, "left", side=tk.LEFT, padx=5, pady=5)
        self._create_camera_view(frame_top, "right", side=tk.LEFT, padx=5, pady=5)

        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(fill="both", expand=True, pady=10)

        button_frame = tk.Frame(frame_bottom)
        button_frame.pack(side=tk.LEFT, padx=10, pady=5)

        Button(
            button_frame,
            text="Start/Stop Recording",
            command=self._toggle_recording,
            height=2,
            width=20,
        ).pack()
        Label(button_frame, textvariable=self.status_text, font=("Helvetica", 14)).pack(pady=10)
        Label(button_frame, textvariable=self.count_text, font=("Helvetica", 14, "bold")).pack(pady=4)

        legend_frame = tk.Frame(frame_bottom)
        legend_frame.pack(side=tk.LEFT, padx=20, pady=5)

        for name, color in self.recorder.border_colors.items():
            lf = tk.Frame(legend_frame)
            lf.pack(anchor="w", pady=5)
            tk.Label(lf, width=2, height=1, bg=color).pack(side=tk.LEFT)
            tk.Label(lf, text=f"{name.capitalize()} Camera").pack(side=tk.LEFT, padx=5)

        self._create_camera_view(frame_bottom, "ball", side=tk.LEFT, padx=10, pady=5)

    def _create_camera_view(self, parent, name: str, **pack_kwargs) -> None:
        wrapper = tk.Frame(parent)
        info_label = tk.Label(wrapper, text=self.recorder._camera_info_text(name), font=("Helvetica", 12))
        info_label.pack()

        img_label = Label(wrapper)
        img_label.pack()

        wrapper.pack(fill="both", expand=True, **pack_kwargs)
        self.labels[name] = img_label

    def _toggle_recording(self) -> None:
        if not self.recorder.recording:
            dims = {
                "left": self.recorder.crop_size,
                "right": self.recorder.crop_size,
                "ball": self.recorder.ball_uncropped_res,
            }
            self.recorder.start_recording(dims)
            label = f"{self.recorder.name_prefix}{self.recorder.throw_count:0{self.recorder.pad_width}d}"
            self.status_text.set(f"Recording {label}")
            self._refresh_count_text()
            return

        self.recorder.stop_recording()
        self.status_text.set("Status: Idle")
        self._refresh_count_text()

    def _refresh_count_text(self) -> None:
        total = self.recorder.recorded_total
        self.count_text.set(f"Recorded Free Throws: {total}")

    def _update_gui(self) -> None:
        boxes = self.recorder._get_display_boxes(self.root)

        for name in self.recorder.frames:
            with self.recorder.frame_locks[name]:
                frame = self.recorder.frames[name]

            if frame is None:
                continue

            max_w, max_h = boxes[name]
            frame_display, _ = self.recorder._resize_to_fit(frame, max_w, max_h)

            frame_rgb = cv.cvtColor(frame_display, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = ImageOps.expand(
                pil_img,
                border=self.recorder.border_thickness,
                fill=self.recorder.border_colors[name],
            )
            img = ImageTk.PhotoImage(pil_img)

            self.images[name] = img
            self.labels[name].configure(image=img)

        if not self.recorder.stop_event.is_set():
            self.root.after(self.recorder.gui_refresh_ms, self._update_gui)

    def _on_close(self) -> None:
        if self.recorder.recording:
            self.recorder.stop_recording()

        self.recorder.stop_event.set()
        time.sleep(0.03)
        self.root.destroy()


if __name__ == "__main__":
    RecordingCaptureGui().run()
