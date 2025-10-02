#!/usr/bin/env python3
"""
Real-Time Display layout (MediaPipe 2D Pose + 5s record-with-overlay + playback controls)

Row -1: Title bar
Row  0: [Left Camera] [3D Skeleton (placeholder)] [Right Camera]
Row  1: [Angles]  |  [Navigation (live)  OR  Playback Controls (during review)]
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import yaml
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
import time

# === MediaPipe Pose (optional) ===
_HAS_MP = True
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
except Exception as _e:
    _HAS_MP = False
    mp_pose = mp_draw = mp_style = None
    print("Note: mediapipe not available; pose overlays disabled:", _e)

# Big/bright drawing specs
if _HAS_MP:
    POSE_LMK_SPEC = mp_draw.DrawingSpec(thickness=4, circle_radius=4)
    POSE_CON_SPEC = mp_draw.DrawingSpec(thickness=3, circle_radius=0)
else:
    POSE_LMK_SPEC = POSE_CON_SPEC = None

# =========================
# Theme / constants
# =========================
BLUE = "#648AB6"
GRAY = "#6D737A"
BG_COLOR = BLUE
TRIM_COLOR = GRAY
BORDER = "#555"
PAD = 10
TITLE_FONT_SIZE = 28
SMALL_TITLE = 16

REC_BORDER = "#AA2B2B"  # red frame during recording
HUD_OK = "#00ff00"
HUD_BAD = "#ff4444"
HUD_ACCENT = "#00ff88"

# =========================
# Repo root + config
# =========================
def find_repo_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / "project_config.yaml").exists():
            return p
    return start

ROOT = find_repo_root()
CFG_PATH = ROOT / "project_config.yaml"
cfg = {}
if CFG_PATH.exists():
    try:
        with open(CFG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: failed to parse {CFG_PATH}: {e}")

# legacy numeric strings
for k in ("camera_left_source", "camera_right_source"):
    v = cfg.get(k)
    if isinstance(v, str) and v.isdigit():
        cfg[k] = int(v)

LEFT_SRC  = cfg.get("left_cam_index", 0)
RIGHT_SRC = cfg.get("right_cam_index", 1)
UI_FPS    = int(cfg.get("player_tracking_fps", 15)) if cfg.get("player_tracking_fps") is not None else 15

# =========================
# Minimal framed box helper
# =========================
class SetupGui:
    def __init__(self, border=BORDER, BG_COLOR=BG_COLOR):
        self.BORDER = border
        self.BG_COLOR = BG_COLOR

    def framed_box(self, parent, title, font_size, subtitle: str | None = None):
        outer = tk.Frame(parent, bg=self.BORDER, highlightthickness=0)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text=title, bg=self.BG_COLOR, font=("Helvetica", font_size, "bold"),
                 anchor="n", justify="center").pack(fill="x", pady=(4, 2))
        if subtitle:
            tk.Label(inner, text=subtitle, bg=self.BG_COLOR, font=("Helvetica", 12),
                     anchor="n", justify="center").pack(fill="x", pady=(0, 6))
        outer.inner = inner
        return outer

# =========================
# Video player widget
# (Live / Recording / Playback)
# =========================
class VideoPlayerWidget(tk.Frame):
    def __init__(self, parent, border="#555", color=TRIM_COLOR, fps=UI_FPS,
                 crop_center_square: bool = False, square_size: int = 640, title="Camera"):
        super().__init__(parent, bg=border, highlightbackground=border, highlightthickness=2)
        self.border, self.color = border, color
        self.cap = None
        self.playing = False
        self.photo = None
        self._fps_ms = max(1, int(1000 / max(fps, 1)))
        self._last_frame_rgb = None
        self._source = None

        # display options
        self.crop_center_square = bool(crop_center_square)
        self.square_size = int(max(1, square_size))
        self.title = title

        # Pose state
        self.draw_pose = True
        self._pose_found = False
        self._last_norm_landmarks = None
        self._dbg_cnt = 0
        self._pose = None
        if _HAS_MP:
            self._pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
            )

        # Modes / state
        self.mode = "live"         # "live" | "recording" | "playback"
        self._countdown_sec = 0    # integer for 3/2/1
        self._countdown_text = None  # "3"/"2"/"1"/"GO" during pre-roll
        self._record_left_ms = 0
        self._recorded_frames = [] # list[np.ndarray RGB]
        self._play_idx = 0
        self._play_speed = 1.0
        self._play_fps = 30.0

        # UI structure
        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.holder = tk.Frame(inner, bg="black", padx=1, pady=1)
        self.holder.pack(fill="both", expand=True, padx=1, pady=1)
        self.holder.pack_propagate(False)

        self.canvas = tk.Canvas(self.holder, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # Keep visual box square-ish
        self.holder.bind("<Configure>", lambda e: self._enforce_square_holder())

        self.info = tk.StringVar(value="No video loaded")
        tk.Label(inner, textvariable=self.info, bg=self.color, font=("Helvetica", 11)).pack(fill="x")

    # public toggles
    def toggle_draw_pose(self): self.draw_pose = not self.draw_pose
    def toggle_crop(self): self.crop_center_square = not self.crop_center_square

    # ------------ camera plumbing ------------
    def load(self, source):
        self.release()
        self._source = source
        try:
            self.cap = cv2.VideoCapture(str(source)) if isinstance(source, (str, Path)) else cv2.VideoCapture(int(source))
        except Exception:
            self.cap = cv2.VideoCapture(str(source))
        if not self.cap or not self.cap.isOpened():
            backend = int(self.cap.get(cv2.CAP_PROP_BACKEND)) if self.cap else -1
            print(f"[LOAD FAIL] source={source!r} cap={self.cap} backend={backend}")
            self.info.set(f"Failed to open: {source}")
            self.cap = None
            return

        # tame the cam (best-effort; may be ignored by backend)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        self._fps_ms = max(1, int(1000 / fps))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.info.set(f"{self.title}: {source}  {w}x{h} @ {fps:.1f} fps")
        self.play()

    def play(self):
        if not self.cap: return
        if not self.playing:
            self.playing = True
            self.after(0, self._loop)

    def pause(self): self.playing = False

    def toggle(self):
        self.pause() if self.playing else self.play()

    def release(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._pose is not None:
            try: self._pose.close()
            except Exception: pass
            self._pose = None
        self._last_frame_rgb = None
        self.photo = None
        self.canvas.delete("all")

    def _loop(self):
        if not self.playing or not self.cap: return

        if self.mode == "live":
            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self._source, (str, Path)):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.after(self._fps_ms, self._loop); return
            self._display_live(frame)
            self.after(self._fps_ms, self._loop)

        elif self.mode == "recording":
            # timed recorder drives frames; just keep HUD refreshing cadence
            self.after(self._fps_ms, self._loop)

        elif self.mode == "playback":
            if self._recorded_frames:
                frame = self._recorded_frames[self._play_idx]
                self._last_frame_rgb = frame
                self._draw_rgb(frame, playback=True)
                self._play_idx = (self._play_idx + 1) % len(self._recorded_frames)
                delay = int(1000 / max(1e-3, self._play_fps * self._play_speed))
                self.after(delay, self._loop)
            else:
                self.set_mode_live()

    def _on_resize(self, _evt):
        if self._last_frame_rgb is not None:
            self._draw_rgb(self._last_frame_rgb)

    def _enforce_square_holder(self):
        w = self.holder.winfo_width() or 1
        h = self.holder.winfo_height() or 1
        side = min(w, h)
        self.holder.configure(width=side, height=side)

    def _center_square_bgr(self, frame_bgr, size):
        h, w = frame_bgr.shape[:2]
        side = min(h, w, size)
        cy, cx = h // 2, w // 2
        half = side // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.shape[0] != size or crop.shape[1] != size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return crop

    # ------------ display paths ------------
    def _display_live(self, frame_bgr):
        if self.crop_center_square: frame_bgr = self._center_square_bgr(frame_bgr, self.square_size)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        if self.draw_pose and _HAS_MP and self._pose is not None:
            rgb.flags.writeable = False
            results = self._pose.process(rgb)
            rgb.flags.writeable = True
            if results and results.pose_landmarks:
                self._pose_found = True
                self._last_norm_landmarks = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
                mp_draw.draw_landmarks(rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=POSE_LMK_SPEC, connection_drawing_spec=POSE_CON_SPEC)
            else:
                self._pose_found = False
                self._last_norm_landmarks = None

        self._last_frame_rgb = rgb
        self._draw_rgb(rgb)

    def _draw_rgb(self, rgb, playback=False):
        # red border if recording
        self.configure(highlightbackground=REC_BORDER if self.mode == "recording" else self.border)

        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = Image.fromarray(cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        img_cx, img_cy = cw // 2, ch // 2
        self.canvas.create_image(img_cx, img_cy, anchor="center", image=self.photo)

        # HUD (mode + size)
        self.canvas.create_text(6, 6, anchor="nw",
                                text=f"{self.title}  {rgb.shape[1]}x{rgb.shape[0]}",
                                fill="white")

        if self.mode == "live":
            self.canvas.create_text(6, 26, anchor="nw",
                                    text=("● Pose" if getattr(self, "_pose_found", False) else "× No Pose"),
                                    fill=(HUD_OK if getattr(self, "_pose_found", False) else HUD_BAD))
        elif self.mode == "recording":
            self.canvas.create_text(cw//2, 24, anchor="n", text="● RECORDING", fill=HUD_BAD,
                                    font=("Helvetica", 20, "bold"))
            self.canvas.create_text(cw//2, 52, anchor="n",
                                    text=f"{max(0,int(self._record_left_ms/1000))}s left",
                                    fill="white", font=("Helvetica", 14, "bold"))
            # countdown / GO overlay
            if self._countdown_text:
                self.canvas.create_text(cw//2, ch//2, anchor="center",
                                        text=str(self._countdown_text),
                                        fill="white", font=("Helvetica", 72, "bold"))
        elif self.mode == "playback":
            self.canvas.create_text(6, 26, anchor="nw",
                                    text=f"▶ Playback  {self._play_speed:.1f}x",
                                    fill=HUD_ACCENT)

        # extra crisp canvas overlay during live (already drawn on image too)
        if self.mode == "live" and getattr(self, "_pose_found", False) and self._last_norm_landmarks:
            x0, y0 = img_cx - nw // 2, img_cy - nh // 2
            def to_px(x, y):
                x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
                return (x0 + int(x * nw), y0 + int(y * nh))
            if _HAS_MP:
                for a, b in mp_pose.POSE_CONNECTIONS:
                    if a < len(self._last_norm_landmarks) and b < len(self._last_norm_landmarks):
                        xa, ya, va = self._last_norm_landmarks[a]
                        xb, yb, vb = self._last_norm_landmarks[b]
                        if (va is None or va >= 0.01) and (vb is None or vb >= 0.01):
                            pa = to_px(xa, ya); pb = to_px(xb, yb)
                            self.canvas.create_line(pa[0], pa[1], pb[0], pb[1], width=3, fill=HUD_ACCENT)
            r = 5
            for (x, y, v) in self._last_norm_landmarks:
                if v is None or v >= 0.01:
                    px, py = to_px(x, y)
                    self.canvas.create_oval(px - r, py - r, px + r, py + r, outline="", fill=HUD_ACCENT)

    # ------------ modes ------------
    def set_mode_live(self):
        self.mode = "live"
        self._recorded_frames = []
        self._play_idx = 0
        self._play_speed = 1.0
        self._countdown_text = None
        self.configure(highlightbackground=self.border)
        if not self.playing: self.play()

    def start_countdown_and_record(self, duration_s=5, fps_hint=30.0):
        """3-2-1, then show 'GO' briefly, then record (with overlay), then switch to playback mode."""
        if not self.cap:
            print("[REC] No capture device"); return
        self._play_fps = float(fps_hint)
        self.mode = "recording"
        self._recorded_frames = []
        self._countdown_sec = 3
        self._countdown_text = "3"
        self._record_left_ms = int(duration_s * 1000)

        def _draw_latest():
            ret, frame = self.cap.read()
            if ret:
                if self.crop_center_square: frame = self._center_square_bgr(frame, self.square_size)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._last_frame_rgb = rgb
                self._draw_rgb(rgb)

        def _tick_countdown():
            if self._countdown_sec > 1:
                self._countdown_sec -= 1
                self._countdown_text = str(self._countdown_sec)
                _draw_latest()
                self.after(1000, _tick_countdown)
            elif self._countdown_sec == 1:
                self._countdown_sec = 0
                self._countdown_text = "GO"
                _draw_latest()
                self.after(500, _start_record)  # brief GO flash then start
            else:
                _start_record()

        def _start_record():
            self._countdown_text = None
            self._do_record_loop(duration_s)

        _draw_latest()
        self.after(1000, _tick_countdown)

    def _do_record_loop(self, duration_s: float):
        """Record for a fixed wall-clock duration, overlaying pose each frame, then set playback fps to the actual capture rate."""
        rec_start = time.time()
        stop_time = rec_start + float(duration_s)
        target_dt = 1.0 / max(1.0, self._play_fps)  # initial hint pacing

        def _step():
            now = time.time()
            # update remaining time for HUD
            self._record_left_ms = int(max(0.0, (stop_time - now)) * 1000)

            # done?
            if now >= stop_time:
                elapsed = max(0.001, now - rec_start)
                frames = max(1, len(self._recorded_frames))
                self._play_fps = max(1.0, frames / elapsed)  # true fps
                self.mode = "playback"
                self._play_idx = 0
                if not self.playing:
                    self.play()
                return

            # grab a frame
            ok, frame = self.cap.read()
            if not ok:
                elapsed = max(0.001, time.time() - rec_start)
                frames = max(1, len(self._recorded_frames))
                self._play_fps = max(1.0, frames / elapsed)
                self.mode = "playback"
                self._play_idx = 0
                if not self.playing:
                    self.play()
                return

            if self.crop_center_square: frame = self._center_square_bgr(frame, self.square_size)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            # pose overlay per recorded frame
            if self.draw_pose and _HAS_MP:
                with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                                  enable_segmentation=False, smooth_landmarks=True,
                                  min_detection_confidence=0.2, min_tracking_confidence=0.2) as tester:
                    res = tester.process(rgb)
                if res and res.pose_landmarks:
                    mp_draw.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=POSE_LMK_SPEC, connection_drawing_spec=POSE_CON_SPEC)

            # store and draw HUD
            self._recorded_frames.append(rgb)
            self._last_frame_rgb = rgb
            self._draw_rgb(rgb)

            # schedule next capture
            self.after(int(target_dt * 1000), _step)

        # ensure we’re in recording mode and running
        self.mode = "recording"
        if not self.playing:
            self.play()
        _step()

    # ------------ playback controls ------------
    def playback_replay(self):
        if self.mode == "playback" and self._recorded_frames:
            self._play_idx = 0
            if not self.playing:
                self.play()

    def playback_set_speed(self, s: float):
        self._play_speed = max(0.1, float(s))

    def playback_exit(self):
        self.set_mode_live()

    # ---------- Debug helpers ----------
    def snapshot_and_test_pose(self, title="Pose Self-Test"):
        if not self.cap:
            print("[DEBUG] No capture device"); return
        ok, frame = self.cap.read()
        if not ok: print("[DEBUG] Failed to grab snapshot"); return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if _HAS_MP:
            tester = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                                  enable_segmentation=False, smooth_landmarks=True,
                                  min_detection_confidence=0.2, min_tracking_confidence=0.2)
            res = tester.process(rgb); tester.close()
        else:
            res = None
        overlay = rgb.copy(); found = False
        if res and res.pose_landmarks:
            found = True
            mp_draw.draw_landmarks(overlay, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=POSE_LMK_SPEC, connection_drawing_spec=POSE_CON_SPEC)
        h, w = overlay.shape[:2]
        img = Image.fromarray(overlay)
        photo = ImageTk.PhotoImage(img)
        top = tk.Toplevel(self); top.title(f"{title} — {'FOUND' if found else 'NONE'}  ({w}x{h})")
        lbl = tk.Label(top, image=photo); lbl.image = photo; lbl.pack()
        tk.Label(top, text=("FOUND landmarks" if found else "NO landmarks"),
                 fg=("#00aa00" if found else "#aa0000")).pack(pady=6)

# =========================
# Static Angles panel (placeholders)
# =========================
class AnglesPanel(tk.Frame):
    PAIRS = [
        ("Shoulder Flex", "Shoulder Flex"),
        ("Elbow Flex",    "Elbow Flex"),
        ("Hip Flex",      "Hip Flex"),
        ("Knee Flex",     "Knee Flex"),
        ("Ankle Flex",    "Ankle Flex"),
    ]
    def __init__(self, parent):
        super().__init__(parent, bg=BORDER)
        inner = tk.Frame(self, bg=TRIM_COLOR); inner.pack(fill="both", expand=True, padx=2, pady=2)
        table = tk.Frame(inner, bg=TRIM_COLOR); table.pack(fill="both", expand=True, padx=10, pady=8)
        table.columnconfigure(0, weight=3); table.columnconfigure(1, weight=1); table.columnconfigure(2, weight=1)
        tk.Label(table, text="Measurement", bg=TRIM_COLOR, anchor="w",
                 font=("Helvetica", 24, "bold")).grid(row=0, column=0, sticky="nsew", pady=(0,6))
        tk.Label(table, text="L", bg=TRIM_COLOR, anchor="e",
                 font=("Helvetica", 24, "bold")).grid(row=0, column=1, sticky="nsew", pady=(0,6))
        tk.Label(table, text="R", bg=TRIM_COLOR, anchor="e",
                 font=("Helvetica", 24, "bold")).grid(row=0, column=2, sticky="nsew", pady=(0,6))
        self._cells = {}
        for i, (label, key) in enumerate(self.PAIRS, start=1):
            tk.Label(table, text=label, bg=TRIM_COLOR, anchor="w",
                     font=("Helvetica", 18)).grid(row=i, column=0, sticky="nsew", pady=3)
            l = tk.Label(table, text="???", bg=TRIM_COLOR, anchor="e", font=("Helvetica", 18, "bold"))
            r = tk.Label(table, text="???", bg=TRIM_COLOR, anchor="e", font=("Helvetica", 18, "bold"))
            l.grid(row=i, column=1, sticky="nsew", pady=3); r.grid(row=i, column=2, sticky="nsew", pady=3)
            self._cells[key] = {"L": l, "R": r}
    def set_pair(self, key: str, left: str, right: str):
        if key in self._cells:
            self._cells[key]["L"].configure(text=left)
            self._cells[key]["R"].configure(text=right)

# =========================
# Navigation (live)
# =========================
class NavigationPanel(tk.Frame):
    def __init__(self, parent, on_toggle_pose=None, on_toggle_crop=None,
                 on_self_test=None, on_record=None):
        super().__init__(parent, bg=BORDER)
        inner = tk.Frame(self, bg=TRIM_COLOR); inner.pack(fill="both", expand=True, padx=2, pady=2)
        btns = tk.Frame(inner, bg=TRIM_COLOR); btns.pack(padx=8, pady=8, fill="x")
        for i in range(3): btns.columnconfigure(i, weight=1, uniform="btns")
        tk.Button(btns, text="Toggle Real-Time / Record Mode").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=4)
        tk.Button(btns, text="Toggle Draw MediaPipe Pose", command=(on_toggle_pose or (lambda: None))
                  ).grid(row=1, column=0, columnspan=3, sticky="nsew", pady=4)
        tk.Button(btns, text="Toggle Crop", command=(on_toggle_crop or (lambda: None))
                  ).grid(row=2, column=1, sticky="nsew", pady=4)
        tk.Button(btns, text="Pose Self-Test (snapshot)", command=(on_self_test or (lambda: None))
                  ).grid(row=2, column=2, sticky="nsew", pady=4)
        tk.Button(btns, text="Record 5s (overlay) — Start", command=(on_record or (lambda: None))
                  ).grid(row=3, column=0, columnspan=3, sticky="nsew", pady=6)

# =========================
# Playback Controls (review)
# =========================
class PlaybackControls(tk.Frame):
    def __init__(self, parent, on_replay, on_speed, on_exit):
        super().__init__(parent, bg=BORDER)
        inner = tk.Frame(self, bg=TRIM_COLOR); inner.pack(fill="both", expand=True, padx=2, pady=2)
        row = tk.Frame(inner, bg=TRIM_COLOR); row.pack(padx=8, pady=8, fill="x")
        for i in range(6): row.columnconfigure(i, weight=1, uniform="pc")

        tk.Label(row, text="Playback Controls", bg=TRIM_COLOR, font=("Helvetica", 18, "bold")
                 ).grid(row=0, column=0, columnspan=6, sticky="nsew", pady=(0,8))

        tk.Button(row, text="Replay", command=on_replay
                 ).grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        for j, s in enumerate([0.5, 1.0, 2.0], start=1):
            tk.Button(row, text=f"{s}×", command=lambda sp=s: on_speed(sp)
                     ).grid(row=1, column=j, sticky="nsew", padx=4, pady=4)

        tk.Button(row, text="Exit Review", command=on_exit
                 ).grid(row=1, column=5, sticky="nsew", padx=4, pady=4)

# =========================
# App entry
# =========================
def main():
    print(f"[VERSIONS] Python {sys.version.split()[0]}  OpenCV {cv2.__version__}  MediaPipe {getattr(mp, '__version__', 'n/a')}")
    root = tk.Tk(); root.title("Real Time Display")

    for c in range(3): root.columnconfigure(c, weight=1, uniform="col")
    root.rowconfigure(0, weight=0, minsize=70); root.rowconfigure(1, weight=6); root.rowconfigure(2, weight=2)

    factory = SetupGui()
    title_box = factory.framed_box(root, "Real Time Display", font_size=TITLE_FONT_SIZE)
    title_box.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD, PAD//2))

    left_box  = factory.framed_box(root, "Left Camera",  font_size=SMALL_TITLE)
    skel_box  = factory.framed_box(root, "3D Skeleton (33 kps)",   font_size=SMALL_TITLE, subtitle="placeholder")
    right_box = factory.framed_box(root, "Right Camera", font_size=SMALL_TITLE)

    left_box.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD//2)
    skel_box.grid(row=1, column=1, sticky="nsew", padx=PAD, pady=PAD//2)
    right_box.grid(row=1, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    left_player  = VideoPlayerWidget(left_box.inner, title="Left")
    right_player = VideoPlayerWidget(right_box.inner, title="Right")
    left_player.pack(fill="both", expand=True, padx=6, pady=6)
    right_player.pack(fill="both", expand=True, padx=6, pady=6)

    skel_placeholder = tk.Frame(skel_box.inner, bg=TRIM_COLOR)
    skel_placeholder.pack(fill="both", expand=True, padx=6, pady=6)
    tk.Label(skel_placeholder, text="3D skeleton(placeholder)", bg=TRIM_COLOR,
             font=("Helvetica", 14)).place(relx=0.5, rely=0.5, anchor="center")

    bottom = tk.Frame(root, bg=root.cget("bg"))
    bottom.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))
    bottom.columnconfigure(0, weight=1, uniform="bot"); bottom.columnconfigure(1, weight=1, uniform="bot")
    bottom.rowconfigure(0, weight=1)

    angles_box = factory.framed_box(bottom, "Angles", font_size=30,
                                    subtitle="Record ▶ Process ▶ Review")
    nav_box    = factory.framed_box(bottom, "Controls", font_size=30)
    angles_box.grid(row=0, column=0, sticky="nsew", padx=PAD, pady=0)
    nav_box.grid(row=0, column=1, sticky="nsew", padx=PAD, pady=0)

    angles_panel = AnglesPanel(angles_box.inner); angles_panel.pack(fill="both", expand=True, padx=6, pady=6)

    # --- panels that we swap in/out in nav_box.inner ---
    nav_container = tk.Frame(nav_box.inner, bg=TRIM_COLOR)
    nav_container.pack(fill="both", expand=True, padx=6, pady=6)

    def show_nav_panel():
        for w in nav_container.winfo_children(): w.destroy()
        NavigationPanel(
            nav_container,
            on_toggle_pose=lambda: (left_player.toggle_draw_pose(), right_player.toggle_draw_pose()),
            on_toggle_crop=lambda: (left_player.toggle_crop(), right_player.toggle_crop()),
            on_self_test=lambda: (left_player.snapshot_and_test_pose("Left Self-Test"),
                                  right_player.snapshot_and_test_pose("Right Self-Test")),
            on_record=lambda: start_record_both()
        ).pack(fill="both", expand=True)

    def show_playback_controls():
        for w in nav_container.winfo_children(): w.destroy()
        PlaybackControls(
            nav_container,
            on_replay=lambda: (left_player.playback_replay(), right_player.playback_replay()),
            on_speed=lambda s: (left_player.playback_set_speed(s), right_player.playback_set_speed(s)),
            on_exit=lambda: exit_playback_both()
        ).pack(fill="both", expand=True)

    # --- recording orchestration for both players ---
    def start_record_both():
        left_player.start_countdown_and_record(duration_s=5, fps_hint=30.0)
        right_player.start_countdown_and_record(duration_s=5, fps_hint=30.0)

        # poll until both are in playback mode, then show controls
        def _poll_ready():
            if left_player.mode == "playback" and right_player.mode == "playback":
                show_playback_controls()
            else:
                root.after(200, _poll_ready)
        _poll_ready()

    def exit_playback_both():
        left_player.playback_exit()
        right_player.playback_exit()
        show_nav_panel()

    # initial panel
    show_nav_panel()

    # Menus
    menubar = tk.Menu(root)
    filem = tk.Menu(menubar, tearoff=0)
    def open_left():
        path = filedialog.askopenfilename(title="Open Left Source",
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if path: left_player.load(path)
    def open_right():
        path = filedialog.askopenfilename(title="Open Right Source",
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if path: right_player.load(path)
    filem.add_command(label="Open Left…", command=open_left)
    filem.add_command(label="Open Right…", command=open_right)
    filem.add_separator(); filem.add_command(label="Quit", command=root.destroy)
    menubar.add_cascade(label="File", menu=filem)
    root.config(menu=menubar)

    # Shortcuts
    root.bind("<space>", lambda _e: (left_player.toggle(), right_player.toggle()))
    root.bind("q", lambda _e: root.destroy())
    root.bind("<Escape>", lambda _e: root.destroy())

    # Open initial sources
    left_player.load(LEFT_SRC); right_player.load(RIGHT_SRC)
    print("[BOOT] LEFT_SRC =", LEFT_SRC, "RIGHT_SRC =", RIGHT_SRC)

    # Window size
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")
    root.after(250, lambda: (left_player._on_resize(None), right_player._on_resize(None)))

    root.mainloop()

if __name__ == "__main__":
    main()
