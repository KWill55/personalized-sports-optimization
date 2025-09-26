#!/usr/bin/env python3
"""
Real-Time Display layout (no functionality wiring yet):

Row -1: Title bar
Row  0: [Left Camera] [3D Skeleton (placeholder)] [Right Camera]
Row  1: container spanning full width with two equal boxes:
        [Angles (subtitle + list with ???)]  |  [Navigation Controls (buttons only)]

Reads optional sources from project_config.yaml at repo root:
  camera_left_source, camera_right_source, ui_fps
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import yaml
import cv2
from PIL import Image, ImageTk

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

# Normalize numeric-like strings to int camera indices
for k in ("camera_left_source", "camera_right_source"):
    v = cfg.get(k)
    if isinstance(v, str) and v.isdigit():
        cfg[k] = int(v)

LEFT_SRC  = cfg.get("left_cam_index")
RIGHT_SRC = cfg.get("right_cam_index")
UI_FPS    = int(cfg.get("player_tracking_fps"))

# =========================
# Minimal framed box helper
# =========================
class SetupGui:
    def __init__(self, border=BORDER, BG_COLOR=BG_COLOR):
        self.BORDER = border
        self.BG_COLOR = BG_COLOR

    def framed_box(self, parent, title, font_size, subtitle: str | None = None):
        outer = tk.Frame(parent, bg=self.BORDER)
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
# Video player widget (crops center square 640x640 by default)
# =========================
class VideoPlayerWidget(tk.Frame):
    def __init__(self, parent, border="#555", color=TRIM_COLOR, fps=UI_FPS,
                 crop_center_square: bool = True, square_size: int = 640):
        super().__init__(parent, bg=border)
        self.border, self.color = border, color
        self.cap = None
        self.playing = False
        self.photo = None
        self._fps_ms = max(1, int(1000 / max(fps, 1)))
        self._last_frame_rgb = None
        self._source = None

        # NEW: square-crop controls
        self.crop_center_square = bool(crop_center_square)
        self.square_size = int(max(1, square_size))

        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.holder = tk.Frame(inner, bg="black", padx=1, pady=1)
        self.holder.pack(fill="both", expand=True, padx=1, pady=1)
        self.holder.pack_propagate(False)

        self.canvas = tk.Canvas(self.holder, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # NEW: keep the visual box roughly square
        self.holder.bind("<Configure>", lambda e: self._enforce_square_holder())

        self.info = tk.StringVar(value="No video loaded")
        tk.Label(inner, textvariable=self.info, bg=self.color, font=("Helvetica", 11)).pack(fill="x")

    def load(self, source):
        self.release()
        self._source = source
        try:
            if isinstance(source, (str, Path)):
                self.cap = cv2.VideoCapture(str(source))
            else:
                self.cap = cv2.VideoCapture(int(source))
        except Exception:
            self.cap = cv2.VideoCapture(str(source))
        if not self.cap or not self.cap.isOpened():
            self.info.set(f"Failed to open: {source}")
            self.cap = None
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps_ms = max(1, int(1000 / fps))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.info.set(f"{source}  {w}x{h} @ {fps:.1f} fps")
        self.play()

    def play(self):
        if not self.cap:
            return
        if not self.playing:
            self.playing = True
            self.after(0, self._loop)

    def pause(self):
        self.playing = False

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def release(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
        self._last_frame_rgb = None
        self.photo = None
        self.canvas.delete("all")

    def _loop(self):
        if not self.playing or not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            if isinstance(self._source, (str, Path)):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after(self._fps_ms, self._loop)
            return
        self._display(frame)
        self.after(self._fps_ms, self._loop)

    def _on_resize(self, _evt):
        if self._last_frame_rgb is not None:
            self._draw_rgb(self._last_frame_rgb)

    # NEW: keep the holder square so the feed sits in a square window
    def _enforce_square_holder(self):
        w = self.holder.winfo_width() or 1
        h = self.holder.winfo_height() or 1
        side = min(w, h)
        self.holder.configure(width=side, height=side)

    # NEW: crop the centered square from the incoming BGR frame
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

    def _display(self, frame_bgr):
        # center 640x640 crop by default
        if self.crop_center_square:
            frame_bgr = self._center_square_bgr(frame_bgr, self.square_size)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._last_frame_rgb = rgb
        self._draw_rgb(rgb)

    def _draw_rgb(self, rgb):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.photo)


# =========================
# Static Angles panel (placeholders only)
# =========================
class AnglesPanel(tk.Frame):
    ANGLES = [
        "Shoulder Flex (L)", "Shoulder Flex (R)",
        "Elbow (L)", "Elbow (R)",
        "Hip Flex (L)", "Hip Flex (R)",
        "Knee (L)", "Knee (R)",
        "Ankle (L)", "Ankle (R)",
    ]
    def __init__(self, parent):
        super().__init__(parent, bg=BORDER)
        inner = tk.Frame(self, bg=TRIM_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text="Angles", bg=TRIM_COLOR, font=("Helvetica", SMALL_TITLE, "bold")).pack(pady=(6, 2))
        tk.Label(inner, text="Mode: Real Time  |  Recorded", bg=TRIM_COLOR, font=("Helvetica", 12)).pack()
        grid = tk.Frame(inner, bg=TRIM_COLOR)
        grid.pack(fill="both", expand=True, padx=10, pady=8)
        grid.columnconfigure(0, weight=3)
        grid.columnconfigure(1, weight=1)
        for r, name in enumerate(self.ANGLES):
            tk.Label(grid, text=name, bg=TRIM_COLOR, anchor="w", font=("Helvetica", 13)).grid(
                row=r, column=0, sticky="nsew", pady=3)
            tk.Label(grid, text="???", bg=TRIM_COLOR, anchor="e", font=("Helvetica", 13, "bold")).grid(
                row=r, column=1, sticky="nsew", pady=3)

# =========================
# Navigation Controls (buttons only, no behavior yet)
# =========================
class NavigationPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BORDER)
        inner = tk.Frame(self, bg=TRIM_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text="Navigation Controls", bg=TRIM_COLOR,
                 font=("Helvetica", SMALL_TITLE, "bold")).pack(pady=(6, 6))
        btns = tk.Frame(inner, bg=TRIM_COLOR)
        btns.pack(padx=8, pady=8, fill="x")
        for i in range(3):
            btns.columnconfigure(i, weight=1, uniform="btns")
        tk.Button(btns, text="Toggle Real-Time / Record Mode").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=4)
        tk.Button(btns, text="Toggle Draw MediaPipe Pose").grid(row=1, column=0, columnspan=3, sticky="nsew", pady=4)
        tk.Button(btns, text="(Placeholder Button)").grid(row=2, column=0, columnspan=3, sticky="nsew", pady=4)

# =========================
# App entry
# =========================

def main():
    root = tk.Tk()
    root.title("Real Time Display")

    # 3 columns for top row (video/skeleton/video)
    for c in range(3):
        root.columnconfigure(c, weight=1, uniform="col")
    # Title, top row, bottom container rows
    root.rowconfigure(0, weight=0, minsize=70)   # title
    # Make the videos row significantly larger than the bottom row
    root.rowconfigure(1, weight=6)               # videos + skeleton (bigger)
    root.rowconfigure(2, weight=2)               # bottom two boxes container (smaller)

    factory = SetupGui()

    # Title bar
    title_box = factory.framed_box(root, "Real Time Display", font_size=TITLE_FONT_SIZE)
    title_box.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD, PAD//2))

    # Row 1: Left / Skeleton / Right
    left_box   = factory.framed_box(root,  "Left Camera (640x640)",  font_size=SMALL_TITLE)
    skel_box   = factory.framed_box(root,  "3D Skeleton (placeholder)", font_size=SMALL_TITLE,
                                    subtitle="Same size as camera feeds")
    right_box  = factory.framed_box(root,  "Right Camera (640x640)", font_size=SMALL_TITLE)

    left_box.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD//2)
    skel_box.grid(row=1, column=1, sticky="nsew", padx=PAD, pady=PAD//2)
    right_box.grid(row=1, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    left_player  = VideoPlayerWidget(left_box.inner)
    right_player = VideoPlayerWidget(right_box.inner)
    left_player.pack(fill="both", expand=True, padx=6, pady=6)
    right_player.pack(fill="both", expand=True, padx=6, pady=6)

    # Skeleton placeholder (just a centered label for now)
    skel_placeholder = tk.Frame(skel_box.inner, bg=TRIM_COLOR)
    skel_placeholder.pack(fill="both", expand=True, padx=6, pady=6)
    tk.Label(skel_placeholder, text="3D skeleton(placeholder)", bg=TRIM_COLOR,
             font=("Helvetica", 14)).place(relx=0.5, rely=0.5, anchor="center")

    # Row 2: Bottom container spanning all 3 columns, with 2 equal boxes inside
    bottom = tk.Frame(root, bg=root.cget("bg"))
    bottom.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))
    bottom.columnconfigure(0, weight=1, uniform="bot")
    bottom.columnconfigure(1, weight=1, uniform="bot")
    bottom.rowconfigure(0, weight=1)

    angles_box = factory.framed_box(bottom, "Angles", font_size=SMALL_TITLE,
                                    subtitle="Mode: Real Time or switch to Recorded values")
    nav_box    = factory.framed_box(bottom, "Navigation Controls", font_size=SMALL_TITLE)
    angles_box.grid(row=0, column=0, sticky="nsew", padx=PAD, pady=0)
    nav_box.grid(row=0, column=1, sticky="nsew", padx=PAD, pady=0)

    angles_panel = AnglesPanel(angles_box.inner)
    angles_panel.pack(fill="both", expand=True, padx=6, pady=6)

    nav_panel = NavigationPanel(nav_box.inner)
    nav_panel.pack(fill="both", expand=True, padx=6, pady=6)

    # Menu shortcuts to open sources
    menubar = tk.Menu(root)
    filem = tk.Menu(menubar, tearoff=0)
    def open_left():
        path = filedialog.askopenfilename(title="Open Left Source",
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if path:
            left_player.load(path)
    def open_right():
        path = filedialog.askopenfilename(title="Open Right Source",
                                          filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")])
        if path:
            right_player.load(path)
    filem.add_command(label="Open Left…", command=open_left)
    filem.add_command(label="Open Right…", command=open_right)
    filem.add_separator()
    filem.add_command(label="Quit", command=root.destroy)
    menubar.add_cascade(label="File", menu=filem)
    root.config(menu=menubar)

    # Shortcuts
    root.bind("<space>", lambda _e: (left_player.toggle(), right_player.toggle()))
    root.bind("q", lambda _e: root.destroy())
    root.bind("<Escape>", lambda _e: root.destroy())

    # Open initial sources from config
    left_player.load(LEFT_SRC)
    right_player.load(RIGHT_SRC)

    # Size ~90% of screen
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")

    root.mainloop()


if __name__ == "__main__":
    main()

