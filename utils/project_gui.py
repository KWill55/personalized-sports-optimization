import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

from pathlib import Path
import yaml

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import cv2
from PIL import Image, ImageTk

"""
TODO add ATHLETE and SESSION as drop down boxes --> changes value of ATHLETE and SESSION also make label that tells what the drop down box is
TODO draw trajectory to ball tracking 
TODO make buttons work on navigation (up/down for next previous clip; left right for frames, etc)
"""


# =========================
# Config
# =========================
config_path = Path(__file__).resolve().parents[1] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
FPS = cfg["player_tracking_fps"]

# =========================
# Paths and Directories
# =========================
base_dir = Path(__file__).resolve().parents[1]
session_dir = base_dir / "data" / ATHLETE / SESSION

LAVENDER = "#d6cee6"
BLUE = "#648AB6"  
GRAY = "#6D737A" 
WHITE="#d6cee6"
BG_COLOR = BLUE
TRIM_COLOR = GRAY


BORDER   = "#555"
PAD      = 10

TITLE_FONT_SIZE = 50


# =========================
# Reusable video widget (Canvas-based, aspect-fit)
# =========================
class VideoPlayerWidget(tk.Frame):
    """
    Lightweight video widget for Tkinter.
    Public API:
      - load(Path) : open a video and show first frame
      - play() / pause()
      - next_frame() / prev_frame()
      - release()
    """
    def __init__(self, parent, title="Video", border="#555", color=BG_COLOR):
        super().__init__(parent, bg=border)
        self.border, self.color = border, color
        self.cap = None
        self.playing = False
        self.photo = None

        # inner content 
        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner, text=title, font=("Helvetica", 16, "bold"),
                 bg=self.color, anchor="n", justify="center").pack(fill="x", pady=(6, 2))

        # display area (black border look)
        holder = tk.Frame(inner, bg="black", padx=5, pady=5)
        holder.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(holder, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # info/status (optional)
        self.info = tk.StringVar(value="No video loaded")
        tk.Label(inner, textvariable=self.info, bg=self.color, font=("Helvetica", 11)).pack(fill="x", pady=(0, 6))

        # redraw state
        self._last_frame_rgb = None
        self._fps_ms = 33

    # -------- public controls --------
    def load(self, path: Path):
        self.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.info.set(f"Failed to open: {path.name}")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps_ms = max(1, int(1000 / fps))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.info.set(f"{path.name}  {w}x{h} @ {fps:.1f} fps")
        self.playing = False
        self._show_current()

    def play(self):
        if not self.cap:
            return
        if not self.playing:
            self.playing = True
            self._loop()

    def pause(self):
        self.playing = False

    def restart(self):
        if not self.cap:
            return
        self.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._show_current()

    def next_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self._display(frame)

    def prev_frame(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
        self.next_frame()

    def release(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None
        self._last_frame_rgb = None
        self.photo = None
        self.canvas.delete("all")

    # -------- internals --------
    def _loop(self):
        if not self.playing or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            # stop at end
            self.playing = False
            return
        self._display(frame)
        self.after(self._fps_ms, self._loop)

    def _show_current(self):
        if not self.cap:
            return
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
        ret, frame = self.cap.read()
        if ret:
            self._display(frame)

    def _on_resize(self, _evt):
        if self._last_frame_rgb is not None:
            self._draw_rgb(self._last_frame_rgb)

    def _display(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._last_frame_rgb = rgb
        self._draw_rgb(rgb)

    def _draw_rgb(self, rgb):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        # center in canvas (letterbox)
        x = (cw - nw) // 2
        y = (ch - nh) // 2

        img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)

class Skeleton3DWidget(tk.Frame):
    """
    Minimal 3D skeleton renderer for MediaPipe 33 (embedded Matplotlib).
    Public API: load_from_session(session_dir), next_frame(), prev_frame(), play(), pause(), restart()
    """
    def __init__(self, parent, border="#555", color=WHITE, fps=30):
        super().__init__(parent, bg=border)
        self.border, self.color = border, color
        self.fps = fps

        self.points = None   # (T,33,3)
        self.T = 0
        self.t = 0
        self.playing = False

        self._view_initialized = False   
        self._preserve_limits = False

        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        # Matplotlib figure
        self.fig = Figure(figsize=(4,3), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_axis_off()
        self.ax.set_box_aspect((1,1,1))
        self.canvas = FigureCanvasTkAgg(self.fig, master=inner)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)

        self.canvas_widget.bind("<Configure>", lambda e: self._draw())

    # ----- loading -----
    def load_from_session(self, session_dir: Path):
        key_dir = session_dir / "metrics" / "3d_keypoints"
        csv = next((p for p in sorted(key_dir.glob("*.csv"))), None)
        if not csv:
            self._draw_text(f"No 3D keypoints in:\n{key_dir}")
            return
        self.load_file(csv)

    def load_file(self, path: Path):
        try:
            P = load_mp33_csv_compact(path)    # (T,33,3)
        except Exception as e:
            self._draw_text(f"Failed to load:\n{path.name}\n{e}")
            return
        self.points = P
        self.T = P.shape[0]
        self.t = 0
        self._draw()

    # ----- navigation API (wire to your nav buttons) -----
    def restart(self):
        if self.points is None: return
        self.playing = False
        self.t = 0
        self._draw()

    def play(self):
        if self.points is None: return
        if not self.playing:
            self.playing = True
            self._loop()

    def pause(self):
        self.playing = False

    def next_frame(self):
        if self.points is None: return
        self.playing = False
        self.t = min(self.T - 1, self.t + 1)
        self._draw()

    def prev_frame(self):
        if self.points is None: return
        self.playing = False
        self.t = max(0, self.t - 1)
        self._draw()

    # ----- internals -----
    def _loop(self):
        if not self.playing: return
        self.t += 1
        if self.t >= self.T:
            self.playing = False
            self.t = self.T - 1
            self._draw()
            return
        self._draw()
        self.after(max(1, int(1000/self.fps)), self._loop)

    def _draw(self):
        # --- cache current view before clearing ---
        elev, azim = self.ax.elev, self.ax.azim
        if self._preserve_limits:
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()

        self.ax.cla()
        self.ax.set_axis_off()

        if self.points is None:
            self._draw_text("No data loaded")
            return

        P = self.points[self.t]  # (33,3)
        x, y, z = P[:,0], P[:,1], P[:,2]

        # draw points & sticks
        self.ax.scatter(x, y, z, s=12, c="k")
        for a, b in MP33_EDGES:
            xa, ya, za = P[a]
            xb, yb, zb = P[b]
            self.ax.plot([xa, xb], [ya, yb], [za, zb], linewidth=2, color="tab:blue")

        # only auto-fit when (a) first draw or (b) you don't want to preserve user zoom/pan
        if (not self._preserve_limits) or (not self._view_initialized):
            mins = np.min(P, axis=0); maxs = np.max(P, axis=0)
            c = (mins + maxs) / 2
            span = max((maxs - mins).max(), 1e-6)
            r = span * 0.6
            self.ax.set_xlim(c[0]-r, c[0]+r)
            self.ax.set_ylim(c[1]-r, c[1]+r)
            self.ax.set_zlim(c[2]-r, c[2]+r)

        # restore (or set initial) camera
        if not self._view_initialized:
            # one-time default view
            self.ax.view_init(elev=20, azim=-70)
            self._view_initialized = True
        else:
            # keep whatever the user rotated to
            self.ax.view_init(elev=elev, azim=azim)
            if self._preserve_limits:
                self.ax.set_xlim3d(xlim)
                self.ax.set_ylim3d(ylim)
                self.ax.set_zlim3d(zlim)

        self.canvas.draw_idle()


    def _draw_text(self, text):
        self.ax.cla(); self.ax.set_axis_off()
        self.ax.text2D(0.5, 0.5, text, transform=self.ax.transAxes,
                       ha="center", va="center", fontsize=11)
        self.canvas.draw_idle()


# =========================
# Your GUI factory
# =========================
class SetupGui:
    def __init__(self, border=BORDER, BG_COLOR=BG_COLOR):
        self.BORDER = border
        self.BG_COLOR = BG_COLOR

    def framed_box(self, parent, title, font_size, subtitle=None):
        outer = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        # expose inner so you can mount widgets inside this box
        outer.inner = inner

        lines = [title] + ([] if subtitle is None else subtitle)
        tk.Label(inner,
                 text="\n".join(lines),
                 bg=self.BG_COLOR,
                 font=("Helvetica", font_size, "bold"),
                 anchor="n", justify="center").pack(fill="x", pady=(6, 2))
        return outer

    def measurement_box(self, parent, title):
        outer = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner, text=f"Chosen Measurement: {title}",
                 bg=self.BG_COLOR,
                 font=("Helvetica", 16, "bold")).pack(pady=(8, 4))

        tf = tk.Frame(inner, bg=self.BG_COLOR)
        text = tk.Text(tf, height=10, wrap="word")
        sb = ttk.Scrollbar(tf, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sb.set)

        text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tf.pack(fill="both", expand=True, padx=10, pady=10)

        text.insert("end", "\n".join(f"Metric {i+1}: …" for i in range(20)))
        return outer

    def selection_box(self, parent):
        outer = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner,
                 text="Configure data here!",
                 bg=self.BG_COLOR,
                 font=("Helvetica", 20, "bold"),
                 justify="center").pack(pady=(8, 0))

        btns = tk.Frame(inner, bg=self.BG_COLOR)
        btn_font = ("Helvetica", 15, "bold")

        # 3 equal columns
        btns.columnconfigure(0, weight=1, uniform="btns")
        btns.columnconfigure(1, weight=1, uniform="btns")
        btns.columnconfigure(2, weight=1, uniform="btns")

        # Row 0: Horizontal divider
        line = tk.Frame(btns, height=2, bg="black", bd=0, relief="solid")
        line.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        # Row 1: label + 2 buttons
        tk.Label(btns,
                 text="Select Session Data:",
                 font=btn_font,
                 bg=self.BG_COLOR).grid(row=1, column=0, padx=6, pady=6, sticky="w")

        tk.Button(btns, text="Choose ATHLETE", font=btn_font).grid(row=1, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Choose SESSION", font=btn_font).grid(row=1, column=2, padx=6, pady=6, sticky="nsew")

        # Row 2: Select Visuals
        tk.Label(btns,
                 text="Select Data:",
                 font=btn_font,
                 bg=self.BG_COLOR).grid(row=2, column=0, padx=6, pady=6, sticky="w")
        tk.Button(btns, text="Pick Measurement 1", font=btn_font).grid(row=2, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Select Graph", font=btn_font).grid(row=2, column=2, padx=6, pady=6, sticky="nsew")

        # Row 3: Future 
        tk.Label(btns,
                 text="Future buttons:",
                 font=btn_font,
                 bg=self.BG_COLOR).grid(row=3, column=0, padx=6, pady=6, sticky="w")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=2, padx=6, pady=6, sticky="nsew")

        btns.pack(pady=6)
        return outer

    def navigation_box(self, parent,
                   on_prev_clip=None, on_restart_clip=None, on_next_clip=None,
                   on_prev_frame=None, on_playpause=None, on_next_frame=None):
        
        # default to no-ops so it's always safe to call
        on_prev_clip    = on_prev_clip     or (lambda: None)
        on_restart_clip = on_restart_clip  or (lambda: None)
        on_next_clip    = on_next_clip     or (lambda: None)
        on_prev_frame   = on_prev_frame    or (lambda: None)
        on_playpause    = on_playpause     or (lambda: None)
        on_next_frame   = on_next_frame    or (lambda: None)

        outer = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner,
                 text="Navigation Control Center",
                 bg=self.BG_COLOR,
                 font=("Helvetica", 25, "bold"),
                 justify="center").pack(pady=(8, 0))

        btns = tk.Frame(inner, bg=self.BG_COLOR)

        # 3 equal columns
        btns.columnconfigure(0, weight=1, uniform="btns")
        btns.columnconfigure(1, weight=1, uniform="btns")
        btns.columnconfigure(2, weight=1, uniform="btns")

        # Row 0: Horizontal divider
        line = tk.Frame(btns, height=2, bg="black", bd=0, relief="solid")
        line.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        # Row 1
        btn_font_big = ("Helvetica", 20, "bold")
        tk.Button(btns, text="Previous Clip", font=btn_font_big, command=on_prev_clip).grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Restart Clip",  font=btn_font_big, command=on_restart_clip).grid(row=1, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Next Clip",     font=btn_font_big, command=on_next_clip).grid(row=1, column=2, padx=6, pady=6, sticky="nsew")

        # Row 2
        btn_font = ("Helvetica", 15, "bold")
        tk.Button(btns, text="Previous Frame", font=btn_font, command=on_prev_frame).grid(row=2, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Pause/Play",     font=btn_font, command=on_playpause).grid(row=2, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Next Frame",     font=btn_font, command=on_next_frame).grid(row=2, column=2, padx=6, pady=6, sticky="nsew")

        # Row 3
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=2, padx=6, pady=6, sticky="nsew")

        btns.pack(pady=6)
        return outer

    def load_folder(self, title, initialdir=None):
        folder = filedialog.askdirectory(initialdir=initialdir or session_dir, title=title)
        if not folder:
            print("[WARNING] No folder found")
            return
        return Path(folder)
    


# ---------- Helpers not in classes F0R VIDEOS ----------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

def _first_video_in(dir_path: Path):
    if not dir_path.exists():
        return None
    files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return files[0] if files else None

def load_videos_for_current_session(video2d_player, ball_player):
    """
    Loads the first videos found in:
      - session_dir / videos / player_tracking / 2d
      - session_dir / videos / ball_tracking / raw
    """
    two_d_dir = session_dir / "videos" / "player_tracking" / "2d"
    ball_dir  = session_dir / "videos" / "ball_tracking" / "raw"

    two_d_vid = _first_video_in(two_d_dir)
    ball_vid  = _first_video_in(ball_dir)

    if not two_d_vid:
        messagebox.showinfo("2D video not found", f"No video found in:\n{two_d_dir}")
    else:
        video2d_player.load(two_d_vid)

    if not ball_vid:
        messagebox.showinfo("Ball tracking video not found", f"No video found in:\n{ball_dir}")
    else:
        ball_player.load(ball_vid)


# ---- MediaPipe 33 names and edges (subset is fine) ----
MP33_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]
MP33_IDX = {n:i for i,n in enumerate(MP33_NAMES)}

# BlazePose-like sticks: shoulders/torso/arms/legs
MP33_EDGES = [
    (11,13),(13,15),        # left arm
    (12,14),(14,16),        # right arm
    (11,12), (23,24),       # shoulders, hips
    (11,23),(12,24),        # torso
    (23,25),(25,27),(27,29),(29,31),  # left leg/foot
    (24,26),(26,28),(28,30),(30,32),  # right leg/foot
]

def load_mp33_csv_compact(path: Path) -> np.ndarray:
    """
    Read CSV with columns like 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', etc.
    Returns (T, 33, 3) float32.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    cols = []
    for name in MP33_NAMES:
        for a in ("x","y","z"):
            col = f"{name}_{a}"
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path.name}")
            cols.append(col)
    arr = df[cols].to_numpy(float).reshape((-1, 33, 3)).astype(np.float32)
    return arr


# =========================
# App entry
# =========================
def main():

    root = tk.Tk()
    root.title("Personalized Sports Optimization GUI")

    # Start large
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")

    factory = SetupGui()

    # 3 columns, same width
    for c in range(3):
        root.columnconfigure(c, weight=1, uniform="col")

    # Row sizing: make rows 1–2 dominant; title + bottom shorter
    ROW_WEIGHTS  = [1, 5, 5, 1]
    ROW_MINSIZE  = [70, 260, 260, 120]
    for r in range(4):
        root.rowconfigure(r, weight=ROW_WEIGHTS[r], minsize=ROW_MINSIZE[r])

    # ----- Title spans across all columns -----
    title = factory.framed_box(root, "Personalized Sports Optimization GUI", font_size=TITLE_FONT_SIZE)
    title.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD, PAD//2))

    # ----- Row 1: video / ball video / measurement 1 -----
    video2d_box = factory.framed_box(root, "2D video", subtitle=["1280x640"], font_size=14)
    video2d_box.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD//2)

    ballvideo_box = factory.framed_box(root, "Ball Tracking Video", subtitle=["1280x720"], font_size=14)
    ballvideo_box.grid(row=1, column=1, sticky="nsew", padx=PAD, pady=PAD//2)

    meas1 = factory.measurement_box(root, "mystery gun")
    meas1.grid(row=1, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    # ----- Mount video players inside those boxes -----
    video2d_player = VideoPlayerWidget(video2d_box.inner, title="2D video", border=BORDER, color=BG_COLOR)
    video2d_player.pack(fill="both", expand=True, padx=4, pady=4)

    ball_player = VideoPlayerWidget(ballvideo_box.inner, title="Ball Tracking Video", border=BORDER, color=TRIM_COLOR)
    ball_player.pack(fill="both", expand=True, padx=4, pady=4)

    load_videos_for_current_session(video2d_player, ball_player)


    # ----- Row 2: 3D skeleton / graph1 / measurement 2 -----

    # create tkinter box to place skeleton in
    skel3d = factory.framed_box(root, "3D skeleton", font_size=14)
    skel3d.grid(row=2, column=0, sticky="nsew", padx=PAD, pady=PAD//2)

    # add the 3D skeleton widget into the box and load data
    skeleton = Skeleton3DWidget(skel3d.inner, border=BORDER, color=TRIM_COLOR, fps=FPS or 30)
    skeleton.pack(fill="both", expand=True, padx=4, pady=4)
    skeleton.load_from_session(session_dir)

    info = factory.framed_box(root, "Info", font_size=TITLE_FONT_SIZE)
    info.grid(row=2, column=1, sticky="nsew", padx=PAD, pady=PAD//2)

    graph1 = factory.framed_box(root, "Chosen Graph", font_size=14)
    graph1.grid(row=2, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    def playpause_all():
        # toggles videos (simple) and skeleton
        for vp in (video2d_player, ball_player):
            if hasattr(vp, "playing") and vp.playing:
                vp.pause()
            else:
                vp.play()
        # skeleton
        if skeleton.playing:
            skeleton.pause()
        else:
            skeleton.play()

    def restart_all():
        # videos: seek to start if you have restart() on your VideoPlayerWidget; otherwise reload frame 0
        for vp in (video2d_player, ball_player):
            if hasattr(vp, "restart"):
                vp.restart()
            else:
                # fallback: pause then prev to 0; you may add a proper restart() on the player
                vp.pause()
        skeleton.restart()

    def prev_frame_all():
        for vp in (video2d_player, ball_player):
            vp.prev_frame()
        skeleton.prev_frame()

    def next_frame_all():
        for vp in (video2d_player, ball_player):
            vp.next_frame()
        skeleton.next_frame()

    # ----- Row 3: shorter bottom row -----
    nav = factory.navigation_box(
        root,
        on_prev_clip=None,     # (wire later if you implement multi-clip at GUI level)
        on_restart_clip=restart_all,
        on_next_clip=None,
        on_prev_frame=prev_frame_all,
        on_playpause=playpause_all,
        on_next_frame=next_frame_all
    )
    nav.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))

    select = factory.selection_box(root)
    select.grid(row=3, column=2, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))


    root.mainloop()


if __name__ == "__main__":
    main()
