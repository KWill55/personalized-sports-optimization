import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

from pathlib import Path
import yaml
import re

import numpy as np
import matplotlib as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cv2
import pandas as pd
from PIL import Image, ImageTk

import seaborn as sns
sns.set_theme(style="whitegrid")


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

# # =========================
# # Get project root 
# # =========================

# locate the repo root
def find_repo_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / 'project_config.yaml').exists():
            return p
    return start

ROOT = find_repo_root()
print('Repo root:', ROOT)

# Load config 
CFG_PATH = ROOT / 'project_config.yaml'
if CFG_PATH.exists():
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

# ---- ATHLETE / FPS defaults ----
ATHLETE = cfg.get('athlete', 'athlete_01')  
FPS = int(cfg.get('player_tracking_fps', 30))

DATA_DIR = ROOT / 'data' / ATHLETE
assert DATA_DIR.exists(), f'DATA_DIR not found: {DATA_DIR}'

# Discover sessions for this athlete (folders under data/<ATHLETE>)
SESSIONS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])

# Create boolean flags per session and a dict to track inclusion
SESSION_INCLUDE = {}
for s in SESSIONS:
    var = 'include_' + re.sub(r'[^0-9a-zA-Z_]', '_', s)
    globals()[var] = True  # default True
    SESSION_INCLUDE[s] = True

print('ATHLETE:', ATHLETE, 'FPS:', FPS)
print('Sessions discovered:', SESSIONS)
print('Per-session booleans created in globals():', [k for k in globals() if k.startswith('include_')])

def selected_sessions():
    # Returns the list of sessions with their boolean 'include_*' variable set to True.
    out = []
    for s in SESSIONS:
        var = 'include_' + re.sub(r'[^0-9a-zA-Z_]', '_', s)
        if globals().get(var, False):
            out.append(s)
    return out

print('Initially selected sessions:', selected_sessions())


# # =========================
# # Plotting functions 
# # =========================

def get_release_rows(angles_long_df: pd.DataFrame,
                     phases_df: pd.DataFrame,
                     sessions: str | list[str]) -> pd.DataFrame:
    """
    For each session/clip/file, take angle values at the clip's release_frame.
    Returns long DF with ['athlete','session','clip','file','angle','value'] at release.
    """
    if isinstance(sessions, str):
        sessions = [sessions]
    ang = angles_long_df[angles_long_df["session"].isin(sessions)].copy()
    ph  = phases_df[phases_df["session"].isin(sessions)].copy()

    need_ang = {"athlete","session","clip","file","frame","angle","value"}
    need_ph  = {"athlete","session","clip","file","release_frame"}
    if not need_ang.issubset(ang.columns):
        missing = need_ang - set(ang.columns)
        raise KeyError(f"angles_long_df missing columns: {missing}")
    if not need_ph.issubset(ph.columns):
        missing = need_ph - set(ph.columns)
        raise KeyError(f"phases_df missing columns: {missing}")

    m = ang.merge(
        ph[["athlete","session","clip","file","release_frame"]],
        on=["athlete","session","clip","file"],
        how="inner"
    )
    m = m[m["frame"] == m["release_frame"]].copy()
    m = (m
         .drop_duplicates(subset=["athlete","session","clip","file","angle"])
         .sort_values(["session","clip","file","angle"])
        )
    out = m[["athlete","session","clip","file","angle","value"]].reset_index(drop=True)
    return out


def _ensure_clip_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric 'clip' exists; derive from 'file' if needed."""
    if 'clip' not in df.columns:
        if 'file' in df.columns:
            clip = df['file'].astype(str).str.extract(r'(\d+)')[0]
            df['clip'] = pd.to_numeric(clip, errors='coerce')
        else:
            raise KeyError("Dataframe needs 'clip' or 'file' to sort throws.")
    return df


def list_angle_csvs(athlete: str, sessions: list[str], root: Path = ROOT) -> pd.DataFrame:
    # Return a DataFrame: ['session', 'clip', 'path'] for all *_angles.csv under data/<ATHLETE>/<SESSION>/metrics/3d_angles/
    rows = []
    base = root / 'data' / athlete
    for s in sessions:
        angles_dir = base / s / 'metrics' / '3d_angles'
        if not angles_dir.exists():
            continue
        for p in sorted(angles_dir.glob('*_angles.csv')):
            # Try to extract a clip id/number from the filename
            m = re.search(r'(\d+)', p.stem)
            clip = int(m.group(1)) if m else None
            rows.append({'session': s, 'clip': clip, 'path': p})
    return pd.DataFrame(rows)

def load_angles_long(angle_files: pd.DataFrame,
                     downcast_float32: bool = True) -> pd.DataFrame:
    """
    Read all *_angles.csv into a long/tidy table with columns:
    ['athlete','session','clip','file','frame','time_s','angle','value']
    """
    dfs = []
    for _, row in angle_files.iterrows():
        p = Path(row['path'])
        df = pd.read_csv(p)

        # ensure frame exists
        if 'frame' not in df.columns:
            df.insert(0, 'frame', np.arange(len(df), dtype=int))

        # melt wide → long
        long = df.melt(id_vars=['frame'], var_name='angle', value_name='value')

        # attach metadata
        long['athlete'] = ATHLETE
        long['session'] = row['session']
        long['clip']    = row['clip']
        long['file']    = p.name   # or p.stem if you want to drop ".csv"

        if downcast_float32:
            long['value'] = pd.to_numeric(long['value'], errors='coerce').astype('float32')

        dfs.append(long)

    if not dfs:
        return pd.DataFrame(columns=['athlete','session','clip','file','frame','time_s','angle','value'])

    out = pd.concat(dfs, ignore_index=True)

    # compute time from frame using FPS
    out['time_s'] = out['frame'] / float(max(FPS, 1))

    # order columns
    out = out[['athlete','session','clip','file','frame','time_s','angle','value']]
    return out

# #defintions for gathering phases 

def list_phase_csvs(athlete: str, sessions: list[str], root: Path = ROOT) -> pd.DataFrame:
    """
    Discover freethrow_phases.csv under each selected session.
    Returns DataFrame with columns: ['session','path']
    """
    rows = []
    base = root / 'data' / athlete
    for s in sessions:
        session_root = base / s
        for p in session_root.rglob('freethrow_phases.csv'):
            rows.append({'session': s, 'path': p})
    return pd.DataFrame(rows)

def extract_clip_id_from_file_str(file_str: str):
    """'clip_003.mp4' -> 3, else None."""
    try:
        stem = Path(str(file_str)).stem
    except Exception:
        stem = str(file_str)
    m = re.search(r'(\d+)', stem)
    return int(m.group(1)) if m else None

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

def extract_clip_id_from_any_name(name: str) -> int | None:
    m = re.search(r'(\d+)', str(name))
    return int(m.group(1)) if m else None

def list_videos_with_clips(dir_path: Path) -> dict[int, Path]:
    """Return {clip_id:int -> Path} for all videos in dir_path."""
    out = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.iterdir()):
        if p.suffix.lower() in VIDEO_EXTS:
            cid = extract_clip_id_from_any_name(p.name)
            if cid is not None:
                out[cid] = p
    return out

def load_phases_table(phase_files: pd.DataFrame,
                      athlete: str,
                      fps: float = 60) -> pd.DataFrame:
    """
    Load all freethrow_phases.csv files into a single table with:
    ['athlete','session','clip','file','windup_start','release_frame','followthrough_end',
     'windup_t','release_t','followthrough_t'].
    """
    dfs = []
    for _, row in phase_files.iterrows():
        p = Path(row['path'])
        df = pd.read_csv(p)

        # normalize names (case-insensitive)
        cols = {c.lower(): c for c in df.columns}
        need = ['file','windup_start','release_frame','followthrough_end']
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f"Missing columns in {p}: {missing}")

        # select with original casing, then standardize names
        df = df[[cols[c] for c in need]].copy()
        df.columns = need

        # align keys with angles_long
        df['athlete'] = athlete
        df['session'] = row['session']
        df['clip']    = df['file'].apply(extract_clip_id_from_file_str)

        # times in seconds
        denom = float(max(fps, 1))
        df['windup_t']         = df['windup_start']      / denom
        df['release_t']        = df['release_frame']     / denom
        df['followthrough_t']  = df['followthrough_end'] / denom

        # enforce numeric (avoids type-mismatch on merge)
        for c in ['windup_start','release_frame','followthrough_end']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[
            'athlete','session','clip','file',
            'windup_start','release_frame','followthrough_end',
            'windup_t','release_t','followthrough_t'
        ])

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=['clip'])  # optional
    out = out[['athlete','session','clip','file',
               'windup_start','release_frame','followthrough_end',
               'windup_t','release_t','followthrough_t']]
    return out

# create data frames for phases and angles 

# #phases data frames
phase_files_df = list_phase_csvs(ATHLETE, selected_sessions())
print(f'Found {len(phase_files_df)} phase files across {phase_files_df['session'].nunique()} sessions.')

phases_df = load_phases_table(phase_files_df, athlete=ATHLETE, fps=FPS)
print(f'Loaded {len(phases_df)} phase rows across {phases_df["session"].nunique()} sessions.')


# #angles data frames 
angle_files_df = list_angle_csvs(ATHLETE, selected_sessions())
print(f'Found {len(angle_files_df)} angle files across {angle_files_df['session'].nunique()} sessions.')

angles_long_df = load_angles_long(angle_files_df)
print(f'Loaded {len(angles_long_df)} angle rows across {angles_long_df["session"].nunique()} sessions.')

release_rows = get_release_rows(angles_long_df, phases_df, sessions=SESSION)


class AnglesPlotWidget(tk.Frame):
    """
    Fast Matplotlib plot for one clip at a time.
    Shows all angles vs frame for the selected clip.
    Call .show_clip(file_name) to update.
    """
    def __init__(self, parent, angles_long_df, session_id, *, bg="#ffffff"):
        super().__init__(parent, bg=bg)

        self.angles_long_df = angles_long_df[angles_long_df["session"] == session_id].copy()
        # Pre-split by file for quick updates
        self.by_file = {}
        for f, g in self.angles_long_df.groupby("file"):
            # Ensure frames are sorted and numeric
            gg = g.copy().sort_values("frame")
            gg["frame"] = pd.to_numeric(gg["frame"], errors="coerce")
            self.by_file[f] = gg

        # Matplotlib figure
        self.fig = Figure(figsize=(4,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Angles vs Frame (per clip)")
        self.ax.grid(True, alpha=0.3)

        # Embed into Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Keep handles so we can update without re-creating legends
        self._lines = {}  # angle -> Line2D

    def show_clip(self, file_name: str):
        """Update the plot to show the given clip file."""
        if file_name not in self.by_file:
            # Clear axes if empty/missing
            self.ax.cla()
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("Angle (deg)")
            self.ax.set_title(f"No data for {file_name}")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
            return

        df = self.by_file[file_name]
        # Redraw fresh for simplicity (still fast)
        self.ax.cla()
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title(f"Angles vs Frame — {file_name}")
        self.ax.grid(True, alpha=0.3)

        # One line per angle
        for angle, g in df.groupby("angle"):
            self.ax.plot(g["frame"].values, g["value"].values, linewidth=1.5, label=str(angle))

        # Nice legend (outside to avoid overlap)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        self.fig.tight_layout()
        self.canvas.draw_idle()

class ReleaseAnglesWidget(tk.Frame):
    """
    Matplotlib plot: release-time angle values per throw (clip) for one session.
    One line per angle; X = clip (ordered); Y = value at release.
    Call .set_current_clip(clip_id_int) to highlight the selected throw.
    """
    def __init__(self, parent, release_rows: pd.DataFrame, session_id: str, *, bg="#ffffff"):
        super().__init__(parent, bg=bg)

        df = release_rows[release_rows['session'] == session_id].copy()
        df = _ensure_clip_col(df)
        if df.empty:
            self.df = pd.DataFrame(columns=["clip","angle","value","file"])
        else:
            # sort by numeric clip and keep only valid
            df = df.dropna(subset=["clip"]).copy()
            df["clip"] = df["clip"].astype(int)
            df = df.sort_values("clip")
            self.df = df

        # X domain & mapping to positions (0..n-1) so Matplotlib is quick
        self.clips = sorted(self.df["clip"].unique().tolist())
        self.pos_by_clip = {c:i for i,c in enumerate(self.clips)}

        # Matplotlib
        self.fig = Figure(figsize=(4,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Throw (clip id)")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Release Angle per Throw")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self._vline = None
        self._highlight_scats = []  # small markers on the highlighted x

        self._draw_static()

    def _draw_static(self):
        self.ax.cla()
        self.ax.set_xlabel("Throw (clip id)")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Release Angle per Throw")
        self.ax.grid(True, alpha=0.3)

        if self.df.empty or not self.clips:
            self.ax.text(0.5, 0.5, "No release data", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        # plot one line per angle
        for angle, g in self.df.groupby("angle"):
            # map clips -> x positions
            x = g["clip"].map(self.pos_by_clip).to_numpy()
            y = g["value"].to_numpy()
            self.ax.plot(x, y, marker="o", linewidth=1.5, label=str(angle))

        # xticks: show clip ids as labels
        self.ax.set_xticks(range(len(self.clips)))
        self.ax.set_xticklabels([str(c) for c in self.clips], rotation=0)

        # legend outside
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_current_clip(self, clip_id: int | None):
        """Draw a thin vertical highlight at the selected clip and dot markers there."""
        # remove old highlight
        if self._vline is not None:
            self._vline.remove()
            self._vline = None
        for s in self._highlight_scats:
            s.remove()
        self._highlight_scats.clear()

        if clip_id is None or clip_id not in self.pos_by_clip:
            self.canvas.draw_idle()
            return

        x0 = self.pos_by_clip[clip_id]

        # vertical line
        self._vline = self.ax.axvline(x=x0, linestyle="--", linewidth=1.0, alpha=0.6)

        # scatter markers for each angle at this x (if that clip has a release value)
        dfc = self.df[self.df["clip"] == clip_id]
        if not dfc.empty:
            yvals = dfc["value"].to_numpy()
            xs = np.full_like(yvals, x0, dtype=float)
            scat = self.ax.scatter(xs, yvals, s=30, zorder=5)
            self._highlight_scats.append(scat)

        self.canvas.draw_idle()






# fig = plot_angles_over_frames(angles_long_df, SESSION, use_seconds=False)
# fig.show()


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
    def __init__(self, parent, border="#555", color=BG_COLOR):
        super().__init__(parent, bg=border)
        self.border, self.color = border, color
        self.cap = None
        self.playing = False
        self.photo = None

        # inner content 
        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
 
        # display area (black border look)
        # holder = tk.Frame(inner, bg="black", padx=1, pady=1)  # thinner padding
        # holder.pack(fill="both", expand=True, padx=1, pady=1)

        self.holder = tk.Frame(inner, bg="black", padx=1, pady=1)
        self.holder.pack(fill="both", expand=True, padx=1, pady=1)
        self.holder.pack_propagate(False)  

        self.canvas = tk.Canvas(self.holder, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.aspect = None

        self.canvas.bind("<Configure>", self._on_resize)

        # info/status (optional)
        self.info = tk.StringVar(value="No video loaded")
        tk.Label(inner, textvariable=self.info, bg=self.color, font=("Helvetica", 11)).pack(fill="x", pady=(0, 0))

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
        self.aspect = (w / h) if (w and h) else None
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
        # keep holder at the video aspect
        if self.aspect:
            pw = self.holder.winfo_width() or self.holder.winfo_reqwidth()
            ph = self.holder.winfo_height() or self.holder.winfo_reqheight()
            # but we need the size of inner container; use the parent's size
            parent = self.holder.master
            cw = parent.winfo_width() or 1
            ch = parent.winfo_height() or 1
            target_w = cw
            target_h = int(target_w / self.aspect)
            if target_h > ch:  # too tall, limit by height
                target_h = ch
                target_w = int(target_h * self.aspect)
            # apply
            self.holder.configure(width=target_w, height=target_h)

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

        img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        # draw centered in the canvas
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.photo)


class Skeleton3DWidget(tk.Frame):
    """
    3D skeleton viewer for MediaPipe-33 keypoints (embedded Matplotlib).

    Expected input:
      self.points : np.ndarray of shape (T, 33, 3)  (T = frames)

    Public API:
      - load_from_session(session_dir: Path)
      - load_file(csv_path: Path)                 # loads (T,33,3)
      - play(), pause(), restart()
      - next_frame(), prev_frame()
    """

    def __init__(self, parent, *, bg_border="#555", bg_panel="#d6cee6", fps=30,
                 point_size=30, line_width=3, use_global_limits=True):
        """
        bg_border/bg_panel: colors for outer frame and inner panel
        fps:               playback speed for play()
        point_size:        scatter size for joints
        line_width:        line width for bones
        use_global_limits: if True, fix axis limits using all frames (no size jitter)
        """
        super().__init__(parent, bg=bg_border)
        self.fps = fps
        self.point_size = point_size
        self.line_width = line_width
        self.use_global_limits = use_global_limits

        # data / state
        self.points = None     # (T,33,3)
        self.T = 0             # number of frames
        self.t = 0             # current frame index
        self.playing = False
        self.global_R = None   # axis radius for fixed limits (computed on load)

        # ---- UI: one inner panel + a full-bleed Matplotlib axes ----
        panel = tk.Frame(self, bg=bg_panel)
        panel.pack(fill="both", expand=True, padx=2, pady=2)

        self.fig = Figure(figsize=(3, 3), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_axis_off()

        # Make axes fill the whole figure so centering is visually true
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_position([0, 0, 1, 1])
        self.ax.set_box_aspect((1, 1, 1))

        self.canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)

        # On resize, just redraw current frame
        self.canvas_widget.bind("<Configure>", lambda _e: self._draw())

        # Keep an initial, pleasant view angle
        self._has_set_initial_view = False

    # ---------------- Loading ----------------
    def load_from_session(self, session_dir: Path):
        """Convenience: load the first CSV from metrics/3d_keypoints/."""
        key_dir = session_dir / "metrics" / "3d_keypoints"
        csv = next((p for p in sorted(key_dir.glob("*.csv"))), None)
        if not csv:
            self._draw_text(f"No 3D keypoints in:\n{key_dir}")
            return
        self.load_file(csv)

    def load_file(self, path: Path):
        """Read CSV → (T,33,3); prepare limits; draw first frame."""
        try:
            P = load_mp33_csv_compact(path)  # your helper: returns (T,33,3)
        except Exception as e:
            self._draw_text(f"Failed to load:\n{path.name}\n{e}")
            return

        self.points = P.astype(np.float32)
        self.T = int(P.shape[0])
        self.t = 0

        # Optionally compute global axis radius so size is stable during playback.
        if self.use_global_limits:
            torso_ids = [
                MP33_IDX["left_shoulder"], MP33_IDX["right_shoulder"],
                MP33_IDX["left_hip"],      MP33_IDX["right_hip"]
            ]
            # Center each frame on its torso centroid, then find the maximum span.
            centroids = self.points[:, torso_ids, :].mean(axis=1)         # (T,3)
            Pc_all = self.points - centroids[:, None, :]                  # (T,33,3)
            span = np.max(np.ptp(Pc_all, axis=1))                         # max over frames
            self.global_R = float(max(span, 1e-6) * 0.55)                 # 0.55 = tight padding
        else:
            self.global_R = None

        self._draw()

    # ---------------- Playback controls ----------------
    def restart(self):
        if self.points is None: return
        self.playing = False
        self.t = 0
        self._draw()

    def play(self):
        if self.points is None or self.playing: return
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

    # ---------------- Internals ----------------
    def _loop(self):
        if not self.playing: return
        self.t += 1
        if self.t >= self.T:
            self.t = self.T - 1
            self.playing = False
        self._draw()
        # schedule next frame
        self.after(max(1, int(1000 / max(self.fps, 1))), self._loop)

    def _draw(self):
        """Render current frame centered and scaled in a symmetric cube."""
        self.ax.cla()
        self.ax.set_axis_off()

        if self.points is None:
            self._draw_text("No data loaded")
            return

        P = self.points[self.t]  # (33,3)

        # ---- Center on torso centroid (shoulders + hips) ----
        torso_ids = [
            MP33_IDX["left_shoulder"], MP33_IDX["right_shoulder"],
            MP33_IDX["left_hip"],      MP33_IDX["right_hip"]
        ]
        c = P[torso_ids].mean(axis=0)   # (3,)
        Pc = P - c                      # recentered joints

        # ---- Draw joints and bones ----
        self.ax.scatter(Pc[:, 0], Pc[:, 1], Pc[:, 2], s=self.point_size, c="k")
        for a, b in MP33_EDGES:
            xa, ya, za = Pc[a]
            xb, yb, zb = Pc[b]
            self.ax.plot([xa, xb], [ya, yb], [za, zb],
                         linewidth=self.line_width, color="tab:blue")

        # ---- Set symmetric cubic limits so it's centered & big ----
        if self.global_R is not None:
            R = self.global_R
        else:
            span = float(max(np.ptp(Pc, axis=0).max(), 1e-6))
            R = span * 0.55  # lower => bigger; 0.50–0.65 is a good range

        self.ax.set_xlim(-R, R)
        self.ax.set_ylim(-R, R)
        self.ax.set_zlim(-R, R)
        self.ax.set_box_aspect((1, 1, 1))

        # ---- Nice default view; preserve if user rotates ----
        if not self._has_set_initial_view:
            self.ax.view_init(elev=20, azim=-70)
            self._has_set_initial_view = True

        # No ticks/grid to avoid extra margins
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.ax.grid(False)

        self.canvas.draw_idle()

    def _draw_text(self, text: str):
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
                 anchor="n", justify="center").pack(fill="x", pady=(2, 2))
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
                 text="Select configurations here:",
                 bg=self.BG_COLOR,
                 font=("Helvetica", 20, "bold"),
                 justify="center").pack(pady=(8, 0))

        btns = tk.Frame(inner, bg=self.BG_COLOR)
        btn_font = ("Helvetica", 15, "bold")

        # 3 equal columns
        btns.columnconfigure(0, weight=1, uniform="btns")
        btns.columnconfigure(1, weight=1, uniform="btns")

        # Row 0: Horizontal divider
        line = tk.Frame(btns, height=2, bg="black", bd=0, relief="solid")
        line.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        # Row 1: Session buttons
        tk.Button(btns, text="ATHLETE", font=btn_font).grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="SESSION", font=btn_font).grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        # Row 2: Select Visuals
        tk.Button(btns, text="Graph 1", font=btn_font).grid(row=2, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Graph 2", font=btn_font).grid(row=2, column=1, padx=6, pady=6, sticky="nsew")

        # Row 3: Future 
        tk.Button(btns, text="Measurement", font=btn_font).grid(row=3, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="...", font=btn_font).grid(row=3, column=1, padx=6, pady=6, sticky="nsew")

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
                 font=("Helvetica", 18, "bold"),
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
    root.title("Personalized Sports Optimization")

    # Start large
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")

    factory = SetupGui()

    # --- set column widths: videos wider than metrics ---
    # col 0 = left video, col 1 = right video, col 2 = metrics
    root.columnconfigure(0, weight=2, uniform="col")
    root.columnconfigure(1, weight=2, uniform="col")
    root.columnconfigure(2, weight=1, uniform="col")

    # Row sizing: make rows 1–2 dominant; title + bottom shorter
    ROW_WEIGHTS  = [1, 5, 5, 1]
    ROW_MINSIZE  = [70, 260, 260, 80]
    for r in range(4):
        root.rowconfigure(r, weight=ROW_WEIGHTS[r], minsize=ROW_MINSIZE[r])

    # ----- Title spans across all columns -----
    title = factory.framed_box(root, "Personalized Sports Optimization GUI", font_size=TITLE_FONT_SIZE)
    title.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD, PAD//2))

    # ----- Row 1: video / ball video / skeleton -----
    video2d_box = factory.framed_box(root, "2D video (1280x640)", font_size=14)
    video2d_box.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD//2)

    ballvideo_box = factory.framed_box(root, "Ball Tracking Video (1280x720)", font_size=14)
    ballvideo_box.grid(row=1, column=1, sticky="nsew", padx=PAD, pady=PAD//2)

    skel3d_box = factory.framed_box(root, "3D skeleton", font_size=14)
    skel3d_box.grid(row=1, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    # Mount video players inside those boxes 
    video2d_player = VideoPlayerWidget(video2d_box.inner, border=BORDER, color=TRIM_COLOR)
    video2d_player.pack(fill="both", expand=True, padx=1, pady=1)
    ball_player = VideoPlayerWidget(ballvideo_box.inner, border=BORDER, color=TRIM_COLOR)
    ball_player.pack(fill="both", expand=True, padx=1, pady=1)
    load_videos_for_current_session(video2d_player, ball_player)

    # add the 3D skeleton widget into the box and load data
    skeleton = Skeleton3DWidget(skel3d_box.inner, bg_border=BORDER, bg_panel=TRIM_COLOR, fps=FPS or 30)
    skeleton.pack(fill="both", expand=True, padx=4, pady=4)
    skeleton.load_from_session(session_dir)

    # ----- Row 2: 3D skeleton / graph1 / measurement 2 -----

    info = factory.framed_box(root, "Graph 1", font_size=14)
    info.grid(row=2, column=0, sticky="nsew", padx=PAD, pady=PAD//2)

    graph1 = factory.framed_box(root, "Graph 2", font_size=14)
    graph1.grid(row=2, column=1, sticky="nsew", padx=PAD, pady=PAD//2)

    meas1 = factory.measurement_box(root, "Measurement")
    meas1.grid(row=2, column=2, sticky="nsew", padx=PAD, pady=PAD//2)

    # --- Graph 1 content: Plotly preview + open button ---

# --- Graph 1 content: fast Seaborn/Matplotlib widget synced to clips ---
    # Build list of clip files for the current SESSION (from angles_long_df)
    session_clip_files = sorted(angles_long_df.loc[angles_long_df["session"] == SESSION, "file"].unique())
    current_clip_idx = tk.IntVar(value=0)

    # The plotting widget lives inside info.inner
    angles_plot = AnglesPlotWidget(info.inner, angles_long_df, SESSION, bg=info.inner.cget("bg"))
    angles_plot.pack(fill="both", expand=True, padx=6, pady=6)

    # Show first clip (if any)
    if session_clip_files:
        angles_plot.show_clip(session_clip_files[current_clip_idx.get()])
    else:
        # No angles found — clear the plot
        angles_plot.show_clip("")

    # Mount Graph 2 (release angles per throw) into the "Graph 2" box
    release_plot = ReleaseAnglesWidget(
        graph1.inner,
        release_rows=release_rows,
        session_id=SESSION,
        bg=graph1.inner.cget("bg")
    )
    release_plot.pack(fill="both", expand=True, padx=6, pady=6)

    # Helper to go to a specific clip index safely
    def _show_clip_at(idx: int):
        if not session_clip_files:
            return
        idx = max(0, min(idx, len(session_clip_files)-1))
        current_clip_idx.set(idx)
        angles_plot.show_clip(session_clip_files[idx])

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

    def go_prev_clip():
        _show_clip_at(current_clip_idx.get() - 1)

    def go_next_clip():
        _show_clip_at(current_clip_idx.get() + 1)

    # ----- Row 3: shorter bottom row -----
    nav = factory.navigation_box(
        root,
        on_prev_clip=go_prev_clip,
        on_restart_clip=lambda: (_show_clip_at(0), skeleton.restart(), video2d_player.restart(), ball_player.restart()),
        on_next_clip=go_next_clip,
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
