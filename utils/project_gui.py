import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

from pathlib import Path
import yaml
import re

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cv2
import pandas as pd
from PIL import Image, ImageTk

#TODO rename to session_gui or something 


# =========================
# Theme / constants
# =========================
BLUE = "#648AB6"
GRAY = "#6D737A"
BG_COLOR = BLUE
TRIM_COLOR = GRAY

BORDER = "#555"
PAD = 10
TITLE_FONT_SIZE = 50

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

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
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

ATHLETE = cfg.get("athlete", "athlete_01")
SESSION = cfg.get("session", "session_01")
FPS = int(cfg.get("player_tracking_fps", 30))

DATA_DIR = ROOT / "data" / ATHLETE
assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
session_dir = DATA_DIR / SESSION

# Discover sessions for this athlete (folders under data/<ATHLETE>)
SESSIONS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])

# Create boolean flags per session and a dict to track inclusion
SESSION_INCLUDE = {}
for s in SESSIONS:
    var = "include_" + re.sub(r"[^0-9a-zA-Z_]", "_", s)
    globals()[var] = True  # default True
    SESSION_INCLUDE[s] = True

def selected_sessions():
    out = []
    for s in SESSIONS:
        var = "include_" + re.sub(r"[^0-9a-zA-Z_]", "_", s)
        if globals().get(var, False):
            out.append(s)
    return out


# =========================
# Data I/O helpers
# =========================
def load_summary_stats(session_dir: Path) -> pd.DataFrame:
    """
    Reads metrics/summary_stats_merged.csv and returns a DataFrame with an int 'clip' column.
    Tries a few common id columns; falls back to extracting digits from 'file' if needed.
    """
    p = session_dir / "metrics" / "summary_stats_merged.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)

    # Find/normalize a clip id column
    id_candidates = ["clip", "clip_id", "free_throw", "throw", "ft", "id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)

    if id_col is None and "file" in df.columns:
        # pull digits from file names like "freethrow001_angles.csv"
        df["clip"] = pd.to_numeric(df["file"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    elif id_col is not None:
        df["clip"] = pd.to_numeric(df[id_col], errors="coerce")
    else:
        # nothing usable
        df["clip"] = pd.NA

    df = df.dropna(subset=["clip"]).copy()
    df["clip"] = df["clip"].astype(int)
    return df

def list_angle_csvs(athlete: str, sessions: list[str], root: Path = ROOT) -> pd.DataFrame:
    """Return DataFrame: ['session','clip','path'] for all *_angles.csv."""
    rows = []
    base = root / "data" / athlete
    for s in sessions:
        angles_dir = base / s / "metrics" / "3d_angles"
        if not angles_dir.exists():
            continue
        for p in sorted(angles_dir.glob("*_angles.csv")):
            m = re.search(r"(\d+)", p.stem)
            clip = int(m.group(1)) if m else None
            rows.append({"session": s, "clip": clip, "path": p})
    return pd.DataFrame(rows)

def load_angles_long(angle_files: pd.DataFrame, downcast_float32: bool = True) -> pd.DataFrame:
    """
    Read all *_angles.csv into long/tidy columns:
    ['athlete','session','clip','file','frame','time_s','angle','value']
    """
    dfs = []
    for _, row in angle_files.iterrows():
        p = Path(row["path"])
        df = pd.read_csv(p)

        if "frame" not in df.columns:
            df.insert(0, "frame", np.arange(len(df), dtype=int))

        long = df.melt(id_vars=["frame"], var_name="angle", value_name="value")
        long["athlete"] = ATHLETE
        long["session"] = row["session"]
        long["clip"] = row["clip"]
        long["file"] = p.name

        if downcast_float32:
            long["value"] = pd.to_numeric(long["value"], errors="coerce").astype("float32")

        dfs.append(long)

    if not dfs:
        return pd.DataFrame(columns=["athlete","session","clip","file","frame","time_s","angle","value"])

    out = pd.concat(dfs, ignore_index=True)
    out["time_s"] = out["frame"] / float(max(FPS, 1))
    out = out[["athlete","session","clip","file","frame","time_s","angle","value"]]
    return out

def list_phase_csvs(athlete: str, sessions: list[str], root: Path = ROOT) -> pd.DataFrame:
    """Discover freethrow_phases.csv under each selected session."""
    rows = []
    base = root / "data" / athlete
    for s in sessions:
        session_root = base / s
        for p in session_root.rglob("freethrow_phases.csv"):
            rows.append({"session": s, "path": p})
    return pd.DataFrame(rows)

def extract_clip_id_from_file_str(file_str: str):
    """'clip_003.mp4' -> 3, else None."""
    try:
        stem = Path(str(file_str)).stem
    except Exception:
        stem = str(file_str)
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None

def extract_clip_id_from_any_name(name: str) -> int | None:
    m = re.search(r"(\d+)", str(name))
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

def load_phases_table(phase_files: pd.DataFrame, athlete: str, fps: float = 60) -> pd.DataFrame:
    """
    Output columns:
    ['athlete','session','clip','file','windup_start','release_frame','followthrough_end',
     'windup_t','release_t','followthrough_t']
    """
    dfs = []
    for _, row in phase_files.iterrows():
        p = Path(row["path"])
        df = pd.read_csv(p)

        cols = {c.lower(): c for c in df.columns}
        need = ["file","windup_start","release_frame","followthrough_end"]
        missing = [c for c in need if c not in cols]
        if missing:
            raise ValueError(f"Missing columns in {p}: {missing}")

        df = df[[cols[c] for c in need]].copy()
        df.columns = need

        df["athlete"] = athlete
        df["session"] = row["session"]
        df["clip"] = df["file"].apply(extract_clip_id_from_file_str)

        denom = float(max(fps, 1))
        df["windup_t"] = df["windup_start"] / denom
        df["release_t"] = df["release_frame"] / denom
        df["followthrough_t"] = df["followthrough_end"] / denom

        for c in ["windup_start","release_frame","followthrough_end"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=[
            "athlete","session","clip","file",
            "windup_start","release_frame","followthrough_end",
            "windup_t","release_t","followthrough_t"
        ])

    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=["clip"])
    out = out[[
        "athlete","session","clip","file",
        "windup_start","release_frame","followthrough_end",
        "windup_t","release_t","followthrough_t"
    ]]
    return out

def list_keypoints_with_clips(dir_path: Path) -> dict[int, Path]:
    """
    Return {clip_id -> Path} for keypoint CSVs in dir_path.
    Assumes filenames contain the clip number (e.g., clip_003_keypoints.csv).
    """
    out = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.glob("*.csv")):
        cid = extract_clip_id_from_any_name(p.name)
        if cid is not None:
            out[cid] = p
    return out



# =========================
# Analytics helpers
# =========================
def _ensure_clip_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric 'clip' exists; derive from 'file' if needed."""
    if "clip" not in df.columns:
        if "file" in df.columns:
            clip = df["file"].astype(str).str.extract(r"(\d+)")[0]
            df["clip"] = pd.to_numeric(clip, errors="coerce")
        else:
            raise KeyError("Dataframe needs 'clip' or 'file' to sort throws.")
    return df

def get_release_rows(angles_long_df: pd.DataFrame,
                     phases_df: pd.DataFrame,
                     sessions: str | list[str]) -> pd.DataFrame:
    """
    For each session/clip/file, take angle values at the clip's release_frame.
    Returns ['athlete','session','clip','file','angle','value'] at release.
    """
    if isinstance(sessions, str):
        sessions = [sessions]
    ang = angles_long_df[angles_long_df["session"].isin(sessions)].copy()
    ph = phases_df[phases_df["session"].isin(sessions)].copy()

    need_ang = {"athlete","session","clip","file","frame","angle","value"}
    need_ph = {"athlete","session","clip","file","release_frame"}
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
    m = (
        m.drop_duplicates(subset=["athlete","session","clip","file","angle"])
         .sort_values(["session","clip","file","angle"])
    )
    out = m[["athlete","session","clip","file","angle","value"]].reset_index(drop=True)
    return out


# =========================
# Widgets
# =========================
class AnglesPlotWidget(tk.Frame):
    """Plot all angles vs frame for one clip at a time."""
    def __init__(self, parent, angles_long_df, session_id, *, bg="#ffffff"):
        super().__init__(parent, bg=bg)

        self.angles_long_df = angles_long_df[angles_long_df["session"] == session_id].copy()
        self.by_file = {}
        for f, g in self.angles_long_df.groupby("file"):
            gg = g.copy().sort_values("frame")
            gg["frame"] = pd.to_numeric(gg["frame"], errors="coerce")
            self.by_file[f] = gg

        self.visible_angles = None   # None => show all
        self.current_file = None
        self.fig = Figure(figsize=(4, 3), dpi=100, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Angles vs Frame (per clip)")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def set_visible_angles(self, angles: set[str] | None):
        """Update which angles should be shown, then redraw current clip."""
        self.visible_angles = set(angles) if angles is not None else None
        if self.current_file is not None:
            self.show_clip(self.current_file)

    def show_clip(self, file_name: str):
        self.current_file = file_name
        if file_name not in self.by_file:
            self.ax.cla()
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("Angle (deg)")
            self.ax.set_title(f"No data for {file_name}")
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
            return
        
        df = self.by_file[file_name]
        self.ax.cla()
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title(f"Angles vs Frame — {file_name}")
        self.ax.grid(True, alpha=0.3)

        # filter by visible angles if provided
        if self.visible_angles is not None:
            df = df[df["angle"].isin(self.visible_angles)]

        for angle, g in df.groupby("angle"):
            self.ax.plot(g["frame"].values, g["value"].values, linewidth=1.5, label=str(angle))

        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        self.fig.tight_layout()
        self.canvas.draw_idle()


class ReleaseAnglesWidget(tk.Frame):
    """
    Release-time angle values per throw for one session.
    One line per angle; X = clip id; Y = value at release.
    """
    def __init__(self, parent, release_rows: pd.DataFrame, session_id: str, *, bg="#ffffff"):
        super().__init__(parent, bg=bg)

        df = release_rows[release_rows["session"] == session_id].copy()
        df = _ensure_clip_col(df)
        if df.empty:
            self.df = pd.DataFrame(columns=["clip","angle","value","file"])
        else:
            df = df.dropna(subset=["clip"]).copy()
            df["clip"] = df["clip"].astype(int)
            df = df.sort_values("clip")
            self.df = df

        self.clips = sorted(self.df["clip"].unique().tolist())
        self.pos_by_clip = {c: i for i, c in enumerate(self.clips)}

        self.visible_angles = None
        self._last_highlight_clip = None
        self.fig = Figure(figsize=(4, 3), dpi=100, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Throw (clip id)")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Release Angle per Throw")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self._vline = None
        self._highlight_scats = []

        self._draw_static()

    def set_visible_angles(self, angles: set[str] | None):
        """Update which angles should be shown and redraw."""
        self.visible_angles = set(angles) if angles is not None else None
        self._draw_static()
        # restore highlight if we had one
        if self._last_highlight_clip is not None:
            self.set_current_clip(self._last_highlight_clip)

    def _draw_static(self):
        self.ax.cla()
        self.ax.set_xlabel("Throw (clip id)")
        self.ax.set_ylabel("Angle (deg)")
        self.ax.set_title("Release Angle per Throw")
        self.ax.grid(True, alpha=0.3)

        # apply filter
        df = self.df
        if self.visible_angles is not None:
            df = df[df["angle"].isin(self.visible_angles)]

        # check the FILTERED df, not self.df
        if df.empty or not self.clips:
            self.ax.text(0.5, 0.5, "No angles selected", ha="center", va="center",
                         transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        # plot FILTERED df
        for angle, g in df.groupby("angle"):
            x = g["clip"].map(self.pos_by_clip).to_numpy()
            y = g["value"].to_numpy()
            self.ax.plot(x, y, marker="o", linewidth=1.5, label=str(angle))

        self.ax.set_xticks(range(len(self.clips)))
        self.ax.set_xticklabels([str(c) for c in self.clips], rotation=0)

        # only show legend if we actually drew lines
        if len(self.ax.lines) > 0:
            self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        # DO NOT call self.fig.tight_layout() here
        self.canvas.draw_idle()

    def set_current_clip(self, clip_id: int | None):
        self._last_highlight_clip = clip_id

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
        self._vline = self.ax.axvline(x=x0, linestyle="--", linewidth=1.0, alpha=0.6)

        # use the same filter that _draw_static used
        df = self.df
        if self.visible_angles is not None:
            df = df[df["angle"].isin(self.visible_angles)]

        dfc = df[df["clip"] == clip_id]
        if not dfc.empty:
            yvals = dfc["value"].to_numpy()
            xs = np.full_like(yvals, x0, dtype=float)
            scat = self.ax.scatter(xs, yvals, s=30, zorder=5)
            self._highlight_scats.append(scat)

        self.canvas.draw_idle()


class AngleFilterPanel(tk.Frame):
    """
    A scrollable grid of Checkbuttons for angles with 'All' / 'None' controls.
    Calls on_change(selected_set) whenever the selection changes.
    """
    def __init__(self, parent, angles: list[str], on_change):
        super().__init__(parent, bg=BG_COLOR)
        self.on_change = on_change
        self.vars: dict[str, tk.BooleanVar] = {}

        # top controls
        top = tk.Frame(self, bg=BG_COLOR)
        top.pack(fill="x", padx=6, pady=(6, 0))
        tk.Label(top, text="Visible angles", bg=BG_COLOR, font=("Helvetica", 13, "bold")).pack(side="left")
        tk.Button(top, text="All", command=self._select_all).pack(side="right", padx=4)
        tk.Button(top, text="None", command=self._select_none).pack(side="right")

        # scrollable area
        container = tk.Frame(self, bg=BG_COLOR)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        canvas = tk.Canvas(container, highlightthickness=0)
        vsb = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=BG_COLOR)

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # lay out checkbuttons in two columns
        cols = 2
        for i, angle in enumerate(angles):
            var = tk.BooleanVar(value=True)
            self.vars[angle] = var
            cb = tk.Checkbutton(self.inner, text=str(angle), variable=var, bg=BG_COLOR,
                                command=self._notify, anchor="w", justify="left")
            r, c = divmod(i, cols)
            self.inner.grid_columnconfigure(c, weight=1)
            cb.grid(row=r, column=c, sticky="ew", padx=4, pady=2)

    def _notify(self):
        selected = {a for a, v in self.vars.items() if v.get()}
        self.on_change(selected)

    def _select_all(self):
        for v in self.vars.values():
            v.set(True)
        self._notify()

    def _select_none(self):
        for v in self.vars.values():
            v.set(False)
        self._notify()



class VideoPlayerWidget(tk.Frame):
    """Lightweight video widget for Tkinter."""
    def __init__(self, parent, border="#555", color=BG_COLOR):
        super().__init__(parent, bg=border)
        self.border, self.color = border, color
        self.cap = None
        self.playing = False
        self.photo = None

        inner = tk.Frame(self, bg=self.color)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.holder = tk.Frame(inner, bg="black", padx=1, pady=1)
        self.holder.pack(fill="both", expand=True, padx=1, pady=1)
        self.holder.pack_propagate(False)

        self.canvas = tk.Canvas(self.holder, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.aspect = None
        self.canvas.bind("<Configure>", self._on_resize)

        self.info = tk.StringVar(value="No video loaded")
        tk.Label(inner, textvariable=self.info, bg=self.color, font=("Helvetica", 11)).pack(fill="x", pady=(0, 0))

        self._last_frame_rgb = None
        self._fps_ms = 33

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

    def _loop(self):
        if not self.playing or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
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
        if self.aspect:
            parent = self.holder.master
            cw = parent.winfo_width() or 1
            ch = parent.winfo_height() or 1
            target_w = cw
            target_h = int(target_w / self.aspect)
            if target_h > ch:
                target_h = ch
                target_w = int(target_h * self.aspect)
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
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.photo)


# ---- MediaPipe 33 names and edges ----
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
MP33_IDX = {n: i for i, n in enumerate(MP33_NAMES)}
MP33_EDGES = [
    (11,13),(13,15), (12,14),(14,16),
    (11,12),(23,24), (11,23),(12,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32),
]

def load_mp33_csv_compact(path: Path) -> np.ndarray:
    """
    CSV columns like 'left_shoulder_x','left_shoulder_y','left_shoulder_z', etc.
    Returns (T, 33, 3) float32.
    """
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    cols = []
    for name in MP33_NAMES:
        for a in ("x", "y", "z"):
            col = f"{name}_{a}"
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path.name}")
            cols.append(col)
    arr = df[cols].to_numpy(float).reshape((-1, 33, 3)).astype(np.float32)
    return arr


class Skeleton3DWidget(tk.Frame):
    """3D skeleton viewer for MediaPipe-33 keypoints (embedded Matplotlib)."""
    TORSO_NAMES = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    TORSO_IDS = [MP33_IDX[n] for n in TORSO_NAMES]

    def __init__(self, parent, *, bg_border="#555", bg_panel=TRIM_COLOR, fps=FPS,
                point_size=60, line_width=7, use_global_limits=False, fixed_R=None,
                init_view=(90, 90),   # (elev, azim): up/down tilt, rotation around Z axis
                zoom_factor=0.42,      #  smaller values = more zoomed in
                bias_axis = "y",
                height_bias=0.62, # fraction of R to shift upward 
                roll_deg=-180, 
                roll_axis='y'
                ): 
        super().__init__(parent, bg=bg_border)
        # Config
        self.fps = fps
        self.point_size = point_size
        self.line_width = line_width
        self.use_global_limits = use_global_limits
        self.fixed_R = fixed_R          # if not None, overrides all R calc
        self.init_view = init_view      # (elev, azim)
        self.zoom_factor = zoom_factor  # used when computing R from span
        self.roll_deg = float(roll_deg)
        self.roll_axis = roll_axis
        self.bias_axis = bias_axis
        self.height_bias = float(height_bias)

        # State
        self.points = None
        self.T = 0
        self.t = 0
        self.playing = False
        self.global_R = None
        self._has_set_initial_view = False

        # Panel + status
        panel = tk.Frame(self, bg=bg_panel)
        panel.pack(fill="both", expand=True, padx=2, pady=2)

        self.info_var = tk.StringVar(value="No skeleton loaded")
        tk.Label(panel, textvariable=self.info_var, bg=bg_panel, font=("Helvetica", 11)).pack(
            fill="x", pady=(0, 0)
        )

        # Figure / Axes
        self.fig = Figure(figsize=(3, 3), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_position([0, 0, 1, 1])
        self.ax.set_box_aspect((1, 1, 1))

        self.canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas_widget.bind("<Configure>", lambda _e: self._draw())

    # --------- Data loading ---------
    def load_from_session(self, session_dir: Path):
        key_dir = session_dir / "metrics" / "3d_keypoints"
        csv = next((p for p in sorted(key_dir.glob("*.csv"))), None)
        if not csv:
            self._draw_text(f"No 3D keypoints in:\n{key_dir}")
            return
        self.load_file(csv)

    def load_file(self, path: Path):
        try:
            P = load_mp33_csv_compact(path).astype(np.float32)
        except Exception as e:
            self._draw_text(f"Failed to load:\n{Path(path).name}\n{e}")
            self.info_var.set(f"Failed to load: {Path(path).name}")
            return

        # Reset state
        self.points = P
        self.T = int(P.shape[0])
        self.t = 0
        self._has_set_initial_view = False

        # Pre-compute a global R if requested (based on torso-centered cloud across all frames)
        if self.use_global_limits and self.points is not None and self.points.size > 0:
            centroids = self.points[:, self.TORSO_IDS, :].mean(axis=1)
            Pc_all = self.points - centroids[:, None, :]
            span_all = np.max(np.ptp(Pc_all, axis=1))  # max across frames of overall extent
            self.global_R = float(max(span_all, 1e-6) * self.zoom_factor)
        else:
            self.global_R = None

        self.info_var.set(f"{Path(path).name} — {self.T} frames")
        self._draw()

    # --------- Playback controls ---------
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

    def _loop(self):
        if not self.playing: return
        self.t += 1
        if self.t >= self.T:
            self.t = self.T - 1
            self.playing = False
        self._draw()
        self.after(max(1, int(1000 / max(self.fps, 1))), self._loop)

    # --------- Draw helpers ---------
    def _compute_R(self, Pc_frame: np.ndarray) -> float:
        """Choose display radius R based on priority: fixed_R > global_R > per-frame span."""
        if self.fixed_R is not None:
            return float(self.fixed_R)
        if self.global_R is not None:
            return float(self.global_R)
        span = float(max(np.ptp(Pc_frame, axis=0).max(), 1e-6))
        return span * self.zoom_factor

    def _clear_axes(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.set_box_aspect((1, 1, 1))

    @staticmethod
    def _rotate_about_axis(points, angle_deg, axis='z'):
        """Rotate Nx3 points by angle around an axis (x, y, or z)."""
        a = np.deg2rad(angle_deg)
        ca, sa = np.cos(a), np.sin(a)
        if axis == 'x':
            R = np.array([[1, 0, 0],
                        [0, ca, -sa],
                        [0, sa,  ca]], dtype=float)
        elif axis == 'y':
            R = np.array([[ ca, 0, sa],
                        [  0, 1,  0],
                        [-sa, 0, ca]], dtype=float)
        else:  # 'z'
            R = np.array([[ca, -sa, 0],
                        [sa,  ca, 0],
                        [ 0,   0, 1]], dtype=float)
        return points @ R.T

    def _draw(self):
        self._clear_axes()
        if self.points is None:
            self._draw_text("No data loaded")
            return

        # center on torso
        P = self.points[self.t]
        c = P[self.TORSO_IDS].mean(axis=0)
        Pc = P - c  # centered
        if self.roll_deg:
            Pc = self._rotate_about_axis(Pc, self.roll_deg, axis=self.roll_axis)

        # draw points & bones
        self.ax.scatter(Pc[:, 0], Pc[:, 1], Pc[:, 2], s=self.point_size, c="k")
        for a, b in MP33_EDGES:
            xa, ya, za = Pc[a]; xb, yb, zb = Pc[b]
            self.ax.plot([xa, xb], [ya, yb], [za, zb], linewidth=self.line_width, color="tab:blue")

        # limits with upward bias
        R = self._compute_R(Pc)
        b = self.height_bias * R

        if self.bias_axis == "x":
            self.ax.set_xlim(-R + b, R + b)
            self.ax.set_ylim(-R, R)
            self.ax.set_zlim(-R, R)
        elif self.bias_axis == "y":
            self.ax.set_xlim(-R, R)
            self.ax.set_ylim(-R + b, R + b)
            self.ax.set_zlim(-R, R)
        else:  # "z" (default)
            self.ax.set_xlim(-R, R)
            self.ax.set_ylim(-R, R)
            self.ax.set_zlim(-R + b, R + b)

        if not self._has_set_initial_view:
            elev, azim = self.init_view
            self.ax.view_init(elev=elev, azim=azim)
            self._has_set_initial_view = True

        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.ax.grid(False)
        self.canvas.draw_idle()

    def _draw_text(self, text: str):
        self._clear_axes()
        self.ax.text2D(0.5, 0.5, text, transform=self.ax.transAxes,
                       ha="center", va="center", fontsize=11)
        self.canvas.draw_idle()



# =========================
# GUI factory
# =========================
class SetupGui:
    def __init__(self, border=BORDER, BG_COLOR=BG_COLOR):
        self.BORDER = border
        self.BG_COLOR = BG_COLOR

    def framed_box(self, parent, title, font_size, subtitle=None):
        outer = tk.Frame(parent, bg=self.BORDER)
        inner = tk.Frame(outer, bg=self.BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
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
                bg=self.BG_COLOR, font=("Helvetica", 16, "bold")).pack(pady=(8, 4))
        tf = tk.Frame(inner, bg=self.BG_COLOR)

        text = tk.Text(tf, height=10, wrap="word")
        sb = ttk.Scrollbar(tf, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sb.set)

        text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tf.pack(fill="both", expand=True, padx=10, pady=10)

        return outer, text  


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

        btns.columnconfigure(0, weight=1, uniform="btns")
        btns.columnconfigure(1, weight=1, uniform="btns")

        line = tk.Frame(btns, height=2, bg="black", bd=0, relief="solid")
        line.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        tk.Button(btns, text="ATHLETE", font=btn_font).grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="SESSION", font=btn_font).grid(row=1, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Graph 1", font=btn_font).grid(row=2, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Graph 2", font=btn_font).grid(row=2, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Measurement", font=btn_font).grid(row=3, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="...", font=btn_font).grid(row=3, column=1, padx=6, pady=6, sticky="nsew")

        btns.pack(pady=6)
        return outer


# =========================
# Load data tables
# =========================
phase_files_df = list_phase_csvs(ATHLETE, selected_sessions())
print(f"Found {len(phase_files_df)} phase files across {phase_files_df['session'].nunique() if not phase_files_df.empty else 0} sessions.")
phases_df = load_phases_table(phase_files_df, athlete=ATHLETE, fps=FPS)
print(f"Loaded {len(phases_df)} phase rows across {phases_df['session'].nunique() if not phases_df.empty else 0} sessions.")

angle_files_df = list_angle_csvs(ATHLETE, selected_sessions())
print(f"Found {len(angle_files_df)} angle files across {angle_files_df['session'].nunique() if not angle_files_df.empty else 0} sessions.")
angles_long_df = load_angles_long(angle_files_df)
print(f"Loaded {len(angles_long_df)} angle rows across {angles_long_df['session'].nunique() if not angles_long_df.empty else 0} sessions.")

release_rows = get_release_rows(angles_long_df, phases_df, sessions=SESSION)


# =========================
# App entry
# =========================
def main():
    root = tk.Tk()
    root.title("Personalized Sports Optimization")

    def on_angles_changed(selected: set[str]):
        # push the selection to both plots
        angles_plot.set_visible_angles(selected if selected else set())
        release_plot.set_visible_angles(selected if selected else set())

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")

    # --- Build clip->video maps for syncing with selected clip ---
    two_d_dir = session_dir / "videos" / "player_tracking" / "2d"
    ball_dir  = session_dir / "videos" / "ball_tracking" / "raw"
    two_d_map = list_videos_with_clips(two_d_dir)   # dict[int, Path]
    ball_map  = list_videos_with_clips(ball_dir)    # dict[int, Path]
    keypoints_dir = session_dir / "metrics" / "3d_keypoints"
    kp_map = list_keypoints_with_clips(keypoints_dir)

    factory = SetupGui()

    # Layout: videos wider than metrics
    root.columnconfigure(0, weight=2, uniform="col")
    root.columnconfigure(1, weight=2, uniform="col")
    root.columnconfigure(2, weight=1, uniform="col")

    ROW_WEIGHTS = [1, 5, 5, 1]
    ROW_MINSIZE = [70, 260, 260, 80]
    for r in range(4):
        root.rowconfigure(r, weight=ROW_WEIGHTS[r], minsize=ROW_MINSIZE[r])

    # --- create tkinter boxes to display widgets in ---
    
    #title
    title = factory.framed_box(root, "Personalized Sports Optimization", font_size=TITLE_FONT_SIZE)
    title.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=PAD, pady=(PAD, PAD//2))
    #row 1
    video2d_box = factory.framed_box(root, "2D video (1280x640)", font_size=14)
    video2d_box.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD//2)
    ballvideo_box = factory.framed_box(root, "Ball Tracking Video (1280x720)", font_size=14)
    ballvideo_box.grid(row=1, column=1, sticky="nsew", padx=PAD, pady=PAD//2)
    skel3d_box = factory.framed_box(root, "3D skeleton", font_size=14)
    skel3d_box.grid(row=1, column=2, sticky="nsew", padx=PAD, pady=PAD//2)
    #row 2
    graph1 = factory.framed_box(root, "Angles over Frames per Free Throw", font_size=14)
    graph1.grid(row=2, column=0, sticky="nsew", padx=PAD, pady=PAD//2)
    graph2 = factory.framed_box(root, "Average Release Angles per Free Throw", font_size=14)
    graph2.grid(row=2, column=1, sticky="nsew", padx=PAD, pady=PAD//2)
    meas_box, meas_text = factory.measurement_box(root, "Summary stats")
    meas_box.grid(row=2, column=2, sticky="nsew", padx=PAD, pady=PAD//2)
    #row 3
    filters_box = factory.framed_box(root, "Angle Filters", font_size=14)
    filters_box.grid(row=3, column=1, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))
    select = factory.selection_box(root)
    select.grid(row=3, column=2, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))
    # navigation is farther below 

    # Mount video players
    video2d_player = VideoPlayerWidget(video2d_box.inner, border=BORDER, color=TRIM_COLOR)
    video2d_player.pack(fill="both", expand=True, padx=1, pady=1)
    ball_player = VideoPlayerWidget(ballvideo_box.inner, border=BORDER, color=TRIM_COLOR)
    ball_player.pack(fill="both", expand=True, padx=1, pady=1)

    # Mount 3D skeleton
    skeleton = Skeleton3DWidget(skel3d_box.inner, bg_border=BORDER, bg_panel=TRIM_COLOR, fps=FPS or 30)
    skeleton.pack(fill="both", expand=True, padx=4, pady=4)
    skeleton.load_from_session(session_dir)

    # Load summary stats 
    summary_df = load_summary_stats(session_dir)

    # get session info 
    session_clip_files = sorted(angles_long_df.loc[angles_long_df["session"] == SESSION, "file"].unique())
    current_clip_idx = tk.IntVar(value=0)
    session_angles = sorted(map(str, angles_long_df.loc[angles_long_df["session"] == SESSION, "angle"].unique()))

    # Plot angles over frames per free throw
    angles_plot = AnglesPlotWidget(graph1.inner, angles_long_df, SESSION, bg=graph1.inner.cget("bg"))
    angles_plot.pack(fill="both", expand=True, padx=6, pady=6)

    # Plot average release angles per throw 
    release_plot = ReleaseAnglesWidget(
        graph2.inner,
        release_rows=release_rows,
        session_id=SESSION,
        bg=graph1.inner.cget("bg")
    )
    release_plot.pack(fill="both", expand=True, padx=6, pady=6)

    # Mount the panel into the box
    angle_panel = AngleFilterPanel(filters_box.inner, session_angles, on_change=on_angles_changed)
    angle_panel.pack(fill="both", expand=True, padx=6, pady=6)
    on_angles_changed(set(session_angles))

    def render_summary_for_clip(clip_id: int | None):
        meas_text.configure(state="normal")
        meas_text.delete("1.0", "end")

        if clip_id is None or summary_df.empty:
            meas_text.insert("end", "No summary stats available.\n")
            meas_text.configure(state="disabled")
            return

        row = summary_df.loc[summary_df["clip"] == clip_id]
        if row.empty:
            meas_text.insert("end", f"No stats for clip {clip_id}.\n")
            meas_text.configure(state="disabled")
            return

        # Take first match if duplicates
        s = row.iloc[0].to_dict()

        # Columns to ignore in the printout
        ignore = {"clip", "file", "session", "athlete"}
        keys = [k for k in s.keys() if k not in ignore]

        # Stable, readable order
        for k in sorted(keys):
            v = s[k]
            # Nice numeric formatting
            if isinstance(v, (int, np.integer)):
                line = f"{k}: {int(v)}\n"
            elif isinstance(v, (float, np.floating)):
                line = f"{k}: {v:.3f}\n"
            else:
                line = f"{k}: {v}\n"
            meas_text.insert("end", line)

        meas_text.configure(state="disabled")


    def _show_clip_at(idx: int):
        if not session_clip_files:
            return
        idx = max(0, min(idx, len(session_clip_files) - 1))
        current_clip_idx.set(idx)

        file_name = session_clip_files[idx]
        angles_plot.show_clip(file_name)

        # highlight on release graph
        clip_id = extract_clip_id_from_any_name(file_name)
        if clip_id is not None:
            release_plot.set_current_clip(clip_id)

        if clip_id is not None:
            render_summary_for_clip(clip_id)

        # load matching videos if present
        if clip_id is not None:
            if clip_id in two_d_map:
                video2d_player.load(two_d_map[clip_id])
            if clip_id in ball_map:
                ball_player.load(ball_map[clip_id])

        if clip_id in kp_map:
            skeleton.load_file(kp_map[clip_id])
        else:
            # if no per-clip skeleton exists, just restart the skeleton
            skeleton.restart()

    # Initialize first clip (shows angles + loads matching videos)
    if session_clip_files:
        _show_clip_at(0)
    else:
        angles_plot.show_clip("")

    # --- Playback / navigation helpers ---
    def playpause_all():
        for vp in (video2d_player, ball_player):
            if getattr(vp, "playing", False):
                vp.pause()
            else:
                vp.play()
        if skeleton.playing:
            skeleton.pause()
        else:
            skeleton.play()

    def restart_all():
        # keep the current clip index
        idx = current_clip_idx.get()

        # restart videos & skeleton (seek to frame 0 for each)
        for vp in (video2d_player, ball_player):
            vp.restart()
        skeleton.restart()

        # re-render graphs for the SAME clip (no index change)
        if session_clip_files:
            file_name = session_clip_files[idx]
            angles_plot.show_clip(file_name)
            clip_id = extract_clip_id_from_any_name(file_name)
            if clip_id is not None:
                release_plot.set_current_clip(clip_id)


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

    # Row 3: navigation + selection
    def navigation_box(parent,
                       on_prev_clip=None, on_restart_clip=None, on_next_clip=None,
                       on_prev_frame=None, on_playpause=None, on_next_frame=None):
        on_prev_clip    = on_prev_clip     or (lambda: None)
        on_restart_clip = on_restart_clip  or (lambda: None)
        on_next_clip    = on_next_clip     or (lambda: None)
        on_prev_frame   = on_prev_frame    or (lambda: None)
        on_playpause    = on_playpause     or (lambda: None)
        on_next_frame   = on_next_frame    or (lambda: None)

        outer = tk.Frame(parent, bg=BORDER)
        inner = tk.Frame(outer, bg=BG_COLOR)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner,
                 text="Navigation Control Center",
                 bg=BG_COLOR,
                 font=("Helvetica", 18, "bold"),
                 justify="center").pack(pady=(8, 0))

        btns = tk.Frame(inner, bg=BG_COLOR)
        btns.columnconfigure(0, weight=1, uniform="btns")
        btns.columnconfigure(1, weight=1, uniform="btns")
        btns.columnconfigure(2, weight=1, uniform="btns")

        line = tk.Frame(btns, height=2, bg="black", bd=0, relief="solid")
        line.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(4, 8))

        btn_font_big = ("Helvetica", 20, "bold")
        tk.Button(btns, text="Previous Clip", font=btn_font_big, command=on_prev_clip).grid(row=1, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Restart Clip",  font=btn_font_big, command=on_restart_clip).grid(row=1, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Next Clip",     font=btn_font_big, command=on_next_clip).grid(row=1, column=2, padx=6, pady=6, sticky="nsew")

        btn_font = ("Helvetica", 18, "bold")
        tk.Button(btns, text="Previous Frame", font=btn_font, command=on_prev_frame).grid(row=2, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Pause/Play",     font=btn_font, command=on_playpause).grid(row=2, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Next Frame",     font=btn_font, command=on_next_frame).grid(row=2, column=2, padx=6, pady=6, sticky="nsew")

        # future row placeholders (kept for layout)
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(btns, text="Future Button", font=btn_font).grid(row=3, column=2, padx=6, pady=6, sticky="nsew")

        btns.pack(pady=6)
        return outer

    nav = navigation_box(
        root,
        on_prev_clip=go_prev_clip,
        on_restart_clip=restart_all,
        on_next_clip=go_next_clip,
        on_prev_frame=prev_frame_all,
        on_playpause=playpause_all,
        on_next_frame=next_frame_all
    )
    nav.grid(row=3, column=0, sticky="nsew", padx=PAD, pady=(PAD//2, PAD))

    root.mainloop()


if __name__ == "__main__":
    main()
