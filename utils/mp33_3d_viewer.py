# mp33_3d_viewer.py
import tkinter as tk
from tkinter import filedialog, Label, Button
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import yaml
import re

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
default_angles_dir = session_dir / "metrics" / "angles"  # overrideable in UI

# Heuristics for where phases might live (tried in order)
PHASE_FILE_CANDIDATES = [
    session_dir / "phases" / "freethrow_phases.csv",
    session_dir / "metrics" / "phases" / "freethrow_phases.csv",
    session_dir / "freethrow_phases.csv",
]

# =========================
# MediaPipe 33 metadata
# =========================
NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]
IDX = {n:i for i,n in enumerate(NAMES)}

# Minimal skeleton for MediaPipe (0-based indices)
EDGES = np.array([
    [11,13],[13,15],   # left arm
    [12,14],[14,16],   # right arm
    [11,12],           # shoulders
    [23,24],           # hips
    [11,23],[12,24],   # torso
    [23,25],[25,27],   # left leg
    [24,26],[26,28],   # right leg
    [0,11],[0,12],     # head to shoulders
], dtype=int)

# =========================
# Keypoint I/O
# =========================
def load_mp33_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)

    cols = []
    for name in NAMES:
        req = [f"{name}_x", f"{name}_y", f"{name}_z"]
        for r in req:
            if r not in df.columns:
                raise ValueError(f"Missing column '{r}' in {path.name}")
        cols.extend(req)

    arr = df[cols].to_numpy(float)  # (T, 99)
    T = arr.shape[0]
    arr = arr.reshape(T, len(NAMES), 3)  # (T,33,3)
    return arr

# =========================
# Angle computation
# =========================
def _series_points(frames, name):
    return frames[:, IDX[name], :]  # (T,3)

def angle_series(frames, a, b, c):
    A = _series_points(frames, a)
    B = _series_points(frames, b)
    C = _series_points(frames, c)

    v1 = A - B
    v2 = C - B
    valid = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    valid &= denom > 1e-8

    ang = np.full(len(frames), np.nan, float)
    if np.any(valid):
        cosang = np.einsum('ij,ij->i', v1[valid], v2[valid]) / denom[valid]
        cosang = np.clip(cosang, -1.0, 1.0)
        ang[valid] = np.degrees(np.arccos(cosang))
    return ang

def compute_angles(frames):
    ang = {}
    # Arms
    ang["elbow_flex_l"]     = angle_series(frames, "left_shoulder",  "left_elbow",  "left_wrist")
    ang["elbow_flex_r"]     = angle_series(frames, "right_shoulder", "right_elbow", "right_wrist")
    ang["shoulder_flex_l"]  = angle_series(frames, "left_hip",       "left_shoulder","left_elbow")
    ang["shoulder_flex_r"]  = angle_series(frames, "right_hip",      "right_shoulder","right_elbow")
    # Legs
    ang["hip_flex_l"]       = angle_series(frames, "left_shoulder",  "left_hip",   "left_knee")
    ang["hip_flex_r"]       = angle_series(frames, "right_shoulder", "right_hip",  "right_knee")
    ang["knee_flex_l"]      = angle_series(frames, "left_hip",       "left_knee",  "left_ankle")
    ang["knee_flex_r"]      = angle_series(frames, "right_hip",      "right_knee", "right_ankle")
    ang["ankle_flex_l"]     = angle_series(frames, "left_knee",      "left_ankle", "left_foot_index")
    ang["ankle_flex_r"]     = angle_series(frames, "right_knee",     "right_ankle","right_foot_index")
    return ang

# =========================
# Angles I/O
# =========================
def name_match_candidates(keypoints_csv: Path):
    """Return possible angle filenames (stemwise) derived from a *_3d.csv."""
    stem = keypoints_csv.stem
    base = stem.replace("_3d", "")
    return [f"{base}_angles.csv", f"{base}.csv"]

def find_angles_for_file(keypoints_csv: Path, preferred_dir: Path | None, fallback_dir: Path):
    """Try preferred dir first (if set), then fallback dir, with multiple candidates."""
    candidates = name_match_candidates(keypoints_csv)
    search_dirs = []
    if preferred_dir is not None:
        search_dirs.append(preferred_dir)
    search_dirs.append(fallback_dir)

    for d in search_dirs:
        for c in candidates:
            p = (d / c)
            if p.exists():
                return p
    return None

def load_angles_csv(angles_csv: Path):
    df = pd.read_csv(angles_csv)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
    cols = [c for c in df.columns if c != "frame"]
    ang = {c: df[c].to_numpy(float) for c in cols}
    return ang, cols, len(df)

# =========================
# Phases I/O (triplets per clip)
# =========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())

def _canonical_clip_key(path_or_name: str) -> str:
    """
    Canonicalize a file name/stem so that:
      freethrow001.csv
      freethrow001_3d.csv
      freethrow001_angles.csv
    all map to 'freethrow001'.
    """
    s = _norm(Path(path_or_name).stem)
    s = re.sub(r"_(3d|angles)$", "", s)  # drop trailing markers
    return s

def _auto_phase_file_hints(current_keypoints_file: Path | None) -> list[Path]:
    hints = []
    hints.extend(PHASE_FILE_CANDIDATES)
    if current_keypoints_file is not None:
        hints.append(current_keypoints_file.parent / "freethrow_phases.csv")
        hints.append(current_keypoints_file.parent.parent / "freethrow_phases.csv")
    seen, out = set(), []
    for p in hints:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def _build_phase_array_triplets(T: int, df: pd.DataFrame, file_stem: str | None) -> tuple[np.ndarray, str]:
    """
    Expected columns (case-insensitive / space-insensitive):
      file, windup_start, release_frame, followthrough_end
    Only the row matching the current clip is used (by filename).
    """
    dfl = df.copy()
    dfl.columns = [_norm(c) for c in dfl.columns]

    needed = {"file", "windup_start", "release_frame", "followthrough_end"}
    if not needed.issubset(set(dfl.columns)):
        raise ValueError("freethrow_phases.csv must have columns: "
                         "'file', 'windup_start', 'release_frame', 'followthrough_end'.")

    if file_stem is None:
        raise ValueError("No current clip to match phases against.")

    want_key = _canonical_clip_key(file_stem)
    dfl["_key"] = dfl["file"].astype(str).apply(_canonical_clip_key)
    sub = dfl[dfl["_key"] == want_key]
    if sub.empty:
        raise ValueError(f"No phase row found for clip '{file_stem}' (key '{want_key}').")

    r = sub.iloc[0]
    ws = int(r["windup_start"])
    rf = int(r["release_frame"])
    fe = int(r["followthrough_end"])

    if not (0 <= ws <= rf <= fe):
        raise ValueError(f"Invalid triplet order for '{r['file']}': "
                         f"windup_start={ws}, release_frame={rf}, followthrough_end={fe}")

    phases = np.full(T, "", dtype=object)

    def rng(a, b):
        a = max(0, int(a))
        b = min(T - 1, int(b))
        if b >= a:
            return a, b
        return None

    # windup: ws .. rf-1
    wb = rng(ws, rf - 1)
    if wb:
        a, b = wb
        phases[a:b+1] = "windup"

    # release: rf
    if 0 <= rf < T:
        phases[rf] = "release"

    # followthrough: rf+1 .. fe
    fb = rng(rf + 1, fe)
    if fb:
        a, b = fb
        phases[a:b+1] = "followthrough"

    return phases, "triplets (windup/release/followthrough)"

def try_autoload_phases(current_keypoints: Path | None, T: int) -> tuple[np.ndarray | None, str]:
    """
    Autoload phases for the current clip using triplet-per-clip CSV if found.
    """
    file_stem = current_keypoints.stem if current_keypoints is not None else None
    for cand in _auto_phase_file_hints(current_keypoints):
        if cand.exists():
            try:
                df = pd.read_csv(cand)
                arr, mode = _build_phase_array_triplets(T, df, file_stem)
                return arr, f"{cand.name} ({mode})"
            except Exception:
                continue
    return None, "(none)"

# =========================
# Viewer
# =========================
class MP33Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("MP33 3D Viewer")

        self.files = []
        self.i = 0
        self.frames = None   # (T,33,3)
        self.T = 0

        # angles
        self.angles = None   # dict name -> (T,)
        self.angle_names = []
        self.angle_source = tk.StringVar(value="angles: (none)")
        self.angles_dir_override = None

        # phases
        self.phases = None
        self.phase_source = tk.StringVar(value="phases: (none)")

        # playback
        self.t = 0
        self.playing = False
        self.after_id = None

        # view state
        self.zoom_scale = 1.0
        self.init_elev = 15
        self.init_azim = -90
        self._view_initialized = False

        Label(root, text="3D Pose Viewer (MediaPipe 33)", font=("Helvetica", 18, "bold")).pack(pady=(8,2))

        # top row: plot + side panel
        top = tk.Frame(root); top.pack(padx=6, pady=6, fill="both", expand=True)
        fig_frame = tk.Frame(top); fig_frame.pack(side="left", fill="both", expand=True)
        side_panel = tk.Frame(top); side_panel.pack(side="right", fill="y", padx=(10,0))

        self.fig = plt.Figure(figsize=(7.0,5.5), dpi=100)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self.ax.view_init(self.init_elev, self.init_azim)

        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, fig_frame, pack_toolbar=True)

        # artists
        self.scatter = self.ax.scatter([], [], [], s=30)
        self.lines = [self.ax.plot([], [], [], linewidth=2, color="gray")[0] for _ in range(len(EDGES))]

        # PHASE display
        Label(side_panel, text="Current Phase", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0,2))
        self.phase_now = tk.StringVar(value="—")
        ph_lbl = Label(side_panel, textvariable=self.phase_now, font=("Helvetica", 14, "bold"))
        ph_lbl.pack(anchor="w", pady=(0,4))
        Label(side_panel, textvariable=self.phase_source, font=("Helvetica", 10, "italic")).pack(anchor="w", pady=(0,8))

        # Angles panel
        Label(side_panel, text="Angles (deg)", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.angles_text = tk.Text(side_panel, width=28, height=24, font=("Menlo", 11))
        self.angles_text.pack(fill="y")
        self.angles_text.configure(state="disabled")
        self.angles_text.tag_configure("inc", foreground="#1a7f37")  # green
        self.angles_text.tag_configure("dec", foreground="#d73a49")  # red
        Label(side_panel, textvariable=self.angle_source, font=("Helvetica", 10, "italic")).pack(anchor="w", pady=(6,0))

        # controls
        controls = tk.Frame(root); controls.pack(pady=6)
        Button(controls, text="Load 3D Folder",      command=self.load_folder).grid(row=0, column=0, padx=5)
        Button(controls, text="Angles Folder…",       command=self.choose_angles_folder).grid(row=0, column=1, padx=5)
        Button(controls, text="Angles File (one)…",   command=self.load_angles_file).grid(row=0, column=2, padx=5)
        Button(controls, text="Previous File",        command=self.prev_file).grid(row=0, column=3, padx=5)
        Button(controls, text="Play/Pause",           command=self.toggle_play).grid(row=0, column=4, padx=5)
        Button(controls, text="Next File",            command=self.next_file).grid(row=0, column=5, padx=5)
        Button(controls, text="<< Frame",             command=self.prev_frame).grid(row=0, column=6, padx=5)
        Button(controls, text="Frame >>",             command=self.next_frame).grid(row=0, column=7, padx=5)
        Button(controls, text="Zoom +",               command=lambda: self.zoom(0.8)).grid(row=0, column=8, padx=8)
        Button(controls, text="Zoom −",               command=lambda: self.zoom(1.25)).grid(row=0, column=9, padx=4)
        Button(controls, text="Reset View",           command=self.reset_view).grid(row=0, column=10, padx=8)
        Button(controls, text="Load Phases…",         command=self.load_phases_file).grid(row=0, column=11, padx=10)

        self.info = tk.StringVar(value="No file loaded")
        Label(root, textvariable=self.info, font=("Helvetica", 12)).pack(pady=(0,8))

        # hotkeys
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Left>",  lambda e: self.prev_frame())
        root.bind("<Right>", lambda e: self.next_frame())
        root.bind("+",       lambda e: self.zoom(0.8))
        root.bind("-",       lambda e: self.zoom(1.25))
        root.bind("r",       lambda e: self.reset_view())

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------- file handling --------
    def load_folder(self):
        folder = filedialog.askdirectory(initialdir=session_dir, title="Select folder with *_3d.csv files")
        if not folder: return
        p = Path(folder)
        cands = sorted(list(p.glob("*_3d.csv")))
        self.files = []
        for f in cands:
            try:
                _ = load_mp33_csv(f)
                self.files.append(f)
            except Exception:
                pass
        if not self.files:
            self.info.set(f"No valid *_3d.csv files in {p}")
            return
        self.i = 0
        self.open_file(self.files[self.i])

    def open_file(self, path: Path):
        try:
            self.frames = load_mp33_csv(path)  # (T,33,3)
        except Exception as e:
            self.info.set(f"Failed to load {path.name}: {e}")
            return

        self.T = self.frames.shape[0]
        self.t = 0
        self.zoom_scale = 1.0
        self._view_initialized = False

        # angles: try matching external CSV, else compute
        ap = find_angles_for_file(path, self.angles_dir_override, default_angles_dir)
        if ap is not None:
            try:
                ang, names, _ = load_angles_csv(ap)
                for k in ang:
                    if len(ang[k]) > self.T:
                        ang[k] = ang[k][:self.T]
                self.angles, self.angle_names = ang, names
                source_label = (f"{ap.parent.name}/" if ap.parent else "") + ap.name
                self.angle_source.set(f"angles: external ({source_label})")
            except Exception as e:
                self.angles = compute_angles(self.frames)
                self.angle_names = list(self.angles.keys())
                self.angle_source.set(f"angles: computed (failed to read external: {e})")
        else:
            self.angles = compute_angles(self.frames)
            self.angle_names = list(self.angles.keys())
            self.angle_source.set("angles: computed")

        # phases: try to autoload for this clip
        self.phases, src = try_autoload_phases(path, self.T)
        self.phase_source.set(f"phases: {src}")
        if self.phases is None:
            self.phase_now.set("—")

        self.redraw()
        self.info.set(f"{path.name}  |  frames: {self.T}  |  joints: 33")

    # -------- phases loading (manual) --------
    def load_phases_file(self):
        initial_dirs = [d.parent for d in PHASE_FILE_CANDIDATES if d.parent.exists()]
        initial = initial_dirs[0] if initial_dirs else session_dir
        f = filedialog.askopenfilename(
            initialdir=initial if initial.exists() else session_dir,
            title="Select freethrow_phases.csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not f:
            return
        try:
            if self.frames is None:
                raise ValueError("Load a 3D file first so I can match the clip row.")
            df = pd.read_csv(f)
            file_stem = self.files[self.i].stem if self.files else None
            arr, mode = _build_phase_array_triplets(self.T, df, file_stem)
            self.phases = arr
            self.phase_source.set(f"phases: {Path(f).name} ({mode})")
            self.redraw()
        except Exception as e:
            self.phases = None
            self.phase_source.set(f"phases: failed to load ({Path(f).name}: {e})")
            self.phase_now.set("—")

    # -------- angles folder select --------
    def choose_angles_folder(self):
        folder = filedialog.askdirectory(
            initialdir=default_angles_dir if default_angles_dir.exists() else session_dir,
            title="Select folder with angles CSVs"
        )
        if not folder:
            return
        self.angles_dir_override = Path(folder)
        if self.files:
            self.open_file(self.files[self.i])

    def load_angles_file(self):
        initial = self.angles_dir_override or default_angles_dir
        f = filedialog.askopenfilename(
            initialdir=initial if initial.exists() else session_dir,
            title="Select angles CSV",
            filetypes=[("CSV", "*.csv")]
        )
        if not f:
            return
        try:
            ang, names, _ = load_angles_csv(Path(f))
            if self.frames is not None:
                for k in ang:
                    if len(ang[k]) > self.T:
                        ang[k] = ang[k][:self.T]
            self.angles, self.angle_names = ang, names
            self.angle_source.set(f"angles: external ({Path(f).name})")
            self.redraw()
        except Exception as e:
            self.angle_source.set(f"angles: failed to load ({e})")

    # -------- drawing --------
    def redraw(self):
        if self.frames is None:
            return
        pts = self.frames[self.t]  # (33,3)

        # valid mask
        valid = np.isfinite(pts).all(axis=1) & ~(pts == -1).any(axis=1)

        # scatter + lines
        if valid.any():
            x, y, z = pts[valid,0], pts[valid,1], pts[valid,2]
            self.scatter._offsets3d = (x, y, z)

            for ln, (a, b) in zip(self.lines, EDGES):
                if valid[a] and valid[b]:
                    xa, ya, za = pts[a]
                    xb, yb, zb = pts[b]
                    ln.set_data([xa, xb], [ya, yb])
                    ln.set_3d_properties([za, zb])
                    ln.set_visible(True)
                else:
                    ln.set_data([], []); ln.set_3d_properties([]); ln.set_visible(True)

            # fit box with zoom
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            zmin, zmax = float(z.min()), float(z.max())
            if xmax == xmin: xmax += 1; xmin -= 1
            if ymax == ymin: ymax += 1; ymin -= 1
            if zmax == zmin: zmax += 1; zmin -= 1
            cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
            r = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.6
            r = max(r, 1e-6) * self.zoom_scale
            self.ax.set_xlim(cx-r, cx+r)
            self.ax.set_ylim(cy-r, cy+r)
            self.ax.set_zlim(cz-r, cz+r)
        else:
            self.scatter._offsets3d = ([], [], [])
            for ln in self.lines:
                ln.set_data([], []); ln.set_3d_properties([])

        if not self._view_initialized:
            self.ax.view_init(self.init_elev, self.init_azim)
            self._view_initialized = True

        # update phase label
        if self.phases is not None and 0 <= self.t < len(self.phases) and isinstance(self.phases[self.t], str) and self.phases[self.t] != "":
            self.phase_now.set(self.phases[self.t])
        else:
            self.phase_now.set("—")

        # update angles panel
        self.update_angles_panel()

        self.canvas.draw_idle()
        if self.files:
            self.info.set(f"{self.files[self.i].name}  |  frame: {self.t+1}/{self.T}  |  joints: 33")

    def update_angles_panel(self):
        self.angles_text.configure(state="normal")
        self.angles_text.delete("1.0", "end")

        if not self.angles:
            self.angles_text.insert("1.0", "(no angles loaded)")
            self.angles_text.configure(state="disabled")
            return

        t = self.t
        for name in self.angle_names:
            arr = self.angles[name]
            val = arr[t] if t < len(arr) else np.nan
            tag = None
            if t > 0 and t < len(arr) and np.isfinite(val) and np.isfinite(arr[t-1]):
                d = val - arr[t-1]
                if d > 0:
                    tag = "inc"  # increasing -> green
                elif d < 0:
                    tag = "dec"  # decreasing -> red
            line = f"{name:16s} {val:7.2f}\n" if np.isfinite(val) else f"{name:16s}    NaN\n"
            if tag:
                self.angles_text.insert("end", line, (tag,))
            else:
                self.angles_text.insert("end", line)

        self.angles_text.configure(state="disabled")

    # -------- view / zoom --------
    def zoom(self, factor):
        self.zoom_scale *= factor
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        self.redraw()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.ax.view_init(self.init_elev, self.init_azim)
        self._view_initialized = True
        self.redraw()

    # -------- navigation --------
    def toggle_play(self):
        if self.frames is None: return
        self.playing = not self.playing
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing or self.frames is None: return
        self.t += 1
        if self.t >= self.T:
            self.playing = False
            return
        self.redraw()
        self.root.after(int(1000 / FPS), self._tick)

    def prev_frame(self):
        if self.frames is None: return
        self.playing = False
        self.t = max(0, self.t - 1)
        self.redraw()

    def next_frame(self):
        if self.frames is None: return
        self.playing = False
        self.t = min(self.T - 1, self.t + 1)
        self.redraw()

    def next_file(self):
        if not self.files: return
        self.playing = False
        self.i = (self.i + 1) % len(self.files)
        self.open_file(self.files[self.i])

    def prev_file(self):
        if not self.files: return
        self.playing = False
        self.i = (self.i - 1) % len(self.files)
        self.open_file(self.files[self.i])

    def on_close(self):
        self.root.quit()

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = MP33Viewer(root)
    root.mainloop()
