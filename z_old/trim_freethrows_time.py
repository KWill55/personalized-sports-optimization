"""
Dual Video Trimmer — Time-based, Two-Folder, Non-Destructive
- Folder A (required) and Folder B (optional)
- A and B preview side-by-side (scaled to ~640px wide each)
- Time-based scrubbing + trimming only
- Controls: Play/Pause, Prev Clip, Next Clip, Set Start, Set End, Save Trim
- Saves to <folder>/_trimmed/ (originals untouched); .hevc outputs as .mp4
- Pairing by index: A[i] ↔ B[i]; B optional
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import yaml
import math

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".hevc", ".m4v", ".mkv"}
DEFAULT_FPS_FALLBACK = 30.0

# Preview target sizes (width, height max). We keep aspect ratio.
PREVIEW_MAX_A = (640, 320)   # for 1280x640 this becomes ~640x320
PREVIEW_MAX_B = (640, 360)   # for 1280x720 this becomes ~640x360

# Time nudges (in seconds)
NUDGE_SMALL = 0.05    # ~1.5 frames @30fps, ~3 frames @60fps
NUDGE_MED   = 0.1     # ~3 frames @30fps,  ~6 frames @60fps
NUDGE_LARGE = 0.5     # ~15 frames @30fps, ~30 frames @60fps
NUDGE_XL    = 1.0     # ~30 frames @30fps, ~60 frames @60fps

# Pick nudge for arrow keys
NUDGE_POWER = NUDGE_SMALL 

# ---------- helper: initial directory from project_config if present ----------
def guess_session_dir():
    try:
        config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        ATHLETE = str(cfg.get("athlete", ""))
        SESSION = str(cfg.get("session", ""))
        base_dir = Path(__file__).resolve().parents[3]
        return base_dir / "data" / ATHLETE / SESSION
    except Exception:
        return Path.home()

SESSION_HINT = guess_session_dir()


class DualTimeTrimmer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Video Trimmer — Time-based")

        # Folders / files
        self.dir_a: Path | None = None
        self.dir_b: Path | None = None
        self.out_a: Path | None = None
        self.out_b: Path | None = None
        self.files_a = []
        self.files_b = []
        self.idx = 0

        # Captures (we preview from A; B is sampled by seek-per-frame for sync)
        self.cap_a = None
        self.playing = False

        # Current A props
        self.fps_a = DEFAULT_FPS_FALLBACK
        self.frames_a = 0
        self.w_a = 0
        self.h_a = 0
        self.dur_a = 0.0  # seconds

        # Current B props (queried on demand)
        # We don't hold cap_b open; we seek it per refresh to match time.
        self.info_b = {"fps": None, "frames": None, "w": None, "h": None, "dur": None}

        # Trim marks in seconds (time-based!)
        self.start_sec: float | None = None
        self.end_sec: float | None = None

        # --- UI ---
        self.build_ui()
        self.update_status("Load Folder A to begin.")

    # ---------- UI ----------
    def build_ui(self):
        # Top: Folder buttons
        top = tk.Frame(self.root)
        top.pack(pady=6)
        tk.Button(top, text="Load Folder A (required)", command=self.load_folder_a).grid(row=0, column=0, padx=6)
        tk.Button(top, text="Load Folder B (optional)", command=self.load_folder_b).grid(row=0, column=1, padx=6)

        # Middle: Previews side-by-side
        vids = tk.Frame(self.root, bg="black", padx=6, pady=6)
        vids.pack(pady=(6, 2))

        left_col = tk.Frame(vids, bg="black")
        left_col.grid(row=0, column=0, padx=4)
        tk.Label(left_col, text="Folder A", font=("Helvetica", 12, "bold")).pack(pady=(0,4))
        self.lbl_a = tk.Label(left_col, bg="black")
        self.lbl_a.pack()

        right_col = tk.Frame(vids, bg="black")
        right_col.grid(row=0, column=1, padx=4)
        tk.Label(right_col, text="Folder B (preview)", font=("Helvetica", 12, "bold")).pack(pady=(0,4))
        self.lbl_b = tk.Label(right_col, bg="black")
        self.lbl_b.pack()

        # Clip info
        info = tk.Frame(self.root)
        info.pack(pady=(2, 6))
        self.info_text = tk.StringVar(value="No clip loaded.")
        tk.Label(info, textvariable=self.info_text, font=("Helvetica", 11)).pack()

        # Seekbar (in seconds)
        seek = tk.Frame(self.root)
        seek.pack(pady=4, fill="x")
        tk.Label(seek, text="Time (s):").pack(side="left", padx=(4,6))
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seekbar = tk.Scale(
            seek, from_=0.0, to=1.0, orient="horizontal",
            resolution=0.01,  # 10ms resolution; adjust as needed
            variable=self.seek_var,
            command=self.on_seek,
            length=700
        )
        self.seekbar.pack(side="left", padx=6)
        self.time_label = tk.StringVar(value="0.00 / 0.00 s")
        tk.Label(seek, textvariable=self.time_label).pack(side="left", padx=8)

        # Controls
        ctrls = tk.Frame(self.root)
        ctrls.pack(pady=6)
        tk.Button(ctrls, text="Prev Clip", command=self.prev_clip).grid(row=0, column=0, padx=6)
        tk.Button(ctrls, text="Play / Pause", command=self.toggle_play).grid(row=0, column=1, padx=6)
        tk.Button(ctrls, text="Next Clip", command=self.next_clip).grid(row=0, column=2, padx=6)
        tk.Button(ctrls, text="Set Start", command=self.set_start).grid(row=0, column=3, padx=12)
        tk.Button(ctrls, text="Set End", command=self.set_end).grid(row=0, column=4, padx=6)
        tk.Button(ctrls, text="Save Trim", command=self.save_trim).grid(row=0, column=5, padx=12)

        # Status line
        self.status = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status, font=("Helvetica", 10)).pack(pady=(2,10))

        # Keys
        # Playback & scrubbing
        self.root.bind("<space>", lambda e: self.toggle_play())      # Play/Pause
        self.root.bind("<Right>", lambda e: self.nudge_time(NUDGE_POWER))   # +50 ms
        self.root.bind("<Left>",  lambda e: self.nudge_time(-NUDGE_POWER))  # -50 ms

        # Clip navigation
        self.root.bind("<Up>",   lambda e: self.next_clip())         # Next clip
        self.root.bind("<Down>", lambda e: self.prev_clip())         # Previous clip

        # Markers
        self.root.bind("1", lambda e: self.set_start())              # Set Start
        self.root.bind("2", lambda e: self.set_end())                # Set End

        # (Optional) keep these if you like
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- Folder loading ----------
    def load_folder_a(self):
        folder = filedialog.askdirectory(initialdir=str(SESSION_HINT), title="Select Folder A")
        if not folder:
            return
        self.dir_a = Path(folder)
        self.out_a = self.dir_a / "_trimmed"
        self.out_a.mkdir(parents=True, exist_ok=True)
        self.files_a = self.scan_videos(self.dir_a)
        if not self.files_a:
            messagebox.showerror("No videos", "No supported videos found in Folder A.")
            return
        self.idx = 0
        self.open_clip(self.idx)

    def load_folder_b(self):
        if self.dir_a is None:
            messagebox.showinfo("Tip", "Load Folder A first, then choose Folder B.")
        initial = self.dir_a or SESSION_HINT
        folder = filedialog.askdirectory(initialdir=str(initial), title="Select Folder B (optional)")
        if not folder:
            self.dir_b = None
            self.out_b = None
            self.files_b = []
            self.update_status("Folder B cleared.")
            self.refresh_previews(force=True)  # clear right pane
            return
        self.dir_b = Path(folder)
        self.out_b = self.dir_b / "_trimmed"
        self.out_b.mkdir(parents=True, exist_ok=True)
        self.files_b = self.scan_videos(self.dir_b)
        if not self.files_b:
            self.dir_b = None
            self.out_b = None
            self.files_b = []
            messagebox.showwarning("Folder B", "No supported videos found. Clearing B.")
        if self.dir_a and self.files_a:
            if len(self.files_b) != len(self.files_a):
                self.update_status(f"⚠ Pair counts differ (A={len(self.files_a)}, B={len(self.files_b)}). Pairing by index.")

        # if a clip is already open, refresh B preview for current time
        self.refresh_previews(force=True)

    def scan_videos(self, directory: Path):
        return sorted(
            [directory / f for f in os.listdir(directory)
             if (directory / f).is_file() and (directory / f).suffix.lower() in VIDEO_EXTENSIONS],
            key=lambda p: p.name.lower()
        )

    # ---------- Clip open / info ----------
    def open_clip(self, index: int):
        if not self.files_a:
            return
        index = max(0, min(len(self.files_a) - 1, index))
        self.idx = index

        # Release previous cap
        if self.cap_a:
            self.cap_a.release()
            self.cap_a = None

        src_a = self.files_a[self.idx]
        self.cap_a = cv2.VideoCapture(str(src_a))
        if not self.cap_a.isOpened():
            messagebox.showwarning("OpenCV", f"Could not open A: {src_a.name}")
            return

        fps = self.cap_a.get(cv2.CAP_PROP_FPS)
        self.fps_a = float(fps) if fps and fps > 0 else DEFAULT_FPS_FALLBACK
        self.frames_a = int(self.cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w_a = int(self.cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_a = int(self.cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dur_a = self.frames_a / self.fps_a if self.fps_a > 0 else 0.0

        # Reset marks & seekbar
        self.start_sec = None
        self.end_sec = None
        self.seekbar.configure(from_=0.0, to=max(self.dur_a, 0.01))
        self.seek_var.set(0.0)
        self.update_time_label(0.0)
        self.playing = False

        # Clear B info cache, will re-query lazily
        self.info_b = {"fps": None, "frames": None, "w": None, "h": None, "dur": None}

        # Position A at t=0 and draw both
        self.cap_a.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.refresh_previews(force=True)
        self.update_info_text()

    def update_info_text(self):
        a_name = self.files_a[self.idx].name if self.files_a else "—"
        b_name = (self.files_b[self.idx].name if (self.dir_b and self.idx < len(self.files_b))
                  else "(no B or out of range)")
        msg = (
            f"A[{self.idx+1}/{len(self.files_a)}]: {a_name} | "
            f"{self.w_a}x{self.h_a}, {self.frames_a} frames, {self.fps_a:.2f} fps, {self.dur_a:.2f}s\n"
            f"B[{self.idx+1}/?]: {b_name}"
        )
        if self.start_sec is not None or self.end_sec is not None:
            s = f" | Start: {self.start_sec:.3f}s" if self.start_sec is not None else ""
            e = f" | End: {self.end_sec:.3f}s" if self.end_sec is not None else ""
            msg += s + e
        self.info_text.set(msg)

    # ---------- Playback / seek ----------
    def toggle_play(self):
        if not self.cap_a:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing:
            return
        # Read next frame from A
        ok, frame = self.cap_a.read()
        if not ok:
            self.playing = False
            return

        # Compute current time from A's position
        # POS_FRAMES returns next frame to be read; current displayed = pos-1
        pos = int(self.cap_a.get(cv2.CAP_PROP_POS_FRAMES))
        t = max(0.0, min(self.dur_a, (pos - 1) / self.fps_a))

        # Update seek var without triggering extra seeks
        self.seek_var.set(t)
        self.update_time_label(t)

        # Draw both previews at this time
        self.draw_frame_on_label(frame, self.lbl_a, PREVIEW_MAX_A)
        self.draw_b_at_time(t)

        # schedule next tick
        delay = int(1000 / (self.fps_a if self.fps_a > 0 else DEFAULT_FPS_FALLBACK))
        self.root.after(delay, self.play_loop)

    def on_seek(self, _val):
        if not self.cap_a:
            return
        t = float(self.seek_var.get())
        t = max(0.0, min(self.dur_a, t))
        # Seek A to the exact frame for that time
        idx = int(round(t * self.fps_a))
        idx = max(0, min(self.frames_a - 1, idx))
        self.cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap_a.read()
        if ok:
            self.draw_frame_on_label(frame, self.lbl_a, PREVIEW_MAX_A)
        self.draw_b_at_time(t)
        self.update_time_label(t)

    def nudge_time(self, dt):
        if not self.cap_a:
            return
        t = float(self.seek_var.get()) + dt
        self.seek_var.set(max(0.0, min(self.dur_a, t)))
        self.on_seek(None)

    def update_time_label(self, t):
        self.time_label.set(f"{t:.2f} / {self.dur_a:.2f} s")

    # ---------- Drawing helpers ----------
    def draw_frame_on_label(self, bgr, label, max_wh):
        # scale to fit within max_wh, preserving aspect
        h, w = bgr.shape[:2]
        max_w, max_h = max_wh
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w != w or new_h != h:
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.configure(image=img)
        label.image = img

    def draw_b_at_time(self, t):
        # Clear B if not loaded/paired
        if not (self.dir_b and self.files_b and self.idx < len(self.files_b)):
            self.lbl_b.configure(image="")
            self.lbl_b.image = None
            return

        src_b = self.files_b[self.idx]
        # Lazy query props
        if self.info_b["fps"] is None:
            cap = cv2.VideoCapture(str(src_b))
            if not cap.isOpened():
                self.info_b = {"fps": 0, "frames": 0, "w": 0, "h": 0, "dur": 0.0}
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else DEFAULT_FPS_FALLBACK
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dur = frames / fps if fps > 0 else 0.0
            cap.release()
            self.info_b.update({"fps": fps, "frames": frames, "w": w, "h": h, "dur": dur})

        # Clamp time to B duration
        fpsB = self.info_b["fps"] or DEFAULT_FPS_FALLBACK
        framesB = self.info_b["frames"] or 0
        t_clamped = max(0.0, min(self.info_b["dur"] or 0.0, t))
        idxB = int(round(t_clamped * fpsB))
        idxB = max(0, min(max(framesB - 1, 0), idxB))

        cap = cv2.VideoCapture(str(src_b))
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxB)
        ok, frameB = cap.read()
        cap.release()
        if ok:
            self.draw_frame_on_label(frameB, self.lbl_b, PREVIEW_MAX_B)
        else:
            self.lbl_b.configure(image="")
            self.lbl_b.image = None

    def refresh_previews(self, force=False):
        if not self.cap_a:
            # clear both
            self.lbl_a.configure(image="")
            self.lbl_a.image = None
            self.lbl_b.configure(image="")
            self.lbl_b.image = None
            return
        # draw current A frame (read current frame)
        pos = int(self.cap_a.get(cv2.CAP_PROP_POS_FRAMES))
        pos = max(0, min(self.frames_a - 1, pos))
        self.cap_a.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = self.cap_a.read()
        if ok or force:
            if not ok:
                # jump to first frame if needed
                self.cap_a.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap_a.read()
            if ok:
                self.draw_frame_on_label(frame, self.lbl_a, PREVIEW_MAX_A)
                t = (max(0, int(self.cap_a.get(cv2.CAP_PROP_POS_FRAMES)) - 1) / self.fps_a) if self.fps_a > 0 else 0.0
                self.seek_var.set(t)
                self.update_time_label(t)
                self.draw_b_at_time(t)

    # ---------- Clip navigation ----------
    def next_clip(self):
        if not self.files_a:
            return
        self.idx = (self.idx + 1) % len(self.files_a)
        self.open_clip(self.idx)

    def prev_clip(self):
        if not self.files_a:
            return
        self.idx = (self.idx - 1) % len(self.files_a)
        self.open_clip(self.idx)

    # ---------- Marks (time-based) ----------
    def set_start(self):
        if not self.cap_a:
            return
        self.start_sec = float(self.seek_var.get())
        self.update_info_text()
        self.update_status(f"Start set at {self.start_sec:.3f}s")

    def set_end(self):
        if not self.cap_a:
            return
        self.end_sec = float(self.seek_var.get())
        self.update_info_text()
        self.update_status(f"End set at {self.end_sec:.3f}s")

    # ---------- Save (time-based to frames per file) ----------
    def save_trim(self):
        if not self.files_a:
            return
        if self.start_sec is None or self.end_sec is None:
            messagebox.showwarning("Missing marks", "Please set both Start and End (time in seconds).")
            return
        a, b = sorted([self.start_sec, self.end_sec])
        if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-3) or b <= a:
            messagebox.showwarning("Zero length", "Start and End times are identical.")
            return

        # Pause during write
        self.playing = False
        srcA = self.files_a[self.idx]
        okA = self.write_time_trim(srcA, a, b, self.out_a)

        okB = True
        if self.dir_b and self.idx < len(self.files_b):
            srcB = self.files_b[self.idx]
            outB = self.out_b or (self.dir_b / "_trimmed")
            okB = self.write_time_trim(srcB, a, b, outB)  # time-based mapping

        if okA and okB:
            messagebox.showinfo("Saved", "Trim(s) saved to _trimmed/ in loaded folder(s).")
        elif okA and not okB and self.dir_b:
            messagebox.showwarning("Partial", "Saved A. B failed or missing (see warnings).")
        elif not okA and okB:
            messagebox.showwarning("Partial", "Saved B. A failed (unexpected).")
        else:
            messagebox.showerror("Failed", "No trims saved.")

        # Reload current clip for continuity
        self.open_clip(self.idx)

    def fourcc_for_ext(self, ext: str):
        ext = ext.lower()
        if ext == ".avi":
            return cv2.VideoWriter_fourcc(*"MJPG")
        if ext in (".mp4", ".mov", ".m4v", ".mkv"):
            return cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter_fourcc(*"mp4v")

    def safe_dest(self, out_dir: Path, stem: str, ext: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        cand = out_dir / f"{stem}{ext}"
        k = 1
        while cand.exists():
            cand = out_dir / f"{stem} ({k}){ext}"
            k += 1
        return cand

    def write_time_trim(self, src: Path, t_start: float, t_end: float, out_dir: Path) -> bool:
        if not src.exists():
            return False
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            messagebox.showwarning("OpenCV", f"Could not open: {src.name}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS_FALLBACK
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0 or total <= 0:
            cap.release()
            messagebox.showerror("Property error", f"Invalid properties for: {src.name}")
            return False
        dur = total / fps

        a_sec = max(0.0, min(dur, t_start))
        b_sec = max(0.0, min(dur, t_end))
        if b_sec <= a_sec:
            cap.release()
            messagebox.showwarning("Selection", f"Invalid selection for: {src.name}")
            return False

        a_idx = int(round(a_sec * fps))
        b_idx = int(round(b_sec * fps))
        a_idx = max(0, min(total - 1, a_idx))
        b_idx = max(0, min(total - 1, b_idx))
        if a_idx >= b_idx:
            cap.release()
            messagebox.showwarning("Selection", f"Zero/invalid length for: {src.name}")
            return False

        # Choose output ext (.hevc -> .mp4)
        ext = src.suffix.lower()
        out_ext = ".mp4" if ext == ".hevc" else ext
        dest = self.safe_dest(out_dir, src.stem, out_ext)

        cap.set(cv2.CAP_PROP_POS_FRAMES, a_idx)
        fourcc = self.fourcc_for_ext(out_ext)
        writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            messagebox.showerror("Writer", f"Could not open writer for: {dest.name}")
            return False

        ok = True
        frame_idx = a_idx
        while frame_idx <= b_idx:
            ok, frame = cap.read()
            if not ok:
                break
            # safety: if codec reports odd sizes
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frame_idx += 1

        writer.release()
        cap.release()

        if frame_idx <= b_idx:
            try:
                if dest.exists():
                    dest.unlink()
            except Exception:
                pass
            messagebox.showerror("Read error", f"Stopped early at frame {frame_idx-1} of {b_idx} for: {src.name}")
            return False

        print(f"✅ Saved: {dest}")
        return True

    # ---------- status / cleanup ----------
    def update_status(self, msg):
        self.status.set(msg)

    def on_close(self):
        self.playing = False
        if self.cap_a:
            self.cap_a.release()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DualTimeTrimmer(root)
    root.mainloop()
