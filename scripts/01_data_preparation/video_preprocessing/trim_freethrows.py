"""
Video Trimmer (Player-style UI) — In-Place Overwrite
- Choose an input folder at startup
- Player-style controls (Play/Pause, Prev/Next clip, Frame stepping)
- Shows filename, resolution, duration, FPS, total frames
- Full-resolution display (no resizing)
- Trims OVERWRITE the current file (with confirmation)
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from pathlib import Path
from PIL import Image, ImageTk  # pip install pillow
import yaml

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".hevc"}
FPS = 60

# ========================================
# Initial config (only used to hint initialdir)
# ========================================
try:
    config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    ATHLETE = str(cfg.get("athlete", ""))
    SESSION = str(cfg.get("session", ""))
    base_dir = Path(__file__).resolve().parents[3]
    session_dir = base_dir / "data" / ATHLETE / SESSION
except Exception:
    ATHLETE = SESSION = ""
    session_dir = Path.home()

# ========================================
# Trimmer App
# ========================================
class VideoTrimmerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer")

        # Video state
        self.video_files = []
        self.current_index = 0
        self.cap = None
        self.playing = False
        self.fps = FPS
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.input_dir = None

        # Trim markers (frame indices)
        self.start_frame = None
        self.end_frame = None

        # ===== UI =====
        Label(root, text="Video Trimmer (In-Place)", font=("Helvetica", 24, "bold")).pack(pady=(10, 2))

        # Video container (nice border)
        video_frame_container = tk.Frame(root, bg="black", padx=5, pady=5)
        video_frame_container.pack(pady=10)

        self.video_label = Label(video_frame_container)  # where frames render
        self.video_label.pack()

        # Controls row 1
        control_frame = tk.Frame(root)
        control_frame.pack(pady=8)
        Button(control_frame, text="Load Folder", command=self.load_folder).grid(row=0, column=0, padx=5)
        Button(control_frame, text="Previous Clip", command=self.prev_video).grid(row=0, column=1, padx=5)
        Button(control_frame, text="Play/Pause", command=self.toggle_play).grid(row=0, column=2, padx=5)
        Button(control_frame, text="Next Clip", command=self.next_video).grid(row=0, column=3, padx=5)

        # Controls row 2 (frame stepping & markers)
        frame_control = tk.Frame(root)
        frame_control.pack(pady=5)
        Button(frame_control, text="<< Frame", command=self.prev_frame).grid(row=0, column=0, padx=5)
        Button(frame_control, text="Frame >>", command=self.next_frame).grid(row=0, column=1, padx=5)
        Button(frame_control, text="Set Start", command=self.set_start).grid(row=0, column=2, padx=12)
        Button(frame_control, text="Set End", command=self.set_end).grid(row=0, column=3, padx=5)
        Button(frame_control, text="Clear Marks", command=self.clear_marks).grid(row=0, column=4, padx=12)
        Button(frame_control, text="Save Trim (Overwrite)", command=self.save_trim).grid(row=0, column=5, padx=5)

        # Info container (like your player)
        info_container = tk.Frame(root, highlightbackground="black", highlightthickness=5, bd=0, padx=10, pady=10)
        info_container.pack(pady=15)
        Label(info_container, text="Video Clip Info", font=("Helvetica", 18, "bold")).pack(pady=(5, 8))

        self.video_info = tk.StringVar(value="No videos loaded")
        Label(info_container, textvariable=self.video_info, font=("Helvetica", 14)).pack(pady=(0, 6))

        self.status = tk.StringVar(value="Press 'Load Folder' to select videos")
        Label(info_container, textvariable=self.status, font=("Helvetica", 14)).pack()

        # Key bindings
        self.root.bind("<space>", lambda e: self.toggle_play())  # Space = play/pause
        self.root.bind("<Right>", lambda e: self.jump_frames(1))
        self.root.bind("<Left>", lambda e: self.jump_frames(-1))
        self.root.bind("<Shift-Right>", lambda e: self.jump_frames(10))
        self.root.bind("<Shift-Left>", lambda e: self.jump_frames(-10))
        self.root.bind("<Next>", lambda e: self.jump_frames(60))   # PageDown
        self.root.bind("<Prior>", lambda e: self.jump_frames(-60)) # PageUp
        self.root.bind("<Home>", lambda e: self.seek_frame(0))
        self.root.bind("<End>", self.seek_end)
        self.root.bind("<Return>", self.enter_flow)  # Enter: Start -> End -> Save

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Prompt immediately
        self.load_folder()

    # ---------- Folder & file handling ----------
    def load_folder(self):
        initial = session_dir if session_dir.exists() else Path.home()
        folder = filedialog.askdirectory(initialdir=str(initial), title="Select Folder with Videos")
        if not folder:
            return

        self.input_dir = Path(folder)
        files = [self.input_dir / f for f in os.listdir(self.input_dir)
                 if (self.input_dir / f).is_file() and (self.input_dir / f).suffix.lower() in VIDEO_EXTENSIONS]
        self.video_files = sorted(files, key=lambda p: p.name.lower())

        if not self.video_files:
            messagebox.showerror("No videos", "No supported videos found in the selected folder.")
            return

        self.current_index = 0
        self.open_video(self.video_files[self.current_index])
        self.status.set(f"Loaded {len(self.video_files)} videos from: {self.input_dir}")

    def open_video(self, path: Path):
        # release previous
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showwarning("OpenCV error", f"Could not open: {path.name}")
            self.next_video()
            return

        # properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (self.total_frames / self.fps) if self.fps > 0 else 0.0

        # reset markers
        self.start_frame = None
        self.end_frame = None
        self.playing = False

        # show first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._read_and_show()

        idx = self.current_index + 1
        total = len(self.video_files)
        self.video_info.set(
            f"Filename: {path.name}  (Clip {idx}/{total})\n"
            f"Resolution: {self.width}x{self.height}  |  Frames: {self.total_frames}  |  "
            f"Duration: {duration:.2f}s  |  FPS: {self.fps:.2f}"
        )
        self._update_status()

    # ---------- Display helpers ----------
    def _read_and_show(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=img)
        self.video_label.image = img
        self._update_status()

    def _current_frame_index(self):
        """OpenCV's POS_FRAMES returns the index of the *next* frame to be read.
        After a successful read, subtract 1 to get the displayed frame index.
        """
        if not self.cap:
            return 0
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, min(self.total_frames - 1, pos - 1))

    def _update_status(self):
        i = self._current_frame_index()
        t = (i / self.fps) if self.fps > 0 else 0.0
        sel = ""
        if self.start_frame is not None:
            sel += f"  |  Start: {self.start_frame} ({self.start_frame / self.fps:.2f}s)"
        if self.end_frame is not None:
            sel += f"  |  End: {self.end_frame} ({self.end_frame / self.fps:.2f}s)"
        if self.start_frame is not None and self.end_frame is not None:
            a, b = sorted([self.start_frame, self.end_frame])
            sel_len = (b - a + 1) / self.fps
            sel += f"  |  Len: {sel_len:.2f}s"

        self.status.set(f"Current Frame: {i}/{self.total_frames - 1}  |  t={t:.2f}s{sel}")

    # ---------- Playback loop ----------
    def _play_loop(self):
        if not self.playing:
            return
        self._read_and_show()
        delay = int(1000 / (self.fps if self.fps > 0 else 30.0))
        self.root.after(delay, self._play_loop)

    def toggle_play(self):
        if not self.cap:
            return
        # restart if at end
        if self._current_frame_index() >= self.total_frames - 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.playing = not self.playing
        if self.playing:
            self._play_loop()

    # ---------- Navigation ----------
    def next_video(self):
        if not self.video_files:
            return
        self.current_index = (self.current_index + 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def prev_video(self):
        if not self.video_files:
            return
        self.current_index = (self.current_index - 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def next_frame(self):
        if not self.cap:
            return
        self.playing = False
        self._read_and_show()

    def prev_frame(self):
        if not self.cap:
            return
        self.playing = False
        # go back two, then read one
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
        self._read_and_show()

    def jump_frames(self, n):
        if not self.cap:
            return
        self.playing = False
        i = self._current_frame_index()
        target = max(0, min(self.total_frames - 1, i + n))
        self.seek_frame(target)

    def seek_frame(self, idx):
        if not self.cap:
            return
        self.playing = False
        idx = max(0, min(self.total_frames - 1, int(idx)))
        # Seek to idx, then read to display idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self._read_and_show()

    def seek_end(self, _evt=None):
        if not self.cap:
            return
        self.seek_frame(self.total_frames - 1)

    # ---------- Markers & flow ----------
    def set_start(self):
        self.start_frame = self._current_frame_index()
        self._update_status()
        print(f"Start frame set: {self.start_frame}")

    def set_end(self):
        self.end_frame = self._current_frame_index()
        self._update_status()
        print(f"End frame set: {self.end_frame}")

    def clear_marks(self):
        self.start_frame = None
        self.end_frame = None
        self._update_status()

    def enter_flow(self, _evt=None):
        # Enter: if no start → set start; else if no end → set end; else save
        if self.start_frame is None:
            self.set_start()
        elif self.end_frame is None:
            self.set_end()
        else:
            self.save_trim()

    # add helper
    def _tmp_path_for(self, src: Path) -> Path:
        return src.with_name(src.stem + "_tmp" + src.suffix)


    # ---------- Save trim (in-place) ----------
    def _fourcc_for_suffix(self, suffix: str):
        s = suffix.lower()
        if s == ".avi":
            # QuickTime dislikes XVID; MJPG is broadly supported
            return cv2.VideoWriter_fourcc(*"MJPG")
        if s in (".mp4", ".mov", ".m4v"):
            # MPEG-4 Part 2; QuickTime opens this reliably from OpenCV
            return cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter_fourcc(*"mp4v")


    def save_trim(self):
        # 1) Validate selection
        if self.start_frame is None or self.end_frame is None:
            messagebox.showwarning("Missing marks", "Please set both Start and End frames.")
            return
        a, b = sorted([int(self.start_frame), int(self.end_frame)])
        if a == b:
            messagebox.showwarning("Zero length", "Start and End are the same frame.")
            return

        # 2) Validate source / container
        src = self.video_files[self.current_index]
        ext = src.suffix.lower()
        if ext == ".hevc":
            messagebox.showerror(
                "Unsupported",
                "In-place overwrite for .hevc is not supported by OpenCV. Convert to .mp4/.mov/.avi first."
            )
            return

        # 3) Confirm destructive overwrite
        if not messagebox.askyesno(
            "Confirm overwrite",
            f"This will permanently replace:\n\n{src.name}\n\nwith the trimmed segment ({a}..{b}). Continue?"
        ):
            return

        # 4) Release the display cap so the file can be reopened
        if self.cap:
            self.cap.release()
            self.cap = None

        tmp_path = self._tmp_path_for(src)  # e.g., clip_tmp.mp4

        try:
            # 5) Reopen source for the actual copy
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                messagebox.showwarning("OpenCV error", f"Could not re-open: {src.name}")
                return

            # Fresh properties from THIS handle
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_val) if fps_val and fps_val > 0 else (self.fps or 30.0)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                cap.release()
                messagebox.showerror("Property error", "Invalid frame size reported by codec.")
                return

            # Seek to first frame to copy
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)

            fourcc = self._fourcc_for_suffix(ext)
            writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                messagebox.showerror(
                    "Writer error",
                    f"Could not open writer for: {tmp_path.name}\n"
                    "Tip: try .mp4/.mov with mp4v or .avi with MJPG."
                )
                return

            # 6) Write frames [a..b] inclusive
            frame_idx = a
            ok = True
            while frame_idx <= b:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                frame_idx += 1

            writer.release()
            cap.release()

            if frame_idx <= b:
                # Didn’t reach requested end; clean up and report
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                messagebox.showerror(
                    "Read error",
                    f"Stopped early at frame {frame_idx-1} of {b}. "
                    "Source may be variable-FPS or corrupted."
                )
                return

            # 7) Atomic replace and reload
            os.replace(str(tmp_path), str(src))
            print(f"✅ Overwrote: {src}")
            self.open_video(src)

        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            messagebox.showerror("Error", f"Failed to overwrite file:\n{e}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            
            if self.start_frame is None or self.end_frame is None:
                messagebox.showwarning("Missing marks", "Please set both Start and End frames.")
                return
            a, b = sorted([self.start_frame, self.end_frame])
            if a == b:
                messagebox.showwarning("Zero length", "Start and End are the same frame.")
                return

            src = self.video_files[self.current_index]
            ext = src.suffix.lower()

            if ext == ".hevc":
                messagebox.showerror(
                    "Unsupported",
                    "In-place overwrite for .hevc is not supported by OpenCV. Convert to .mp4/.mov/.avi first."
                )
                return

            # Confirm destructive overwrite
            if not messagebox.askyesno(
                "Confirm overwrite",
                f"This will permanently replace:\n\n{src.name}\n\nwith the trimmed segment ({a}..{b}). Continue?"
            ):
                return

            # Ensure no open handle on the file before replacing
            if self.cap:
                self.cap.release()
                self.cap = None

            tmp_path = self._tmp_path_for(src) 

            try:
                # Re-open to copy only [a..b]
                cap = cv2.VideoCapture(str(src))
                if not cap.isOpened():
                    messagebox.showwarning("OpenCV error", f"Could not re-open: {src.name}")
                    return
                cap.set(cv2.CAP_PROP_POS_FRAMES, a)

                fourcc = self._fourcc_for_suffix(ext)
                if fourcc is None:
                    messagebox.showerror("Codec error", f"No suitable codec for extension {ext}.")
                    cap.release()
                    return

                writer = cv2.VideoWriter(str(tmp_path), fourcc, self.fps, (self.width, self.height))
                if not writer.isOpened():
                    cap.release()
                    messagebox.showerror("Writer error", f"Could not open writer for: {tmp_path.name}")
                    return

                frame_idx = a
                ok, frame = cap.read()
                while ok and frame_idx <= b:
                    writer.write(frame)
                    frame_idx += 1
                    ok, frame = cap.read()

                writer.release()
                cap.release()

                # Atomic replace
                os.replace(str(tmp_path), str(src))

                print(f"✅ Overwrote: {src}")
                # Reload the now-trimmed clip
                self.open_video(src)

            except Exception as e:
                # Cleanup tmp if something went wrong
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                messagebox.showerror("Error", f"Failed to overwrite file:\n{e}")

    # ---------- Cleanup ----------
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.quit()

# ========================================
# Main
# ========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmerApp(root)
    root.mainloop()
