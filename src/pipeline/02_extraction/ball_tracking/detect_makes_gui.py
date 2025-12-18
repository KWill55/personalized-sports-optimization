import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from pathlib import Path
from PIL import Image, ImageTk
import yaml
import csv
import math
import os

# =========================
# Config loaders (project + session)
# =========================
def load_configs():
    # Expect these YAMLs next to where you run the script (like your detect_makes.py)
    proj_path = Path("project_config.yaml")
    sess_path = Path("session_config.yaml")
    if not proj_path.exists():
        raise FileNotFoundError("project_config.yaml not found")
    if not sess_path.exists():
        raise FileNotFoundError("session_config.yaml not found")

    with open(proj_path, "r") as f:
        project_cfg = yaml.safe_load(f)
    with open(sess_path, "r") as f:
        session_cfg = yaml.safe_load(f)

    ATHLETE = str(project_cfg["athlete"])
    SESSION = str(project_cfg["session"])
    FRAME_WIDTH = int(project_cfg["original_frame_width"])
    FRAME_HEIGHT = int(project_cfg["original_frame_height"])
    CROP_SIZE = tuple(project_cfg["crop_size"])
    FPS_DEFAULT = float(project_cfg["player_tracking_fps"])

    SESSION_INFO = session_cfg["athletes"][ATHLETE][SESSION]
    UPPER = SESSION_INFO["hoop_regions"]["upper"]          # ((x1,y1),(x2,y2))
    LOWER = SESSION_INFO["hoop_regions"]["lower"]
    HSV_LOWER = np.array(SESSION_INFO["hsv_ranges"]["lower"], dtype=np.uint8)
    HSV_UPPER = np.array(SESSION_INFO["hsv_ranges"]["upper"], dtype=np.uint8)
    AREA_MIN = float(SESSION_INFO["ball_area_px"]["min"])
    AREA_MAX = float(SESSION_INFO["ball_area_px"]["max"])
    CIRC_MIN = float(SESSION_INFO["circularity_min"])
    FILL_MIN = float(SESSION_INFO["fill_ratio_min"])

    # Repo-style paths (like your script)
    try:
        BASE_DIR = Path(__file__).resolve().parents[3]
    except Exception:
        BASE_DIR = Path.cwd()

    SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
    INPUT_FOLDER = SESSION_DIR / "videos" / "ball_tracking" / "raw"
    OUTPUT_PATH = SESSION_DIR / "analysis" / "outcomes.csv"

    return {
        "ATHLETE": ATHLETE,
        "SESSION": SESSION,
        "FRAME_WIDTH": FRAME_WIDTH,
        "FRAME_HEIGHT": FRAME_HEIGHT,
        "CROP_SIZE": CROP_SIZE,
        "FPS_DEFAULT": FPS_DEFAULT,
        "UPPER_HOOP_REGION": tuple(map(tuple, UPPER)),
        "LOWER_HOOP_REGION": tuple(map(tuple, LOWER)),
        "HSV_LOWER": HSV_LOWER,
        "HSV_UPPER": HSV_UPPER,
        "AREA_MIN": AREA_MIN,
        "AREA_MAX": AREA_MAX,
        "CIRC_MIN": CIRC_MIN,
        "FILL_MIN": FILL_MIN,
        "INPUT_FOLDER": INPUT_FOLDER,
        "OUTPUT_PATH": OUTPUT_PATH,
        "BASE_DIR": BASE_DIR,
        "SESSION_DIR": SESSION_DIR,
    }


# =========================
# Geometry helpers
# =========================
def is_inside_rect(pt, rect_tl, rect_br):
    if pt is None:
        return False
    x, y = pt
    x1, y1 = rect_tl
    x2, y2 = rect_br
    return min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2)

def is_make(trajectory, upper_box, lower_box):
    in_upper = False
    waiting_for_next = False
    for i in range(1, len(trajectory)):
        prev, curr = trajectory[i-1], trajectory[i]
        if curr is None:
            continue
        # Step 1: Entered upper box from above (descending)
        if not in_upper and is_inside_rect(curr, *upper_box):
            if prev and curr[1] > prev[1]:
                in_upper = True
                waiting_for_next = True
            continue
        # Step 2: after upper entry, next visible point check
        if in_upper and waiting_for_next:
            if is_inside_rect(curr, *lower_box):
                return True
            elif not is_inside_rect(curr, *upper_box):
                return False
    return False

# =========================
# Detection (with “prev_center” smoothing)
# =========================
class BallDetector:
    def __init__(self, hsv_lo, hsv_hi, area_min, area_max, circ_min, fill_min):
        self.hsv_lo = hsv_lo
        self.hsv_hi = hsv_hi
        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.fill_min = fill_min
        self.prev_center = None

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.area_min < area < self.area_max):
                continue
            per = cv2.arcLength(cnt, True)
            if per <= 0:
                continue
            circ = 4 * math.pi * area / (per * per)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius <= 0:
                continue
            fill = area / (math.pi * radius * radius)
            if circ < self.circ_min or fill < self.fill_min:
                continue
            if self.prev_center is not None:
                dx = abs(x - self.prev_center[0])
                dy = abs(y - self.prev_center[1])
                if dx < 2 and dy < 2:
                    continue
            score = circ * fill
            if score > best_score:
                best_score = score
                best = (int(x), int(y))

        if best is not None:
            self.prev_center = best
        return best, mask

    def update_params(self, hsv_lo, hsv_hi, area_min, area_max, circ_min, fill_min):
        self.hsv_lo = hsv_lo
        self.hsv_hi = hsv_hi
        self.area_min = area_min
        self.area_max = area_max
        self.circ_min = circ_min
        self.fill_min = fill_min
        self.prev_center = None

# =========================
# GUI App
# =========================
class FreeThrowReviewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Free Throw Reviewer (Detect Makes)")

        # Load config
        self.cfg = load_configs()

        # IO
        self.video_files = []
        self.current_index = 0
        self.results = {}  # {video_name: label}

        # Playback state
        self.cap = None
        self.playing = False
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0

        # Analysis state per clip
        self.trajectory = []          # per-frame list (points or None)
        self.detector = BallDetector(
            self.cfg["HSV_LOWER"], self.cfg["HSV_UPPER"],
            self.cfg["AREA_MIN"], self.cfg["AREA_MAX"],
            self.cfg["CIRC_MIN"], self.cfg["FILL_MIN"]
        )
        self.auto_verdict = None
        self.manual_label = None

        # UI toggles
        self.show_traj = True
        self.show_mask = False

        # ===== UI =====
        Label(root, text="Detect Makes — Player Viewer", font=("Helvetica", 22, "bold")).pack(pady=(8, 0))

        vf = tk.Frame(root, bg="black", padx=4, pady=4)
        vf.pack(pady=8)
        self.video_label = Label(vf)
        self.video_label.pack()

        # Controls 1
        c1 = tk.Frame(root)
        c1.pack(pady=6)
        Button(c1, text="Load Folder", command=self.load_folder).grid(row=0, column=0, padx=5)
        Button(c1, text="Previous Clip", command=self.prev_video, 
            bg="orange", activebackground="darkorange").grid(row=0, column=1, padx=5)
        Button(c1, text="Play/Pause", command=self.toggle_play).grid(row=0, column=2, padx=5)
        Button(c1, text="Next Clip", command=self.next_video, 
            bg="orange", activebackground="darkorange").grid(row=0, column=3, padx=5)
        Button(c1, text="Restart Clip", command=self.restart_clip).grid(row=0, column=4, padx=5)

        # Controls 2
        c2 = tk.Frame(root)
        c2.pack(pady=6)
        Button(c2, text="<< Frame", command=self.prev_frame).grid(row=0, column=0, padx=5)
        Button(c2, text="Frame >>", command=self.next_frame).grid(row=0, column=1, padx=5)
        Button(c2, text="Reload YAML (R)", command=self.reload_yaml).grid(row=0, column=4, padx=12)

        # Labels/Status
        info = tk.Frame(root)
        info.pack(pady=8)
        self.video_info = tk.StringVar(value="No videos loaded")
        Label(info, textvariable=self.video_info, font=("Helvetica", 13)).pack()

        verdict_box = tk.Frame(root)
        verdict_box.pack(pady=6)
        self.verdict_info = tk.StringVar(value="Auto: —    Label: —")
        Label(verdict_box, textvariable=self.verdict_info, font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(0,6))
        Button(verdict_box, text="Mark MAKE (M)", command=lambda: self.set_label("made")).grid(row=1, column=0, padx=6)
        Button(verdict_box, text="Mark MISS (S)", command=lambda: self.set_label("miss")).grid(row=1, column=1, padx=6)
        Button(verdict_box, text="Mark UNKNOWN (U)", command=lambda: self.set_label("unknown")).grid(row=1, column=2, padx=6)

        self.status = tk.StringVar(value="Press ‘Load Folder’ (defaults to session's ball_tracking/raw). Space=Play/Pause.")
        Label(root, textvariable=self.status, font=("Helvetica", 12)).pack(pady=(4, 10))

        # Key bindings
        root.bind("<space>", lambda e: self.toggle_play())
        root.bind("<Right>", lambda e: self.jump_frames(1))
        root.bind("<Left>",  lambda e: self.jump_frames(-1))
        root.bind("<Shift-Right>", lambda e: self.jump_frames(10))
        root.bind("<Shift-Left>",  lambda e: self.jump_frames(-10))
        root.bind("<Next>",  lambda e: self.jump_frames(60))   # PageDown
        root.bind("<Prior>", lambda e: self.jump_frames(-60))  # PageUp
        root.bind("<Home>",  lambda e: self.seek_frame(0))
        root.bind("<End>",   lambda e: self.seek_frame(self.total_frames - 1))

        root.bind("<r>",  lambda e: self.reload_yaml())
        root.bind("<R>",  lambda e: self.reload_yaml())

        root.bind("<m>",  lambda e: self.set_label("made"))
        root.bind("<s>",  lambda e: self.set_label("miss"))
        root.bind("<u>",  lambda e: self.set_label("unknown"))

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Prompt on launch
        self.load_folder(initial=self.cfg["INPUT_FOLDER"])

    # ---------- Folder / navigation ----------
    def load_folder(self, initial=None):
        init_dir = str(initial) if (initial and Path(initial).exists()) else str(Path.home())
        folder = filedialog.askdirectory(initialdir=init_dir, title="Select Folder with Videos")
        if not folder:
            return
        p = Path(folder)
        vids = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".mp4",".avi",".mov"}]
        self.video_files = sorted(vids, key=lambda q: q.name.lower())
        if not self.video_files:
            messagebox.showerror("No videos", "No .mp4/.mov/.avi files found.")
            return
        self.current_index = 0
        self.open_video(self.video_files[self.current_index])
        self.status.set(f"Loaded {len(self.video_files)} videos from: {p}")

    def open_video(self, path: Path):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showwarning("OpenCV", f"Could not open: {path.name}")
            self.next_video()
            return

        # properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else (self.cfg["player_tracking_fps"])
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # reset analysis state
        self.trajectory = []
        self.detector.prev_center = None
        self.auto_verdict = None
        # prefill manual label if previously set (so you can revisit)
        self.manual_label = self.results.get(path.name, None)

        # show first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._read_process_show()

        idx = self.current_index + 1
        total = len(self.video_files)
        duration = (self.total_frames / self.fps) if self.fps > 0 else 0.0
        self.video_info.set(
            f"Filename: {path.name}  (Clip {idx}/{total})  |  "
            f"{self.width}x{self.height}, {self.total_frames} frames, {self.fps:.2f} FPS, {duration:.2f}s"
        )
        self._update_verdict_label()

    def next_video(self):
        if not self.video_files: return
        self._save_current_result()
        self.current_index = (self.current_index + 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def prev_video(self):
        if not self.video_files: return
        self._save_current_result()
        self.current_index = (self.current_index - 1) % len(self.video_files)
        self.open_video(self.video_files[self.current_index])

    def restart_clip(self):
        if not self.cap: return
        self.playing = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.trajectory = []
        self.detector.prev_center = None
        self.auto_verdict = None
        self._read_process_show()
        self._update_verdict_label()

    # ---------- Playback / stepping ----------
    def _play_loop(self):
        if not self.playing:
            return
        # if at end, compute auto-verdict once and stop
        if self._current_frame_index() >= self.total_frames - 1:
            self._compute_auto_verdict_if_needed()
            self.playing = False
            return
        self._read_process_show()
        delay = int(1000 / (self.fps if self.fps > 0 else 30.0))
        self.root.after(delay, self._play_loop)

    def toggle_play(self):
        if not self.cap: return
        if self._current_frame_index() >= self.total_frames - 1:
            self.seek_frame(0)
        self.playing = not self.playing
        if self.playing:
            self._play_loop()

    def next_frame(self):
        if not self.cap: return
        self.playing = False
        self._read_process_show()

    def prev_frame(self):
        if not self.cap: return
        self.playing = False
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
        # NOTE: stepping backward does not undo trajectory; for accurate auto, restart the clip.
        self._read_process_show()

    def jump_frames(self, n):
        if not self.cap: return
        self.playing = False
        target = max(0, min(self.total_frames - 1, self._current_frame_index() + n))
        self.seek_frame(target)

    def seek_frame(self, idx):
        if not self.cap: return
        self.playing = False
        idx = max(0, min(self.total_frames - 1, int(idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # Seeking breaks sequential trajectory; show frame but do not append to trajectory.
        self._read_process_show(update_trajectory=False)

    def _current_frame_index(self):
        if not self.cap: return 0
        # POS_FRAMES is "next to read", so subtract 1 after a read:
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, min(self.total_frames - 1, pos - 1))

    # ---------- Analysis / draw ----------
    def _read_process_show(self, update_trajectory=True):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            # End reached
            self._compute_auto_verdict_if_needed()
            return

        # Detect + optional append to trajectory (append None when no detection)
        center, mask = self.detector.detect(frame)
        if update_trajectory:
            self.trajectory.append(center if center is not None else None)

        # Overlays: hoop regions, detection, trajectory, optional HSV mask
        out = frame.copy()

        # Optional HSV mask overlay
        if self.show_mask:
            overlay = out.copy()
            # colorize mask yellow-ish for visibility
            color = np.zeros_like(out)
            color[:, :] = (0, 255, 255)  # BGR
            overlay = np.where(mask[..., None] > 0, color, overlay)
            cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

        # Draw hoops
        cv2.rectangle(out, self.cfg["UPPER_HOOP_REGION"][0], self.cfg["UPPER_HOOP_REGION"][1], (255, 0, 0), 2)
        cv2.rectangle(out, self.cfg["LOWER_HOOP_REGION"][0], self.cfg["LOWER_HOOP_REGION"][1], (0, 0, 255), 2)

        # Draw current center
        if center:
            cv2.circle(out, center, 6, (0, 255, 0), -1)

        # Draw trajectory
        if self.show_traj and len(self.trajectory) > 1:
            for pt in self.trajectory:
                if pt is not None:
                    cv2.circle(out, pt, 2, (0, 200, 255), -1)

        # Header/status
        idx = self.current_index + 1
        total = len(self.video_files)
        name = self.video_files[self.current_index].name
        header = f"Clip {idx}/{total} — {name}"
        cv2.putText(out, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # Convert to Tk image
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=img)
        self.video_label.image = img

        # Status line
        i = self._current_frame_index()
        t = (i / self.fps) if self.fps > 0 else 0.0
        self.status.set(f"Frame {i}/{self.total_frames-1}  |  t={t:.2f}s  |  "
                        f"Traj points: {sum(1 for p in self.trajectory if p is not None)}")
        self._update_verdict_label()

    def _compute_auto_verdict_if_needed(self):
        if self.auto_verdict is None and len(self.trajectory) > 1:
            self.auto_verdict = "made" if is_make(self.trajectory, self.cfg["UPPER_HOOP_REGION"], self.cfg["LOWER_HOOP_REGION"]) else "miss"
            self._update_verdict_label()

    def _update_verdict_label(self):
        auto_txt = self.auto_verdict if self.auto_verdict is not None else "—"
        label_txt = self.manual_label if self.manual_label is not None else "—"
        self.verdict_info.set(f"Auto: {auto_txt}    Label: {label_txt}")

    # ---------- Toggles / actions ----------

    def reload_yaml(self):
        try:
            new = load_configs()
            self.cfg.update(new)
            self.detector.update_params(
                self.cfg["HSV_LOWER"], self.cfg["HSV_UPPER"],
                self.cfg["AREA_MIN"], self.cfg["AREA_MAX"],
                self.cfg["CIRC_MIN"], self.cfg["FILL_MIN"]
            )
            self.status.set("Reloaded YAML params.")
        except Exception as e:
            messagebox.showerror("YAML reload error", str(e))

    def set_label(self, label):
        # "made" | "miss" | "unknown"
        self.manual_label = label
        self._update_verdict_label()

    # ---------- Save results ----------
    def _save_current_result(self):
        if not self.video_files:
            return
        name = self.video_files[self.current_index].name
        # Priority: manual, otherwise auto, otherwise unknown
        label = self.manual_label or self.auto_verdict or "unknown"
        self.results[name] = label

        # Write/append to CSV in the same schema your analysis expects
        out_path = self.cfg["OUTPUT_PATH"]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build full rows: map video name -> feature_key + normalized label
        base = Path(name).stem
        core = base.split("_")[0]  # e.g., "freethrow001"
        feature_key = f"{core}_angles.csv"
        normalized = {"made":"made", "miss":"miss"}.get(label, None)
        if normalized is None:
            # skip unknown to keep dataset clean
            return
        
        existing = {}
        if out_path.exists():
            with open(out_path, "r", newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if len(row) >= 2:
                        existing[row[0]] = row[1]

        existing[feature_key] = normalized
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "outcome"])
            for k,v in sorted(existing.items()):
                w.writerow([k, v])

    # ---------- Cleanup ----------
    def on_close(self):
        # Save the current clip result before exiting
        try:
            self._save_current_result()
        except Exception:
            pass
        if self.cap:
            self.cap.release()
        self.root.quit()

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = FreeThrowReviewerApp(root)
    root.mainloop()