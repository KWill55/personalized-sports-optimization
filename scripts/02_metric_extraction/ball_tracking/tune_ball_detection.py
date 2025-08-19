#!/usr/bin/env python3
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
from pathlib import Path
import yaml
import math

# -------------------------------
# Config: Load from project/session YAMLs
# -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]

PROJECT_PATH = BASE_DIR / "project_config.yaml"
SESSION_PATH = BASE_DIR / "session_config.yaml"

with open(PROJECT_PATH, "r") as f:
    CONFIG1 = yaml.safe_load(f)
with open(SESSION_PATH, "r") as f:
    CONFIG2 = yaml.safe_load(f)

ATHLETE = str(CONFIG1["athlete"])
SESSION = str(CONFIG1["session"])
FRAME_WIDTH  = int(CONFIG1["original_frame_width"])
FRAME_HEIGHT = int(CONFIG1["original_frame_height"])

SESSION_INFO = CONFIG2["athletes"][ATHLETE][SESSION]
UPPER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["upper"]))   # ((x1,y1),(x2,y2))
LOWER_HOOP_REGION = tuple(map(tuple, SESSION_INFO["hoop_regions"]["lower"]))

# Defaults pulled from session config (fall back to reasonable values)
HSV_LOWER_DEFAULT = np.array(SESSION_INFO.get("hsv_ranges", {}).get("lower", [5,100,100]), dtype=np.uint8)
HSV_UPPER_DEFAULT = np.array(SESSION_INFO.get("hsv_ranges", {}).get("upper", [30,255,255]), dtype=np.uint8)
AREA_MIN_DEFAULT  = float(SESSION_INFO.get("ball_area_px", {}).get("min", 30))
AREA_MAX_DEFAULT  = float(SESSION_INFO.get("ball_area_px", {}).get("max", 2000))
CIRC_MIN_DEFAULT  = float(SESSION_INFO.get("circularity_min", 0.6))
FILL_MIN_DEFAULT  = float(SESSION_INFO.get("fill_ratio_min", 0.6))

# Optional extras (if present)
MIN_MOTION_DEFAULT   = int(SESSION_INFO.get("min_motion", SESSION_INFO.get("min_motion_px", 0)))
ERODE_ITERS_DEFAULT  = int(SESSION_INFO.get("erode_iters", SESSION_INFO.get("morph", {}).get("erode", 0)))
DILATE_ITERS_DEFAULT = int(SESSION_INFO.get("dilate_iters", SESSION_INFO.get("morph", {}).get("dilate", 2)))
BLUR_KSIZE_DEFAULT   = int(SESSION_INFO.get("blur_ksize", 1))  # 1 = off

# -------------------------------
# Default session video dir (used by "Load Folder")
# -------------------------------
SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION
DEFAULT_VIDEO_DIR = SESSION_DIR / "videos" / "ball_tracking" / "raw"
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

# ===============================================
# Global video state (list + current capture)
# ===============================================
video_files = []          # [Path, ...]
current_vid_idx = -1      # index into video_files
cap = None                # cv2.VideoCapture
frame_count = 1
fps = 30.0
cur_idx = [0]             # current frame index (int)
frame_resized = None
hsv_img = None            # HSV buffer for current frame

# =========================
# Tk App
# =========================
root = tk.Tk()
root.title("Tune Ball Tracking (HSV + Shape + Morph)")

style = ttk.Style(root)
try:
    style.theme_use("clam")
except tk.TclError:
    pass

# -------------------------------
# Tk Variables (sliders)
# -------------------------------
# HSV
H_MIN = tk.IntVar(value=int(HSV_LOWER_DEFAULT[0]))
H_MAX = tk.IntVar(value=int(HSV_UPPER_DEFAULT[0]))
S_MIN = tk.IntVar(value=int(HSV_LOWER_DEFAULT[1]))
S_MAX = tk.IntVar(value=int(HSV_UPPER_DEFAULT[1]))
V_MIN = tk.IntVar(value=int(HSV_LOWER_DEFAULT[2]))
V_MAX = tk.IntVar(value=int(HSV_UPPER_DEFAULT[2]))

# Shape filters
AREA_MIN = tk.IntVar(value=int(AREA_MIN_DEFAULT))
AREA_MAX = tk.IntVar(value=int(AREA_MAX_DEFAULT))
CIRC_MIN = tk.DoubleVar(value=float(CIRC_MIN_DEFAULT))
FILL_MIN = tk.DoubleVar(value=float(FILL_MIN_DEFAULT))

# Stabilization / Morph / Blur
MIN_MOTION = tk.IntVar(value=int(MIN_MOTION_DEFAULT))       # px threshold
ERODE_ITERS = tk.IntVar(value=int(ERODE_ITERS_DEFAULT))     # 0..6
DILATE_ITERS = tk.IntVar(value=int(DILATE_ITERS_DEFAULT))   # 0..6
BLUR_KSIZE = tk.IntVar(value=int(BLUR_KSIZE_DEFAULT))       # 1 (off), 3,5,7,...

# Toggles
SHOW_MASK = tk.BooleanVar(value=True)
SHOW_TRAJ = tk.BooleanVar(value=True)  # trajectory while you scrub
MASK_MODE = tk.StringVar(value="HSV")  # "HSV" or "FILTERED"

# Active slider & registries
active_name = [None]
label_vars = {}      # name -> StringVar text
slider_vars = {}     # name -> associated Tk var
slider_ranges = {}   # name -> (from_, to_, is_float)

# Keep a short trajectory for visualization across scrubs
traj_points = []

# -------------------------------
# Video helpers (folder + open + load frame)
# -------------------------------
def list_videos_in(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
                  key=lambda x: x.name.lower())

def close_cap():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def open_video(idx: int):
    """Open video at index; update frame_count/fps, reset seek, traj, labels."""
    global cap, frame_count, fps, cur_idx, frame_resized, hsv_img, current_vid_idx, traj_points

    if not video_files:
        return False
    idx = max(0, min(idx, len(video_files)-1))
    current_vid_idx = idx

    close_cap()
    path = video_files[idx]
    cap_local = cv2.VideoCapture(str(path))
    if not cap_local.isOpened():
        messagebox.showwarning("OpenCV", f"Could not open: {path.name}")
        return False

    cap = cap_local
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    fps_local = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_local) if fps_local and fps_local > 0 else 30.0
    cur_idx[0] = 0
    frame_resized = None
    hsv_img = None
    traj_points = []  # reset on new clip

    # update seek slider range + labels
    seek_slider.configure(to=max(0, frame_count-1))
    clip_label_var.set(f"Clip {current_vid_idx+1}/{len(video_files)} — {path.name}")

    # load first frame for display
    ok = load_frame(0)
    if ok:
        update_image()
    return ok

def load_frame(idx: int) -> bool:
    """Seek to frame idx and refresh frame_resized/hsv_img globals."""
    global frame_resized, hsv_img
    if cap is None:
        return False

    idx = max(0, min(idx, frame_count - 1))
    cur_idx[0] = idx
    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        # Fallback: brute-forward from 0 (rare with some codecs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(idx + 1):
            ret, frame = cap.read()
            if not ret:
                break
    if not ret:
        print(f"❌ Could not load frame {idx} from: {video_files[current_vid_idx].name}")
        return False

    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    hsv_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    return True

# -------------------------------
# Detection Logic
# -------------------------------
_prev_center = [None]

def odd_ksize(k):
    """Force odd kernel sizes; 1 means 'no blur'."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1

def detect_best(frame_bgr):
    """
    Returns: (center, raw_mask, filtered_mask, metrics)
      - center: (x,y) or None
      - raw_mask: HSV thresholded mask (uint8)
      - filtered_mask: mask of accepted contours only (uint8)
      - metrics: {'area','circ','fill','radius','x','y'} for best, or None
    """
    # 1) Preprocess
    hmin, smin, vmin = H_MIN.get(), S_MIN.get(), V_MIN.get()
    hmax, smax, vmax = H_MAX.get(), S_MAX.get(), V_MAX.get()
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    hsv = hsv_img  # already computed for current frame
    if BLUR_KSIZE.get() > 1:
        k = odd_ksize(BLUR_KSIZE.get())
        hsv = cv2.GaussianBlur(hsv_img, (k, k), 0)

    raw_mask = cv2.inRange(hsv, lower, upper)

    mask = raw_mask.copy()
    if ERODE_ITERS.get() > 0:
        mask = cv2.erode(mask, None, iterations=int(ERODE_ITERS.get()))
    if DILATE_ITERS.get() > 0:
        mask = cv2.dilate(mask, None, iterations=int(DILATE_ITERS.get()))

    # 2) Find best contour by circ*fill score and gate by area/circ/fill
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_min = max(0, int(AREA_MIN.get()))
    area_max = max(area_min+1, int(AREA_MAX.get()))
    circ_min = max(0.0, min(1.0, float(CIRC_MIN.get())))
    fill_min = max(0.0, min(1.0, float(FILL_MIN.get())))

    best_score = -1.0
    best_center = None
    best_metrics = None
    accepted = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if not (area_min < area < area_max):
            continue
        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue
        circ = 4.0 * math.pi * area / (per * per)

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r <= 0:
            continue
        fill = float(area) / (math.pi * r * r)

        if circ < circ_min or fill < fill_min:
            continue

        accepted.append(cnt)

        score = circ * fill
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))
            best_metrics = {"area": area, "circ": circ, "fill": fill, "radius": r, "x": x, "y": y}

    # Build filtered mask from accepted contours only (intersect with raw for cleanliness)
    filtered_mask = np.zeros_like(mask)
    if accepted:
        cv2.drawContours(filtered_mask, accepted, -1, 255, thickness=cv2.FILLED)
        filtered_mask = cv2.bitwise_and(filtered_mask, raw_mask)

    if best_center is not None:
        _prev_center[0] = best_center

    return best_center, raw_mask, filtered_mask, best_metrics

# -------------------------------
# UI pieces (sliders, labels, folder controls)
# -------------------------------
def set_active(name):
    active_name[0] = name

def update_label_text(name, is_float):
    v = slider_vars[name].get()
    if is_float:
        label_vars[name].set(f"{name}: {float(v):.3f}")
    else:
        label_vars[name].set(f"{name}: {int(v)}")

def on_slider(name=None, is_float=False):
    if name is not None:
        update_label_text(name, is_float)
    update_image()

def clamp(val, lo, hi, is_float):
    if is_float:
        return max(lo, min(hi, float(val)))
    return int(max(lo, min(hi, int(val))))

def prompt_value(name):
    """Double-click label (or press Enter) -> prompt for exact value."""
    if name not in slider_vars:
        return
    var = slider_vars[name]
    lo, hi, is_float = slider_ranges[name]

    if is_float:
        val = simpledialog.askfloat("Set value", f"{name} [{lo}..{hi}]:",
                                    initialvalue=float(var.get()),
                                    minvalue=float(lo), maxvalue=float(hi), parent=root)
    else:
        val = simpledialog.askinteger("Set value", f"{name} [{int(lo)}..{int(hi)}]:",
                                      initialvalue=int(var.get()),
                                      minvalue=int(lo), maxvalue=int(hi), parent=root)
    if val is None:
        return

    val = clamp(val, lo, hi, is_float)
    var.set(val)

    # Keep paired bounds sane
    def ensure_pair(min_name, max_name, require_strict=False):
        if name == min_name:
            vmin = slider_vars[min_name].get()
            vmax = slider_vars[max_name].get()
            if (vmin > vmax) or (require_strict and vmin >= vmax):
                slider_vars[max_name].set(vmin + (1 if not is_float else 0.0))
        elif name == max_name:
            vmin = slider_vars[min_name].get()
            vmax = slider_vars[max_name].get()
            if (vmax < vmin) or (require_strict and vmax <= vmin):
                slider_vars[min_name].set(vmax - (1 if not is_float else 0.0))

    # HSV pairs (allow equality)
    ensure_pair("H Min", "H Max", require_strict=False)
    ensure_pair("S Min", "S Max", require_strict=False)
    ensure_pair("V Min", "V Max", require_strict=False)
    # Area pairs (strict inequality)
    ensure_pair("Area Min (px)", "Area Max (px)", require_strict=True)

    update_label_text(name, is_float)
    set_active(name)
    update_image()

def prompt_value_for_active(_evt=None):
    if active_name[0]:
        prompt_value(active_name[0])

def build_slider(parent, name, var, from_, to_, is_float=False, step=1.0):
    slider_vars[name] = var
    slider_ranges[name] = (from_, to_, is_float)

    row = ttk.Frame(parent); row.pack(fill="x", pady=2)
    label_var = tk.StringVar()
    label_vars[name] = label_var

    lab = ttk.Label(row, textvariable=label_var, width=22)
    lab.pack(side="left")

    scale = ttk.Scale(row, from_=from_, to=to_, orient="horizontal",
                      variable=var, command=lambda _=None: on_slider(name, is_float))
    scale.pack(side="left", expand=True, fill="x", padx=6)

    lab.bind("<Button-1>",  lambda e, n=name: set_active(n))      # single click = active
    lab.bind("<Double-Button-1>", lambda e, n=name: prompt_value(n))  # double click = type exact
    update_label_text(name, is_float)

# --- Keyboard nudge handler for the active slider ---
def on_key_adjust(event):
    """
    Left/Right arrow to tweak the currently 'active' slider.
    - Int sliders move by 1 and clamp to their [min, max].
    - Float sliders move by 0.01 and clamp to their [min, max].
    """
    name = active_name[0]
    if not name:
        return

    var = slider_vars[name]
    lo, hi, is_float = slider_ranges[name]

    if is_float:
        step = 0.01
        if event.keysym == "Left":
            var.set(max(float(lo), float(var.get()) - step))
        elif event.keysym == "Right":
            var.set(min(float(hi), float(var.get()) + step))
        update_label_text(name, True)
    else:
        if event.keysym == "Left":
            var.set(max(int(lo), int(var.get()) - 1))
        elif event.keysym == "Right":
            var.set(min(int(hi), int(var.get()) + 1))
        update_label_text(name, False)

    update_image()


# -------------------------------
# Image Display
# -------------------------------
metrics_var = tk.StringVar(value="area: —  circ: —  fill: —  r: —")
clip_label_var = tk.StringVar(value="No folder loaded")

def update_image():
    if frame_resized is None:
        return
    center, raw_mask, filt_mask, metrics = detect_best(frame_resized)

    annotated = frame_resized.copy()

    # Optional mask overlay (choose raw HSV vs filtered)
    if SHOW_MASK.get():
        use_mask = raw_mask if MASK_MODE.get() == "HSV" else filt_mask
        overlay = np.zeros_like(annotated)
        overlay[:] = (0, 255, 255)  # yellow
        annotated = np.where(use_mask[..., None] > 0,
                             cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0),
                             annotated)

    # Draw hoop boxes (context)
    cv2.rectangle(annotated, UPPER_HOOP_REGION[0], UPPER_HOOP_REGION[1], (255, 0, 0), 2)
    cv2.rectangle(annotated, LOWER_HOOP_REGION[0], LOWER_HOOP_REGION[1], (0, 0, 255), 2)

    # Draw best candidate
    if center is not None:
        cv2.circle(annotated, center, 6, (0, 255, 0), -1)
        if SHOW_TRAJ.get():
            traj_points.append(center)
            if len(traj_points) > 200:
                del traj_points[:len(traj_points)-200]
    else:
        traj_points.append(None)

    if SHOW_TRAJ.get() and len(traj_points) > 1:
        for pt in traj_points:
            if pt is not None:
                cv2.circle(annotated, pt, 2, (0, 200, 255), -1)

    # HUD
    t = cur_idx[0] / max(fps, 1.0)
    cv2.putText(annotated, f"Frame {cur_idx[0]} / {frame_count-1}  ({t:.2f}s)",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Ball Tracking Preview", annotated)
    cv2.waitKey(1)

    if metrics is not None:
        metrics_var.set(f"area: {metrics['area']:.1f}   circ: {metrics['circ']:.3f}   "
                        f"fill: {metrics['fill']:.3f}   r: {metrics['radius']:.1f}")
    else:
        metrics_var.set("area: —   circ: —   fill: —   r: —")

# -------------------------------
# Scrubber + Keybinds + Video nav
# -------------------------------
def on_seek(val):
    idx = int(float(val))
    if load_frame(idx):
        update_image()

def step_frames(delta: int):
    new_idx = max(0, min(cur_idx[0] + delta, frame_count - 1))
    if load_frame(new_idx):
        seek_slider.set(new_idx)
        update_image()

def on_video_keys(event):
    if event.keysym in ("comma",):       step_frames(-1)
    elif event.keysym in ("period",):    step_frames(+1)
    elif event.keysym in ("j", "J"):     step_frames(-int(fps) if fps >= 1 else -10)
    elif event.keysym in ("k", "K"):     step_frames(+int(fps) if fps >= 1 else +10)
    elif event.keysym in ("n", "N"):     next_clip()
    elif event.keysym in ("p", "P"):     prev_clip()

def next_clip():
    if not video_files:
        return
    idx = (current_vid_idx + 1) % len(video_files)
    if open_video(idx):
        seek_slider.set(0)
        update_image()

def prev_clip():
    if not video_files:
        return
    idx = (current_vid_idx - 1) % len(video_files)
    if open_video(idx):
        seek_slider.set(0)
        update_image()

# -------------------------------
# Flat YAML snippet output (terminal)
# -------------------------------
def _fmt_float_short(x: float, places=3) -> str:
    s = f"{float(x):.{places}f}".rstrip("0").rstrip(".")
    if s.startswith("0.") and 0.0 < float(x) < 1.0:
        s = s[1:]
    if s == "-0":
        s = "0"
    return s

def build_flat_yaml_text() -> str:
    hmin, smin, vmin = int(H_MIN.get()), int(S_MIN.get()), int(V_MIN.get())
    hmax, smax, vmax = int(H_MAX.get()), int(S_MAX.get()), int(V_MAX.get())
    area_min, area_max = int(AREA_MIN.get()), int(AREA_MAX.get())
    circ_min = _fmt_float_short(float(CIRC_MIN.get()))
    fill_min = _fmt_float_short(float(FILL_MIN.get()))
    min_motion = int(MIN_MOTION.get())
    erode = int(ERODE_ITERS.get())
    dilate = int(DILATE_ITERS.get())
    blurk = int(BLUR_KSIZE.get())

    lines = [
        "hsv_ranges:",
        f"  lower: [{hmin}, {smin}, {vmin}]",
        f"  upper: [{hmax}, {smax}, {vmax}]",
        "",
        f"ball_area_px: {{min: {area_min}, max: {area_max}}}",
        f"circularity_min: {circ_min}",
        f"fill_ratio_min: {fill_min}",
        "",
        f"min_motion: {min_motion}",
        f"erode_iters: {erode}",
        f"dilate_iters: {dilate}",
        f"blur_ksize: {blurk}",
    ]
    return "\n".join(lines)

def print_flat_yaml():
    text = build_flat_yaml_text()
    print("\n# --- Paste this into your session_config.yaml for this athlete/session ---")
    print(text)
    try:
        root.clipboard_clear()
        root.clipboard_append(text)
    except Exception:
        pass

# -------------------------------
# Folder loader UI
# -------------------------------
def load_folder():
    initial = str(DEFAULT_VIDEO_DIR if DEFAULT_VIDEO_DIR.exists() else Path.home())
    chosen = filedialog.askdirectory(initialdir=initial, title="Select Folder with Videos")
    if not chosen:
        return
    folder = Path(chosen)
    files = list_videos_in(folder)
    if not files:
        messagebox.showerror("No videos", "No .mp4/.mov/.avi files found in that folder.")
        return

    global video_files
    video_files = files
    if open_video(0):
        seek_slider.set(0)
        update_image()

# -------------------------------
# Build GUI
# -------------------------------
# Top bar: folder + clip info + nav
top_row = ttk.Frame(root); top_row.pack(fill="x", pady=6, padx=6)
ttk.Button(top_row, text="Load Folder", command=load_folder).pack(side="left")
ttk.Button(top_row, text="Prev Clip (P)", command=prev_clip).pack(side="left", padx=(8,4))
ttk.Button(top_row, text="Next Clip (N)", command=next_clip).pack(side="left", padx=4)
ttk.Label(top_row, textvariable=clip_label_var).pack(side="left", padx=12)

# Seek bar
seek_frame = ttk.Frame(root); seek_frame.pack(fill="x", pady=6, padx=6)
ttk.Label(seek_frame, text="Frame").pack(side="left")
seek_slider = ttk.Scale(seek_frame, from_=0, to=max(0, frame_count-1),
                        orient="horizontal", command=on_seek)
seek_slider.pack(side="left", expand=True, fill="x", padx=6)
frame_label_var = tk.StringVar(value=f"0 / {frame_count-1}")
ttk.Label(seek_frame, textvariable=frame_label_var, width=12).pack(side="right")

# Sections
wrap = ttk.Frame(root); wrap.pack(fill="x", padx=6, pady=6)

# HSV
ttk.Label(wrap, text="HSV Range", font=("Helvetica", 12, "bold")).pack(anchor="w")
build_slider(wrap, "H Min", H_MIN, 0, 179, is_float=False)
build_slider(wrap, "H Max", H_MAX, 0, 179, is_float=False)
build_slider(wrap, "S Min", S_MIN, 0, 255, is_float=False)
build_slider(wrap, "S Max", S_MAX, 0, 255, is_float=False)
build_slider(wrap, "V Min", V_MIN, 0, 255, is_float=False)
build_slider(wrap, "V Max", V_MAX, 0, 255, is_float=False)

# Shape gates
ttk.Label(wrap, text="Shape Filters", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8,2))
build_slider(wrap, "Area Min (px)", AREA_MIN, 0, 10000, is_float=False)
build_slider(wrap, "Area Max (px)", AREA_MAX, 1, 20000, is_float=False)
build_slider(wrap, "Circularity Min", CIRC_MIN, 0.0, 1.0, is_float=True)
build_slider(wrap, "Fill Ratio Min", FILL_MIN, 0.0, 1.0, is_float=True)

# Stabilization / Morph / Blur
ttk.Label(wrap, text="Stabilize / Morphology", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(8,2))
build_slider(wrap, "Min Motion (px)", MIN_MOTION, 0, 20, is_float=False)
build_slider(wrap, "Erode iters", ERODE_ITERS, 0, 6, is_float=False)
build_slider(wrap, "Dilate iters", DILATE_ITERS, 0, 6, is_float=False)
build_slider(wrap, "Blur ksize (odd)", BLUR_KSIZE, 1, 21, is_float=False)

# Toggles + Metrics
tog_row = ttk.Frame(root); tog_row.pack(fill="x", padx=6, pady=6)
ttk.Checkbutton(tog_row, text="Show mask", variable=SHOW_MASK, command=update_image).pack(side="left", padx=6)
ttk.Checkbutton(tog_row, text="Show trajectory", variable=SHOW_TRAJ, command=update_image).pack(side="left", padx=6)
ttk.Radiobutton(tog_row, text="HSV mask", value="HSV", variable=MASK_MODE, command=update_image).pack(side="left", padx=6)
ttk.Radiobutton(tog_row, text="Filtered mask", value="FILTERED", variable=MASK_MODE, command=update_image).pack(side="left", padx=6)
ttk.Label(tog_row, textvariable=metrics_var, width=48).pack(side="right")

# Buttons
btn_row = ttk.Frame(root); btn_row.pack(pady=8)
ttk.Button(btn_row, text="Print flat YAML to console", command=print_flat_yaml).pack(side="left", padx=6)

# Keybinds
root.bind("<Left>", on_key_adjust)
root.bind("<Right>", on_key_adjust)
root.bind("<Return>", prompt_value_for_active)  # Enter edits active slider
root.bind(",", on_video_keys)
root.bind(".", on_video_keys)
root.bind("j", on_video_keys); root.bind("J", on_video_keys)
root.bind("k", on_video_keys); root.bind("K", on_video_keys)
root.bind("n", on_video_keys); root.bind("N", on_video_keys)
root.bind("p", on_video_keys); root.bind("P", on_video_keys)

# Initialize with default folder if available
if DEFAULT_VIDEO_DIR.exists():
    video_files = list_videos_in(DEFAULT_VIDEO_DIR)
    if video_files:
        open_video(0)
        seek_slider.set(0)

def on_tick():
    # Keep the frame label in sync with the seek bar even if user drags fast
    if frame_count > 0:
        frame_label_var.set(f"{int(seek_slider.get())} / {frame_count-1}")
    root.after(100, on_tick)

root.after(100, on_tick)

root.protocol("WM_DELETE_WINDOW", lambda: (close_cap(), root.destroy()))
root.mainloop()
close_cap()
cv2.destroyAllWindows()
