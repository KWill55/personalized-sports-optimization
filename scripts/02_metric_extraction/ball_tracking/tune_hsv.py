import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import yaml

# -------------------------------
# Config: Load Frame from Project
# -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]

# --- Load athlete/session from project_config.yaml ---
CONFIG_PATH = BASE_DIR / "project_config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = str(cfg["athlete"])
SESSION = str(cfg["session"])
FRAME_WIDTH = cfg["original_frame_width"] # iphone records 720p (1280x720)
FRAME_HEIGHT = cfg["original_frame_height"]

# Build session_dir AFTER loading athlete/session
SESSION_DIR = BASE_DIR / "data" / ATHLETE / SESSION

# Where to look for videos
VIDEO_DIR = SESSION_DIR / "videos" / "ball_tracking" / "raw"

# Find first .avi or .mp4 in VIDEO_DIR
video_files = sorted([*VIDEO_DIR.glob("*.mp4"), *VIDEO_DIR.glob("*.avi")])
if not video_files:
    raise FileNotFoundError(f"No .mp4 or .avi video found in {VIDEO_DIR}")
VIDEO_PATH = video_files[0]

# ===============================================
# Video I/O (keep open) + Seeking/Scrubbing state
# ===============================================
### [ADDED] keep capture open so we can seek to arbitrary frames
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open: {VIDEO_PATH}")

### [ADDED] basic metadata for scrubber/labels/hotkeys
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

### [ADDED] mutable holders we can edit inside callbacks
cur_idx = [0]  # current frame index
frame_resized = None  # will be set by load_frame
hsv = None            # will be set by load_frame

### [ADDED] central loader that seeks and updates globals
def load_frame(idx: int) -> bool:
    """Seek to frame idx and refresh frame_resized/hsv globals."""
    idx = max(0, min(idx, frame_count - 1))
    cur_idx[0] = idx

    ok = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        # fallback: rewind and advance
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(idx + 1):
            ret, frame = cap.read()
            if not ret:
                break
    if not ret:
        print(f"❌ Could not load frame {idx} from: {VIDEO_PATH}")
        return False

    global frame_resized, hsv
    frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    return True
# ===============================================

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("HSV Tuner (Click + Arrows)")

# HSV values (after root)
h_min, h_max = tk.IntVar(value=5), tk.IntVar(value=30)
s_min, s_max = tk.IntVar(value=100), tk.IntVar(value=255)
v_min, v_max = tk.IntVar(value=100), tk.IntVar(value=255)

# Use name strings as keys instead of IntVar objects
label_vars = {}         # e.g., "H Min" → StringVar
slider_vars = {}        # e.g., "H Min" → IntVar
active_name = [None]    # Name of the currently focused slider

# -------------------------------
# Image Display Logic
# -------------------------------
def update_image():
    lower = np.array([h_min.get(), s_min.get(), v_min.get()], dtype=np.uint8)
    upper = np.array([h_max.get(), s_max.get(), v_max.get()], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame_resized.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(annotated, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    ### [ADDED] HUD showing frame index and time
    t = cur_idx[0] / max(fps, 1.0)
    cv2.putText(annotated,
                f"Frame {cur_idx[0]} / {frame_count-1}  ({t:.2f}s)",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    if 'frame_label_var' in globals():
        frame_label_var.set(f"{cur_idx[0]} / {frame_count-1}  |  {t:.2f}s")

    cv2.imshow("Annotated HSV", annotated)

# -------------------------------
# Slider Callbacks
# -------------------------------
def on_slider(_=None):
    update_labels()
    update_image()

# -------------------------------
# Arrow Key Adjustments
# -------------------------------
def on_key(event):
    if not active_name[0]:
        return
    name = active_name[0]
    var = slider_vars[name]
    max_val = 179 if "H" in name else 255

    if event.keysym == 'Left':
        var.set(max(var.get() - 1, 0))
    elif event.keysym == 'Right':
        var.set(min(var.get() + 1, max_val))

    update_labels()
    update_image()

# -------------------------------
# Build GUI Sliders
# -------------------------------
def build_slider(name, var):
    slider_vars[name] = var
    label_var = tk.StringVar()
    label_vars[name] = label_var

    frame = ttk.Frame(root)
    frame.pack(fill="x", pady=2)

    label = ttk.Label(frame, textvariable=label_var, width=20)
    label.pack(side="left")

    slider = ttk.Scale(
        frame,
        from_=0,
        to=179 if "H" in name else 255,
        orient="horizontal",
        variable=var,
        command=on_slider
    )
    slider.pack(side="left", expand=True, fill="x", padx=5)

    label.bind("<Button-1>", lambda e: active_name.__setitem__(0, name))
    label_var.set(f"{name}: {var.get()}")

def update_labels():
    for name, var in slider_vars.items():
        label_vars[name].set(f"{name}: {var.get()}")

# -------------------------------
# Print HSV Button
# -------------------------------
def save_hsv():
    lower = [h_min.get(), s_min.get(), v_min.get()]
    upper = [h_max.get(), s_max.get(), v_max.get()]
    txt = f"lower: {lower}\nupper: {upper}"
    print(txt)
    try:
        root.clipboard_clear()
        root.clipboard_append(txt)
    except Exception:
        pass

# ===============================================
# Frame Scrubber UI + Hotkeys
# ===============================================
### [ADDED] seek callback (slider -> frame)
def on_seek(val):
    idx = int(float(val))
    if load_frame(idx):
        update_image()

### [ADDED] build a small seek bar with a label
seek_frame = ttk.Frame(root)
seek_frame.pack(fill="x", pady=6)
ttk.Label(seek_frame, text="Frame").pack(side="left")

seek_slider = ttk.Scale(
    seek_frame,
    from_=0,
    to=max(0, frame_count - 1),
    orient="horizontal",
    command=on_seek
)
seek_slider.pack(side="left", expand=True, fill="x", padx=6)

frame_label_var = tk.StringVar(value=f"0 / {frame_count-1} | 0.00s")
ttk.Label(seek_frame, textvariable=frame_label_var, width=20).pack(side="right")

### [ADDED] stepping functions + keybinds (comma/period for ±1, j/k for ~1s)
STEP_SMALL = 1
STEP_BIG = int(fps) if fps >= 1 else 10

def step_frames(delta: int):
    new_idx = max(0, min(cur_idx[0] + delta, frame_count - 1))
    if load_frame(new_idx):
        seek_slider.set(new_idx)  # keep slider in sync
        update_image()

def on_video_keys(event):
    if event.keysym in ("comma",):       step_frames(-STEP_SMALL)
    elif event.keysym in ("period",):    step_frames(+STEP_SMALL)
    elif event.keysym in ("j", "J"):     step_frames(-STEP_BIG)
    elif event.keysym in ("k", "K"):     step_frames(+STEP_BIG)

root.bind(",", on_video_keys)
root.bind(".", on_video_keys)
root.bind("j", on_video_keys)
root.bind("k", on_video_keys)
root.bind("J", on_video_keys)
root.bind("K", on_video_keys)
# ===============================================

# -------------------------------
# Build Sliders + Run
# -------------------------------
build_slider("H Min", h_min)
build_slider("H Max", h_max)
build_slider("S Min", s_min)
build_slider("S Max", s_max)
build_slider("V Min", v_min)
build_slider("V Max", v_max)

ttk.Button(root, text="Print HSV to Console", command=save_hsv).pack(pady=10)

root.bind("<Left>", on_key)
root.bind("<Right>", on_key)

### [CHANGED] we no longer grab an initial frame at the top.
###           Instead, load the first frame via our loader so the scrubber stays consistent.
if not load_frame(0):
    cap.release()
    print("Exiting due to load failure.")
    raise SystemExit(1)

update_labels()
update_image()
cv2.waitKey(1)
root.mainloop()

### [ADDED] clean up capture on exit
cap.release()
cv2.destroyAllWindows()
