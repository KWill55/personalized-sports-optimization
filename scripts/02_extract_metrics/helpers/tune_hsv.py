import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# -------------------------------
# Config: Load Frame from Project
# -------------------------------
SESSION = "freethrow_tests"
VIDEO_FILE = "freethrow028_trimmed.mp4"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]  # go up to project root
VIDEO_PATH = BASE_DIR / "data" / SESSION / "angled" / "trimmed" / VIDEO_FILE

cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ Could not load frame from: {VIDEO_PATH}")
    exit()

frame_resized = cv2.resize(frame, (960, 540))
hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

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
    lower = np.array([h_min.get(), s_min.get(), v_min.get()])
    upper = np.array([h_max.get(), s_max.get(), v_max.get()])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame_resized.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(annotated, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    cv2.imshow("Annotated HSV", annotated)

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
        variable=var
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
    print("\n📋 Current HSV Range:")
    print(f"LOWER = np.array([{h_min.get()}, {s_min.get()}, {v_min.get()}])")
    print(f"UPPER = np.array([{h_max.get()}, {s_max.get()}, {v_max.get()}])")

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

update_labels()
update_image()
cv2.waitKey(1)
root.mainloop()
cv2.destroyAllWindows()
