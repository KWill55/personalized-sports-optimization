import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# ========== CONFIG ==========

# Load 3D keypoint CSV
csv_path = "../../data/tests/player_tracking_tests/player_tracking_1/02_process_data/triangulated/freethrow1_sync_3d.csv"
df = pd.read_csv(csv_path)

# Full landmark names from columns
landmark_names = [col[:-2] for col in df.columns if col.endswith("_x")]
num_frames = len(df)

# Only keep meaningful upper & lower body joints (no feet)
landmarks_to_plot = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Map name to new index (shorter list)
name_to_idx = {name: i for i, name in enumerate(landmarks_to_plot)}

# Remap POSE_CONNECTIONS using landmark names
raw_connections = [
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('left_shoulder', 'right_shoulder'),
    ('left_hip', 'right_hip'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('nose', 'left_shoulder'),
    ('nose', 'right_shoulder'),
]
POSE_CONNECTIONS = [(name_to_idx[a], name_to_idx[b]) for a, b in raw_connections if a in name_to_idx and b in name_to_idx]

# ========== PLOT SETUP ==========

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_frame(i):
    ax.clear()
    ax.set_title(f"Frame {i}")

    xs, ys, zs = [], [], []

    # Center reference: midpoint between hips
    center_x = (df.at[i, "left_hip_x"] + df.at[i, "right_hip_x"]) / 2
    center_y = (df.at[i, "left_hip_y"] + df.at[i, "right_hip_y"]) / 2
    center_z = (df.at[i, "left_hip_z"] + df.at[i, "right_hip_z"]) / 2

    for name in landmarks_to_plot:
        x = df.at[i, f"{name}_x"]
        y = df.at[i, f"{name}_y"]
        z = df.at[i, f"{name}_z"]
        if -1 in (x, y, z):
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
        else:
            xs.append(x - center_x)
            ys.append(y - center_y)
            zs.append(z - center_z)

    zoom = 1.5
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_zlim(-zoom, zoom)

    ax.set_xticks(np.linspace(-zoom, zoom, 3))
    ax.set_yticks(np.linspace(-zoom, zoom, 3))
    ax.set_zticks(np.linspace(-zoom, zoom, 3))
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_facecolor('white')

    ax.scatter(xs, ys, zs, c='red', s=20)

    for a, b in POSE_CONNECTIONS:
        if not np.isnan(xs[a]) and not np.isnan(xs[b]):
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], c='blue')


# ========== ANIMATION CONTROLS ==========

start_animation = False
def on_start_clicked(event):
    global start_animation
    start_animation = True

button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
start_button = Button(button_ax, 'Start')
start_button.on_clicked(on_start_clicked)

update_frame(0)
plt.pause(0.01)

while not start_animation:
    plt.pause(0.1)

while True:
    for i in range(num_frames):
        update_frame(i)
        plt.pause(0.03)
