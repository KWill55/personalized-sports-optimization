import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# MediaPipe Pose landmark connections (simplified for clarity)
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23), (12, 24),  # torso
    (0, 11), (0, 12),    # head to shoulders
]

# Load 3D keypoint CSV
csv_path = "../../data/freethrows1/02_process_data/triangulated/freethrow2_sync_3d.csv"
df = pd.read_csv(csv_path)

# Get landmark names (exclude frame)
landmark_names = [col[:-2] for col in df.columns if col.endswith("_x")]
num_frames = len(df)
num_landmarks = len(landmark_names)


# Quick sanity checks
# print(df.head(2))  # Make sure the CSV loaded
# print(df.columns[:10])  # Are the column names correct (e.g., 'nose_x', etc.)?
# print(df.iloc[0, 1:10])  # Check actual 3D values


# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_frame(i):
    ax.clear()
    ax.set_title(f"Frame {i}")

    xs, ys, zs = [], [], []

    for name in landmark_names:
        x = df.at[i, f"{name}_x"]
        y = df.at[i, f"{name}_y"]
        z = df.at[i, f"{name}_z"]
        if -1 in (x, y, z):  # skip invalid
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
        else:
            xs.append(x)
            ys.append(y)
            zs.append(z)

    center_x = np.nanmean(xs)
    center_y = np.nanmean(ys)
    center_z = np.nanmean(zs)
    zoom = 2  # try 1–5 to experiment with tightness

    ax.set_xlim(center_x - zoom, center_x + zoom)
    ax.set_ylim(center_y - zoom, center_y + zoom)
    ax.set_zlim(center_z - zoom, center_z + zoom)

    important = {'left_wrist', 'right_wrist', 'left_elbow', 'right_elbow', 'left_ankle', 'right_ankle'}
    for j, name in enumerate(landmark_names):
        if name in important and not np.isnan(xs[j]):
            ax.text(xs[j], ys[j] + 0.05, zs[j], name, fontsize=6, color='black')

    # Draw points
    ax.scatter(xs, ys, zs, c='red', s=20)

    # Draw skeleton lines
    for a, b in POSE_CONNECTIONS:
        if not np.isnan(xs[a]) and not np.isnan(xs[b]):
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]], c='blue')


# === Animation control flag ===
start_animation = False

def on_start_clicked(event):
    global start_animation
    start_animation = True

# === Add a Start button ===
button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
start_button = Button(button_ax, 'Start')
start_button.on_clicked(on_start_clicked)

# === Initial dummy frame ===
update_frame(0)
plt.pause(0.01)

# === Wait for button press ===
while not start_animation:
    plt.pause(0.1)

# === Animation loop ===
while True:
    for i in range(num_frames):
        update_frame(i)
        plt.pause(0.03)

