import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ========================================
# Configuration Parameters
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"
CLIP_NAME = "freethrow1_sync"

# ========================================
# Paths and Directories 
# ========================================

script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parents[2]
session_dir = base_dir / ATHLETE / SESSION
csv_path = session_dir / "02_process_data" / "triangulated" / f"{CLIP_NAME}_3d.csv"

# ========================================
# MediaPipe-style POSE_CONNECTIONS
# ========================================

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

# ========================================
# Load data
# ========================================

df = pd.read_csv(csv_path)
frames = df.drop(columns=["frame"]).values.reshape(len(df), 33, 3)  # shape: (num_frames, 33, 3)

# ========================================
# Setup Plot
# ========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"3D Pose Animation - {CLIP_NAME}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=15, azim=-90)

# Set limits (tweak if needed)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

scatter = ax.scatter([], [], [], color='red', s=20)
lines = [ax.plot([], [], [], color='blue', linewidth=2)[0] for _ in POSE_CONNECTIONS]

# ========================================
# Animation function
# ========================================

def update(frame_idx):
    keypoints = frames[frame_idx]

    # Mask invalid points
    mask = ~(keypoints == -1).any(axis=1)
    visible_pts = keypoints[mask]

    if visible_pts.size == 0:
        scatter.set_offsets([])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return scatter, *lines

    x, y, z = visible_pts[:, 0], visible_pts[:, 1], visible_pts[:, 2]
    scatter._offsets3d = (x, y, z)

    for i, (a, b) in enumerate(POSE_CONNECTIONS):
        if mask[a] and mask[b]:
            xa, ya, za = keypoints[a]
            xb, yb, zb = keypoints[b]
            lines[i].set_data([xa, xb], [ya, yb])
            lines[i].set_3d_properties([za, zb])
        else:
            lines[i].set_data([], [])
            lines[i].set_3d_properties([])

    return scatter, *lines

# ========================================
# Run animation
# ========================================

anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
plt.tight_layout()
plt.show()
