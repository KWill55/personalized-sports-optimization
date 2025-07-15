"""
Title: Visualize 2D Keypoints on Player Tracking Videos

Description:
    This script visualizes 2D keypoints on synchronized player tracking videos using previously
    extracted CSV keypoint data. It draws skeletal landmarks and saves the annotated stitched video.

Inputs:
    - Original synchronized video (left + right views combined, 1280x640)
    - Two CSV files containing 2D keypoints for left and right views

Usage: 
    - Running the script produces the visualizations 

Outputs:
    - Annotated video with drawn skeletons for both views side by side
"""

import cv2
import pandas as pd
from pathlib import Path

# ========================================
# Configuration
# ========================================
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12),  # Shoulders
    (23, 24),  # Hips
    (11, 23), (12, 24),  # Torso
    (23, 25), (25, 27), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 32)   # Right leg
]

COLORS = {
    "skeleton": (255, 0, 0),  # Blue Lines
    "joint": (0, 0, 255)      # Red points
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# ========================================
# Paths and Parameters
# ========================================

base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / "kenny" / "session_test"

videos_dir = session_dir / "videos/player_tracking/synchronized"
keypoints_dir = session_dir / "metrics/2d_keypoints"
output_dir = session_dir / "videos/player_tracking/2d"
output_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Keypoint Visualizer Class
# ========================================
class KeypointVisualizer:
    def __init__(self, pose_connections=POSE_CONNECTIONS, colors=COLORS, point_radius=4, line_thickness=2):
        self.pose_connections = pose_connections
        self.colors = colors
        self.point_radius = point_radius
        self.line_thickness = line_thickness

    def load_keypoints(self, csv_path):
        return pd.read_csv(csv_path)

    def draw_skeleton(self, frame, keypoints_row):
        h, w, _ = frame.shape
        points = []

        for name in LANDMARK_NAMES:
            x = keypoints_row[f"{name}_x"]
            y = keypoints_row[f"{name}_y"]
            if x != -1 and y != -1:
                points.append((int(x * w), int(y * h)))
            else:
                points.append(None)

        # Draw lines
        for start, end in self.pose_connections:
            if points[start] and points[end]:
                cv2.line(frame, points[start], points[end], self.colors["skeleton"], self.line_thickness)

        # Draw points
        for p in points:
            if p:
                cv2.circle(frame, p, self.point_radius, self.colors["joint"], -1)

        return frame

    def visualize_from_csv(self, video_path, left_csv, right_csv, output_path):
        cap = cv2.VideoCapture(str(video_path))
        left_df = self.load_keypoints(left_csv)
        right_df = self.load_keypoints(right_csv)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Split frame into left and right
            mid = width // 2
            left_frame = frame[:, :mid].copy()
            right_frame = frame[:, mid:].copy()

            if frame_idx < len(left_df):
                left_frame = self.draw_skeleton(left_frame, left_df.iloc[frame_idx])
            if frame_idx < len(right_df):
                right_frame = self.draw_skeleton(right_frame, right_df.iloc[frame_idx])

            stitched_frame = cv2.hconcat([left_frame, right_frame])
            out.write(stitched_frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"✅ Saved annotated video to {output_path}")


# ========================================
# Batch Processing
# ========================================
if __name__ == "__main__":
    visualizer = KeypointVisualizer()

    for video_path in sorted(videos_dir.glob("*.avi")):
        stem = video_path.stem  # e.g., freethrow1
        left_csv = keypoints_dir / f"{stem}_left.csv"
        right_csv = keypoints_dir / f"{stem}_right.csv"
        output_path = output_dir / f"{stem}_2d.avi"

        if left_csv.exists() and right_csv.exists():
            print(f"Processing {video_path.name}...")
            visualizer.visualize_from_csv(video_path, left_csv, right_csv, output_path)
        else:
            print(f"❌ Missing CSV files for {stem}. Skipping.")