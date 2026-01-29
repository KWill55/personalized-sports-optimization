"""
Title: extract_2d_keypoints.py

Description:
    Extracts 2D keypoints using MediaPipe Pose.
    Adds visibility filtering, Hampel outlier removal, and Butterworth smoothing.
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
import yaml

# ========================================
# Config
# ========================================

config_path = Path(__file__).resolve().parents[4] / "project_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

ATHLETE = config["athlete"]
SESSION = config["session"]

# ========================================
# Paths
# ========================================
# Anchor at project root so we look under data/, not src/data.
base_dir = Path(__file__).resolve().parents[4]
session_dir = base_dir / "data" / ATHLETE / SESSION
videos_dir = session_dir / "videos"
metrics_dir = session_dir / "metrics"

input_video_dir = videos_dir / "player_tracking" / "synchronized"
output_keypoints_dir = metrics_dir / "2d_keypoints"
output_keypoints_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Signal Cleaning Helpers
# ========================================

def hampel_filter(series, window_size=5, n_sigmas=3):
    """Remove spikes with Hampel filter"""
    s = pd.Series(series)
    rolling_median = s.rolling(window=window_size, center=True).median()
    diff = np.abs(s - rolling_median)
    mad = 1.4826 * diff.rolling(window=window_size, center=True).median()
    outliers = diff > (n_sigmas * mad)
    s[outliers] = np.nan
    return s.interpolate(limit_direction="both")

def butterworth_smooth(series, cutoff=0.1, order=2):
    """Apply a low-pass Butterworth filter to smooth the signal"""
    s = pd.Series(series).interpolate(limit_direction="both").bfill().ffill()

    b, a = butter(order, cutoff)
    return pd.Series(filtfilt(b, a, s))

def clean_keypoint_series(x, y, v, vis_thresh=0.6):
    """Clean a single keypoint’s (x, y) trajectory given visibility values."""
    x = np.where(v < vis_thresh, np.nan, x)
    y = np.where(v < vis_thresh, np.nan, y)
    # Apply filters
    x_clean = butterworth_smooth(hampel_filter(x))
    y_clean = butterworth_smooth(hampel_filter(y))
    return x_clean, y_clean, v

# ========================================
# Video Processing
# ========================================
class VideoProcessor:
    """Handles video reading and splitting into left and right frames."""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.frames = self._read_frames()
        self.cap.release()

    def _read_frames(self):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def split_frames(self):
        left, right = [], []
        for frame in self.frames:
            h, w, _ = frame.shape
            mid = w // 2
            left.append(frame[:, :mid])
            right.append(frame[:, mid:])
        return left, right


# ========================================
# Pose Extraction
# ========================================
class PoseExtractor:
    """Uses MediaPipe Pose to extract and clean keypoints."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def extract(self, frames):
        keypoints = []
        all_landmarks = []

        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                pts = []
                for lm in results.pose_landmarks.landmark:
                    x = lm.x * w
                    y = lm.y * h
                    pts.extend([x, y, lm.visibility])
            else:
                pts = [-1] * (33 * 3)

            all_landmarks.append(pts)

        # Convert to DataFrame
        columns = [f"{name}_{axis}" for name in KeypointSaver.LANDMARK_NAMES for axis in ("x", "y", "v")]
        df = pd.DataFrame(all_landmarks, columns=columns)

        # === Clean each keypoint series ===
        for name in KeypointSaver.LANDMARK_NAMES:
            x, y, v = df[f"{name}_x"].values, df[f"{name}_y"].values, df[f"{name}_v"].values
            x_clean, y_clean, v = clean_keypoint_series(x, y, v)
            df[f"{name}_x"], df[f"{name}_y"], df[f"{name}_v"] = x_clean, y_clean, v

        return df


# ========================================
# Saving
# ========================================
class KeypointSaver:
    """Saves extracted keypoints to CSV with headers."""

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

    @staticmethod
    def save_csv(df, output_path):
        df.insert(0, "frame", range(len(df)))
        df.to_csv(output_path, index=False)
        print(f"✅ Saved keypoints: {output_path.name}")


# ========================================
# Main
# ========================================
if __name__ == "__main__":
    print("Begin Processing")
    for video_path in sorted(input_video_dir.glob("*.avi")):
        print(f"Processing {video_path.name}...")

        processor = VideoProcessor(video_path)
        left_frames, right_frames = processor.split_frames()

        extractor = PoseExtractor()
        left_df = extractor.extract(left_frames)
        right_df = extractor.extract(right_frames)

        left_csv = output_keypoints_dir / f"{video_path.stem}_left_2d.csv"
        right_csv = output_keypoints_dir / f"{video_path.stem}_right_2d.csv"

        KeypointSaver.save_csv(left_df, left_csv)
        KeypointSaver.save_csv(right_df, right_csv)
