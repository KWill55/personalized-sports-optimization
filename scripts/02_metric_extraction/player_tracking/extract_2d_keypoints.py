"""
Title: extract_2d_keypoints.py

Description:
    This module's purpose is to extract 2D keypoints from player tracking videos using MediaPipe Pose.
    It processes synchronized player tracking videos, extracts keypoints, and saves them to CSV files.

Inputs:
    - synchronized player tracking videos (1280x640, split into left and right halves)

Usage:
    - Running the script processes the videos, extracts keypoints, and saves them to CSV files

Outputs:
    - CSV files containing 2D keypoints for each frame
"""

import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# ========================================
# Configuration Parameters
# ========================================
ATHLETE = "kenny"
SESSION = "session_test"

# ========================================
# Paths and Directories
# ========================================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

videos_dir = session_dir / "videos"
metrics_dir = session_dir / "metrics"

input_video_dir = videos_dir / "player_tracking" / "synchronized"
output_keypoints_dir = metrics_dir / "2d_keypoints"

# Ensure output directory exists
output_keypoints_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Classes
# ========================================
class VideoProcessor:
    """Handles video reading and splitting into left and right frames."""
    
    # VideoProcessor initialization:
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


class PoseExtractor:
    """Uses MediaPipe Pose to extract keypoints from frames."""
    
    # PoseExtractor initialization:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def extract(self, frames):
        """
        Extract 2D pose keypoints from a list of video frames.

        Args:
            frames (list): A list of frames (BGR format) from a video.

        Returns:
            list: A list of keypoints per frame, where each frame is represented as
                [x1, y1, v1, x2, y2, v2, ..., x33, y33, v33].
                If no keypoints detected in a frame, returns [-1]*99.
        """
        keypoints = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if results.pose_landmarks:
                pts = [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.visibility)]
            else:
                pts = [-1] * (33 * 3)
            keypoints.append(pts)
        return keypoints


class KeypointSaver:
    """Saves extracted keypoints to CSV with appropriate headers."""
    
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
    """List[str]: The 33 human body landmarks used by Google MediaPipe Pose."""

    @staticmethod
    def save_csv(keypoints, output_path):
        columns = ['frame'] + [f'{name}_{axis}' for name in KeypointSaver.LANDMARK_NAMES for axis in ('x', 'y', 'v')]
        df = pd.DataFrame([[i] + kp for i, kp in enumerate(keypoints)], columns=columns)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved keypoints: {output_path}")

# ========================================
# Main Pipeline
# ========================================
if __name__ == "__main__":
    for video_path in sorted(input_video_dir.glob("*.avi")):
        print(f"Processing {video_path.name}...")

        # Load and split video frames
        processor = VideoProcessor(video_path)
        left_frames, right_frames = processor.split_frames()

        # Extract keypoints for both views
        extractor = PoseExtractor()
        left_kps = extractor.extract(left_frames)
        right_kps = extractor.extract(right_frames)

        # Save to CSV
        left_csv = output_keypoints_dir / f"{video_path.stem}_left.csv"
        right_csv = output_keypoints_dir / f"{video_path.stem}_right.csv"
        KeypointSaver.save_csv(left_kps, left_csv)
        KeypointSaver.save_csv(right_kps, right_csv)
