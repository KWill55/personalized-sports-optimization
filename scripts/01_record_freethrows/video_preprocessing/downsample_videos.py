"""
Title: downsample_videos.py

Purpose: Downsample trimmed videos from three different camera angles so that they all have matching FPS 

"""

import cv2
import os
from pathlib import Path


# ========================================
# Constants 
# ========================================

ATHLETE = "kenny"
SESSION = "session_001"
TARGET_FPS = 30


# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

trimmed_left_dir = session_dir / "videos" / "player_tracking" / "trimmed" / "left"
trimmed_right_dir = session_dir / "videos" / "player_tracking" / "trimmed" / "right"
trimmed_ball_dir = session_dir / "videos" / "ball_tracking" / "trimmed"

downsampled_left_dir = session_dir / "videos" / "player_tracking" / "downsampled" / "left"
downsampled_right_dir = session_dir / "videos" / "player_tracking" / "downsampled" / "right"
downsampled_ball_dir = session_dir / "videos" / "ball_tracking" / "downsampled"

# Make sure output dirs exist
for dir_path in [downsampled_left_dir, downsampled_right_dir, downsampled_ball_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================================
# Function for Downsampling videos 
# ========================================

def downsample_videos(input_dir: Path, output_dir: Path, target_fps: int):
    for video_path in input_dir.glob("*.mp4"):
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps < target_fps:
            print(f"[ERROR] {video_path.name} already below target FPS ({original_fps:.2f})")
            cap.release()
            continue

        frame_interval = int(round(original_fps / target_fps))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        output_path = output_dir / video_path.name
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))

        frame_count = 0
        written_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                out.write(frame)
                written_frames += 1
            frame_count += 1

        cap.release()
        out.release()
        print(f"[DONE] {video_path.name}: {original_fps:.2f} -> {target_fps}, {written_frames} frames written")

# ========================================
# Run Downsampling on All Cameras
# ========================================

downsample_videos(trimmed_left_dir, downsampled_left_dir, TARGET_FPS)
downsample_videos(trimmed_right_dir, downsampled_right_dir, TARGET_FPS)
downsample_videos(trimmed_ball_dir, downsampled_ball_dir, TARGET_FPS)



