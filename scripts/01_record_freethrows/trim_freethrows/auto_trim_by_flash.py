

import os
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ========================================
# Configuration Constants
# ========================================

ATHLETE = "Kenny"
SESSION = "session_001"

VIDEO_EXTENSIONS = ['.mp4', '.mov', '.hevc']  # Supported formats
RESIZE_DIMENSIONS = (640, 480)     # Display size in GUI

# ========================================
# Paths and Directories
# ========================================

base_dir = Path(__file__).resolve().parents[3] # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION

left_calib_dir = session_dir / "calibration" / "calib_images" / "left"
right_calib_dir = session_dir / "calibration" / "calib_images" / "right"

# raw video directories 
raw_left_dir = session_dir / "videos" / "player_tracking" / "raw" / "left"
raw_right_dir = session_dir / "videos" / "player_tracking" / "raw" / "right"
raw_ball_dir = session_dir / "videos" / "ball_tracking" / "raw"

# processed video directories
processed_left_dir = session_dir / "videos" / "player_tracking" / "processed" / "left"
processed_right_dir = session_dir / "videos" / "player_tracking" / "processed" / "right"
processed_ball_dir = session_dir / "videos" / "ball_tracking" / "processed"