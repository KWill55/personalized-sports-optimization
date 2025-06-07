
# this script is meant to automatically trim videos 
# using release point as start frame 
# and vertical descent finishing as end frame
# but need to fix this script 


import os
import cv2
import subprocess
import numpy as np
from pathlib import Path
import mediapipe as mp

# ========================================
# Script Configuration
# ========================================

session = "freethrow_tests"
angle = "angled" # angle of the camera view (angled, rear, side)

LOWER_ORANGE = np.array([5, 100, 100])
UPPER_ORANGE = np.array([20, 255, 255])
SECONDS_TO_SKIP = 15  # for ball detection only

# ========================================
# Paths 
# ========================================

script_dir = Path(__file__).resolve().parent
session_dir = script_dir.parents[2] / "data" / session

input_dir = session_dir / angle / "hevc"
raw_dir = session_dir / angle / "raw"
trimmed_dir = session_dir / angle / "trimmed"

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(trimmed_dir, exist_ok=True)

# ========================================
# MediaPipe Setup
# ========================================

mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# ========================================
# Functions
# ========================================

def convert_hevc_to_mp4(input_path, output_path):
    subprocess.run([
        "ffmpeg",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ])

def detect_elbow_lift_start(frames):
    right_elbow_ys = []

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb)

        if results.pose_landmarks:
            elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_elbow_ys.append((i, elbow.y))
        else:
            right_elbow_ys.append((i, None))

    ys = [y for i, y in right_elbow_ys if y is not None]
    if len(ys) < 5:
        return 0  # fallback

    dy = np.diff(ys)
    for i in range(5, len(dy)):
        if dy[i-3] < -0.01 and dy[i-2] < -0.01 and dy[i-1] < -0.01:
            return right_elbow_ys[i][0]

    return 0  # fallback

def detect_ball_trajectory(video_path, skip_initial_sec=1.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(skip_initial_sec * fps)

    frames = []
    positions = []

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if current_frame < skip_frames:
            positions.append((current_frame, None))
            current_frame += 1
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest)
            if radius > 5:
                positions.append((current_frame, y))
            else:
                positions.append((current_frame, None))
        else:
            positions.append((current_frame, None))

        current_frame += 1

    cap.release()
    return frames, positions, fps

def find_release_and_end_frame(positions, fps):
    y_positions = np.array([p[1] if p[1] is not None else np.nan for p in positions])
    smoothed = np.nan_to_num(y_positions)
    dy = np.diff(smoothed)

    release_frame = None
    for i in range(1, len(dy)):
        if dy[i-1] < 0 and dy[i] > 0:
            release_frame = i
            break

    if release_frame is None:
        release_frame = 10  # fallback

    # Step 1: Wait until the ball starts descending after release
    has_started_descending = False
    end_frame = release_frame

    for i in range(release_frame + 1, len(dy)):
        if not has_started_descending:
            if dy[i] > 0.5:  # ball is descending
                has_started_descending = True
        else:
            # Step 2: Ball is done descending (starts leveling off or going up)
            if dy[i] < -0.1:  # ball going back up or bouncing slightly
                end_frame = i
                break
            else:
                end_frame = i  # keep updating as long as it's descending or steady

        
    return release_frame, end_frame

def save_trimmed_video(frames, start_frame, end_frame, output_path, fps):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i in range(start_frame, min(end_frame, len(frames))):
        out.write(frames[i])
    out.release()

# ========================================
# Main Processing Loop
# ========================================

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".hevc") or filename.lower().endswith(".mov"):
        base_name = os.path.splitext(filename)[0]
        hevc_path = os.path.join(input_dir, filename)
        mp4_path = os.path.join(raw_dir, base_name + ".mp4")

        print(f"\nConverting {filename} to .mp4...")
        convert_hevc_to_mp4(hevc_path, mp4_path)

        print(f"Processing {base_name}.mp4...")
        frames, positions, fps = detect_ball_trajectory(mp4_path, skip_initial_sec=SECONDS_TO_SKIP)
        release_frame, end_frame = find_release_and_end_frame(positions, fps)
        # elbow_lift_frame = detect_elbow_lift_start(frames)
        # start_frame = max(0, elbow_lift_frame)
        start_frame = release_frame


        # print(f"Detected elbow lift frame: {elbow_lift_frame}, release_frame: {release_frame}, end_frame: {end_frame}")
        print(f"Release frame: {release_frame}, End frame: {end_frame}")
        print(f"Trim duration: {(end_frame - start_frame)/fps:.2f} seconds")

        trimmed_path = os.path.join(trimmed_dir, base_name + "_trimmed.mp4")
        save_trimmed_video(frames, start_frame, end_frame, trimmed_path, fps)
        print(f"Saved trimmed video: {trimmed_path}")
