# Python Driver 
# scripts/02_extract_metrics.py
# TODO this file and eventually make rule to run this 

# Important: This is just a demo. dont actally use this file yet

import subprocess

# === CONFIGURATION FLAGS ===
RUN_PHASE_DETECTION = True
RUN_PLAYER_TRACKING = True
RUN_BALL_TRACKING   = True
RUN_VISUALIZATION   = True
RUN_COMBINE_METRICS = True

# === PIPELINE EXECUTION ===

if RUN_PHASE_DETECTION:
    print("🔍 Detecting release phases...")
    subprocess.run(["python", "02_extract_metrics/detect_phases.py"], check=True)

if RUN_PLAYER_TRACKING:
    print("🧍 Extracting 3D keypoints...")
    subprocess.run(["python", "02_extract_metrics/player_tracking/detect_3d_keypoints.py"], check=True)

if RUN_BALL_TRACKING:
    print("🏀 Extracting ball metrics...")
    subprocess.run(["python", "02_extract_metrics/ball_tracking/detect_metrics.py"], check=True)

if RUN_VISUALIZATION:
    print("🖼 Visualizing kinematics...")
    subprocess.run(["python", "02_extract_metrics/visualize/plot_release_kinematics.py"], check=True)

if RUN_COMBINE_METRICS:
    print("📊 Combining all metrics into summary CSV...")
    subprocess.run(["python", "02_extract_metrics/finalize_data/combine_release_summaries.py"], check=True)

print("✅ Phase 2 complete.")
