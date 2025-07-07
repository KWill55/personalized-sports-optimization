"""
run_phase2.py

Purpose:
    Run Phase 2 of the free throw analysis pipeline to extract motion and ball metrics,
    producing summarized CSV data ready for analysis and modeling.

Steps:
    1. Split video phases
    2. Run ball tracking and extract metrics (arc, spin, contact, velocity)
    3. Detect 2D and 3D player keypoints
    4. [Optional] Visualize player tracking results
    5. Combine shot metrics into summary CSV
    6. Split summary into features and labels

Control which steps run using the flags in the CONFIG section.
"""

#TODO: maybe add some of the helper scripts like HSV tuning, hoop region creation, etc. to this script

import subprocess

# ======================
# CONFIG
# ======================

RUN_SPLIT_PHASES = True
RUN_BALL_TRACKING = True
RUN_PLAYER_TRACKING = True
RUN_VISUALIZATIONS = False
RUN_COMBINE_SUMMARIES = True
RUN_SPLIT_FEATURES_LABELS = True  

# ======================
# PATHS
# ======================

SCRIPT_PATHS = {
    "split_phases":              "02_metric_extraction/split_phases/split_phases.py",
    "ball_tracking":             "02_metric_extraction/ball_tracking/detect_makes.py",
    "detect_2d_keypoints":       "02_metric_extraction/player_tracking/detect_2d_keypoints.py",
    "detect_3d_keypoints":       "02_metric_extraction/player_tracking/detect_3d_keypoints.py",
    "visualize_2d_keypoints":    "02_metric_extraction/player_tracking/visualize_2d_keypoints.py",
    "visualize_3d_keypoints":    "02_metric_extraction/player_tracking/visualize_3d_keypoints.py",
    "combine_summaries":         "02_metric_extraction/summary_builder/combine_release_summaries.py",
    "split_features_labels":     "02_metric_extraction/summary_builder/split_features_labels.py",  
}

# ======================
# RUNNER FUNCTION
# ======================

def run_script(name, path):
    print(f"\n==============================")
    print(f"🔄 Running: {name}")
    print(f"==============================")
    try:
        subprocess.run(["python", path], check=True)
        print(f"✅ Completed: {name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {name} - {e}")
        exit(1)

# ======================
# PHASE 2 RUNNER
# ======================

if __name__ == "__main__":
    if RUN_SPLIT_PHASES:
        run_script("Split Video Phases", SCRIPT_PATHS["split_phases"])

    if RUN_BALL_TRACKING:
        run_script("Ball Tracking + Metric Extraction", SCRIPT_PATHS["ball_tracking"])

    if RUN_PLAYER_TRACKING:
        run_script("Detect 2D Keypoints", SCRIPT_PATHS["detect_2d_keypoints"])
        run_script("Detect 3D Keypoints", SCRIPT_PATHS["detect_3d_keypoints"])

    if RUN_VISUALIZATIONS:
        run_script("Visualize 2D Keypoints", SCRIPT_PATHS["visualize_2d_keypoints"])
        run_script("Visualize 3D Keypoints", SCRIPT_PATHS["visualize_3d_keypoints"])

    if RUN_COMBINE_SUMMARIES:
        run_script("Combine Summary CSVs", SCRIPT_PATHS["combine_summaries"])

    if RUN_SPLIT_FEATURES_LABELS:
        run_script("Split Features & Labels", SCRIPT_PATHS["split_features_labels"])

    print("\n🎉 Phase 2 completed successfully!")
