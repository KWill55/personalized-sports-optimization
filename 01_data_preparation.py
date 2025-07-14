"""
run_phase1.py

Purpose:
    Run Phase 1 of the free throw analysis pipeline by executing the following scripts in order:

    tune intrinsics for each camera through... 
    1. Capture checkerboard pairs
    2. Stereo calibration
    3. Record free throws
    4. Trim videos by flash
    5. Downsample videos
    6. Synchronize videos

Control which scripts run using the flags in the CONFIG section.
"""

import subprocess


# ========================================
# CONFIGURATION PARAMTERS
# ========================================

RUN_TUNE_INTRINSICS = True 
RUN_CAPTURE_CB_PAIRS = True
RUN_CALIBRATE_STEREO = True
RUN_RECORD_FREETHROWS = True
RUN_TRIM_BY_FLASH = True
RUN_DOWNSAMPLE_VIDEOS = True
RUN_SYNCHRONIZE_VIDEOS = True

# Optional: Paths to scripts (customize these if your paths differ)
SCRIPT_PATHS = {
    "tune_intrinsics":       "01_data_preparation/helpers/tune_intrinsics.py",
    "capture_cb_pairs":      "01_data_preparation/player_calibration/capture_cb_pairs.py",
    "calibrate_stereo":      "01_data_preparation/player_calibration/calibrate_stereo.py",
    "record_freethrows":     "01_data_preparation/record_freethrows/record_freethrows.py",
    "trim_by_flash":         "01_data_preparation/video_preprocessing/trim_by_flash.py",
    "downsample_videos":     "01_data_preparation/video_preprocessing/downsample_videos.py",
    "synchronize_videos":    "01_data_preparation/video_preprocessing/synchronize_videos.py"
}

# ========================================
# SCRIPTS RUNNER FUNCTION
# ========================================

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


# ========================================
# PHASE 1 RUNNER
# ========================================

if __name__ == "__main__":

    # Use identify_cameras.py to find the correct camera indices

    if RUN_TUNE_INTRINSICS: 
        run_script("Tune Intrinsics", SCRIPT_PATHS("tune_intrinsics"))
    
    if RUN_CAPTURE_CB_PAIRS:
        run_script("Capture Checkerboard Pairs", SCRIPT_PATHS["capture_cb_pairs"])

    if RUN_CALIBRATE_STEREO:
        run_script("Calibrate Stereo", SCRIPT_PATHS["calibrate_stereo"])

    # Use inspect_npz.py to verify calibration results 

    if RUN_RECORD_FREETHROWS:
        run_script("Record Free Throws", SCRIPT_PATHS["record_freethrows"])

    # Verify FPS on recorded videos using detect_video_fps.py

    if RUN_TRIM_BY_FLASH:
        run_script("Trim Videos by Flash", SCRIPT_PATHS["trim_by_flash"])

    if RUN_DOWNSAMPLE_VIDEOS:
        run_script("Downsample Videos", SCRIPT_PATHS["downsample_videos"])

    if RUN_SYNCHRONIZE_VIDEOS:
        run_script("Synchronize Videos", SCRIPT_PATHS["synchronize_videos"])

    print("\n🎉 Phase 1 completed successfully!")
