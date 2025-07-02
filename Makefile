# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# Targets
all: process split plot analyze

# ======================================== 
# help
# ========================================

.PHONY: help

help:
	@echo ""
	@echo "\033[1mMakefile Command Reference\033[0m"
	@echo "-------------------------------------"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?##' Makefile | \
	awk -F: '{ \
		if ($$1 ~ /-header$$/) { \
			sub(/.*## */, "", $$2); \
			printf "\n\033[1m%s\033[0m\n", $$2; \
		} else { \
			sub(/.*## */, "", $$2); \
			printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2; \
		} \
	}'
	@echo ""


# ======================================== 
# Phase 1: Recording Freethrows 
# ========================================
divider1-header: ## -------------------------------------
	@:
phase1-header: ## PHASE 1 SCRIPTS (Record Freethrows)
	@:
divider2-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories  
# ----------------------------------------
record_dir := scripts/01_record_freethrows
calibrate_dir := $(record_dir)/player_calibration
preprocessing_dir := $(record_dir)/video_preprocessing
helpers_dir := $(record_dir)/helpers

# ----------------------------------------
# Player calibration
# ----------------------------------------
calibrate-header: ## 🔧 Calibrate Camera System
	@: 

capture_cb_pairs: ## Step 1 - Capture calibration pairs
	@echo "Capturing image pairs for calibration..."
	python $(calibrate_dir)/capture_cb_pairs.py

calibrate_stereo: ## Step 2 - Stereo Calibration (int/ext)
	@echo "Calibrating stereo cameras..."
	python $(calibrate_dir)/calibrate_stereo.py

# ----------------------------------------
# Record freethrows
# ----------------------------------------
record-header: ## 🎥 Record Freethrows
	@:

record_freethrows: ## Record a freethrow session
	@echo "Recording a freethrow..."
	python $(record_dir)/record_freethrows/record_freethrows.py


# ----------------------------------------
# Preprocessing freethrow videows 
# ----------------------------------------
preprocessing-header: ## ✂️  Preprocssing Videos
	@:

trim_by_flash: ## Automatic trimming via flashes
	@echo "Trimming video clips via flash..."
	python $(preprocessing_dir)/trim_by_flash.py

downsample_videos: ## Downsample trimmed videos 
	@echo "Downsampling video clips..."
	python $(preprocessing_dir)/downsample_videos.py

synchronize_videos: ## synchronize downsampled videos
	@echo "Synchronizing video clips..."
	python $(preprocessing_dir)/synchronize_videos.py

# ----------------------------------------
# Helpers
# ----------------------------------------
helpers1-header: ## 👋 Helpers (Phase 1)
	@:

identify_cameras: ## Camera identification GUI
	@echo "Opening GUI to identify camera indices..."
	python $(helpers_dir)/identify_cameras.py

detect_video_fps: ## Detect FPS of a video
	@echo "Detecting video FPS..."
	python $(helpers_dir)/detect_video_fps.py

inspect_npz: ## Inspect .npz file contents
	@echo "Inspecting .npz file..."
	python $(helpers_dir)/inspect_npz.py

test_fps: ## Test FPS extraction
	@echo "Testing FPS script..."
	python $(helpers_dir)/test_fps.py

test_dual_cameras: ## open 2 camera streams
	@echo "Opening 2 camera streams..."
	python $(helpers_dir)/test_dual_cameras.py


# ======================================== 
# Phase 2: Extract player and ball metrics
# ========================================
divider3-header: ## -------------------------------------
	@:
phase2-header: ## PHASE 2 SCRIPTS (Prepare Data for ML)
	@:
divider4-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories  
# ----------------------------------------
extract_dir := scripts/02_extract_metrics
ball_dir := $(extract_dir)/ball_tracking
metrics_dir := $(ball_dir)/metrics
player_dir := $(extract_dir)/player_tracking
helpers2_dir := $(extract_dir)/helpers
summary_dir := $(extract_dir)/summary_builder
phases_dir := $(extract_dir)/split_phases

# ----------------------------------------
# Detect Phases
# ----------------------------------------
phases-header: ## 🗂️  Split Phases
	@:

detect_phases: ## Detect motion phases
	python $(phases_dir)/split_phases.py

# ----------------------------------------
# Player Tracking
# ----------------------------------------
player-header: ## ⛹️  Player Tracking
	@:

detect_2d_keypoints: ## Detect 2D keypoints
	python $(player_dir)/detect_2d_keypoints.py

detect_3d_keypoints: ## Triangulate 3D keypoints
	python $(player_dir)/detect_3d_keypoints.py

visualize_2d_keypoints: ## Visualize 2D keypoints
	python $(player_dir)/visualize_2d_keypoints.py

visualize_3d_keypoints: ## Visualize 3D keypoints
	python $(player_dir)/visualize_3d_keypoints.py

plot_kin_timeseries: ## Plot time-series kinematics
	python $(player_dir)/plot_kinematics_time_series.py

plot_release_kinematics: ## Plot release metrics
	python $(player_dir)/plot_release_kinematics.py

plot_vel_timeseries: ## Plot velocity profiles
	python $(player_dir)/plot_velocities_time_series.py

# ----------------------------------------
# Ball Tracking
# ----------------------------------------
ball-header: ## 🏀 Ball Tracking
	@:

detect_metrics: ## Detect ball metrics
	python $(ball_dir)/detect_metrics.py

backboard_contact: ## Detect backboard contact
	python $(metrics_dir)/backboard_contact.py

backspin: ## Estimate backspin
	python $(metrics_dir)/backspin.py

elbow_release_frame: ## Detect elbow release frame
	python $(metrics_dir)/elbow_release_frame.py

release_angle: ## Measure release angle
	python $(metrics_dir)/release_angle.py

rim_contact: ## Detect rim contact
	python $(metrics_dir)/rim_contact.py

track_ball_arc: ## Track ball arc
	python $(metrics_dir)/track_ball_arc.py

velocity: ## Calculate velocity
	python $(metrics_dir)/velocity.py

# ----------------------------------------
# Summary Builder
# ----------------------------------------
summary-header: ## 🧮 Summary Builder
	@:

combine_release_summaries: ## Combine release summaries
	python $(summary_dir)/combine_release_summaries.py

# ----------------------------------------
# Helpers
# ----------------------------------------
helpers2-header: ## 👋 Helpers (Phase 2)
	@:

create_hoop_regions: ## Create hoop detection regions
	python $(helpers2_dir)/create_hoop_regions.py

process_release: ## Process release phases
	python $(helpers2_dir)/process_release.py

tune_hsv: ## HSV color tuning
	python $(helpers2_dir)/tune_hsv.py

# ======================================== 
# Phase 3: Analyze Data
# ========================================
divider5-header: ## -------------------------------------
	@:
phase3-header: ## PHASE 3 SCRIPTS (Analyze Data)
	@:
divider6-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories 
# ----------------------------------------
analyze_dir := scripts/03_analyze_data/train_models

# ----------------------------------------
# Train Models
# ----------------------------------------
train-header: ## 📊 Train Models
	@:

analyze_release_averages: ## Analyze average release data
	python $(analyze_dir)/analyze_release_averages.py

prepare_release_features: ## Create training features
	python $(analyze_dir)/prepare_release_features.py

train_baseline_models: ## Train ML models
	python $(analyze_dir)/train_baseline_models.py

# ----------------------------------------
# Helpers (phase 3)
# ----------------------------------------
helpers3-header: ## 👋 Helpers (Phase 3)
	@:


# ======================================== 
# Tools
# ========================================
tools-header: ## 🛠  Tools
	@:

rename_freethrows: ## Rename files in bulk
	python tools/rename_freethrows.py

# ======================================== 
# clean 
# ========================================

clean:
	@echo "Cleaning output files..."