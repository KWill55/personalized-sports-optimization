# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# Targets
all: process split plot analyze

.PHONY: train_phase_models_cv

# Constants
ATHLETE := $(shell python3 -c "import yaml; print(yaml.safe_load(open('project_config.yaml'))['athlete'])")
SESSION := $(shell python3 -c "import yaml; print(yaml.safe_load(open('project_config.yaml'))['session'])")

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
# Phase 1: Data Preparation 
# ========================================
divider1-header: ## -------------------------------------
	@:
phase1-header: ## PHASE 1 SCRIPTS
	@:
divider2-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories  
# ----------------------------------------
prep_dir := scripts/01_data_preparation
calibrate_dir := $(prep_dir)/player_calibration
preprocessing_dir := $(prep_dir)/video_preprocessing
helpers_dir := $(prep_dir)/helpers

# ----------------------------------------
# Player calibration
# ----------------------------------------
calibrate-header: ## 🔧 Calibrate Camera System
	@: 

generate_grid: ## Step 0 - print a calibration grid
	@echo "Generating calibration grid..."
	python $(calibrate_dir)/generate_grid.py

check_cb_detection: ## Step 1 - Ensure cameras can see CB grid. 
	@echo "Opening camera feeds to ensure cb detection"
	python $(calibrate_dir)/check_cb_detection.py

capture_cb_mono: ## Step 2 - Capture calibration of CB grid.
	@echo "Capturing images for calibration..."
	python $(calibrate_dir)/capture_cb_mono.py

capture_cb_pairs: ## Step 2 - Capture calibration pairs of CB grid.
	@echo "Capturing image pairs for calibration..."
	python $(calibrate_dir)/capture_cb_pairs.py

calibrate_stereo: ## Step 3 - Stereo Calibration (int/ext)
	@echo "Calibrating stereo cameras..."
	python $(calibrate_dir)/calibrate_stereo.py

estimate_intrinsics: ## Estimate intrinsics from calibration images
	@echo "Estimating intrinsics from calibration images..."
	python $(calibrate_dir)/estimate_intrinsics.py

estimate_extrinsics: ## Estimate extrinsics from calibration images
	@echo "Estimating extrinsics from calibration images..."
	python $(calibrate_dir)/estimate_extrinsics.py

inspect_calibration: ## Step 4 - Ensure calibration was successful
	@echo "Printing parameters to terminal and opening images..."
	python $(calibrate_dir)/inspect_calibration.py

# ----------------------------------------
# Record freethrows
# ----------------------------------------
record-header: ## 🎥 Record Freethrows
	@:

record_freethrows: ## Record a freethrow session
	@echo "Recording a freethrow..."
	python $(prep_dir)/record_freethrows/record_freethrows.py


# ----------------------------------------
# Preprocessing freethrow videows 
# ----------------------------------------
preprocessing-header: ## ✂️  Preprocssing Videos
	@:

trim_freethrows_time: ## GUI to trim freethrows manually
	@echo "Opening GUI to trim freethrows..."
	python $(preprocessing_dir)/trim_freethrows_time.py

trim_freethrows_frames: ## GUI to trim freethrows manually
	@echo "Opening GUI to trim freethrows..."
	python $(preprocessing_dir)/trim_freethrows_frames.py

combine_player_feeds: ## combine player feeds
	@echo "Combining left and right player feeds..."
	python $(preprocessing_dir)/combine_player_feeds.py

# ----------------------------------------
# Helpers
# ----------------------------------------
helpers1-header: ## 👋 Helpers (Phase 1)
	@:

identify_cameras: ## Camera identification GUI
	@echo "Opening GUI to identify camera indices..."
	python $(helpers_dir)/identify_cameras.py


# ======================================== 
# Phase 2: Extract player and ball metrics
# ========================================
divider3-header: ## -------------------------------------
	@:
phase2-header: ## PHASE 2 SCRIPTS 
	@:
divider4-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories  
# ----------------------------------------
extract_dir := scripts/02_metric_extraction
ball_dir := $(extract_dir)/ball_tracking
metrics_dir := $(ball_dir)/metrics
player_dir := $(extract_dir)/player_tracking
helpers2_dir := $(extract_dir)/helpers
summary_dir := $(extract_dir)/summary_builder

# ----------------------------------------
# Player Tracking
# ----------------------------------------
player-header: ## ⛹️  Player Tracking
	@:

extract_2d_keypoints: ## Extract 2D keypoints from videos
	python $(player_dir)/extract_2d_keypoints.py

extract_3d_keypoints: ## Triangulate 3D keypoints from 2D keypoints
	python $(player_dir)/extract_3d_keypoints.py

compute_3d_angles: ## Compute 3D angles from 3D keypoints
	python $(player_dir)/compute_3d_angles.py

compute_3d_vel_acc: ## Compute 3D velocity and acceleration from 3D keypoints
	python $(player_dir)/compute_3d_vel_acc.py

draw_2d_keypoints: ## Draw 2D keypoints onto videos
	python $(player_dir)/draw_2d_keypoints.py

split_phases: ## Detect motion phases
	python $(player_dir)/split_phases.py

# ----------------------------------------
# Ball Tracking
# ----------------------------------------
ball-header: ## 🏀 Ball Tracking
	@:

detect_makes: ## Detect ball metrics
	python $(ball_dir)/detect_makes.py

detect_makes_gui: ## Detect ball metrics
	python $(ball_dir)/detect_makes_gui.py

create_hoop_regions: ## Create hoop detection regions
	python $(ball_dir)/create_hoop_regions.py

tune_ball_detection: ## Tune HSV, circularity, and size thresholds for ball detection
	python $(ball_dir)/tune_ball_detection.py

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

combine_releases: ## Process release data into single CSV
	python $(helpers2_dir)/combine_releases.py

# ======================================== 
# Phase 3: Analyze Data
# ========================================
divider5-header: ## -------------------------------------
	@:
phase3-header: ## PHASE 3 SCRIPTS
	@:
divider6-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories 
# ----------------------------------------
feature_engineering_dir := scripts/03_analysis_and_modeling/feature_engineering
exploratory_analysis_dir := scripts/03_analysis_and_modeling/exploratory_analysis
train_models_dir := scripts/03_analysis_and_modeling/train_models
evaluation_dir := scripts/03_analysis_and_modeling/evaluation

# ----------------------------------------
# Feature Engineering 
# ----------------------------------------
feature-engineering-header: ## 🛠 Feature Engineering
	@:

extract_phase_features: ## Extract features from phases
	@echo "Extracting features from phases..."
	python $(feature_engineering_dir)/extract_phase_features.py

prepare_phase_dataset: ## Merge features and labels for analysis 
	@echo "Merging features.csv and outcomes.csv..."
	python $(feature_engineering_dir)/prepare_phase_dataset.py

# ----------------------------------------
# Exploratory Analysis 
# ----------------------------------------
exploratory-analysis-header: ## 🛠 Exploratory Analysis 
	@:

# ----------------------------------------
# Train Models
# ----------------------------------------
train-header: ## 📊 Train Models
	@:

training: ## Train ML models with CV
	@echo "Training phase models for $(ATHLETE) / $(SESSION) with cross-validation..."
	./scripts/03_analysis_and_modeling/train_models/training.sh $(ATHLETE) $(SESSION)

# ----------------------------------------
# Evaluation
# ----------------------------------------
evaluation-header: ## 🛠 Evaluation 
	@:

# ----------------------------------------
# Helpers (phase 3)
# ----------------------------------------
helpers3-header: ## 👋 Helpers (Phase 3)
	@:

# ======================================== 
# Utilities
# ========================================
divider7-header: ## -------------------------------------
	@:
util-header: ## UTILITIES
	@:
divider8-header: ## -------------------------------------
	@:

# ----------------------------------------
# Paths and Directories  
# ----------------------------------------
util_dir := utils


util1-header: ## 🛠 Utilities
	@: 

play_avi_videos: ## GUI for interacting with AVI videos in a folder
	@echo "Opening GUI to interact with AVI videos..."
	python $(util_dir)/play_avi_videos.py

3d_viewer: ## GUI for viewing 3D data
	@echo "Opening 3D viewer GUI..."
	python $(util_dir)/mp33_3d_viewer.py

# ======================================== 
# clean 
# ========================================

clean:
	@echo "Cleaning output files..."