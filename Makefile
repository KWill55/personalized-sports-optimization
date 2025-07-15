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

check_cb_detection: ## Step 1 - Ensure cameras can see CB grid. 
	@echo "Opening camera feeds to ensure cb detection"
	python $(calibrate_dir)/check_cb_detection.py

capture_cb_pairs: ## Step 2 - Capture calibration pairs of CB grid.
	@echo "Capturing image pairs for calibration..."
	python $(calibrate_dir)/capture_cb_pairs.py

calibrate_stereo: ## Step 3 - Stereo Calibration (int/ext)
	@echo "Calibrating stereo cameras..."
	python $(calibrate_dir)/calibrate_stereo.py

inspect_calibration: ## Step 4 - Inspect .npz file contents
	@echo "Printing calibration file paramters to terminal..."
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

trim_freethrows: ## GUI to trim freethrows manually
	@echo "Opening GUI to trim freethrows..."
	python $(preprocessing_dir)/trim_freethrows.py

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

extract_2d_keypoints: ## Extract 2D keypoints from videos
	python $(player_dir)/extract_2d_keypoints.py

extract_3d_keypoints: ## Triangulate 3D keypoints from 2D keypoints
	python $(player_dir)/extract_3d_keypoints.py

visualize_2d_keypoints: ## Visualize 2D keypoints
	python $(player_dir)/visualize_2d_keypoints.py

visualize_3d_keypoints: ## Visualize 3D keypoints
	python $(player_dir)/visualize_3d_keypoints.py

# ----------------------------------------
# Ball Tracking
# ----------------------------------------
ball-header: ## 🏀 Ball Tracking
	@:

detect_makes: ## Detect ball metrics
	python $(ball_dir)/detect_makes.py

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
phase3-header: ## PHASE 3 SCRIPTS
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

# ======================================== 
# clean 
# ========================================

clean:
	@echo "Cleaning output files..."