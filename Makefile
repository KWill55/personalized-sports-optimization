# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# ---------------------------
# GLOBAL CONSTANTS
# ---------------------------
SRC := src

ATHLETE := $(shell python3 -c "import yaml; print(yaml.safe_load(open('project_config.yaml'))['athlete'])")
SESSION := $(shell python3 -c "import yaml; print(yaml.safe_load(open('project_config.yaml'))['session'])")

export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.PHONY: help clean

# ======================================== 
# HELP COMMAND
# ========================================
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
# PHASE 1 — DATA PREPARATION
# ========================================
divider1-header: ## -------------------------------------
	@:
phase1-header: ## PHASE 1 SCRIPTS
	@:
divider2-header: ## -------------------------------------
	@:

# Directories
prep_dir := $(SRC)/01_data_preparation
calibrate_dir := $(prep_dir)/player_calibration
preprocessing_dir := $(prep_dir)/video_preprocessing
helpers1_dir := $(prep_dir)/helpers


# ----- Player calibration -----
calibrate-header: ## 🔧 Calibrate Camera System
	@: 

generate_grid: ## Step 0 - print calibration grid
	python $(calibrate_dir)/generate_grid.py

setup_cameras: ## Step 1 - Ensure cameras detect chessboard grid
	python $(calibrate_dir)/setup_cameras.py

capture_cb_mono: ## Step 2a - Capture mono calibration
	python $(calibrate_dir)/capture_cb_mono.py

capture_cb_pairs: ## Step 2b - Capture stereo calibration pairs
	python $(calibrate_dir)/capture_cb_pairs.py

capture_calib_images: ## Alternate capture script
	python $(calibrate_dir)/capture_calib_images.py

calibrate_stereo: ## Step 3 - Stereo Calibration
	python $(calibrate_dir)/calibrate_stereo.py

estimate_intrinsics:
	python $(calibrate_dir)/estimate_intrinsics.py

estimate_extrinsics:
	python $(calibrate_dir)/estimate_extrinsics.py

inspect_calibration:
	python $(calibrate_dir)/inspect_calibration.py


# ----- Record freethrows -----
record-header: ## 🎥 Record Freethrows
	@:

record_freethrows:
	python $(prep_dir)/record_freethrows/record_freethrows.py


# ----- Preprocessing -----
preprocessing-header: ## ✂️  Preprocessing Videos
	@:

trim_freethrows_time:
	python $(preprocessing_dir)/trim_freethrows_time.py

trim_freethrows_frames:
	python $(preprocessing_dir)/trim_freethrows_frames.py

combine_player_feeds:
	python $(preprocessing_dir)/combine_player_feeds.py


# ----- Phase 1 Helpers -----
helpers1-header: ## 👋 Helpers (Phase 1)
	@:

identify_cameras:
	python $(helpers1_dir)/identify_cameras.py



# ======================================== 
# PHASE 2 — PLAYER + BALL METRICS
# ========================================
divider3-header: ## -------------------------------------
	@:
phase2-header: ## PHASE 2 SCRIPTS
	@:
divider4-header: ## -------------------------------------
	@:

# Directories
extract_dir := $(SRC)/02_metric_extraction
ball_dir := $(extract_dir)/ball_tracking
metrics_dir := $(ball_dir)/metrics
player_dir := $(extract_dir)/player_tracking
helpers2_dir := $(extract_dir)/helpers
summary_dir := $(extract_dir)/summary_builder


# ----- Player Tracking -----
player-header: ## ⛹️  Player Tracking
	@:

extract_2d_keypoints:
	python $(player_dir)/extract_2d_keypoints.py

extract_3d_keypoints:
	python $(player_dir)/extract_3d_keypoints.py

compute_3d_angles:
	python $(player_dir)/compute_3d_angles.py

compute_3d_vel_acc:
	python $(player_dir)/compute_3d_vel_acc.py

draw_2d_keypoints:
	python $(player_dir)/draw_2d_keypoints.py

verify_stereo_accuracy:
	python $(player_dir)/verify_stereo_accuracy.py

split_phases:
	python $(player_dir)/split_phases.py

label_phases_gui:
	python -m src.utils.label_phases_gui

label_ball_phases_gui:
	python -m src.utils.label_ball_phases_gui


# ----- Ball Tracking -----
ball-header: ## 🏀 Ball Tracking
	@:

detect_makes:
	python $(ball_dir)/detect_makes.py

detect_makes_gui:
	python $(ball_dir)/detect_makes_gui.py

simple_mog2:
	python $(ball_dir)/simple_mog2.py

create_hoop_regions:
	python $(ball_dir)/create_hoop_regions.py

tune_ball_detection:
	python $(ball_dir)/tune_ball_detection.py


# ----- Summary Builder -----
summary-header: ## 🧮 Summary Builder
	@:

combine_release_summaries:
	python $(summary_dir)/combine_release_summaries.py


# ----- Phase 2 Helpers -----
helpers2-header: ## 👋 Helpers (Phase 2)
	@:

combine_releases:
	python $(helpers2_dir)/combine_releases.py



# ======================================== 
# PHASE 3 — ANALYSIS & MODELING
# ========================================
divider5-header: ## -------------------------------------
	@:
phase3-header: ## PHASE 3 SCRIPTS
	@:
divider6-header: ## -------------------------------------
	@:

# Directories
analysis_dir := $(SRC)/03_analysis_and_modeling
feature_engineering_dir := $(analysis_dir)/feature_engineering
exploratory_analysis_dir := $(analysis_dir)/exploratory_analysis
train_models_dir := $(analysis_dir)/train_models
evaluation_dir := $(analysis_dir)/evaluation


# ----- Feature Engineering -----
feature-engineering-header: ## 🛠 Feature Engineering
	@:

extract_phase_features:
	python $(feature_engineering_dir)/extract_phase_features.py

extract_phase_features_time_series:
	python $(feature_engineering_dir)/extract_phase_features_time_series.py

prepare_phase_dataset:
	python $(feature_engineering_dir)/prepare_phase_dataset.py

detect_outliers:
	python $(feature_engineering_dir)/detect_outliers.py --roots data --glob "**/merged.csv"


# ----- Train Models -----
train-header: ## 📊 Train Models
	@:

training:
	./$(train_models_dir)/training.sh $(ATHLETE) $(SESSION)


# ======================================== 
# UTILITIES
# ========================================
divider7-header: ## -------------------------------------
	@:
util-header: ## UTILITIES
	@:
divider8-header: ## -------------------------------------
	@:

util_dir := $(SRC)/utils

video_player:
	python $(util_dir)/video_player.py

project_gui:
	python $(util_dir)/project_gui.py

real_time_display:
	python $(util_dir)/real_time_display.py


# ======================================== 
# CLEAN
# ========================================
clean:
	@echo "Cleaning output files..."
