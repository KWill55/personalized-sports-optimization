# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# Targets
all: process split plot analyze


# ======================================== 
# Phase 1: Recording Freethrows 
# ========================================


# ===== paths and directories =====
record_dir := scripts/01_record_freethrows
calibrate_dir := $(record_dir)/player_tracking_calibration
trim_dir := $(record_dir)/trim_freethrows
helpers_dir := $(record_dir)/helpers


# ===== helpers =====

identify_cameras: ## Camera identifiication gui
	@echo "opening gui to identify camera indicies"
	python $(helpers_dir)/identify_cameras.py

## ------------------------------
# ===== Calibration =====
## ------------------------------

calibration-header: ## 🔧 Calibration
	@:


tune_intrinsics: ## Step 1: Tune the physical intrinsics
	@echo "Opening GUI for intrinsic tuning..."
	python $(calibrate_dir)/01_tune_intrinsics.py


capture_calibration_pairs: ## Step 2: Capture calibration pairs
	@echo "Capturing image pairs for calibration..."
	python $(calibrate_dir)/02_capture_calibration_pairs.py


calibrate_stereo: ## Step 3: Calibrate stereo cameras
	@echo "Calibrating cameras using captured pairs..."
	python $(calibrate_dir)/03_calibrate_stereo.py


rectify_stereo: ## Step 4: Rectify stereo images
	@echo "Rectifying stereo images..."
	python $(calibrate_dir)/04_rectify_stereo.py


## ------------------------------
# ===== Trimming freethrows =====
## ------------------------------

auto_trim_by_flash: ## automatic trimming via flashes
	@echo "Trimming video clips via flash..."
	python $(trim_dir)/auto_trim_by_flash.py

# ===== Recording Freethrows =====

record_freethrows:
	@echo "Recording a freethrow..."
	python $(record_dir)/record_freethrows.py


# ======================================== 
# Phase 2: Extract player and ball metrics
# ========================================


# ===== paths and directories =====
extract_dir := scripts/02_extract_metrics
ball_dir := extract_dir

player_dir := extract_dir / player_tracking
combine_dir := extract_dir / combine_extracted_metrics


helpers_dir := $(extract_metrics_dir)/helpers


# ===== Helpers =====

process_release: ## 
	@echo "Processing .mot files into release_summary.csv..."
	python scripts/02_process_data/release/process_release.py

# ===== Extract Metrics for Player Tracking =====

# ===== Extract Metrics for Ball Tracking =====

# ===== Combine Extracted Metrics =====

#TODO make this script usable for both release and time series data
split_release: $(RELEASE_CSV)
	@echo "Splitting release_summary.csv into separate csvs (core,legs,arms)..."
	python scripts/02_process_data/release/split_release.py

# ===== Time Series Data Processing =====

detect_phases:
	@echo "Detecting phases in .mot files..."
	python scripts/02_extract_features/detect_phases.py

process_time_series:
	@echo "Processing .mot files into time series data..."
	python scripts/02_process_data/time_series/save_time_series.py

#doesn't work yet
trim_freethrow_automatic:
	@echo "Producing a trimmed free throw for evaluation..."
	python scripts/02_process_data/time_series/trim_freethrow_automatic.py

trim_freethrow_manual:
	@echo "Opening up GUI for trimming free throw data..."
	python scripts/02_process_data/time_series/trim_freethrow_manual.py

create_hoop_regions:
	@echo "Opening GUI to define hoop regions (above and below)..."
	python scripts/02_process_data/time_series/create_hoop_regions.py

find_hsv_region:
	@echo "Opening GUI to define HSV region..."
	python scripts/02_process_data/time_series/tune_hsv.py

detect_metrics:
	@echo "Detecting metrics from free throw data..."
	python scripts/02_extract_features/ball_tracking/detect_metrics.py 



# ======================================== 
# Phase 3: 03_analyze_data
# ========================================

# ===== Release Data Processing =====

plot_release:
	@echo "Generating 3 kinematic plots from release_summary.csv..."
	python scripts/03_visualize_data/player_tracking/plot_release_kinematics.py

# ===== Time Series Data Processing =====

plot_kinematics_time_series:
	@echo "Saving kinematic plots for each free throw..."
	python scripts/03_visualize_data/player_tracking/plot_kinematics_time_series.py

plot_velocities_time_series: 
	@echo "Saving velocity plots for each free throw..."
	python scripts/03_visualize_data/player_tracking/plot_velocities_time_series.py

# ===== Release Data Processing =====

analyze_release: $(RELEASE_CSV)
	@echo "Analyzing average kinematics at release by outcome..."
	python scripts/04_analyze_data/analyze_release_averages.py

# model training 
combine_release_summaries: $(RELEASE_CSV)
	@echo "Combining multiple release_summary.csv files..."
	python scripts/04_analyze_data/release/combine_release_summaries.py

prepare_release_features: $(RELEASE_CSV)
	@echo "Creating a CSV for features and labels..."
	python scripts/04_analyze_data/release/prepare_release_features.py

train_baseline_models:
	@echo "Training baseline models on combined release data..."
	python scripts/04_analyze_data/train_baseline_models.py

# ===== Time Series Data Processing =====

# TODO not implemented yet


# ======================================== 
# help
# ========================================

.PHONY: help

help:
	@echo ""
	@echo "🏀 \033[1mMakefile Command Reference\033[0m"
	@echo "--------------------------------------------"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?##' Makefile | sort | \
	awk -F: '{ if ($$1 ~ /-header$$/) { printf "\n\033[1m%s\033[0m\n", substr($$2, index($$2,$$3)) } else { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } }'
	@echo ""



# ======================================== 
# clean 
# ========================================

clean:
	@echo "Cleaning output files..."

.PHONY: all process split plot analyze clean
