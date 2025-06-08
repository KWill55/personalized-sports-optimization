# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# ======================================== 
# Paths (update if needed)  
# ========================================

SESSION=freethrows1
BASE_DIR=data/$(SESSION)/02_process_data/release

RELEASE_CSV=$(BASE_DIR)/release_summary.csv
AVERAGE_CSV=$(BASE_DIR)/average_kinematics_by_outcome.csv


# Targets
all: process split plot analyze

# ======================================== 
# 01_record_data 
# ========================================

# TODO implement later

# ======================================== 
# 02_process_data 
# ========================================

# ===== Release Data Processing =====

process_release: 
	@echo "Processing .mot files into release_summary.csv..."
	python scripts/02_process_data/release/process_release.py

#TODO make this script usable for both release and time series data
split_release: $(RELEASE_CSV)
	@echo "Splitting release_summary.csv into separate csvs (core,legs,arms)..."
	python scripts/02_process_data/release/split_release.py

# ===== Time Series Data Processing =====

detect_phases:
	@echo "Detecting phases in .mot files..."
	python scripts/02_process_data/time_series/detect_phases.py

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

detect_make:
	@echo "Detecting makes and misses from free throw data..."
	python scripts/02_process_data/time_series/detect_make.py


# ======================================== 
# 03_visualize_data 
# ========================================

# ===== Release Data Processing =====

plot_release:
	@echo "Generating 3 kinematic plots from release_summary.csv..."
	python scripts/03_visualize_data/release/plot_release_kinematics.py

# ===== Time Series Data Processing =====

plot_kinematics_time_series: $(RELEASE_CSV)
	@echo "Saving kinematic plots for each free throw in $(SESSION)..."
	python scripts/03_visualize_data/time_series/plot_kinematics_time_series.py

plot_velocities_time_series: $(RELEASE_CSV)
	@echo "Saving velocity plots for each free throw in $(SESSION)..."
	python scripts/03_visualize_data/time_series/plot_velocities_time_series.py

# ======================================== 
# 04_analyze_data 
# ========================================

# ===== Release Data Processing =====

analyze_release: $(RELEASE_CSV)
	@echo "Analyzing average kinematics at release by outcome..."
	python scripts/04_analyze_data/release/analyze_release_averages.py

# model training 
combine_release_summaries: $(RELEASE_CSV)
	@echo "Combining multiple release_summary.csv files..."
	python scripts/04_analyze_data/release/combine_release_summaries.py

prepare_release_features: $(RELEASE_CSV)
	@echo "Creating a CSV for features and labels..."
	python scripts/04_analyze_data/release/prepare_release_features.py

train_baseline_models:
	@echo "Training baseline models on combined release data..."
	python scripts/04_analyze_data/release/train_baseline_models.py

# ===== Time Series Data Processing =====

# TODO not implemented yet

# ======================================== 
# clean 
# ========================================

clean:
	@echo "Cleaning output files..."
	rm -f $(RELEASE_CSV) $(AVERAGE_CSV)
	rm -f ../data/freethrows1/group_*.csv

.PHONY: all process split plot analyze clean
