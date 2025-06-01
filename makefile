# Makefile for Personalized Sports Optimization Project
# Purpose:
# Automate data processing, plotting, and analysis for basketball free throw kinematics

# ======================================== 
# Paths (update if needed)  
# ========================================

MOT_FOLDER=data/freethrows1/mot_files
RELEASE_CSV=data/freethrows1/release/release_summary.csv
AVERAGE_CSV=data/freethrows1/release/average_kinematics_by_outcome.csv

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
	python scripts/02_process_data/release/process_mot_to_csv.py

split_release: $(RELEASE_CSV)
	@echo "Splitting release_summary.csv into separate csvs (core,legs,arms)..."
	python scripts/02_process_data/release/split_csv_by_group.py

# ===== Time Series Data Processing =====

detect_phases:
	@echo "Detecting phases in .mot files..."
	python scripts/02_process_data/time_series/detect_phases.py


# ======================================== 
# 03_plot_data 
# ========================================

# ===== Release Data Processing =====

plot: $(RELEASE_CSV)
	@echo "Generating 3 kinematic plots from release_summary.csv..."
	python scripts/03_plot_data/release/plot_release_kinematics.py

# ===== Time Series Data Processing =====

# TODO not implemented yet

# ======================================== 
# 04_analyze_data 
# ========================================

# ===== Release Data Processing =====

analyze_release: $(RELEASE_CSV)
	@echo "Analyzing average kinematics at release by outcome..."
	python scripts/04_analyze_data/release/analyze_release_averages.py

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
