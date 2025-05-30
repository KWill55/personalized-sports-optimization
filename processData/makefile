# Makefile for Personalized Sports Optimization Project

# Paths (update if needed)
MOT_FOLDER=../data/freethrows1/mot_files
RELEASE_CSV=../data/freethrows1/release_summary.csv
AVERAGE_CSV=../data/freethrows1/average_kinematics_by_outcome.csv

# Targets
all: process split plot analyze

process:
	@echo "Processing .mot files into release_summary.csv..."
	python process_mot_to_csv.py

split: $(RELEASE_CSV)
	@echo "Splitting CSV into grouped body part files..."
	python split_csv_by_group.py

plot: $(RELEASE_CSV)
	@echo "Generating kinematic plots..."
	python plot_kinematics.py

analyze: $(RELEASE_CSV)
	@echo "Analyzing average kinematics by outcome..."
	python analyze_averages.py

clean:
	@echo "Cleaning output files..."
	rm -f $(RELEASE_CSV) $(AVERAGE_CSV)
	rm -f ../data/freethrows1/group_*.csv

.PHONY: all process split plot analyze clean
