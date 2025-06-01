# Makefile for managing stereo camera workflow

# Python scripts
CAPTURE_SCRIPT = captureImagePairs.py
CALIBRATION_SCRIPT = stereo_calibrate.py
RECTIFY_SCRIPT = stereo_rectify.py
RECORD_SCRIPT = record_free_throws.py

# Targets
.PHONY: capture calibrate rectify record clean help

capture:
	python $(CAPTURE_SCRIPT)

calibrate:
	python $(CALIBRATION_SCRIPT)

rectify:
	python $(RECTIFY_SCRIPT)

record:
	python $(RECORD_SCRIPT)

clean:
	rm -f stereo_calib.npz
	rm -rf calib_images/*
	rm -rf throws/*

help:
	@echo "Stereo Free Throw Workflow"
	@echo ""
	@echo "Usage:"
	@echo "  make capture     - Run captureImagePairs.py"
	@echo "  make calibrate   - Run stereo_calibrate.py"
	@echo "  make rectify     - Run stereo_rectify.py"
	@echo "  make record      - Run record_free_throws.py"
	@echo "  make clean       - Delete calibration/results"
