"""
Title: insepct_npz.py

Description:
    The purpose of this module is to display the calibration file's contents

Inputs:
    - calibration .npz file

Usage:
    - Run the script to produce calibration paramters

Outputs:
    - Terminal prints out the calibration file's contents

"""

#K1, K2: Intrinsic camera matrices for both cameras.
# dist1, dist2: Lens distortion coefficients.
# R: Rotation between camera 1 and camera 2.
# T: Translation vector (baseline distance between cameras).
# P1, P2: Projection matrices for rectified stereo pair.
# E, F: Essential and Fundamental matrices (used for epipolar geometry).

import numpy as np
from pathlib import Path

# ============================
# Parameter Constants 
# ============================
ATHLETE = "kenny"
SESSION = "session_001"

# ============================
# Paths and Directories
# ============================
base_dir = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

# Load calibration file
calib_path = session_dir / "calibration" / "stereo_calibration" / "stereo_calib.npz"
calib = np.load(calib_path)

# ============================
# Display contents of calibration file
# ============================

# Settings for clean printing
np.set_printoptions(precision=4, suppress=True)

# Header
print("="*60)
print(f"Stereo Calibration Data: {calib_path.name}")
print("="*60)

# Summary Table
print("\nSummary of Calibration Arrays:")
print(f"{'Key':<20}{'Shape':<15}{'Dtype'}")
print("-" * 60)
for key in calib.files:
    arr = calib[key]
    print(f"{key:<20}{str(arr.shape):<15}{arr.dtype}")

# Detailed Values
print("\nDetailed Matrices:")
for key in calib.files:
    print(f"\n--- {key} ---\n{calib[key]}")
