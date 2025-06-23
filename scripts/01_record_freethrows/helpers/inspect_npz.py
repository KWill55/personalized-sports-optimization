import numpy as np
from pathlib import Path

ATHLETE = "Kenny"
SESSION = "session_001"


base_dir = Path(__file__).resolve().parents[3] # Go up to project root
session_dir = base_dir / "data" / ATHLETE / SESSION


calib_path = session_dir / "calibration" / "stereo_calib.npz"
calib = np.load(calib_path)

print("Keys in calibration file:")
print(calib.files)
