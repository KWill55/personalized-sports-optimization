from dataclasses import dataclass
from typing import Optional

import numpy as np

# =========================================================
# Calibration Objects
# =========================================================

@dataclass
class Intrinsics:
    K: np.ndarray          # 3x3
    dist: np.ndarray       # (k,) OpenCV distortion vector
    rms: float             # single-camera RMS

@dataclass
class Extrinsics:
    R: np.ndarray          # 3x3
    T: np.ndarray          # 3x1
    E: np.ndarray          # 3x3
    F: np.ndarray          # 3x3
    P1: np.ndarray         # 3x4
    P2: np.ndarray         # 3x4
    rms: float             # stereo RMS






