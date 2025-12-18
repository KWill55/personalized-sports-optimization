"""
calibrate_stereo.py

Purpose:
  Perform stereo camera calibration using checkerboard image pairs
  from combined images captured with 'capture_cb_pairs.py'.

Input:
  - Combined images where left and right cameras are stitched side-by-side (1280x640).
  - Checkerboard dimensions and square size must match the capture script (from project_config.yaml).

Output:
  - Intrinsic parameters (K1, K2) and distortion (dist1, dist2)
  - Extrinsic parameters (R, T)
  - Projection matrices (P1, P2)
  - Essential (E) and Fundamental (F) matrices
  - RMS reprojection error
  - Saves all results to stereo_calib.npz (+ stereo_calib_summary.yaml)
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import yaml
import sys


# =========================
# Data containers
# =========================
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


# =========================
# Configuration & IO
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from utils.io_utils import load_config

# =========================
# Config (from YAML)
# =========================
cfg = load_config("project_config.yaml")

def get_paths(cfg: dict) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Returns:
        calib_mono_left_dir, calib_mono_right_dir, calib_pairs_dir, output_dir, session_dir
    """
    base_dir = Path(__file__).resolve().parents[3]
    session_dir = base_dir / "data" / cfg["athlete"] / cfg["session"]

    mono_left_dir       = session_dir / "calibration" / "calib_images" / "mono_left"
    mono_right_dir      = session_dir / "calibration" / "calib_images" / "mono_right"
    stereo_combined_dir = session_dir / "calibration" / "calib_images" / "pairs"
    output_dir          = session_dir / "calibration" / "stereo_calibration"

    # mono_left_dir      = cfg["paths"]["calib_mono_left"]
    # mono_right_dir     = cfg["paths"]["calib_mono_right"]
    # stereo_combined_dir=  cfg["paths"]["calib_pairs"]  
    # output_dir = cfg["paths"]["stereo_calibration"]

    output_dir.mkdir(parents=True, exist_ok=True)
    return mono_left_dir, mono_right_dir, stereo_combined_dir, output_dir, session_dir


# =========================
# Prepare 3D object points (checkerboard world coordinates)
# =========================
def prepare_object_points(
        inner_corners: Tuple[int, int], # inner corners of checkerboard (col, row)
        square_size: float #real world square length in cm
        ) -> np.ndarray: #objpoints: fixed 3D array of all corner coordinates in a grid (world coordinates)
    """
    Purpose: create known 3D coordinates of real world checkerboard 
    
    Notes:
        - z values are always 0 since the checkerboard is a flat plane 
    """

    # Create an array of zeros with shape (num points, 3),
    # where each row will hold (X, Y, Z) coordinates of a checkerboard corner
    objpoints = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)

    # Fill in the X and Y coordinates with a grid:
    # np.mgrid[0:cols, 0:rows] makes a 2D grid of indices,
    # .T.reshape(-1, 2) flattens it into (num_points, 2) pairs
    # [:, :2] take all rows, take first two columns (Z is always 0)
    objpoints[:, :2] = np.mgrid[0:inner_corners[0], 0:inner_corners[1]].T.reshape(-1, 2)

    # Scale the grid by the real-world square size (e.g., cm, mm, meters)
    # so that object points represent actual physical distances
    objpoints *= float(square_size)

    # Return the full set of 3D object points (all Z = 0 since the board is flat)
    return objpoints


# =========================
# Checkerboard detection in images 
# =========================

def collect_mono_detections(calib_dir: Path, pattern_size: Tuple[int, int]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Detect corners in a folder of MONO images (no splitting).
    
    Returns:
        imgpoints: list of (N,1,2) arrays for each image
        image_size: (width, height)
    """
    print(f"[DEBUG] Mono detection dir: {calib_dir}")
    files = sorted(glob.glob(str(calib_dir / "*.png"))) + sorted(glob.glob(str(calib_dir / "*.jpg")))
    print(f"\n[INFO] Found {len(files)} mono images at {calib_dir}\n")

    imgpoints: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    # max 30 iterations
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for fp in files:
        img = cv.imread(fp)
        if img is None:
            print(f"[ERROR] Could not read: {fp}")
            continue

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray, pattern_size, None)
        if ret:
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
        else:
            print(f"[WARNING] Checkerboard not detected: {fp}")

    if image_size is None or len(imgpoints) == 0:
        raise RuntimeError(f"[ERROR] No valid detections in {calib_dir}")

    print(f"\n[INFO] Using {len(imgpoints)} images from {calib_dir} for mono calibration\n")
    return imgpoints, image_size


def collect_stereo_detections_combined(
    calib_images_dir: Path,
    pattern_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    Purpose: Finds 2D pixel coordinates for calibration grid corners
             in each combined stereo image (stitched left|right).
    
    Returns:
        imgpointsL: list of 2D detections for the left camera
        imgpointsR: list of 2D detections for the right camera
        image_size: (width, height) of a single half-frame
    """
    print(f"[DEBUG] Looking for images in: {calib_images_dir}")
    combined_images = sorted(glob.glob(str(calib_images_dir / "pair_*.png")))
    print(f"\n[INFO] Found {len(combined_images)} combined images.\n")
    if len(combined_images) < 10:
        print("[WARNING] Fewer than 10 image pairs may reduce calibration accuracy.")

    imgpointsL: List[np.ndarray] = []
    imgpointsR: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for img_path in combined_images:
        combined = cv.imread(img_path)
        if combined is None:
            print(f"[ERROR] Could not read: {img_path}")
            continue

        h, w = combined.shape[:2]
        if w % 2 != 0:
            print(f"[ERROR] Invalid combined image width (must be even): {img_path}")
            continue

        half = w // 2
        if image_size is None:
            image_size = (half, h)  # per-camera size, not full stitched width

        frameL = combined[:, :half]
        frameR = combined[:, half:]

        grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

        retL, cornersL = cv.findChessboardCornersSB(grayL, pattern_size, None)
        retR, cornersR = cv.findChessboardCornersSB(grayR, pattern_size, None)

        if retL and retR:
            cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
        else:
            print(f"[WARNING] Checkerboard not detected in {img_path}")

    if image_size is None or len(imgpointsL) == 0:
        raise RuntimeError("[ERROR] No valid checkerboard detections found in stereo folder.")

    print(f"\n[INFO] Using {len(imgpointsL)} valid stereo pairs for extrinsics.\n")
    return imgpointsL, imgpointsR, image_size



# =========================
# Calibration steps
# =========================
def calibrate_mono_intrinsics(
    objp_template: np.ndarray,
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[Intrinsics, List[np.ndarray]]:
    """
    Calibrate a single camera from its own mono detections.
    """
    if objp_template.dtype != np.float32:
        objp_template = objp_template.astype(np.float32, copy=False)

    num = len(imgpoints)
    objpoints = [objp_template.copy() for _ in range(num)]

    rms, K, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return Intrinsics(K=K, dist=dist, rms=float(rms)), objpoints


def calibrate_extrinsics(
    objpoints: List[np.ndarray],
    imgpointsL: List[np.ndarray],
    imgpointsR: List[np.ndarray],
    intrinsics_L: Intrinsics,
    intrinsics_R: Intrinsics,
    image_size: Tuple[int, int],
) -> Extrinsics:
    
    flags = cv.CALIB_FIX_INTRINSIC  # keep our intrinsic values
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        intrinsics_L.K, intrinsics_L.dist, intrinsics_R.K, intrinsics_R.dist,
        image_size,
        criteria=criteria,
        flags=flags
    )

    # Projection matrices: P1 = K1 [I|0], P2 = K2 [R|T]
    # Projection matrices define how 3D points map into each camera’s 2D image.
    P1 = intrinsics_L.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = intrinsics_R.K @ np.hstack((R, T))

    return Extrinsics(R=R, T=T, E=E, F=F, P1=P1, P2=P2, rms=float(retval))


# =========================
# Save & report
# =========================
def save_npz(output_file: Path, intr_L: Intrinsics, intr_R: Intrinsics, extr: Extrinsics) -> None:
    np.savez(
        output_file,
        # Intrinsics
        K1=intr_L.K, dist1=intr_L.dist, rms1=np.array([intr_L.rms]),
        K2=intr_R.K, dist2=intr_R.dist, rms2=np.array([intr_R.rms]),
        # Extrinsics
        R=extr.R, T=extr.T, E=extr.E, F=extr.F, P1=extr.P1, P2=extr.P2, rms_stereo=np.array([extr.rms]),
    )


def save_summary_yaml(out_yaml: Path, intr_L: Intrinsics, intr_R: Intrinsics, extr: Extrinsics) -> None:
    def to_list(a: np.ndarray): return a.tolist()
    summary = {
        "intrinsics": {
            "left":  {"K1": to_list(intr_L.K), "dist1": to_list(intr_L.dist.squeeze()), "rms1": intr_L.rms},
            "right": {"K2": to_list(intr_R.K), "dist2": to_list(intr_R.dist.squeeze()), "rms2": intr_R.rms},
        },
        "extrinsics": {
            "R": to_list(extr.R), "T": to_list(extr.T.squeeze()),
            "E": to_list(extr.E), "F": to_list(extr.F),
            "P1": to_list(extr.P1), "P2": to_list(extr.P2),
            "rms_stereo": extr.rms,
        },
    }
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)



def print_report(intr_L: Intrinsics, intr_R: Intrinsics, extr: Extrinsics, output_file: Path, out_yaml: Path) -> None:
    np.set_printoptions(precision=6, suppress=True)
    print("\n================== CALIBRATION REPORT ==================")

    print("INTRINSICS")
    print("  Left camera:")
    print("    K1:\n", intr_L.K)
    print("    dist1:", intr_L.dist.ravel())
    print(f"    rms1:  {intr_L.rms:.6f}")
    print("  Right camera:")
    print("    K2:\n", intr_R.K)
    print("    dist2:", intr_R.dist.ravel())
    print(f"    rms2:  {intr_R.rms:.6f}")

    print("\nEXTRINSICS")
    print("  R:\n", extr.R)
    print("  T:", extr.T.ravel())
    print("  E:\n", extr.E)
    print("  F:\n", extr.F)
    print("  P1:\n", extr.P1)
    print("  P2:\n", extr.P2)
    print(f"  rms_stereo: {extr.rms:.6f}")
    print("\n[INFO] Saved arrays to:", output_file)
    print("[INFO] Summary YAML:   ", out_yaml)

    print("========================================================\n")


# =========================
# Main
# =========================
def main():
    
    # Configuration
    project_cfg = load_config()
    CHECKERBOARD_SIZE = tuple(project_cfg["inner_corners"])   # (cols, rows) — INNER corners
    SQUARE_SIZE       = float(project_cfg["square_size_in"])  # e.g., cm
    mono_left_dir, mono_right_dir, stereo_combined_dir, output_dir, _ = get_paths(project_cfg)
    output_file = output_dir / "stereo_calib.npz"
    out_yaml    = output_dir / "stereo_calib_summary.yaml"

    # Define 3D checkerboard geometry (real world)
    obj_template = prepare_object_points(CHECKERBOARD_SIZE, SQUARE_SIZE)

    # --- Intrinsics from mono folders ---
    imgpointsL_mono, image_size_L = collect_mono_detections(mono_left_dir,  CHECKERBOARD_SIZE)
    imgpointsR_mono, image_size_R = collect_mono_detections(mono_right_dir, CHECKERBOARD_SIZE)

    intrinsics_L, _objL = calibrate_mono_intrinsics(obj_template, imgpointsL_mono, image_size_L)
    intrinsics_R, _objR = calibrate_mono_intrinsics(obj_template, imgpointsR_mono, image_size_R)

    # --- Extrinsics from combined folder ---
    imgpointsL_stereo, imgpointsR_stereo, image_size_stereo = collect_stereo_detections_combined(stereo_combined_dir, CHECKERBOARD_SIZE)
    
    # Build objpoints matched to the number of valid stereo pairs
    objpoints_stereo = [obj_template.copy() for _ in range(len(imgpointsL_stereo))]

    extrinsics = calibrate_extrinsics(
        objpoints_stereo, imgpointsL_stereo, imgpointsR_stereo,
        intrinsics_L, intrinsics_R, image_size_stereo, 
    ) #fixedintrinsic 

    # Save + human-readable summary
    save_npz(output_file, intrinsics_L, intrinsics_R, extrinsics)
    save_summary_yaml(out_yaml, intrinsics_L, intrinsics_R, extrinsics)
    print_report(intrinsics_L, intrinsics_R, extrinsics, output_file, out_yaml)


if __name__ == "__main__":
    main()
