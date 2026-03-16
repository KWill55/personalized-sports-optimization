from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

from domain.types import Intrinsics
from domain.types import Extrinsics

import cv2 as cv
import numpy as np

"""
Responsible for holding functions necessary for calibration 
"""

# =========================
# Prepare 3D object points (checkerboard world coordinates)
# =========================
def prepare_object_points(
    inner_corners: Tuple[int, int],  # (cols, rows) inner corners
    square_size: float,              # real-world square size (units must be consistent everywhere)
) -> np.ndarray:
    """
    Create known 3D coordinates of checkerboard corners on a flat plane (Z=0).
    """
    objpoints = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[0:inner_corners[0], 0:inner_corners[1]].T.reshape(-1, 2)
    objpoints *= float(square_size)
    return objpoints


# =========================
# Checkerboard detection (mono images)
# =========================
def collect_mono_detections(
    calib_dir: Path,
    pattern_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Detect corners in a folder of MONO images (no splitting).

    Returns:
        imgpoints: list of (N,1,2) arrays for each image
        image_size: (width, height)
    """
    files = sorted(glob.glob(str(calib_dir / "*.png"))) + sorted(glob.glob(str(calib_dir / "*.jpg")))

    imgpoints: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for fp in files:
        img = cv.imread(fp)
        if img is None:
            continue

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCornersSB(gray, pattern_size, None)

        if ret:
            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    if image_size is None or len(imgpoints) == 0:
        raise RuntimeError(f"No valid mono detections in {calib_dir}")

    return imgpoints, image_size


# =========================
# Checkerboard detection (stitched stereo images)
# =========================
def collect_stereo_detections_combined(
    calib_images_dir: Path,
    pattern_size: Tuple[int, int],
) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int]]:
    """
    Detect corners in stitched stereo images (left|right).

    Returns:
        imgpointsL: list of 2D detections for the left camera
        imgpointsR: list of 2D detections for the right camera
        image_size: (width, height) of a single half-frame
    """
    combined_images = sorted(glob.glob(str(calib_images_dir / "pair_*.png")))

    imgpointsL: List[np.ndarray] = []
    imgpointsR: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for img_path in combined_images:
        combined = cv.imread(img_path)
        if combined is None:
            continue

        h, w = combined.shape[:2]
        if w % 2 != 0:
            continue

        half = w // 2
        if image_size is None:
            image_size = (half, h)  # per-camera size

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

    if image_size is None or len(imgpointsL) == 0:
        raise RuntimeError(f"No valid stereo detections in {calib_images_dir}")

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

    objpoints = [objp_template.copy() for _ in range(len(imgpoints))]

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
    """
    Stereo calibration using fixed intrinsics.
    """
    flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        intrinsics_L.K, intrinsics_L.dist,
        intrinsics_R.K, intrinsics_R.dist,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    # Projection matrices
    P1 = intrinsics_L.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = intrinsics_R.K @ np.hstack((R, T))

    return Extrinsics(R=R, T=T, E=E, F=F, P1=P1, P2=P2, rms=float(retval))


# =========================
# JSON-friendly conversions
# =========================
def intrinsics_to_dict(intr: Intrinsics, image_size: Tuple[int, int]) -> dict:
    """
    Convert Intrinsics to a JSON-serializable dict.
    image_size is (width, height).
    """
    return {
        "K": intr.K.tolist(),
        "dist": np.asarray(intr.dist).squeeze().tolist(),
        "rms": float(intr.rms),
        "image_size": [int(image_size[0]), int(image_size[1])],
    }


def extrinsics_to_dict(extr: Extrinsics, image_size: Tuple[int, int]) -> dict:
    """
    Convert Extrinsics to a JSON-serializable dict.
    image_size is (width, height) for a single camera half-frame.
    """
    return {
        "R": extr.R.tolist(),
        "T": np.asarray(extr.T).squeeze().tolist(),
        "E": extr.E.tolist(),
        "F": extr.F.tolist(),
        "P1": extr.P1.tolist(),
        "P2": extr.P2.tolist(),
        "rms": float(extr.rms),
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
