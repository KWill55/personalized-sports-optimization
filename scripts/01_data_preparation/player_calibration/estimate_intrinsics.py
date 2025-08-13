#!/usr/bin/env python3
"""
Simple per-camera intrinsics from mono checkerboard images (orientation-robust).

Reads:
  data/<ATHLETE>/<SESSION>/calibration/calib_images/mono_left/*
  data/<ATHLETE>/<SESSION>/calibration/calib_images/mono_right/*

Writes:
  data/<ATHLETE>/<SESSION>/calibration/mono_intrinsics/intrinsics_left.npz
  data/<ATHLETE>/<SESSION>/calibration/mono_intrinsics/intrinsics_right.npz
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import glob
import yaml

# --- Config from YAML ---
cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
CHECKERBOARD = tuple(cfg["inner_corners"])   # (cols, rows)
SQUARE_CM   = float(cfg["square_size_cm"])   # square size in cm

# --- Paths ---
base      = Path(__file__).resolve().parents[3] / "data" / ATHLETE / SESSION
root_in   = base / "calibration" / "calib_images"
left_dir  = root_in / "mono_left"
right_dir = root_in / "mono_right"
out_dir   = base / "calibration" / "mono_intrinsics"
out_dir.mkdir(parents=True, exist_ok=True)

def _flag(mod, name, default=0):
    return getattr(mod, name, default)

def list_images(folder: Path):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files += glob.glob(str(folder / ext))
    return sorted(files)

def detect_board(gray, cols, rows):
    """Try (cols,rows); if fail, try (rows,cols) and reorder to (cols,rows)."""
    det_flags = (_flag(cv, "CALIB_CB_EXHAUSTIVE", 0) |
                 _flag(cv, "CALIB_CB_ACCURACY", 0) |
                 _flag(cv, "CALIB_CB_NORMALIZE_IMAGE", 0))
    ok, pts = cv.findChessboardCornersSB(gray, (cols, rows), flags=det_flags)
    if ok:
        return True, pts
    ok2, pts2 = cv.findChessboardCornersSB(gray, (rows, cols), flags=det_flags)
    if not ok2:
        return False, None
    # reorder from (rows,cols) to (cols,rows)
    pts2 = pts2.reshape(rows, cols, 1, 2).transpose(1, 0, 2, 3).reshape(-1, 1, 2)
    return True, pts2

def calibrate_one(folder: Path, side_name: str, out_name: str, TOP: int = 15):
    cols, rows = CHECKERBOARD

    # Prepare one object grid per view
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_CM

    objpoints, imgpoints = [], []
    accepted_files, sharpness = [], []
    image_size = None

    files = list_images(folder)
    print(f"[INFO] {side_name}: found {len(files)} images in {folder}")
    if not files:
        raise SystemExit(f"[ERROR] No images found for {side_name} at {folder}")

    for p in files:
        img = cv.imread(p)
        if img is None:
            print(f"[WARN] {side_name}: cannot read {p}, skipping")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ok, corners = detect_board(gray, cols, rows)  # orientation-robust
        if not ok:
            # quiet skip to avoid noisy logs
            continue

        cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        accepted_files.append(p)
        sharpness.append(cv.Laplacian(gray, cv.CV_64F).var())
        image_size = gray.shape[::-1]  # (W, H)

    n_accept = len(objpoints)
    if n_accept < 3:
        raise SystemExit(f"[ERROR] {side_name}: need ≥3 valid detections (got {n_accept}).")

    # Keep TOP sharpest views (if we have more than TOP); otherwise keep all
    if TOP is not None and n_accept > TOP:
        idx = np.argsort(np.array(sharpness))[::-1][:TOP]
        objpoints = [objpoints[i] for i in idx]
        imgpoints = [imgpoints[i] for i in idx]
        kept_files = [accepted_files[i] for i in idx]
    else:
        kept_files = accepted_files

    # Simple calibration: 5+k rational model
    flags = cv.CALIB_RATIONAL_MODEL
    rms, K, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)

    # Quick sanity summary
    W, H = image_size
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    off = float(np.hypot(cx - W/2, cy - H/2))
    print(f"[RESULT] {side_name}: used {len(objpoints)} / {n_accept} accepted views  RMS={rms:.3f}px")
    print(f"[RESULT] {side_name}: fx/fy={fx/fy:.3f}  center=({cx:.1f},{cy:.1f})  offset={off:.1f}px")
    print(f"[RESULT] {side_name}: K=\n{K}")
    print(f"[RESULT] {side_name}: dist=\n{dist}")

    out_path = out_dir / out_name
    np.savez(out_path, K=K, dist=dist, image_size=np.array(image_size, np.int32),
             kept_files=np.array(kept_files))
    print(f"[INFO]  {side_name}: saved {out_path}")

    return K, dist


if __name__ == "__main__":
    print("[INFO] Simple intrinsics (orientation-robust)")
    print("[INFO] Athlete/Session:", ATHLETE, SESSION)

    K_left, dist_left   = calibrate_one(left_dir,  "LEFT",  "intrinsics_left.npz")
    K_right, dist_right = calibrate_one(right_dir, "RIGHT", "intrinsics_right.npz")

    print("[INFO] Done. Intrinsics written to:", out_dir)
