#!/usr/bin/env python3
"""
Per-camera intrinsics from mono checkerboard images (orientation-robust).
Runs BOTH 5-param and Rational models, prints per-model tables + a final summary,
and saves NPZ files with suffixes: *_five_param.npz and *_rational.npz.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import glob
import yaml
import math

# ----------------------------
# Config / thresholds
# ----------------------------
CRITERIA = {
    "RMS reprojection error [px]": {"good": (0, 1.5), "warn": (1.5, 3.0)},  # <=1.5 good, <=3 warn
    "fx/fy ratio (≈1)":            {"good": (0.98, 1.02), "warn": (0.95, 1.05)},
    "center offset [px]":          {"good": (0, 30), "warn": (30, 60)},
    "max |distortion|":            {"good": (0, 1.0), "warn": (1.0, 5.0)},
}

def print_criteria():
    print("="*86)
    print("Calibration Criteria (GOOD / WARNING / FAIL)")
    print("Legend: ✅ = GOOD   ⚠️ = WARNING   ❌ = FAIL")
    print("-"*86)
    print(f"{'Metric':39s}  {'GOOD':>14s}  {'WARNING':>14s}  {'FAIL':>13s}")
    print("-"*86)
    print(f"{'RMS reprojection error [px]':39s}  {'≤ 1.5':>14s}  {'≤ 3.0':>14s}  {'> 3.0':>13s}")
    print(f"{'fx/fy ratio (≈1)':39s}          {'0.98–1.02':>14s}  {'0.95–1.05':>14s}  {'outside':>13s}")
    print(f"{'center offset [px]':39s}        {'≤ 30':>14s}  {'≤ 60':>14s}  {'> 60':>13s}")
    print(f"{'max |distortion|':39s}          {'< 1.0':>14s}  {'< 5.0':>14s}  {'≥ 5.0':>13s}")
    print("="*86)
    print("\n") 
    print("\n")

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

# ---- Status helpers (text -> emoji) ----
def _status(metric, value):
    if metric == "fx/fy ratio (≈1)":
        good_lo, good_hi = CRITERIA[metric]["good"]
        warn_lo, warn_hi = CRITERIA[metric]["warn"]
        if good_lo <= value <= good_hi: return "GOOD"
        if warn_lo <= value <= warn_hi: return "WARNING"
        return "FAIL"
    else:
        good_lo, good_hi = CRITERIA[metric]["good"]
        warn_lo, warn_hi = CRITERIA[metric]["warn"]
        if value <= good_hi: return "GOOD"
        if value <= warn_hi: return "WARNING"
        return "FAIL"

def _status_emoji(status: str) -> str:
    return {"GOOD": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(status, "")

# ---- Metrics / reporting ----
def metrics_from(K, dist, rms, image_size):
    W, H = image_size
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    ratio = fx / fy if fy else math.inf
    center_offset = float(np.hypot(cx - W/2, cy - H/2))
    dmax = float(np.max(np.abs(dist)))
    return {
        "rms": rms, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "ratio": ratio, "center_offset": center_offset, "dmax": dmax, "image_size": (W, H)
    }

def pretty_report(side_name, model_name, M):
    W, H = M["image_size"]
    rows_raw = [
        ("RMS reprojection error [px]", M['rms']),
        ("fx/fy ratio (≈1)",            M['ratio']),
        ("center offset [px]",          M['center_offset']),
        ("max |distortion|",            M['dmax']),
    ]
    rows = []
    for metric, val in rows_raw:
        status = _status(metric, val)
        emoji = _status_emoji(status)
        if "RMS" in metric:
            sval = f"{val:.3f}"
        elif "ratio" in metric:
            sval = f"{val:.3f}"
        elif "offset" in metric:
            sval = f"{val:.1f}"
        else:
            sval = f"{val:.2f}"
        rows.append((metric, sval, emoji))

    print(f"\n{side_name} — {model_name} Intrinsics Summary (image {W}x{H})")
    print("-"*86)
    print(f"{'Metric':39s}  {'Value':>14s}  {'Status':>12s}")
    print("-"*86)
    for m, v, e in rows:
        print(f"{m:39s}  {v:>14s}  {e:>12s}")
    print("-"*86)
    print(f"fx={M['fx']:.1f}, fy={M['fy']:.1f}, cx={M['cx']:.1f}, cy={M['cy']:.1f}")
    print("")

def save_npz(out_path, K, dist, image_size, kept_files, flags, model_label):
    np.savez(out_path, K=K, dist=dist, image_size=np.array(image_size, np.int32),
             kept_files=np.array(kept_files), flags=np.array([flags]), model=np.array([model_label]))

def calibrate_core(objpoints, imgpoints, image_size, flags):
    rms, K, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None, flags=flags)
    return rms, K, dist

def calibrate_one(folder: Path, side_name: str, basename: str, TOP: int = 15):
    cols, rows = CHECKERBOARD
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
        ok, corners = detect_board(gray, cols, rows)
        if not ok:
            continue
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                        (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        accepted_files.append(p)
        sharpness.append(cv.Laplacian(gray, cv.CV_64F).var())
        image_size = gray.shape[::-1]

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

    # --------- Run 5-param (flags=0) ----------
    flags_five = 0
    rms5, K5, D5 = calibrate_core(objpoints, imgpoints, image_size, flags_five)
    M5 = metrics_from(K5, D5, rms5, image_size)
    print(f"[RESULT] {side_name} (5-param): used {len(objpoints)} / {n_accept} accepted views  RMS={rms5:.3f}px")
    pretty_report(side_name, "5-param", M5)
    out5 = out_dir / f"{basename}_five_param.npz"
    save_npz(out5, K5, D5, image_size, kept_files, flags_five, "five_param")
    print(f"[INFO]  {side_name}: saved {out5}")

    # --------- Run Rational (flags=CALIB_RATIONAL_MODEL) ----------
    flags_rat = cv.CALIB_RATIONAL_MODEL
    rmsR, KR, DR = calibrate_core(objpoints, imgpoints, image_size, flags_rat)
    MR = metrics_from(KR, DR, rmsR, image_size)
    print(f"[RESULT] {side_name} (Rational): used {len(objpoints)} / {n_accept} accepted views  RMS={rmsR:.3f}px")
    pretty_report(side_name, "Rational", MR)
    outr = out_dir / f"{basename}_rational.npz"
    save_npz(outr, KR, DR, image_size, kept_files, flags_rat, "rational")
    print(f"[INFO]  {side_name}: saved {outr}")

    # --------- Compact side-by-side summary ----------
    def status_emoji(s): return {"GOOD":"✅","WARNING":"⚠️","FAIL":"❌"}.get(s, "")
    print(f"\n{side_name} — Model Comparison Summary (image {image_size[0]}x{image_size[1]})")
    print("-"*86)
    print(f"{'Metric':39s}  {'5-param':>18s}  {'Rational':>18s}")
    print("-"*86)
    rows = [
        ("RMS reprojection error [px]",
         (f"{M5['rms']:.3f}", _status("RMS reprojection error [px]", M5['rms'])),
         (f"{MR['rms']:.3f}", _status("RMS reprojection error [px]", MR['rms']))),
        ("fx/fy ratio (≈1)",
         (f"{M5['ratio']:.3f}", _status("fx/fy ratio (≈1)", M5['ratio'])),
         (f"{MR['ratio']:.3f}", _status("fx/fy ratio (≈1)", MR['ratio']))),
        ("center offset [px]",
         (f"{M5['center_offset']:.1f}", _status("center offset [px]", M5['center_offset'])),
         (f"{MR['center_offset']:.1f}", _status("center offset [px]", MR['center_offset']))),
        ("max |distortion|",
         (f"{M5['dmax']:.2f}", _status("max |distortion|", M5['dmax'])),
         (f"{MR['dmax']:.2f}", _status("max |distortion|", MR['dmax']))),
    ]
    for name, (v5, s5), (vr, sr) in rows:
        print(f"{name:39s}  {v5:>9s} {status_emoji(s5):>3s}      {vr:>9s} {status_emoji(sr):>3s}")
    print("-"*86)

    # --------- Simple recommendation ----------
    rec = "5-param"
    reason = "lower complexity and stable distortion"
    # Prefer rational only if it materially improves RMS AND keeps distortion sane
    if (MR["rms"] + 1e-4) < (M5["rms"] - 0.2) and MR["dmax"] < 5.0 and MR["center_offset"] <= 60:
        rec = "Rational"
        reason = "meaningfully lower RMS with sane coefficients"
    print(f"[CHOICE] {side_name}: Recommend **{rec}** ({reason}).\n")

    return (M5, out5), (MR, outr)

# ----------------------------

if __name__ == "__main__":
    print("[INFO] Simple intrinsics (orientation-robust)")
    print(f"[INFO] Athlete/Session: {ATHLETE} {SESSION}")
    print_criteria()

    # LEFT
    (M5_L, out5_L), (MR_L, outr_L) = calibrate_one(left_dir,  "LEFT",  "intrinsics_left")

    # RIGHT
    (M5_R, out5_R), (MR_R, outr_R) = calibrate_one(right_dir, "RIGHT", "intrinsics_right")

    print("[INFO] Done. Intrinsics written to:", out_dir)
