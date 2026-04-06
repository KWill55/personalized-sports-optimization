"""
Title: inspect_calibration.py

Purpose:
    Display calibration parameters from stereo_calib.npz and validate visually:
    - Undistortion preview
    - Rectification with epipolar lines
    - Optional disparity map for depth check
    - NEW: Intrinsics/Extrinsics quality assessment (OK/WARN/BAD)

Inputs:
    - stereo_calib.npz (calibration results)
    - Combined checkerboard images (pair_XX.png)

Usage:
    - Run script to print parameters + assessment
    - GUI shows undistortion, rectification, and optional disparity map
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import glob
import yaml

# ====================== Helpers ======================
def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "project_config.yaml").exists():
            return parent
    raise FileNotFoundError("project_config.yaml not found")


# ====================== Config ======================
PROJECT_ROOT = find_repo_root()
config_path = PROJECT_ROOT / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
SHOW_DISPARITY = True     # Set to False to skip disparity preview
NUM_PREVIEW_IMAGES = 0    # How many image pairs to visualize/assess

# ====================== Paths =======================
base_dir = PROJECT_ROOT
session_dir = base_dir / "data" / ATHLETE / SESSION
stereo_calib_dir = session_dir / "calibration" / "stereo_calibration"
calib_file = stereo_calib_dir / "stereo_calib.npz"
if not calib_file.exists():
    fallback = stereo_calib_dir / "stereo_calib_manual.npz"
    calib_file = fallback
image_dir = session_dir / "calibration" / "calib_images"

# ================== Load Calibration =================
calib = np.load(calib_file)
K1, K2 = calib["K1"], calib["K2"]
dist1, dist2 = calib["dist1"], calib["dist2"]
R, T = calib["R"], calib["T"]
P1, P2 = calib["P1"], calib["P2"]
E, F = calib["E"], calib["F"]

np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print(f"Stereo Calibration Data: {calib_file.name}")
print("=" * 60)

print("\nSummary of Calibration Arrays:")
print(f"{'Key':<20}{'Shape':<15}{'Dtype'}")
print("-" * 60)
for key in calib.files:
    arr = calib[key]
    print(f"{key:<20}{str(arr.shape):<15}{arr.dtype}")

print("\nDetailed Matrices:")
for key in calib.files:
    print(f"\n--- {key} ---\n{calib[key]}")

# ================== Helper functions =================
def _classify(v, ok, warn):
    """Return 'OK'/'WARN'/'BAD' given thresholds (smaller is better)."""
    return "OK" if v <= ok else ("WARN" if v <= warn else "BAD")

def _reproj_rmse(K, D, rvec, tvec, obj3d, img2d):
    proj, _ = cv.projectPoints(obj3d, rvec, tvec, K, D)
    e = proj.reshape(-1,2) - img2d.reshape(-1,2)
    return float(np.sqrt(np.mean(np.sum(e*e, axis=1))))

def _sampson_errors(F, pts1, pts2):
    """Sampson epipolar error per point (px^2)."""
    x1 = np.concatenate([pts1.reshape(-1,2), np.ones((pts1.shape[0],1))], axis=1)  # Nx3
    x2 = np.concatenate([pts2.reshape(-1,2), np.ones((pts2.shape[0],1))], axis=1)  # Nx3
    Fx1  = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    num = np.sum(x2 * (F @ x1.T).T, axis=1)**2
    den = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2
    return num / np.maximum(den, 1e-12)

def assess_calibration(K1, dist1, K2, dist2, R, T, F,
                       image_paths, checkerboard_size, square_size_in,
                       image_size, max_pairs=5):
    """
    Probe mono & stereo quality on a few checkerboard pairs and print a verdict.
    Returns a dict of metrics.
    """
    cols, rows = map(int, checkerboard_size)
    # 3D board points (units: cm to match your calibration)
    objp = np.zeros((cols*rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_in)

    # Intrinsic heuristics
    w, h = image_size
    cx1, cy1 = float(K1[0,2]), float(K1[1,2])
    cx2, cy2 = float(K2[0,2]), float(K2[1,2])
    fx1, fy1 = float(K1[0,0]), float(K1[1,1])
    fx2, fy2 = float(K2[0,0]), float(K2[1,1])
    fxfy1 = fx1 / max(fy1, 1e-9)
    fxfy2 = fx2 / max(fy2, 1e-9)
    c1_off = float(np.hypot(cx1 - w/2.0, cy1 - h/2.0))
    c2_off = float(np.hypot(cx2 - w/2.0, cy2 - h/2.0))

    # Rectification maps for vertical disparity test
    R1r, R2r, P1r, P2r, _, _, _ = cv.stereoRectify(K1, dist1, K2, dist2, (w, h), R, T,
                                                   flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    pnp_rmse_L, pnp_rmse_R = [], []
    vert_rms, sampson_med = [], []

    for img_path in image_paths[:max_pairs]:
        im = cv.imread(img_path)
        if im is None or im.shape[1] < 2*w:  # need combined 1280x640
            continue
        L = im[:, :w]
        Rimg = im[:, w:]

        gL = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
        gR = cv.cvtColor(Rimg, cv.COLOR_BGR2GRAY)
        okL, ptsL = cv.findChessboardCornersSB(gL, (cols, rows), flags=cv.CALIB_CB_EXHAUSTIVE)
        okR, ptsR = cv.findChessboardCornersSB(gR, (cols, rows), flags=cv.CALIB_CB_EXHAUSTIVE)
        if not (okL and okR):
            continue

        crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cv.cornerSubPix(gL, ptsL, (11,11), (-1,-1), crit)
        cv.cornerSubPix(gR, ptsR, (11,11), (-1,-1), crit)

        # Mono PnP reprojection RMSE (intrinsics sanity)
        ok1, r1, t1, _ = cv.solvePnPRansac(objp, ptsL, K1, dist1, flags=cv.SOLVEPNP_ITERATIVE)
        ok2, r2, t2, _ = cv.solvePnPRansac(objp, ptsR, K2, dist2, flags=cv.SOLVEPNP_ITERATIVE)
        if ok1: pnp_rmse_L.append(_reproj_rmse(K1, dist1, r1, t1, objp, ptsL))
        if ok2: pnp_rmse_R.append(_reproj_rmse(K2, dist2, r2, t2, objp, ptsR))

        # Sampson epipolar error (px)
        se = _sampson_errors(F, ptsL, ptsR)
        sampson_med.append(float(np.sqrt(np.median(se))))

        # Vertical disparity after rectification
        pLr = cv.undistortPoints(ptsL, K1, dist1, R=R1r, P=P1r).reshape(-1,2)
        pRr = cv.undistortPoints(ptsR, K2, dist2, R=R2r, P=P2r).reshape(-1,2)
        dy = pLr[:,1] - pRr[:,1]
        vert_rms.append(float(np.sqrt(np.mean(dy*dy))))

    metrics = {
        "fxfy_L": fxfy1, "fxfy_R": fxfy2,
        "c_off_L_px": c1_off, "c_off_R_px": c2_off,
        "pnp_rmse_L_px": None if not pnp_rmse_L else float(np.median(pnp_rmse_L)),
        "pnp_rmse_R_px": None if not pnp_rmse_R else float(np.median(pnp_rmse_R)),
        "rect_vert_rms_px": None if not vert_rms else float(np.median(vert_rms)),
        "sampson_med_px": None if not sampson_med else float(np.median(sampson_med)),
        "baseline_norm_units": float(np.linalg.norm(T)),
    }

    print("\n================ Calibration Assessment ================")
    print("Intrinsics:")
    print(f"  L fx/fy = {fxfy1:.3f}   -> {_classify(abs(fxfy1-1.0), 0.03, 0.10)} (target ≈1.00)")
    print(f"  R fx/fy = {fxfy2:.3f}   -> {_classify(abs(fxfy2-1.0), 0.03, 0.10)} (target ≈1.00)")
    print(f"  L principal center offset = {c1_off:.1f}px -> {_classify(c1_off, 15, 60)} (≤15px OK)")
    print(f"  R principal center offset = {c2_off:.1f}px -> {_classify(c2_off, 15, 60)} (≤15px OK)")
    if metrics['pnp_rmse_L_px'] is not None:
        print(f"  L PnP RMSE = {metrics['pnp_rmse_L_px']:.2f}px -> {_classify(metrics['pnp_rmse_L_px'], 1.0, 2.0)}")
    if metrics['pnp_rmse_R_px'] is not None:
        print(f"  R PnP RMSE = {metrics['pnp_rmse_R_px']:.2f}px -> {_classify(metrics['pnp_rmse_R_px'], 1.0, 2.0)}")

    print("\nExtrinsics:")
    if metrics['rect_vert_rms_px'] is not None:
        print(f"  Rectified vertical RMS = {metrics['rect_vert_rms_px']:.2f}px -> {_classify(metrics['rect_vert_rms_px'], 0.5, 2.0)} (≤0.5px OK)")
    if metrics['sampson_med_px'] is not None:
        print(f"  Sampson epipolar error = {metrics['sampson_med_px']:.2f}px -> {_classify(metrics['sampson_med_px'], 0.5, 1.5)}")
    print(f"  Baseline ||T|| (units of your T) = {metrics['baseline_norm_units']:.3f}")
    print("========================================================\n")

    return metrics

# =============== Load sample images =================
combined_images = sorted(glob.glob(str(image_dir / "pair_*.png")))
if not combined_images:
    print("[ERROR] No calibration images found for preview.")
    raise SystemExit(1)
combined_images = combined_images[:NUM_PREVIEW_IMAGES]
print(f"\n[INFO] Showing visual validation for {len(combined_images)} pairs...")

# ============== Rectification transforms =============
image_size = (640, 640)  # From capture scripts
R1, R2, P1_rect, P2_rect, Q, _, _ = cv.stereoRectify(K1, dist1, K2, dist2, image_size, R, T)

# Precompute undistortion/rectification maps
map1_L, map2_L = cv.initUndistortRectifyMap(K1, dist1, R1, P1_rect, image_size, cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(K2, dist2, R2, P2_rect, image_size, cv.CV_16SC2)

# ============== NEW: Quality assessment ==============
CHECKERBOARD_SIZE = tuple(cfg["inner_corners"])
square_size_in = cfg["square_size_in"]
_ = assess_calibration(
    K1, dist1, K2, dist2, R, T, F,
    combined_images, CHECKERBOARD_SIZE, square_size_in,
    image_size=image_size, max_pairs=NUM_PREVIEW_IMAGES
)

# ==================== Preview loop ===================
for img_path in combined_images:
    combined = cv.imread(img_path)
    if combined is None or combined.shape[1] < 1280:
        continue

    # Split into left and right
    imgL = combined[:, 0:640]
    imgR = combined[:, 640:1280]

    # Undistort
    undistL = cv.undistort(imgL, K1, dist1)
    undistR = cv.undistort(imgR, K2, dist2)

    # Rectify
    rectL = cv.remap(imgL, map1_L, map2_L, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, map1_R, map2_R, cv.INTER_LINEAR)

    # Draw epipolar lines on rectified pair
    for y in range(0, rectL.shape[0], 50):
        cv.line(rectL, (0, y), (rectL.shape[1], y), (0, 255, 0), 1)
        cv.line(rectR, (0, y), (rectR.shape[1], y), (0, 255, 0), 1)

    # Display windows
    cv.imshow("Original Left | Undistorted Left", np.hstack([imgL, undistL]))
    cv.imshow("Original Right | Undistorted Right", np.hstack([imgR, undistR]))
    cv.imshow("Rectified Pair", np.hstack([rectL, rectR]))

    if SHOW_DISPARITY:
        grayL = cv.cvtColor(rectL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)
        stereo = cv.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(grayL, grayR)
        disp_norm = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        cv.imshow("Disparity Map", disp_norm)

    key = cv.waitKey(0)
    if key == 27:  # ESC
        break

cv.destroyAllWindows()
print("[INFO] Visual validation complete.")
