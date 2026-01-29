import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import yaml

#TODO add kalman filter and smoothing 
# add way to see blocked or missing joints better 

# ========================================
# Config
# ========================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
config_path = PROJECT_ROOT / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
paths_cfg = cfg.get("paths", {})

def cfg_path(key: str) -> Path:
    try:
        template = paths_cfg[key]
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' in project_config.yaml paths") from exc
    return PROJECT_ROOT / Path(template.format(athlete=ATHLETE, session=SESSION))

# ========================================
# Paths
# ========================================
calib_path = cfg_path("stereo_calibration") / "stereo_calib.npz"
kps_dir    = cfg_path("keypoints_2d")
out_dir    = cfg_path("keypoints_3d")
out_dir.mkdir(parents=True, exist_ok=True)

# ========================================
# Load calibration
# ========================================
calib = np.load(calib_path)
K1, D1 = calib["K1"],   calib["dist1"]
K2, D2 = calib["K2"],   calib["dist2"]
R,  T  = calib["R"],    calib["T"].reshape(3,1)

# Pixel-space projection matrices
P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K2 @ np.hstack([R, T])

# Infer frame size (fallback to 640x640 if not saved)
if "image_size" in calib.files:
    sz = calib["image_size"]
    FRAME_W, FRAME_H = int(sz[0]), int(sz[1])
else:
    FRAME_W = FRAME_H = 640

def to_pixels(x, y):
    # Heuristic: if looks like normalized coords, scale to pixels
    if 0.0 <= x <= 2.0 and 0.0 <= y <= 2.0:
        return x * FRAME_W, y * FRAME_H
    return x, y

# ========================================
# Landmarks
# ========================================
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# ========================================
# Pairing helpers (single folder)
# ========================================
def base_name(stem: str) -> str:
    for suf in ("_left_2d", "_right_2d", "_left", "_right"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem

def find_pairs(kdir: Path):
    lefts, rights = {}, {}
    for f in kdir.glob("*.csv"):
        s = f.stem
        if s.endswith(("_left_2d", "_left")):
            lefts[base_name(s)] = f
        elif s.endswith(("_right_2d", "_right")):
            rights[base_name(s)] = f
    bases = sorted(set(lefts.keys()) & set(rights.keys()))
    for b in bases:
        yield b, lefts[b], rights[b]

# ========================================
# Triangulation (with quick QC)
# ========================================
def triangulate_clip(left_csv: Path, right_csv: Path, out_csv: Path):
    dfL = pd.read_csv(left_csv)
    dfR = pd.read_csv(right_csv)

    tri_rows = []
    repro_errs_L = []
    repro_errs_R = []
    neg_z = 0
    total_ok = 0

    for idx in range(min(len(dfL), len(dfR))):
        row = [idx]
        for name in landmark_names:
            lx, ly = dfL.loc[idx, f"{name}_x"], dfL.loc[idx, f"{name}_y"]
            rx, ry = dfR.loc[idx, f"{name}_x"], dfR.loc[idx, f"{name}_y"]

            if -1 in (lx, ly, rx, ry):
                row += [-1, -1, -1]
                continue

            # Ensure pixel coords
            lx, ly = to_pixels(lx, ly)
            rx, ry = to_pixels(rx, ry)

            # Ideal pixel coords (undistorted but still in pixels)
            ptL = np.array([[[lx, ly]]], dtype=np.float32)
            ptR = np.array([[[rx, ry]]], dtype=np.float32)
            uL = cv2.undistortPoints(ptL, K1, D1, P=K1).reshape(2,1)  # 2x1
            uR = cv2.undistortPoints(ptR, K2, D2, P=K2).reshape(2,1)

            # Triangulate in pixel space with P1,P2
            Xh = cv2.triangulatePoints(P1, P2, uL, uR)   # 4x1
            X  = (Xh[:3] / Xh[3]).reshape(3)             # 3D in cam1 coords

            row += [float(X[0]), float(X[1]), float(X[2])]

            # QC: negative depth count
            if X[2] <= 0:
                neg_z += 1
            else:
                total_ok += 1

            # QC: reprojection error vs ideal pixels
            # Left: uL_pred = project K1 * X
            XL = K1 @ X.reshape(3,1)
            uL_pred = (XL[:2] / XL[2]).reshape(2)
            errL = float(np.linalg.norm(uL_pred - uL.reshape(2)))
            repro_errs_L.append(errL)

            # Right: K2 * (R X + T)
            XR = K2 @ (R @ X.reshape(3,1) + T)
            uR_pred = (XR[:2] / XR[2]).reshape(2)
            errR = float(np.linalg.norm(uR_pred - uR.reshape(2)))
            repro_errs_R.append(errR)

        tri_rows.append(row)

    # Save
    cols = ["frame"] + [f"{n}_{ax}" for n in landmark_names for ax in ("x","y","z")]
    pd.DataFrame(tri_rows, columns=cols).to_csv(out_csv, index=False)
    print(f"✅ Saved 3D keypoints: {out_csv.name}")

    # QC summary
    if repro_errs_L and repro_errs_R:
        mL = np.mean(repro_errs_L); mR = np.mean(repro_errs_R)
        frac_neg = neg_z / max(1, (neg_z + total_ok))
        print(f"   ↳ mean repro err (px): left={mL:.2f}, right={mR:.2f} | Z≤0: {100*frac_neg:.1f}%")

# ========================================
# Run
# ========================================
pairs = list(find_pairs(kps_dir))
if not pairs:
    print(f"❌ No left/right 2D keypoint pairs found in {kps_dir}")
else:
    print(f"[INFO] Found {len(pairs)} clip(s) in {kps_dir}")
    for base, lf, rf in pairs:
        out_csv = out_dir / f"{base}_3d.csv"
        triangulate_clip(lf, rf, out_csv)
