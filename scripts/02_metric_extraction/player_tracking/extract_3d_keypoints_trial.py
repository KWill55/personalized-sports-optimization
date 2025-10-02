#!/usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import yaml
from typing import Dict, Tuple, Optional

# ===========================================================
# USER TUNABLES (tiny toggles for fast triage)
# ===========================================================
# If your 2D CSVs are in [0,1] normalized image coordinates, set True.
# If they are already pixel coordinates (0..W-1, 0..H-1), set False.
INPUT_IS_NORMALIZED = True

# If your detector ran on undistorted/rectified images and outputs *already-undistorted* points,
# set True to SKIP cv2.undistortPoints(). (Common if you rectified then detected.)
POINTS_ARE_ALREADY_UNDISTORTED = False

# Enable/disable features to isolate issues quickly
USE_ONE_VIEW_FALLBACK = True
USE_KALMAN = True
DEBUG_PRINTS = False  # quick prints on first frame

# Confidence threshold(s)
CONF_THRESH = 0.50
# Optional stricter gating for left joints on the left camera (if that view is flaky for those joints)
LEFT_JOINTS_STRICTER = True
LEFT_JOINTS_CONF_L = 0.65  # used only when LEFT_JOINTS_STRICTER is True

# Reprojection gate (px, avg of L/R)
REPRO_GATE_PX = 6.0  # loosen slightly while stabilizing

# One-view behavior
MEAS_SIGMA_TWO_VIEW = 6.0          # baseline meas noise for KF (px)
MEAS_SIGMA_ONE_VIEW = 30.0         # noisier one-view updates so KF trusts prediction
ONE_VIEW_MAX_STEP_M = 0.20         # clamp per-frame motion during one-view fallback (meters)

# ===========================================================
# Config / Paths
# ===========================================================
config_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ATHLETE = cfg["athlete"]
SESSION = cfg["session"]
FPS = float(cfg.get("player_tracking_fps", 30))

base_dir    = Path(__file__).resolve().parents[3]
session_dir = base_dir / "data" / ATHLETE / SESSION

calib_path  = session_dir / "calibration" / "stereo_calibration" / "stereo_calib.npz"
kps_dir     = session_dir / "metrics" / "2d_keypoints"
out_dir     = session_dir / "metrics" / "3d_keypoints"
out_dir.mkdir(parents=True, exist_ok=True)

# ===========================================================
# Load stereo calib
# ===========================================================
calib = np.load(calib_path)
K1, D1 = calib["K1"],   calib["dist1"]
K2, D2 = calib["K2"],   calib["dist2"]
R,  T  = calib["R"],    calib["T"].reshape(3,1)

# Frame size (be skeptical about ordering in npz!)
FRAME_W, FRAME_H = 640, 640
if "image_size" in calib.files:
    sz = calib["image_size"]
    # First try (W,H); if your prints look bogus, swap to (H,W).
    FRAME_W, FRAME_H = int(sz[0]), int(sz[1])
    if DEBUG_PRINTS:
        print(f"[INFO] image_size from calib: {sz} -> using (W,H)=({FRAME_W},{FRAME_H})")

def to_pixels_xy(x, y):
    """Convert CSV coords to pixels based on INPUT_IS_NORMALIZED."""
    if INPUT_IS_NORMALIZED:
        return float(x) * FRAME_W, float(y) * FRAME_H
    else:
        return float(x), float(y)

# ===========================================================
# MediaPipe 33 names (+ simple parent map for legs)
# ===========================================================
landmark_names = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

PARENT = {
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_heel": "left_ankle",
    "left_foot_index": "left_ankle",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_heel": "right_ankle",
    "right_foot_index": "right_ankle",
}

# ===========================================================
# Pairing helpers
# ===========================================================
def base_name(stem: str) -> str:
    for suf in ("_left_2d","_right_2d","_left","_right"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem

def find_pairs(kdir: Path):
    lefts, rights = {}, {}
    for f in kdir.glob("*.csv"):
        s = f.stem
        if s.endswith(("_left_2d","_left")):
            lefts[base_name(s)] = f
        elif s.endswith(("_right_2d","_right")):
            rights[base_name(s)] = f
    bases = sorted(set(lefts) & set(rights))
    for b in bases:
        yield b, lefts[b], rights[b]

# ===========================================================
# Confidence / validity / geometry helpers
# ===========================================================
def read_conf_series(df: pd.DataFrame, name: str, default=1.0) -> pd.Series:
    for col in (f"{name}_conf", f"{name}_confidence", f"{name}_visibility"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(default, index=df.index, dtype=float)

def in_bounds(x, y, w, h, m=2):
    return (0-m) <= x <= (w+m) and (0-m) <= y <= (h+m)

# --------- Mode A: normalized rays + bare [R|t] (recommended) ----------
P1n = np.hstack([np.eye(3), np.zeros((3,1))])  # no K
P2n = np.hstack([R, T])

def undist_to_norm(pt_xy, K, D):
    """Return normalized coordinates (no K multiplication)."""
    if POINTS_ARE_ALREADY_UNDISTORTED:
        # If points are already normalized (rare), then convert pixels->norm manually:
        x, y = pt_xy
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        return np.array([[(x - cx)/fx], [(y - cy)/fy]], dtype=np.float32)
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    u = cv2.undistortPoints(pt, K, D)  # normalized coords (1,1,2)
    return u.reshape(2,1)

def triangulate_two_view_norm(uL: np.ndarray, uR: np.ndarray) -> np.ndarray:
    Xh = cv2.triangulatePoints(P1n, P2n, uL, uR)  # 4x1
    X  = (Xh[:3] / Xh[3]).reshape(3)
    return X

# Reprojection in pixel space for gating
def reproj_err_left_px(K1, X, uL_px):
    XL = K1 @ X.reshape(3,1)
    u_pred = (XL[:2] / XL[2]).reshape(2)
    return float(np.linalg.norm(u_pred - uL_px.reshape(2)))

def reproj_err_right_px(K2, R, T, X, uR_px):
    XR = K2 @ (R @ X.reshape(3,1) + T)
    u_pred = (XR[:2] / XR[2]).reshape(2)
    return float(np.linalg.norm(u_pred - uR_px.reshape(2)))

def backproject_ray(K, Rcw, tcw, uv_px: np.ndarray):
    """Back-project a pixel to a world-space ray (left camera is world)."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    xn = np.array([(uv_px[0]-cx)/fx, (uv_px[1]-cy)/fy, 1.0], dtype=float)
    Rc2w = Rcw.T
    C = (-Rc2w @ tcw).reshape(3)
    r = Rc2w @ xn
    n = np.linalg.norm(r)
    if n > 0: r /= n
    return C, r

def choose_depth_along_ray(C, r,
                           parent_prev: Optional[np.ndarray],
                           nominal_len: Optional[float],
                           prev_X: Optional[np.ndarray]) -> np.ndarray:
    """Prefer bone-length to parent; else keep previous depth; clamp step."""
    # Try to satisfy ||(C + t r) - parent|| = nominal_len
    if parent_prev is not None and nominal_len is not None and nominal_len > 0:
        p = parent_prev.reshape(3)
        a = float(np.dot(r, r))
        b = float(2.0 * np.dot(r, C - p))
        c = float(np.dot(C - p, C - p) - nominal_len**2)
        disc = b*b - 4*a*c
        if disc >= 0:
            t1 = (-b + np.sqrt(disc)) / (2*a)
            t2 = (-b - np.sqrt(disc)) / (2*a)
            candidates = [t for t in (t1, t2) if t > 0]
            if candidates:
                if prev_X is not None:
                    tprev = float(np.dot(prev_X - C, r))
                    t = min(candidates, key=lambda ti: abs(ti - tprev))
                else:
                    t = min(candidates, key=abs)
                X = C + t * r
                # clamp step if we have prev
                if prev_X is not None:
                    d = X - prev_X
                    n = np.linalg.norm(d)
                    if n > ONE_VIEW_MAX_STEP_M:
                        X = prev_X + d * (ONE_VIEW_MAX_STEP_M / max(1e-9, n))
                return X
    # Otherwise anchor at previous depth if available
    if prev_X is not None:
        tprev = float(np.dot(prev_X - C, r))
        if tprev > 0:
            X = C + tprev * r
            # clamp step
            d = X - prev_X
            n = np.linalg.norm(d)
            if n > ONE_VIEW_MAX_STEP_M:
                X = prev_X + d * (ONE_VIEW_MAX_STEP_M / max(1e-9, n))
            return X
    # Last resort: fixed 2 m out
    return C + 2.0 * r

# ===========================================================
# Lightweight 3D constant-velocity Kalman filter
# ===========================================================
class CVKalman3D:
    def __init__(self, dt=1/60, process_sigma=1e-2, meas_sigma=8.0):
        self.dt = float(dt)

        self.F = np.eye(6)
        self.F[0,3] = self.dt
        self.F[1,4] = self.dt
        self.F[2,5] = self.dt

        self.H = np.zeros((3,6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0

        q = float(process_sigma)
        self.Q = (q**2) * np.array([
            [self.dt**4/4, 0, 0, self.dt**3/2, 0, 0],
            [0, self.dt**4/4, 0, 0, self.dt**3/2, 0],
            [0, 0, self.dt**4/4, 0, 0, self.dt**3/2],
            [self.dt**3/2, 0, 0, self.dt**2, 0, 0],
            [0, self.dt**3/2, 0, 0, self.dt**2, 0],
            [0, 0, self.dt**3/2, 0, 0, self.dt**2],
        ])

        r = float(meas_sigma)
        self.R_base = (r**2) * np.eye(3)

        self.x = None   # 6x1
        self.P = None   # 6x6

    def reset(self):
        self.x, self.P = None, None

    def predict(self):
        if self.x is None: return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: Optional[np.ndarray], meas_sigma_scale=1.0):
        if z is None:
            return
        R = (meas_sigma_scale**2) * self.R_base
        if self.x is None:
            self.x = np.zeros((6,1))
            self.x[:3,0] = z.reshape(3)
            self.P = np.eye(6) * 1e2
            return
        y = z.reshape(3,1) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def current_pos(self) -> Optional[np.ndarray]:
        if self.x is None: return None
        return self.x[:3,0].copy()

# ===========================================================
# Triangulation per clip
# ===========================================================
def triangulate_clip(left_csv: Path, right_csv: Path, out_csv: Path):
    dfL = pd.read_csv(left_csv)
    dfR = pd.read_csv(right_csv)

    confL: Dict[str,pd.Series] = {n: read_conf_series(dfL, n, default=1.0) for n in landmark_names}
    confR: Dict[str,pd.Series] = {n: read_conf_series(dfR, n, default=1.0) for n in landmark_names}

    # temporal states
    prev_3d: Dict[str, Optional[np.ndarray]] = {n: None for n in landmark_names}
    kf: Dict[str, CVKalman3D] = {
        n: CVKalman3D(dt=1/max(1.0, FPS), process_sigma=1e-2, meas_sigma=MEAS_SIGMA_TWO_VIEW)
        for n in landmark_names
    }
    nominal_len: Dict[Tuple[str,str], float] = {}

    # world poses (left-cam world)
    Rcw1, tcw1 = np.eye(3), np.zeros((3,1))
    Rcw2, tcw2 = R, T

    tri_rows = []
    repro_errs_L, repro_errs_R = [], []
    neg_z = 0; total_ok = 0

    nframes = min(len(dfL), len(dfR))
    for idx in range(nframes):
        row = [idx]

        for name in landmark_names:
            lx, ly = dfL.at[idx, f"{name}_x"], dfL.at[idx, f"{name}_y"]
            rx, ry = dfR.at[idx, f"{name}_x"], dfR.at[idx, f"{name}_y"]

            badL = (lx == -1) or (ly == -1)
            badR = (rx == -1) or (ry == -1)

            if not badL:
                lx, ly = to_pixels_xy(lx, ly)
            if not badR:
                rx, ry = to_pixels_xy(rx, ry)

            cL = float(confL[name].iat[idx]) if not badL else 0.0
            cR = float(confR[name].iat[idx]) if not badR else 0.0

            useL = (not badL) and (cL >= CONF_THRESH) and in_bounds(lx, ly, FRAME_W, FRAME_H)
            useR = (not badR) and (cR >= CONF_THRESH) and in_bounds(rx, ry, FRAME_W, FRAME_H)

            # Optional stricter left-view gating for left-* joints
            if LEFT_JOINTS_STRICTER and name.startswith("left_") and useL:
                useL = (cL >= LEFT_JOINTS_CONF_L) and useL

            # Debug once
            if DEBUG_PRINTS and idx == 0 and name in ("left_hip","left_knee","left_ankle"):
                print(f"[DBG] {name} L(px)=({lx:.1f},{ly:.1f}) useL={useL} cL={cL:.2f} | "
                      f"R(px)=({rx:.1f},{ry:.1f}) useR={useR} cR={cR:.2f}")

            # Kalman predict
            if USE_KALMAN:
                kf[name].predict()

            X_meas = None
            meas_sigma_scale = 1.0
            used_two_view = False

            # ---------- Two-view path ----------
            if useL and useR:
                # normalized inputs for triangulation (Mode A)
                uL_norm = undist_to_norm((lx, ly), K1, D1)
                uR_norm = undist_to_norm((rx, ry), K2, D2)

                X = triangulate_two_view_norm(uL_norm, uR_norm)

                # reprojection gate in pixel space
                eL = reproj_err_left_px(K1, X, np.array([lx, ly]))
                eR = reproj_err_right_px(K2, R, T, X, np.array([rx, ry]))
                if np.isfinite(eL) and np.isfinite(eR) and (eL + eR)*0.5 <= REPRO_GATE_PX and X[2] > 0:
                    X_meas = X
                    used_two_view = True
                    repro_errs_L.append(eL); repro_errs_R.append(eR)
                    meas_sigma_scale = 1.0  # baseline
                    # Learn nominal bone lengths (parent->child) on clean frames
                    parent = PARENT.get(name)
                    if parent is not None and prev_3d.get(parent) is not None:
                        L = float(np.linalg.norm(X_meas - prev_3d[parent]))
                        if L > 1e-6:
                            nominal_len[(parent, name)] = L

            # ---------- One-view fallback ----------
            if X_meas is None:
                if USE_ONE_VIEW_FALLBACK and (useL or useR):
                    if useL:
                        C, r = backproject_ray(K1, Rcw1, tcw1, np.array([lx, ly], dtype=float))
                    else:
                        C, r = backproject_ray(K2, Rcw2, tcw2, np.array([rx, ry], dtype=float))
                    parent = PARENT.get(name)
                    parent_prev = prev_3d.get(parent) if parent is not None else None
                    L_nom = nominal_len.get((parent, name), None) if parent is not None else None
                    prevX = kf[name].current_pos() if (USE_KALMAN and kf[name].current_pos() is not None) else prev_3d.get(name)
                    X_est = choose_depth_along_ray(C, r, parent_prev, L_nom, prevX)
                    X_meas = X_est
                    meas_sigma_scale = MEAS_SIGMA_ONE_VIEW / MEAS_SIGMA_TWO_VIEW
                else:
                    # no measurement this frame
                    if USE_KALMAN:
                        X_pred = kf[name].current_pos()
                        if X_pred is None:
                            row += [-1, -1, -1]
                            continue
                        Xf = X_pred
                        row += [float(Xf[0]), float(Xf[1]), float(Xf[2])]
                        prev_3d[name] = Xf
                        if Xf[2] <= 0: neg_z += 1
                        else: total_ok += 1
                        continue
                    else:
                        row += [-1, -1, -1]
                        continue

            # ---------- Fuse (or pass-through) ----------
            if USE_KALMAN:
                kf[name].update(X_meas.reshape(3), meas_sigma_scale=meas_sigma_scale)
                Xf = kf[name].current_pos()
                if Xf is None:
                    row += [-1, -1, -1]
                    continue
                X_out = Xf
            else:
                X_out = X_meas

            row += [float(X_out[0]), float(X_out[1]), float(X_out[2])]
            prev_3d[name] = X_out

            if X_out[2] <= 0: neg_z += 1
            else: total_ok += 1

        tri_rows.append(row)

    cols = ["frame"] + [f"{n}_{ax}" for n in landmark_names for ax in ("x","y","z")]
    pd.DataFrame(tri_rows, columns=cols).to_csv(out_csv, index=False)
    print(f"✅ Saved 3D keypoints: {out_csv.name}")

    if repro_errs_L and repro_errs_R:
        mL, mR = np.mean(repro_errs_L), np.mean(repro_errs_R)
        frac_neg = neg_z / max(1, (neg_z + total_ok))
        print(f"   ↳ mean repro err (px): left={mL:.2f}, right={mR:.2f} | Z≤0: {100*frac_neg:.1f}%")

# ===========================================================
# Run over pairs
# ===========================================================
def main():
    pairs = list(find_pairs(kps_dir))
    if not pairs:
        print(f"❌ No left/right 2D keypoint pairs found in {kps_dir}")
        return
    print(f"[INFO] Found {len(pairs)} clip(s) in {kps_dir}")
    for base, lf, rf in pairs:
        out_csv = out_dir / f"{base}_3d.csv"
        triangulate_clip(lf, rf, out_csv)

if __name__ == "__main__":
    main()
