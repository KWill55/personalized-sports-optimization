"""3D pose reconstruction"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from utils.io_utils import PROJECT_ROOT

LANDMARK_NAMES: list[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

IDX = {n: i for i, n in enumerate(LANDMARK_NAMES)}

ANGLE_SPECS: dict[str, tuple[str, str, str]] = {
    "elbow_flex_l": ("left_shoulder", "left_elbow", "left_wrist"),
    "elbow_flex_r": ("right_shoulder", "right_elbow", "right_wrist"),
    "shoulder_flex_l": ("left_hip", "left_shoulder", "left_elbow"),
    "shoulder_flex_r": ("right_hip", "right_shoulder", "right_elbow"),
    "hip_flex_l": ("left_shoulder", "left_hip", "left_knee"),
    "hip_flex_r": ("right_shoulder", "right_hip", "right_knee"),
    "knee_flex_l": ("left_hip", "left_knee", "left_ankle"),
    "knee_flex_r": ("right_hip", "right_knee", "right_ankle"),
    "ankle_flex_l": ("left_knee", "left_ankle", "left_foot_index"),
    "ankle_flex_r": ("right_knee", "right_ankle", "right_foot_index"),
}


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _base_name(stem: str) -> str:
    for suffix in ("_left_2d", "_right_2d", "_left", "_right"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_2d_pairs(keypoints_2d_dir: Path) -> list[tuple[str, Path, Path]]:
    lefts: dict[str, Path] = {}
    rights: dict[str, Path] = {}

    for file in keypoints_2d_dir.glob("*.csv"):
        stem = file.stem
        if stem.endswith(("_left_2d", "_left")):
            lefts[_base_name(stem)] = file
        elif stem.endswith(("_right_2d", "_right")):
            rights[_base_name(stem)] = file

    bases = sorted(set(lefts.keys()) & set(rights.keys()))
    return [(b, lefts[b], rights[b]) for b in bases]


def _to_pixels(x: float, y: float, frame_w: int, frame_h: int) -> tuple[float, float]:
    if 0.0 <= x <= 2.0 and 0.0 <= y <= 2.0:
        return x * frame_w, y * frame_h
    return x, y


def _triangulate_clip(
    left_csv: Path,
    right_csv: Path,
    out_csv: Path,
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[int, float, float]:
    df_left = pd.read_csv(left_csv)
    df_right = pd.read_csv(right_csv)

    p1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = K2 @ np.hstack([R, T.reshape(3, 1)])

    repro_err_l: list[float] = []
    repro_err_r: list[float] = []
    tri_rows: list[list[float]] = []

    n_frames = min(len(df_left), len(df_right))

    for idx in range(n_frames):
        row: list[float] = [float(idx)]

        for name in LANDMARK_NAMES:
            lx, ly = float(df_left.loc[idx, f"{name}_x"]), float(df_left.loc[idx, f"{name}_y"])
            rx, ry = float(df_right.loc[idx, f"{name}_x"]), float(df_right.loc[idx, f"{name}_y"])

            if not np.isfinite([lx, ly, rx, ry]).all() or -1.0 in (lx, ly, rx, ry):
                row.extend([np.nan, np.nan, np.nan])
                continue

            lx, ly = _to_pixels(lx, ly, frame_w=frame_w, frame_h=frame_h)
            rx, ry = _to_pixels(rx, ry, frame_w=frame_w, frame_h=frame_h)

            pt_l = np.array([[[lx, ly]]], dtype=np.float32)
            pt_r = np.array([[[rx, ry]]], dtype=np.float32)

            u_l = cv2.undistortPoints(pt_l, K1, D1, P=K1).reshape(2, 1)
            u_r = cv2.undistortPoints(pt_r, K2, D2, P=K2).reshape(2, 1)

            xh = cv2.triangulatePoints(p1, p2, u_l, u_r)
            x = (xh[:3] / xh[3]).reshape(3)

            row.extend([float(x[0]), float(x[1]), float(x[2])])

            # reprojection QC
            xl = K1 @ x.reshape(3, 1)
            ul_pred = (xl[:2] / xl[2]).reshape(2)
            repro_err_l.append(float(np.linalg.norm(ul_pred - u_l.reshape(2))))

            xr = K2 @ (R @ x.reshape(3, 1) + T.reshape(3, 1))
            ur_pred = (xr[:2] / xr[2]).reshape(2)
            repro_err_r.append(float(np.linalg.norm(ur_pred - u_r.reshape(2))))

        tri_rows.append(row)

    columns = ["frame"] + [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y", "z")]
    pd.DataFrame(tri_rows, columns=columns).to_csv(out_csv, index=False)

    mean_l = float(np.mean(repro_err_l)) if repro_err_l else float("nan")
    mean_r = float(np.mean(repro_err_r)) if repro_err_r else float("nan")
    return n_frames, mean_l, mean_r


def run_pose_3d_pipeline(
    cfg: dict[str, Any],
    do_triangulation: bool = True,
    do_kinematics: bool = True,
) -> dict[str, Any]:
    calib_path = _format_path(cfg["paths"]["stereo_calibration"], cfg) / "stereo_calib.npz"
    keypoints_2d_dir = _format_path(cfg["paths"]["keypoints_2d"], cfg)
    keypoints_3d_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)

    keypoints_3d_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, Path, Path]] = []
    if do_triangulation:
        if not calib_path.exists():
            raise FileNotFoundError(f"Stereo calibration file not found: {calib_path}")

        pairs = _find_2d_pairs(keypoints_2d_dir)
        if not pairs:
            raise ValueError(f"No left/right 2D keypoint pairs found in: {keypoints_2d_dir}")

        calib = np.load(calib_path)
        K1, D1 = calib["K1"], calib["dist1"]
        K2, D2 = calib["K2"], calib["dist2"]
        R, T = calib["R"], calib["T"]

        if "image_size" in calib.files:
            image_size = calib["image_size"]
            frame_w, frame_h = int(image_size[0]), int(image_size[1])
        else:
            frame_w, frame_h = 640, 640

    triangulated = 0
    triangulated_skipped = 0
    tri_qc: list[dict[str, Any]] = []
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    if do_triangulation:
        for base, left_csv, right_csv in pairs:
            out_csv = keypoints_3d_dir / f"{base}_3d.csv"
            if out_csv.exists() and not overwrite_existing:
                triangulated_skipped += 1
                print(f"Skipped 3D triangulation for {base}: output already exists")
                continue
            n_frames, mean_l, mean_r = _triangulate_clip(
                left_csv=left_csv,
                right_csv=right_csv,
                out_csv=out_csv,
                K1=K1,
                D1=D1,
                K2=K2,
                D2=D2,
                R=R,
                T=T,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            triangulated += 1
            tri_qc.append({"file": base, "frames": n_frames, "reproj_err_left": mean_l, "reproj_err_right": mean_r})
            print(f"Saved 3D keypoints: {out_csv.name}")

    print(f"Triangulated {triangulated} clips (skipped existing: {triangulated_skipped})")
    delegated_secondary: dict[str, Any] = {}
    if do_kinematics:
        from secondary_measurements.kinematics_3d_lib import run_kinematics_3d_pipeline

        print("Kinematics moved to secondary_measurements; running secondary kinematics pipeline...")
        delegated_secondary = run_kinematics_3d_pipeline(cfg)

    return {
        "pairs_found": len(pairs),
        "clips_triangulated": triangulated,
        "clips_triangulation_skipped_existing": triangulated_skipped,
        "keypoints_2d_dir": str(keypoints_2d_dir),
        "keypoints_3d_dir": str(keypoints_3d_dir),
        "triangulation_qc": tri_qc,
        "secondary_kinematics": delegated_secondary,
    }
