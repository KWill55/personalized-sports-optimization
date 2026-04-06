"""3D kinematics (angles/velocities/accelerations) from existing 3D keypoints."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from primary_measurements.pose_3d_lib import ANGLE_SPECS, IDX, LANDMARK_NAMES
from utils.io_utils import PROJECT_ROOT


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _base_num(path: Path) -> int | float:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else float("inf")


def _load_3d_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        frame_idx = pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        frame_idx = np.arange(len(df), dtype=int)

    cols: list[str] = []
    for name in LANDMARK_NAMES:
        for suffix in ("_x", "_y", "_z"):
            col = f"{name}{suffix}"
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path.name}")
            cols.append(col)

    arr = df[cols].to_numpy(float).reshape(len(df), len(LANDMARK_NAMES), 3)
    return arr, frame_idx


def _angle_series(frames_xyz: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    A = frames_xyz[:, IDX[a], :]
    B = frames_xyz[:, IDX[b], :]
    C = frames_xyz[:, IDX[c], :]

    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2

    valid = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1) & (denom > 1e-8)
    angle = np.full(len(frames_xyz), np.nan, dtype=float)

    if np.any(valid):
        cosang = np.einsum("ij,ij->i", v1[valid], v2[valid]) / denom[valid]
        cosang = np.clip(cosang, -1.0, 1.0)
        angle[valid] = np.degrees(np.arccos(cosang))
    return angle


def _maybe_smooth(series: np.ndarray, window: int) -> np.ndarray:
    if window >= 3 and window % 2 == 1:
        s = pd.Series(series, dtype=float)
        s = s.rolling(window, center=True, min_periods=1).median()
        s = s.rolling(window, center=True, min_periods=1).mean()
        return s.to_numpy(float)
    return series


def _interp_fill_vec(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    s = s.interpolate(limit_direction="both")
    s = s.bfill().ffill()
    return s.to_numpy(float)


def _central_diff_1d(x: np.ndarray, fps: float) -> np.ndarray:
    t_len = len(x)
    if t_len == 0:
        return np.array([], dtype=float)
    if t_len == 1:
        return np.zeros(1, dtype=float)

    dx = np.zeros(t_len, dtype=float)
    dx[0] = (x[1] - x[0]) * fps
    dx[-1] = (x[-1] - x[-2]) * fps
    if t_len > 2:
        dx[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    return dx


def run_kinematics_3d_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    aligned_cropped_keypoints_dir = metrics_dir / "3d_keypoints_aligned_release_cropped"
    default_keypoints_3d_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)
    keypoints_3d_dir = (
        aligned_cropped_keypoints_dir
        if aligned_cropped_keypoints_dir.exists()
        else default_keypoints_3d_dir
    )
    angles_dir = _format_path(cfg["paths"]["angles"], cfg)
    velocities_dir = _format_path(cfg["paths"]["velocities"], cfg)
    accelerations_dir = _format_path(cfg["paths"]["accelerations"], cfg)
    angles_dir.mkdir(parents=True, exist_ok=True)
    velocities_dir.mkdir(parents=True, exist_ok=True)
    accelerations_dir.mkdir(parents=True, exist_ok=True)

    fps = float(cfg.get("player_tracking_fps", 60.0))
    angle_smooth_window = int(cfg.get("angle_smooth_window", 0))
    kin_pos_smooth_window = int(cfg.get("kin_pos_smooth_window", 0))
    kin_vel_smooth_window = int(cfg.get("kin_vel_smooth_window", 0))
    kin_acc_smooth_window = int(cfg.get("kin_acc_smooth_window", 0))
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    # Support both naming styles:
    # - raw 3D output: freethrow001_3d.csv
    # - aligned/cropped output: freethrow001.csv
    keypoint_files = sorted(keypoints_3d_dir.glob("*_3d.csv"), key=_base_num)
    if not keypoint_files:
        keypoint_files = sorted(keypoints_3d_dir.glob("*.csv"), key=_base_num)
    if not keypoint_files:
        raise ValueError(f"No 3D keypoint files found in: {keypoints_3d_dir}")

    angles_written = 0
    angles_skipped = 0
    velacc_written = 0
    velacc_skipped = 0

    for keypoint_file in keypoint_files:
        frames_xyz, frame_idx = _load_3d_array(keypoint_file)
        base_name = keypoint_file.stem.replace("_3d", "")

        angle_path = angles_dir / f"{base_name}_angles.csv"
        if angle_path.exists() and not overwrite_existing:
            angles_skipped += 1
        else:
            angle_out: dict[str, np.ndarray] = {"frame": frame_idx}
            for angle_name, (a, b, c) in ANGLE_SPECS.items():
                series = _angle_series(frames_xyz, a, b, c)
                filled = pd.Series(series, dtype=float).interpolate(limit_direction="both").bfill().ffill().to_numpy(float)
                angle_out[angle_name] = _maybe_smooth(filled, angle_smooth_window)
            pd.DataFrame(angle_out).to_csv(angle_path, index=False)
            angles_written += 1

        vel_path = velocities_dir / f"{base_name}_3d_velocities.csv"
        acc_path = accelerations_dir / f"{base_name}_3d_accelerations.csv"
        if vel_path.exists() and acc_path.exists() and not overwrite_existing:
            velacc_skipped += 1
            continue

        t_len = len(frames_xyz)
        pos = frames_xyz.copy()
        for j in range(len(LANDMARK_NAMES)):
            for d in range(3):
                pos[:, j, d] = _interp_fill_vec(pos[:, j, d])
                pos[:, j, d] = _maybe_smooth(pos[:, j, d], kin_pos_smooth_window)

        vx = np.zeros((t_len, len(LANDMARK_NAMES)), dtype=float)
        vy = np.zeros((t_len, len(LANDMARK_NAMES)), dtype=float)
        vz = np.zeros((t_len, len(LANDMARK_NAMES)), dtype=float)
        for j in range(len(LANDMARK_NAMES)):
            vx[:, j] = _central_diff_1d(pos[:, j, 0], fps)
            vy[:, j] = _central_diff_1d(pos[:, j, 1], fps)
            vz[:, j] = _central_diff_1d(pos[:, j, 2], fps)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)

        vel_cols: dict[str, np.ndarray] = {"frame": frame_idx}
        for j, name in enumerate(LANDMARK_NAMES):
            vel_cols[f"{name}_vx"] = _maybe_smooth(vx[:, j], kin_vel_smooth_window)
            vel_cols[f"{name}_vy"] = _maybe_smooth(vy[:, j], kin_vel_smooth_window)
            vel_cols[f"{name}_vz"] = _maybe_smooth(vz[:, j], kin_vel_smooth_window)
            vel_cols[f"{name}_speed"] = _maybe_smooth(speed[:, j], kin_vel_smooth_window)

        ax = np.zeros_like(vx)
        ay = np.zeros_like(vy)
        az = np.zeros_like(vz)
        for j, name in enumerate(LANDMARK_NAMES):
            ax[:, j] = _central_diff_1d(vel_cols[f"{name}_vx"], fps)
            ay[:, j] = _central_diff_1d(vel_cols[f"{name}_vy"], fps)
            az[:, j] = _central_diff_1d(vel_cols[f"{name}_vz"], fps)
        accel = np.sqrt(ax * ax + ay * ay + az * az)

        acc_cols: dict[str, np.ndarray] = {"frame": frame_idx}
        for j, name in enumerate(LANDMARK_NAMES):
            acc_cols[f"{name}_ax"] = _maybe_smooth(ax[:, j], kin_acc_smooth_window)
            acc_cols[f"{name}_ay"] = _maybe_smooth(ay[:, j], kin_acc_smooth_window)
            acc_cols[f"{name}_az"] = _maybe_smooth(az[:, j], kin_acc_smooth_window)
            acc_cols[f"{name}_accel"] = _maybe_smooth(accel[:, j], kin_acc_smooth_window)

        pd.DataFrame(vel_cols).to_csv(vel_path, index=False)
        pd.DataFrame(acc_cols).to_csv(acc_path, index=False)
        velacc_written += 1

    print(f"Wrote {angles_written} angle files (skipped existing: {angles_skipped})")
    print(f"Wrote {velacc_written} velocity/acceleration file pairs (skipped existing: {velacc_skipped})")

    return {
        "keypoints_3d_dir": str(keypoints_3d_dir),
        "angles_dir": str(angles_dir),
        "velocities_dir": str(velocities_dir),
        "accelerations_dir": str(accelerations_dir),
        "angles_written": angles_written,
        "angles_skipped_existing": angles_skipped,
        "vel_acc_written": velacc_written,
        "vel_acc_skipped_existing": velacc_skipped,
    }
