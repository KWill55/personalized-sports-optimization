"""2D pose extraction pipeline ported from old_pipeline player tracking scripts."""

from __future__ import annotations

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


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _hampel_filter(series: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    rolling_median = s.rolling(window=window_size, center=True).median()
    diff = (s - rolling_median).abs()
    mad = 1.4826 * diff.rolling(window=window_size, center=True).median()
    outliers = diff > (n_sigmas * mad)
    s[outliers] = np.nan
    s = s.interpolate(limit_direction="both")
    return s.to_numpy(float)


def _butterworth_or_fallback(series: np.ndarray, cutoff: float = 0.1, order: int = 2) -> np.ndarray:
    s = pd.Series(series, dtype=float).interpolate(limit_direction="both").bfill().ffill()

    try:
        from scipy.signal import butter, filtfilt  # lazy import to keep startup resilient

        b, a = butter(order, cutoff)
        return pd.Series(filtfilt(b, a, s.to_numpy(float))).to_numpy(float)
    except Exception:
        # If scipy is unavailable, use a light rolling smooth fallback.
        return s.rolling(5, center=True, min_periods=1).mean().to_numpy(float)


def _clean_keypoint_series(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    vis_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.where(v < vis_thresh, np.nan, x)
    y = np.where(v < vis_thresh, np.nan, y)

    x_clean = _butterworth_or_fallback(_hampel_filter(x))
    y_clean = _butterworth_or_fallback(_hampel_filter(y))
    return x_clean, y_clean, v


def _extract_landmarks_for_frame(frame: np.ndarray, pose: Any) -> list[float]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return [-1.0] * (len(LANDMARK_NAMES) * 3)

    pts: list[float] = []
    for lm in result.pose_landmarks.landmark:
        pts.extend([float(lm.x * w), float(lm.y * h), float(lm.visibility)])
    return pts


def _build_dataframe(raw_rows: list[list[float]], vis_thresh: float) -> pd.DataFrame:
    cols = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y", "v")]
    df = pd.DataFrame(raw_rows, columns=cols)

    for name in LANDMARK_NAMES:
        x = df[f"{name}_x"].to_numpy(float)
        y = df[f"{name}_y"].to_numpy(float)
        v = df[f"{name}_v"].to_numpy(float)

        x_clean, y_clean, v_clean = _clean_keypoint_series(x=x, y=y, v=v, vis_thresh=vis_thresh)
        df[f"{name}_x"] = x_clean
        df[f"{name}_y"] = y_clean
        df[f"{name}_v"] = v_clean

    df.insert(0, "frame", np.arange(len(df), dtype=int))
    return df


def _video_stems(input_dir: Path) -> list[Path]:
    exts = {".avi", ".mp4", ".mov", ".mkv"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def run_pose_2d_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import mediapipe as mp
    except Exception as exc:
        raise ImportError(
            "MediaPipe is required for 2D pose extraction. Install with: pip install mediapipe"
        ) from exc

    input_dir = _format_path(cfg["paths"]["player_tracking_sync"], cfg)
    output_dir = _format_path(cfg["paths"]["keypoints_2d"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Synchronized player video directory not found: {input_dir}")

    videos = _video_stems(input_dir)
    if not videos:
        raise ValueError(f"No synchronized player videos found in: {input_dir}")

    vis_thresh = float(cfg.get("pose_visibility_threshold", 0.6))
    min_det_conf = float(cfg.get("pose_min_detection_confidence", 0.5))
    min_track_conf = float(cfg.get("pose_min_tracking_confidence", 0.5))

    mp_pose = mp.solutions.pose

    processed = 0
    skipped_existing = 0
    total_frames = 0
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    ) as pose:
        for video_path in videos:
            left_csv = output_dir / f"{video_path.stem}_left_2d.csv"
            right_csv = output_dir / f"{video_path.stem}_right_2d.csv"
            if left_csv.exists() and right_csv.exists() and not overwrite_existing:
                skipped_existing += 1
                print(f"Skipped 2D extraction for {video_path.name}: outputs already exist")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARNING] Could not open {video_path.name}; skipping")
                continue

            left_rows: list[list[float]] = []
            right_rows: list[list[float]] = []

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                mid = w // 2
                left_frame = frame[:, :mid]
                right_frame = frame[:, mid:]

                left_rows.append(_extract_landmarks_for_frame(left_frame, pose))
                right_rows.append(_extract_landmarks_for_frame(right_frame, pose))
                total_frames += 1

            cap.release()

            if not left_rows or not right_rows:
                print(f"[WARNING] No frames processed for {video_path.name}")
                continue

            left_df = _build_dataframe(left_rows, vis_thresh=vis_thresh)
            right_df = _build_dataframe(right_rows, vis_thresh=vis_thresh)

            left_df.to_csv(left_csv, index=False)
            right_df.to_csv(right_csv, index=False)

            processed += 1
            print(f"Saved 2D keypoints for {video_path.name}")

    print(f"Processed {processed}/{len(videos)} clips (skipped existing: {skipped_existing})")

    return {
        "videos_found": len(videos),
        "videos_processed": processed,
        "videos_skipped_existing": skipped_existing,
        "total_frames_processed": total_frames,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
