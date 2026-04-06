"""Side-camera 2D pose extraction using shared pose_2d_lib helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from primary_measurements.pose_2d_lib import LANDMARK_NAMES, _build_dataframe, _extract_landmarks_for_frame
from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import extract_base_freethrow_name
from utils.view_images import close_all_windows


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _list_videos(folder: Path) -> list[Path]:
    exts = {".avi", ".mp4", ".mov", ".mkv"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _pick_roi_interactive(video_path: Path) -> tuple[float, float, float, float] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    preview = frame.copy()
    cv2.putText(
        preview,
        "Draw shooter ROI and press ENTER/SPACE. ESC cancels.",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    try:
        x, y, w, h = cv2.selectROI("Select Side ROI", preview, showCrosshair=True, fromCenter=False)
        try:
            cv2.destroyWindow("Select Side ROI")
        except Exception:
            pass
        close_all_windows()
    except Exception:
        close_all_windows()
        return None

    if w <= 1 or h <= 1:
        return None
    H, W = frame.shape[:2]
    x0 = float(x) / float(W)
    x1 = float(x + w) / float(W)
    y0 = float(y) / float(H)
    y1 = float(y + h) / float(H)
    return (
        float(np.clip(x0, 0.0, 1.0)),
        float(np.clip(x1, 0.0, 1.0)),
        float(np.clip(y0, 0.0, 1.0)),
        float(np.clip(y1, 0.0, 1.0)),
    )


def _offset_roi_row_to_full(row: list[float], x0: int, y0: int) -> list[float]:
    out = row[:]
    n = len(LANDMARK_NAMES)
    for i in range(n):
        xi = i * 3
        yi = i * 3 + 1
        if out[xi] >= 0 and out[yi] >= 0:
            out[xi] = float(out[xi] + x0)
            out[yi] = float(out[yi] + y0)
    return out


def _extract_side_pose_csv(
    video_path: Path,
    *,
    roi_x_min: float,
    roi_x_max: float,
    roi_y_min: float,
    roi_y_max: float,
    use_roi: bool,
    vis_thresh: float,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> pd.DataFrame:
    try:
        import mediapipe as mp
    except Exception as exc:
        raise ImportError("MediaPipe is required for side pose extraction. Install with: pip install mediapipe") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    raw_rows: list[list[float]] = []
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if use_roi:
                h, w = frame.shape[:2]
                x0 = int(np.clip(round(roi_x_min * w), 0, w - 1))
                x1 = int(np.clip(round(roi_x_max * w), x0 + 1, w))
                y0 = int(np.clip(round(roi_y_min * h), 0, h - 1))
                y1 = int(np.clip(round(roi_y_max * h), y0 + 1, h))
                roi = frame[y0:y1, x0:x1]
                row = _extract_landmarks_for_frame(roi, pose)
                row = _offset_roi_row_to_full(row, x0=x0, y0=y0)
            else:
                row = _extract_landmarks_for_frame(frame, pose)
            raw_rows.append(row)

    cap.release()
    if not raw_rows:
        return pd.DataFrame()
    return _build_dataframe(raw_rows, vis_thresh=vis_thresh)


def _to_hand_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = ["frame"]
    for key in ("left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb"):
        for ax in ("x", "y"):
            c = f"{key}_{ax}"
            if c in df.columns:
                cols.append(c)
    return df[cols].copy()


def run_side_pose_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    side_video_dir = _format_path(cfg["paths"]["ball_tracking_raw"], cfg)
    metrics_dir = _format_path(cfg["paths"]["metrics"], cfg)
    out_dir = metrics_dir / "side_pose_2d"
    compat_hands_dir = metrics_dir / "side_hand_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    compat_hands_dir.mkdir(parents=True, exist_ok=True)

    if not side_video_dir.exists():
        raise FileNotFoundError(f"Missing side ball video folder: {side_video_dir}")

    videos = _list_videos(side_video_dir)
    if not videos:
        raise ValueError(f"No side videos found in: {side_video_dir}")

    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))
    roi_x_min = float(cfg.get("side_release_roi_x_min", 0.25))
    roi_x_max = float(cfg.get("side_release_roi_x_max", 0.75))
    roi_y_min = float(cfg.get("side_release_roi_y_min", 0.10))
    roi_y_max = float(cfg.get("side_release_roi_y_max", 0.95))
    select_roi_interactively = bool(cfg.get("side_release_roi_select_interactively", True))
    use_roi = bool(cfg.get("side_pose_use_roi", True))
    vis_thresh = float(cfg.get("pose_visibility_threshold", 0.6))
    min_det = float(cfg.get("pose_min_detection_confidence", 0.5))
    min_track = float(cfg.get("pose_min_tracking_confidence", 0.5))

    if select_roi_interactively and use_roi and videos:
        picked = _pick_roi_interactive(videos[0])
        if picked is not None:
            roi_x_min, roi_x_max, roi_y_min, roi_y_max = picked
            print(
                "Using interactive ROI: "
                f"x=[{roi_x_min:.3f}, {roi_x_max:.3f}], "
                f"y=[{roi_y_min:.3f}, {roi_y_max:.3f}]"
            )
        else:
            print("ROI selection canceled/failed. Falling back to config ROI values.")

    written = 0
    skipped = 0
    failed = 0
    for idx, video_path in enumerate(videos, start=1):
        base = extract_base_freethrow_name(video_path.stem) or video_path.stem
        out_csv = out_dir / f"{base}.csv"
        compat_csv = compat_hands_dir / f"{base}.csv"
        if out_csv.exists() and compat_csv.exists() and not overwrite_existing:
            skipped += 1
            print(f"[{idx}/{len(videos)}] Skipping {video_path.name}: side pose CSV exists")
            continue
        try:
            print(f"[{idx}/{len(videos)}] Side pose {video_path.name} ...")
            pose_df = _extract_side_pose_csv(
                video_path,
                roi_x_min=roi_x_min,
                roi_x_max=roi_x_max,
                roi_y_min=roi_y_min,
                roi_y_max=roi_y_max,
                use_roi=use_roi,
                vis_thresh=vis_thresh,
                min_detection_confidence=min_det,
                min_tracking_confidence=min_track,
            )
            pose_df.to_csv(out_csv, index=False)
            _to_hand_only(pose_df).to_csv(compat_csv, index=False)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[WARNING] Failed side pose for {video_path.name}: {exc}")

    print(f"Side pose extraction complete. written={written}, skipped={skipped}, failed={failed}")
    return {
        "side_video_dir": str(side_video_dir),
        "output_dir": str(out_dir),
        "compat_hands_output_dir": str(compat_hands_dir),
        "videos_found": len(videos),
        "written": written,
        "skipped_existing": skipped,
        "failed": failed,
        "used_roi": bool(use_roi),
    }

