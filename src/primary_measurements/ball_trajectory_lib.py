"""Ball trajectory extraction pipeline based on notebooks/yolo_ball_detection.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import extract_base_freethrow_name

SPORTS_BALL_CLASS_ID = 32


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _resolve_model_path(cfg: dict[str, Any]) -> str:
    configured = cfg.get("ball_detection_model")
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return str(p)

    candidates = [
        PROJECT_ROOT / "notebooks" / "yolo11m.pt",
        PROJECT_ROOT / "notebooks" / "yolo11s.pt",
        PROJECT_ROOT / "yolo11m.pt",
        PROJECT_ROOT / "yolo11s.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "No YOLO weights found. Set 'ball_detection_model' in project_config.yaml "
        "or place yolo11m.pt/yolo11s.pt in notebooks/."
    )


def _list_ball_videos(raw_video_dir: Path) -> list[Path]:
    exts = {".avi", ".mp4", ".mov", ".mkv"}
    return sorted([p for p in raw_video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _best_ball_detection(
    result: Any,
    prev_center: tuple[float, float] | None,
    max_jump_px: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> tuple[float, float, float, float, float] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_score = float("-inf")

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id != SPORTS_BALL_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        cx = float((x1 + x2) / 2.0 + x_offset)
        cy = float((y1 + y2) / 2.0 + y_offset)
        w = float(x2 - x1)
        h = float(y2 - y1)
        conf = float(box.conf[0]) if box.conf is not None else 0.0

        if prev_center is None:
            score = conf
        else:
            dist = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
            if dist > max_jump_px:
                continue
            score = conf - (dist / max(max_jump_px, 1.0))

        if score > best_score:
            best_score = score
            best = (cx, cy, conf, w, h)

    return best


def _roi_bounds(width: int, height: int, roi_mode: str) -> tuple[int, int, int, int]:
    mode = (roi_mode or "full").strip().lower()
    if mode == "middle_half":
        x0 = int(round(width * 0.25))
        x1 = int(round(width * 0.75))
        x0 = max(0, min(x0, width - 1))
        x1 = max(x0 + 1, min(x1, width))
        return x0, x1, 0, height
    if mode == "middle_top_half":
        x0 = int(round(width * 0.25))
        x1 = int(round(width * 0.75))
        y0 = 0
        y1 = int(round(height * 0.5))
        x0 = max(0, min(x0, width - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(1, min(y1, height))
        return x0, x1, y0, y1
    if mode == "middle_top_75":
        x0 = int(round(width * 0.25))
        x1 = int(round(width * 0.75))
        y0 = 0
        y1 = int(round(height * 0.75))
        x0 = max(0, min(x0, width - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(1, min(y1, height))
        return x0, x1, y0, y1
    return 0, width, 0, height


def _detect_single_video(
    video_path: Path,
    model: Any,
    conf_threshold: float,
    imgsz: int,
    max_jump_px: float,
    roi_mode: str = "full",
    stop_after_miss_frames: int | None = None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    df = pd.DataFrame(
        {
            "frame": np.arange(total_frames, dtype=int),
            "x": np.nan,
            "y": np.nan,
            "conf": np.nan,
            "box_w": np.nan,
            "box_h": np.nan,
        }
    )

    frame_idx = 0
    prev_center: tuple[float, float] | None = None

    detected_frames = 0
    seen_detection = False
    consecutive_misses = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x0, x1, y0, y1 = _roi_bounds(frame.shape[1], frame.shape[0], roi_mode=roi_mode)
        roi_frame = frame[y0:y1, x0:x1]
        result = model(roi_frame, verbose=False, conf=conf_threshold, imgsz=imgsz)[0]
        best = _best_ball_detection(
            result=result,
            prev_center=prev_center,
            max_jump_px=max_jump_px,
            x_offset=float(x0),
            y_offset=float(y0),
        )

        if best is not None:
            cx, cy, score, bw, bh = best
            df.loc[frame_idx, "x"] = cx
            df.loc[frame_idx, "y"] = cy
            df.loc[frame_idx, "conf"] = score
            df.loc[frame_idx, "box_w"] = bw
            df.loc[frame_idx, "box_h"] = bh
            prev_center = (cx, cy)
            detected_frames += 1
            seen_detection = True
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        if frame_idx > 0 and frame_idx % 300 == 0:
            print(
                f"  {video_path.name}: processed {frame_idx}/{total_frames} frames "
                f"(detections so far: {detected_frames})"
            )

        if (
            stop_after_miss_frames is not None
            and stop_after_miss_frames > 0
            and seen_detection
            and consecutive_misses >= stop_after_miss_frames
        ):
            print(
                f"  {video_path.name}: early stop at frame {frame_idx} "
                f"(no detections for {consecutive_misses} consecutive frames)"
            )
            break

        frame_idx += 1

    cap.release()
    print(
        f"  {video_path.name}: finished {total_frames} frames "
        f"(detections: {detected_frames})"
    )
    return df

def _detect_videos_to_dir(
    videos: list[Path],
    out_dir: Path,
    model: Any,
    *,
    conf_threshold: float,
    imgsz: int,
    max_jump_px: float,
    overwrite_existing: bool,
    name_suffix: str = "",
    roi_mode: str = "full",
    stop_after_miss_frames: int | None = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    dfs: dict[str, pd.DataFrame] = {}
    skipped_existing = 0

    for idx, video_path in enumerate(videos, start=1):
        base = extract_base_freethrow_name(video_path.stem) or video_path.stem
        name = f"{base}{name_suffix}"
        out_csv = out_dir / f"{name}.csv"

        if out_csv.exists() and not overwrite_existing:
            dfs[name] = pd.read_csv(out_csv)
            skipped_existing += 1
            print(f"\n[{idx}/{len(videos)}] Skipping {video_path.name}: trajectory exists")
            continue

        print(f"\n[{idx}/{len(videos)}] Processing {video_path.name} ...")
        df = _detect_single_video(
            video_path=video_path,
            model=model,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            max_jump_px=max_jump_px,
            roi_mode=roi_mode,
            stop_after_miss_frames=stop_after_miss_frames,
        )
        df.to_csv(out_csv, index=False)
        dfs[name] = df

    return dfs, skipped_existing


def run_ball_trajectory_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ImportError(
            "Ultralytics is required for ball trajectory extraction. Install with: pip install ultralytics"
        ) from exc

    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    raw_video_dir = _format_path(cfg["paths"]["ball_tracking_raw"], cfg)

    raw_ball_trajectory_dir = metrics_dir / "raw_ball_trajectories"

    # Legacy cleanup-safe behavior: if a file exists at this path, keep it and use a new folder.
    if raw_ball_trajectory_dir.exists() and not raw_ball_trajectory_dir.is_dir():
        raw_ball_trajectory_dir = metrics_dir / "raw_ball_trajectories"

    raw_ball_trajectory_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(_resolve_model_path(cfg))
    conf_threshold = float(cfg.get("ball_detection_conf", 0.4))
    imgsz = int(cfg.get("ball_detection_imgsz", 1280))
    max_jump_px = float(cfg.get("ball_detection_max_jump_px", 220.0))
    print(
        "YOLO settings: "
        f"conf={conf_threshold}, imgsz={imgsz}, max_jump_px={max_jump_px}"
    )

    raw_ball_trajectory_dfs: dict[str, pd.DataFrame] = {}
    skipped_existing = 0
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    videos: list[Path] = []
    if raw_video_dir.exists():
        videos = _list_ball_videos(raw_video_dir)
    if videos:
        print(f"Ball trajectory input directory: {raw_video_dir}")
        print(f"Found {len(videos)} ball-tracking video(s)")
        raw_ball_trajectory_dfs, skipped_existing = _detect_videos_to_dir(
            videos=videos,
            out_dir=raw_ball_trajectory_dir,
            model=model,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            max_jump_px=max_jump_px,
            overwrite_existing=overwrite_existing,
            roi_mode="full",
            stop_after_miss_frames=None,
        )
    else:
        print(f"No side ball-tracking videos found in: {raw_video_dir}")

    total_detected_frames = int(
        sum(df["x"].notna().sum() for df in raw_ball_trajectory_dfs.values())
    )
    total_frames = int(sum(len(df) for df in raw_ball_trajectory_dfs.values()))
    print(f"Processed {len(videos)} side videos (skipped existing: {skipped_existing})")
    print(f"Side ball detections: {total_detected_frames}/{total_frames} frames")
    print(f"Saved side trajectories: {raw_ball_trajectory_dir}")

    return {
        "videos_processed": len(videos),
        "videos_skipped_existing": skipped_existing,
        "raw_trajectory_dir": str(raw_ball_trajectory_dir),
        "cropped_trajectory_dir": "",
        "aligned_trajectory_dir": "",
        "detected_frames": total_detected_frames,
        "total_frames": total_frames,
        "cropped_trials": 0,
        "cropped_skipped_existing": 0,
        "aligned_trials": 0,
        "aligned_skipped_existing": 0,
        "used_freethrow_phases": False,
        "used_ball_phases": False,
        "used_kinematic_phases_for_ball": False,
        "used_shift_table": False,
    }

run_ball_detection_pipeline = run_ball_trajectory_pipeline
