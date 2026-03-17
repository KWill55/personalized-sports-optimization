"""Ball detection pipeline based on notebooks/yolo_ball_detection.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from utils.align_freethrows_utils import apply_shift_to_dataset
from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import crop_to_freethrow, extract_base_freethrow_name

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


def _best_ball_detection(result: Any, prev_center: tuple[float, float] | None, max_jump_px: float) -> tuple[float, float, float, float, float] | None:
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
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
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


def _detect_single_video(
    video_path: Path,
    model: Any,
    conf_threshold: float,
    imgsz: int,
    max_jump_px: float,
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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False, conf=conf_threshold, imgsz=imgsz)[0]
        best = _best_ball_detection(result=result, prev_center=prev_center, max_jump_px=max_jump_px)

        if best is not None:
            cx, cy, score, bw, bh = best
            df.loc[frame_idx, "x"] = cx
            df.loc[frame_idx, "y"] = cy
            df.loc[frame_idx, "conf"] = score
            df.loc[frame_idx, "box_w"] = bw
            df.loc[frame_idx, "box_h"] = bh
            prev_center = (cx, cy)
            detected_frames += 1

        if frame_idx > 0 and frame_idx % 300 == 0:
            print(
                f"  {video_path.name}: processed {frame_idx}/{total_frames} frames "
                f"(detections so far: {detected_frames})"
            )

        frame_idx += 1

    cap.release()
    print(
        f"  {video_path.name}: finished {total_frames} frames "
        f"(detections: {detected_frames})"
    )
    return df


def _save_trajectory_dict(
    dfs: dict[str, pd.DataFrame],
    out_dir: Path,
    *,
    overwrite_existing: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for name, df in dfs.items():
        out_path = out_dir / f"{name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped += 1
            continue
        df.to_csv(out_path, index=False)
    return skipped


def _load_shift_table(path: Path) -> pd.DataFrame:
    shifts = pd.read_csv(path)
    if "file" not in shifts.columns or "shift" not in shifts.columns:
        raise ValueError(f"Invalid shift table format: {path}")
    shifts = shifts.copy()
    shifts["file"] = shifts["file"].apply(extract_base_freethrow_name)
    return shifts.dropna(subset=["file"])


def _align_ball_trajectories(
    cropped_ball_trajectory_dfs: dict[str, pd.DataFrame],
    shift_table_df: pd.DataFrame,
    fps: int,
) -> dict[str, pd.DataFrame]:
    if not cropped_ball_trajectory_dfs:
        return {}

    valid_trials = set(cropped_ball_trajectory_dfs.keys()) & set(shift_table_df["file"].tolist())
    if not valid_trials:
        return {}

    filtered_dfs = {k: v for k, v in cropped_ball_trajectory_dfs.items() if k in valid_trials}
    filtered_shifts = shift_table_df[shift_table_df["file"].isin(valid_trials)].copy()

    cols = ["x", "y", "conf", "box_w", "box_h"]
    existing_cols = [c for c in cols if c in next(iter(filtered_dfs.values())).columns]

    return apply_shift_to_dataset(
        dfs=filtered_dfs,
        log_df=filtered_shifts,
        fps=fps,
        cols=existing_cols,
    )


def run_ball_detection_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ImportError(
            "Ultralytics is required for ball detection. Install with: pip install ultralytics"
        ) from exc

    metrics_dir = _format_path(cfg["paths"]["metrics"], cfg)
    raw_video_dir = _format_path(cfg["paths"]["ball_tracking_raw"], cfg)

    raw_freethrow_phases_path = metrics_dir / "freethrow_phases.csv"
    ball_phases_path = metrics_dir / "ball_phases.csv"
    shift_table_path = metrics_dir / "alignment_shift_table.csv"

    raw_ball_trajectory_dir = metrics_dir / "raw_ball_trajectory"
    cropped_ball_trajectory_dir = metrics_dir / "cropped_ball_trajectory"
    aligned_ball_trajectory_dir = metrics_dir / "aligned_ball_trajectory_unsigned_area"

    # Legacy cleanup-safe behavior: if a file exists at this path, keep it and use a new folder.
    if raw_ball_trajectory_dir.exists() and not raw_ball_trajectory_dir.is_dir():
        raw_ball_trajectory_dir = metrics_dir / "raw_ball_trajectory_csvs"

    raw_ball_trajectory_dir.mkdir(parents=True, exist_ok=True)
    cropped_ball_trajectory_dir.mkdir(parents=True, exist_ok=True)
    aligned_ball_trajectory_dir.mkdir(parents=True, exist_ok=True)

    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Raw ball video directory not found: {raw_video_dir}")

    videos = _list_ball_videos(raw_video_dir)
    if not videos:
        raise ValueError(f"No ball-tracking videos found in: {raw_video_dir}")

    print(f"Ball detection input directory: {raw_video_dir}")
    print(f"Found {len(videos)} ball-tracking video(s)")

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

    for idx, video_path in enumerate(videos, start=1):
        name = extract_base_freethrow_name(video_path.stem) or video_path.stem
        out_csv = raw_ball_trajectory_dir / f"{name}.csv"
        if out_csv.exists() and not overwrite_existing:
            raw_ball_trajectory_dfs[name] = pd.read_csv(out_csv)
            skipped_existing += 1
            print(f"\n[{idx}/{len(videos)}] Skipping {video_path.name}: raw trajectory exists")
            continue

        print(f"\n[{idx}/{len(videos)}] Processing {video_path.name} ...")
        df = _detect_single_video(
            video_path=video_path,
            model=model,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            max_jump_px=max_jump_px,
        )
        raw_ball_trajectory_dfs[name] = df
        df.to_csv(out_csv, index=False)

    cropped_count = 0
    cropped_ball_trajectory_dfs: dict[str, pd.DataFrame] = {}

    if ball_phases_path.exists():
        ball_phases_df = pd.read_csv(ball_phases_path)
        cropped_ball_trajectory_dfs = crop_to_freethrow(
            angles_dfs=raw_ball_trajectory_dfs,
            phases_df=ball_phases_df,
            start_col="raw_start_frame",
            end_col="raw_ball_through_net",
        )
        cropped_skipped_existing = _save_trajectory_dict(
            cropped_ball_trajectory_dfs,
            cropped_ball_trajectory_dir,
            overwrite_existing=overwrite_existing,
        )
        cropped_count = len(cropped_ball_trajectory_dfs)
    else:
        cropped_skipped_existing = 0

    aligned_count = 0
    if cropped_ball_trajectory_dfs and shift_table_path.exists():
        shifts = _load_shift_table(shift_table_path)
        aligned = _align_ball_trajectories(
            cropped_ball_trajectory_dfs=cropped_ball_trajectory_dfs,
            shift_table_df=shifts,
            fps=int(cfg.get("player_tracking_fps", 60)),
        )
        aligned_skipped_existing = _save_trajectory_dict(
            aligned,
            aligned_ball_trajectory_dir,
            overwrite_existing=overwrite_existing,
        )
        aligned_count = len(aligned)
    else:
        aligned_skipped_existing = 0

    total_detected_frames = int(
        sum(df["x"].notna().sum() for df in raw_ball_trajectory_dfs.values())
    )
    total_frames = int(sum(len(df) for df in raw_ball_trajectory_dfs.values()))

    print(f"Processed {len(videos)} videos (skipped existing raw outputs: {skipped_existing})")
    print(f"Ball detections: {total_detected_frames}/{total_frames} frames")
    print(f"Saved raw trajectories: {raw_ball_trajectory_dir}")
    if ball_phases_path.exists():
        print(f"Saved cropped trajectories: {cropped_ball_trajectory_dir}")
    if shift_table_path.exists() and cropped_ball_trajectory_dfs:
        print(f"Saved aligned trajectories: {aligned_ball_trajectory_dir}")

    return {
        "videos_processed": len(videos),
        "videos_skipped_existing": skipped_existing,
        "raw_trajectory_dir": str(raw_ball_trajectory_dir),
        "cropped_trajectory_dir": str(cropped_ball_trajectory_dir),
        "aligned_trajectory_dir": str(aligned_ball_trajectory_dir),
        "detected_frames": total_detected_frames,
        "total_frames": total_frames,
        "cropped_trials": cropped_count,
        "cropped_skipped_existing": cropped_skipped_existing,
        "aligned_trials": aligned_count,
        "aligned_skipped_existing": aligned_skipped_existing,
        "used_freethrow_phases": raw_freethrow_phases_path.exists(),
        "used_ball_phases": ball_phases_path.exists(),
        "used_shift_table": shift_table_path.exists(),
    }
