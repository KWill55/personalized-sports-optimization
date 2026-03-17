"""Preprocessing pipeline for combining left/right player feeds into synchronized stereo videos."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from utils.io_utils import PROJECT_ROOT


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _get_matching_video_pairs(left_dir: Path, right_dir: Path, name_prefix: str) -> list[tuple[Path, Path]]:
    pattern = re.compile(rf"^{re.escape(name_prefix)}(\d+)$")
    left_files = sorted(left_dir.glob(f"{name_prefix}*.avi"))

    matches: list[tuple[Path, Path]] = []
    for left_file in left_files:
        match = pattern.match(left_file.stem)
        if not match:
            continue

        number = match.group(1)
        right_file = right_dir / f"{name_prefix}{number}.avi"
        if right_file.exists():
            matches.append((left_file, right_file))

    return matches


def _validate_capture_size(cap: cv.VideoCapture, expected_size: tuple[int, int]) -> tuple[bool, tuple[int, int]]:
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return (width, height) == expected_size, (width, height)


def _combine_pair(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    expected_size: tuple[int, int],
    default_fps: float,
) -> dict[str, Any]:
    left_cap = cv.VideoCapture(str(left_path))
    right_cap = cv.VideoCapture(str(right_path))

    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        raise RuntimeError(f"Failed to open video pair: {left_path.name}, {right_path.name}")

    left_ok, left_size = _validate_capture_size(left_cap, expected_size)
    right_ok, right_size = _validate_capture_size(right_cap, expected_size)

    if not left_ok or not right_ok:
        left_cap.release()
        right_cap.release()
        return {
            "output": str(output_path),
            "status": "skipped_size_mismatch",
            "left_size": left_size,
            "right_size": right_size,
            "expected_size": expected_size,
            "frames_written": 0,
        }

    fps = left_cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = default_fps

    combined_width = expected_size[0] * 2
    combined_height = expected_size[1]

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(str(output_path), fourcc, fps, (combined_width, combined_height))

    frames_written = 0
    while True:
        left_ok, left_frame = left_cap.read()
        right_ok, right_frame = right_cap.read()
        if not left_ok or not right_ok:
            break

        combined_frame = np.hstack((left_frame, right_frame))
        out.write(combined_frame)
        frames_written += 1

    left_cap.release()
    right_cap.release()
    out.release()

    return {
        "output": str(output_path),
        "status": "ok",
        "left_size": left_size,
        "right_size": right_size,
        "expected_size": expected_size,
        "frames_written": frames_written,
    }


def run_combine_player_feeds_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    name_prefix = str(cfg.get("freethrow_name_prefix", "freethrow"))

    if "paths" not in cfg:
        raise KeyError("Missing 'paths' in config")

    left_dir = _format_path(cfg["paths"]["player_tracking_left"], cfg)
    right_dir = _format_path(cfg["paths"]["player_tracking_right"], cfg)
    output_dir = _format_path(cfg["paths"]["player_tracking_sync"], cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_size = tuple(int(v) for v in cfg.get("cropped_stereo_resolution", cfg.get("uncropped_stereo_resolution", [1280, 720])))
    if len(expected_size) != 2:
        raise ValueError(f"Invalid cropped_stereo_resolution: {expected_size}")

    if not left_dir.exists() or not right_dir.exists():
        existing_combined = sorted(output_dir.glob(f"{name_prefix}*.avi"))
        if existing_combined:
            print(
                "Raw left/right directories are missing, but synchronized videos already exist. "
                "Skipping combine step."
            )
            return {
                "pairs_found": 0,
                "pairs_combined": 0,
                "pairs_skipped": 0,
                "output_dir": str(output_dir),
                "details": [],
                "status": "already_combined",
                "existing_outputs": len(existing_combined),
            }
        raise FileNotFoundError(
            f"Input directories missing. left={left_dir} right={right_dir}"
        )

    pairs = _get_matching_video_pairs(left_dir=left_dir, right_dir=right_dir, name_prefix=name_prefix)
    if not pairs:
        existing_combined = sorted(output_dir.glob(f"{name_prefix}*.avi"))
        if existing_combined:
            print("No left/right pairs found. Existing synchronized outputs detected; nothing to combine.")
            return {
                "pairs_found": 0,
                "pairs_combined": 0,
                "pairs_skipped": 0,
                "output_dir": str(output_dir),
                "details": [],
                "status": "already_combined",
                "existing_outputs": len(existing_combined),
            }
        print(f"No matching '{name_prefix}*.avi' pairs found in {left_dir} and {right_dir}")
        return {
            "pairs_found": 0,
            "pairs_combined": 0,
            "pairs_skipped": 0,
            "output_dir": str(output_dir),
            "details": [],
            "status": "no_pairs",
        }

    default_fps = float(cfg.get("player_tracking_fps", 60))
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))
    details: list[dict[str, Any]] = []
    combined = 0
    skipped = 0

    for left_path, right_path in pairs:
        output_path = output_dir / left_path.name
        if output_path.exists() and not overwrite_existing:
            skipped += 1
            details.append(
                {
                    "left_input": str(left_path),
                    "right_input": str(right_path),
                    "output": str(output_path),
                    "status": "skipped_exists",
                    "frames_written": 0,
                }
            )
            print(f"Skipped {left_path.name}: output already exists")
            continue

        result = _combine_pair(
            left_path=left_path,
            right_path=right_path,
            output_path=output_path,
            expected_size=(expected_size[0], expected_size[1]),
            default_fps=default_fps,
        )
        result["left_input"] = str(left_path)
        result["right_input"] = str(right_path)
        details.append(result)

        if result["status"] == "ok":
            combined += 1
            print(f"Saved: {output_path}")
        else:
            skipped += 1
            print(
                f"Skipped {left_path.name}: size mismatch "
                f"left={result['left_size']} right={result['right_size']} expected={result['expected_size']}"
            )

    print(f"Combined {combined}/{len(pairs)} pairs")

    return {
        "pairs_found": len(pairs),
        "pairs_combined": combined,
        "pairs_skipped": skipped,
        "output_dir": str(output_dir),
        "details": details,
    }
