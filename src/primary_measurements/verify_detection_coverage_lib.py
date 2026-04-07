"""Verification utilities for frame-valid detection coverage"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.io_utils import PROJECT_ROOT, load_csv_folder
from utils.preprocess_utils import extract_base_freethrow_name


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _to_base_name_dict(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        base = extract_base_freethrow_name(name) or name
        out[base] = df
    return out


def _split_keypoints_2d_by_side(
    keypoints_2d_dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    left_map: dict[str, pd.DataFrame] = {}
    right_map: dict[str, pd.DataFrame] = {}
    for name, df in keypoints_2d_dfs.items():
        base = extract_base_freethrow_name(name)
        if not base:
            continue
        lname = name.lower()
        if lname.endswith("_left_2d") or lname.endswith("_left"):
            left_map[base] = df
        elif lname.endswith("_right_2d") or lname.endswith("_right"):
            right_map[base] = df
    return left_map, right_map


def _frame_validity_2d(
    df: pd.DataFrame,
    *,
    visibility_threshold: float,
    min_visible_keypoints: int,
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    total = int(len(df))
    v_cols = [c for c in df.columns if c.endswith("_v")]

    if v_cols:
        vis = df[v_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.sum(vis >= visibility_threshold, axis=1) >= int(min_visible_keypoints)
        return total, int(np.sum(valid))

    # Fallback if visibility columns are not present.
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    xy_cols = sorted(set(x_cols) & set(c[:-2] + "_y" for c in x_cols))
    if not xy_cols:
        return total, 0

    # Build x/y pairs from common keypoint stems.
    keypoint_stems = []
    for x_col in x_cols:
        stem = x_col[:-2]
        y_col = f"{stem}_y"
        if y_col in df.columns:
            keypoint_stems.append(stem)
    if not keypoint_stems:
        return total, 0

    valid_counts = np.zeros(total, dtype=int)
    for stem in keypoint_stems:
        x = pd.to_numeric(df[f"{stem}_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[f"{stem}_y"], errors="coerce").to_numpy(dtype=float)
        valid_counts += np.isfinite(x) & np.isfinite(y)

    valid = valid_counts >= int(min_visible_keypoints)
    return total, int(np.sum(valid))


def _frame_validity_3d(
    df: pd.DataFrame,
    *,
    min_keypoints_3d: int,
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    total = int(len(df))
    stems: list[str] = []
    for col in df.columns:
        if col.endswith("_x"):
            stem = col[:-2]
            if f"{stem}_y" in df.columns and f"{stem}_z" in df.columns:
                stems.append(stem)

    if not stems:
        return total, 0

    valid_counts = np.zeros(total, dtype=int)
    for stem in stems:
        x = pd.to_numeric(df[f"{stem}_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[f"{stem}_y"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(df[f"{stem}_z"], errors="coerce").to_numpy(dtype=float)
        valid_counts += np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    valid = valid_counts >= int(min_keypoints_3d)
    return total, int(np.sum(valid))


def _frame_validity_ball(
    df: pd.DataFrame,
    *,
    min_ball_conf: float,
) -> tuple[int, int]:
    if df.empty:
        return 0, 0

    total = int(len(df))
    x = pd.to_numeric(df.get("x", pd.Series([np.nan] * total)), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get("y", pd.Series([np.nan] * total)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if "conf" in df.columns:
        conf = pd.to_numeric(df["conf"], errors="coerce").to_numpy(dtype=float)
        valid &= np.isfinite(conf) & (conf >= float(min_ball_conf))

    return total, int(np.sum(valid))


def _pct(valid: int, total: int) -> float:
    return float((100.0 * valid / total) if total > 0 else np.nan)


def run_verify_detection_coverage_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    keypoints_2d_dir = _format_path(cfg["paths"]["keypoints_2d"], cfg)
    keypoints_3d_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)
    side_ball_dir = metrics_dir / "raw_ball_trajectories"

    out_dir = metrics_dir / "detection_coverage_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_trial_csv = out_dir / "detection_coverage_per_trial.csv"
    summary_csv = out_dir / "detection_coverage_summary.csv"

    visibility_threshold = float(cfg.get("pose_visibility_threshold", 0.6))
    min_visible_keypoints = int(cfg.get("verification_min_visible_keypoints_2d", 8))
    min_keypoints_3d = int(cfg.get("verification_min_valid_keypoints_3d", 8))
    min_ball_conf = float(cfg.get("verification_min_ball_conf", 0.2))

    keypoints_2d = load_csv_folder(keypoints_2d_dir) if keypoints_2d_dir.exists() else {}
    keypoints_3d = _to_base_name_dict(load_csv_folder(keypoints_3d_dir)) if keypoints_3d_dir.exists() else {}
    side_ball = _to_base_name_dict(load_csv_folder(side_ball_dir)) if side_ball_dir.exists() else {}

    left_2d, right_2d = _split_keypoints_2d_by_side(keypoints_2d)

    trials = sorted(
        set(left_2d.keys())
        | set(right_2d.keys())
        | set(keypoints_3d.keys())
        | set(side_ball.keys())
    )
    if not trials:
        raise ValueError(
            "No data found for coverage verification. "
            "Run 2D/3D extraction and ball trajectory extraction first."
        )

    rows: list[dict[str, Any]] = []
    for trial in trials:
        l2_total, l2_valid = _frame_validity_2d(
            left_2d.get(trial, pd.DataFrame()),
            visibility_threshold=visibility_threshold,
            min_visible_keypoints=min_visible_keypoints,
        )
        r2_total, r2_valid = _frame_validity_2d(
            right_2d.get(trial, pd.DataFrame()),
            visibility_threshold=visibility_threshold,
            min_visible_keypoints=min_visible_keypoints,
        )
        k3_total, k3_valid = _frame_validity_3d(
            keypoints_3d.get(trial, pd.DataFrame()),
            min_keypoints_3d=min_keypoints_3d,
        )
        bs_total, bs_valid = _frame_validity_ball(
            side_ball.get(trial, pd.DataFrame()),
            min_ball_conf=min_ball_conf,
        )
        rows.append(
            {
                "file": f"{trial}.avi",
                "two_d_left_total_frames": l2_total,
                "two_d_left_valid_frames": l2_valid,
                "two_d_left_valid_pct": _pct(l2_valid, l2_total),
                "two_d_right_total_frames": r2_total,
                "two_d_right_valid_frames": r2_valid,
                "two_d_right_valid_pct": _pct(r2_valid, r2_total),
                "three_d_total_frames": k3_total,
                "three_d_valid_frames": k3_valid,
                "three_d_valid_pct": _pct(k3_valid, k3_total),
                "ball_side_total_frames": bs_total,
                "ball_side_valid_frames": bs_valid,
                "ball_side_valid_pct": _pct(bs_valid, bs_total),
            }
        )

    per_trial_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    per_trial_df.to_csv(per_trial_csv, index=False)

    summary_rows = []
    metrics = [
        ("two_d_left", "2D Left"),
        ("two_d_right", "2D Right"),
        ("three_d", "3D"),
        ("ball_side", "Ball Side"),
    ]
    for key, label in metrics:
        total = int(per_trial_df[f"{key}_total_frames"].sum())
        valid = int(per_trial_df[f"{key}_valid_frames"].sum())
        pct = _pct(valid, total)
        trials_with_data = int((per_trial_df[f"{key}_total_frames"] > 0).sum())
        summary_rows.append(
            {
                "metric": label,
                "total_frames": total,
                "valid_frames": valid,
                "valid_pct": pct,
                "trials_with_data": trials_with_data,
                "total_trials": int(len(per_trial_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    print("\nDetection Coverage Summary:")
    for row in summary_rows:
        pct_str = f"{row['valid_pct']:.2f}%" if np.isfinite(row["valid_pct"]) else "n/a"
        print(
            f"  {row['metric']}: {row['valid_frames']}/{row['total_frames']} "
            f"valid frames ({pct_str}); trials with data: {row['trials_with_data']}/{row['total_trials']}"
        )

    return {
        "trials": int(len(per_trial_df)),
        "per_trial_csv": str(per_trial_csv),
        "summary_csv": str(summary_csv),
        "visibility_threshold": visibility_threshold,
        "min_visible_keypoints_2d": min_visible_keypoints,
        "min_valid_keypoints_3d": min_keypoints_3d,
        "min_ball_conf": min_ball_conf,
    }
