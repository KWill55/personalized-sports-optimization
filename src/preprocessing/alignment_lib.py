"""Description: Alignment using release frame found from phases node"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.align_freethrows_utils import (
    apply_shift_to_dataset,
)
from utils.io_utils import load_csv_folder
from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import extract_base_freethrow_name
from utils.curve_comparison_viewer import run_curve_comparison_viewer


KEYPOINT_COLS: list[str] = [
    "nose_x", "nose_y", "nose_z",
    "left_eye_inner_x", "left_eye_inner_y", "left_eye_inner_z",
    "left_eye_x", "left_eye_y", "left_eye_z",
    "left_eye_outer_x", "left_eye_outer_y", "left_eye_outer_z",
    "right_eye_inner_x", "right_eye_inner_y", "right_eye_inner_z",
    "right_eye_x", "right_eye_y", "right_eye_z",
    "right_eye_outer_x", "right_eye_outer_y", "right_eye_outer_z",
    "left_ear_x", "left_ear_y", "left_ear_z",
    "right_ear_x", "right_ear_y", "right_ear_z",
    "mouth_left_x", "mouth_left_y", "mouth_left_z",
    "mouth_right_x", "mouth_right_y", "mouth_right_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_elbow_x", "left_elbow_y", "left_elbow_z",
    "right_elbow_x", "right_elbow_y", "right_elbow_z",
    "left_wrist_x", "left_wrist_y", "left_wrist_z",
    "right_wrist_x", "right_wrist_y", "right_wrist_z",
    "left_pinky_x", "left_pinky_y", "left_pinky_z",
    "right_pinky_x", "right_pinky_y", "right_pinky_z",
    "left_index_x", "left_index_y", "left_index_z",
    "right_index_x", "right_index_y", "right_index_z",
    "left_thumb_x", "left_thumb_y", "left_thumb_z",
    "right_thumb_x", "right_thumb_y", "right_thumb_z",
    "left_hip_x", "left_hip_y", "left_hip_z",
    "right_hip_x", "right_hip_y", "right_hip_z",
    "left_knee_x", "left_knee_y", "left_knee_z",
    "right_knee_x", "right_knee_y", "right_knee_z",
    "left_ankle_x", "left_ankle_y", "left_ankle_z",
    "right_ankle_x", "right_ankle_y", "right_ankle_z",
    "left_heel_x", "left_heel_y", "left_heel_z",
    "right_heel_x", "right_heel_y", "right_heel_z",
    "left_foot_index_x", "left_foot_index_y", "left_foot_index_z",
    "right_foot_index_x", "right_foot_index_y", "right_foot_index_z",
]

def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _to_base_name_dict(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        base: df
        for name, df in dfs.items()
        for base in [extract_base_freethrow_name(name)]
        if base
    }


def _save_csv_dict(
    dfs: dict[str, pd.DataFrame],
    out_dir: Path,
    overwrite_existing: bool,
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for freethrow_name, df in dfs.items():
        out_path = out_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped += 1
            continue
        df.to_csv(out_path, index=False)
        written += 1
    return written, skipped


def _build_release_shift_log(
    phases_df: pd.DataFrame,
    valid_trials: set[str],
    release_col: str,
) -> tuple[pd.DataFrame, int]:
    work = phases_df.copy()
    work["base"] = work["file"].apply(extract_base_freethrow_name)
    work = work[work["base"].isin(valid_trials)].copy()
    work[release_col] = pd.to_numeric(work[release_col], errors="coerce")
    work = work.dropna(subset=[release_col])
    if work.empty:
        return pd.DataFrame(columns=["file", "shift"]), 0

    # If duplicates exist, keep the first valid release frame per trial.
    release_per_trial = (
        work.sort_values("file")
        .drop_duplicates(subset=["base"], keep="first")[["base", release_col]]
        .rename(columns={"base": "file"})
    )
    median_release = int(round(float(np.median(release_per_trial[release_col].to_numpy(dtype=float)))))
    log_df = release_per_trial.copy()
    log_df["shift"] = median_release - pd.to_numeric(log_df[release_col], errors="coerce")
    log_df = log_df[["file", "shift"]].copy()
    log_df["shift"] = pd.to_numeric(log_df["shift"], errors="coerce").fillna(0).astype(int)
    return log_df, median_release


def _build_aligned_release_map(
    phases_df: pd.DataFrame,
    valid_trials: set[str],
    release_col: str,
    log_df: pd.DataFrame,
) -> dict[str, int]:
    if log_df.empty:
        return {}
    work = phases_df.copy()
    work["base"] = work["file"].apply(extract_base_freethrow_name)
    work = work[work["base"].isin(valid_trials)].copy()
    work[release_col] = pd.to_numeric(work[release_col], errors="coerce")
    work = work.dropna(subset=["base", release_col])
    if work.empty:
        return {}

    raw_release_map = (
        work.sort_values("file")
        .drop_duplicates(subset=["base"], keep="first")
        .set_index("base")[release_col]
        .astype(float)
        .to_dict()
    )
    shift_map = (
        log_df.copy()
        .set_index("file")["shift"]
        .astype(float)
        .to_dict()
    )

    out: dict[str, int] = {}
    for base in sorted(set(raw_release_map.keys()) & set(shift_map.keys())):
        out[base] = int(round(float(raw_release_map[base]) + float(shift_map[base])))
    return out


def _reframe_to_release_zero(
    dfs: dict[str, pd.DataFrame],
    aligned_release_map: dict[str, int],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        rel = aligned_release_map.get(name)
        if rel is None:
            out[name] = df
            continue
        new_df = df.copy()
        if "frame" in new_df.columns:
            frame_vals = pd.to_numeric(new_df["frame"], errors="coerce").to_numpy(dtype=float)
            frame_vals = np.where(np.isfinite(frame_vals), frame_vals, np.arange(len(new_df), dtype=float))
        else:
            frame_vals = np.arange(len(new_df), dtype=float)
        aligned_frame = np.rint(frame_vals - float(rel)).astype(int)
        new_df["frame"] = aligned_frame
        new_df["aligned_frame"] = aligned_frame
        out[name] = new_df
    return out


def run_alignment_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    keypoints_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)
    phases_path = _format_path(cfg["paths"]["phases"], cfg)

    keypoints_dfs = _to_base_name_dict(load_csv_folder(keypoints_dir)) if keypoints_dir.exists() else {}
    phases_df = pd.read_csv(phases_path) if phases_path.exists() else pd.DataFrame()

    if not keypoints_dfs or phases_df.empty:
        raise ValueError("Missing alignment inputs. Need 3D keypoints and freethrow phases.")
    if "file" not in phases_df.columns:
        raise ValueError(f"Phases CSV must contain file column: {phases_path}")

    phases_df = phases_df.copy()
    preferred_keypoint_release_col = str(cfg.get("alignment_release_column", "raw_release_frame_stereo"))
    keypoint_candidate_cols = [preferred_keypoint_release_col, "raw_release_frame_stereo", "raw_release_frame"]
    keypoint_release_col = next((c for c in keypoint_candidate_cols if c in phases_df.columns), None)
    if keypoint_release_col is None:
        raise ValueError(
            f"Phases CSV missing keypoint release-frame column. Tried: {keypoint_candidate_cols}. Path: {phases_path}"
        )

    preferred_ball_release_col = str(cfg.get("alignment_ball_release_column", "raw_release_frame_ball_cam"))
    ball_candidate_cols = [preferred_ball_release_col, "raw_release_frame_ball_cam", "raw_release_frame"]
    ball_release_col = next((c for c in ball_candidate_cols if c in phases_df.columns), None)
    if ball_release_col is None:
        raise ValueError(
            f"Phases CSV missing ball release-frame column. Tried: {ball_candidate_cols}. Path: {phases_path}"
        )

    phases_trials = {extract_base_freethrow_name(v) for v in phases_df["file"].tolist()}
    keypoint_trials = set(keypoints_dfs.keys()) & phases_trials
    if not keypoint_trials:
        raise ValueError("No overlapping trials across 3D keypoints/phases.")

    keypoints_dfs = {k: v for k, v in keypoints_dfs.items() if k in keypoint_trials}
    phases_df = phases_df[phases_df["file"].apply(extract_base_freethrow_name).isin(phases_trials)].copy()

    aligned_keypoints_release_dir = metrics_dir / "3d_keypoints_aligned_release"
    release_shift_table_path = metrics_dir / "alignment_release_shift_table.csv"
    ball_release_shift_table_path = metrics_dir / "alignment_release_shift_table_ball_cam.csv"
    aligned_ball_release_dir = metrics_dir / "aligned_ball_trajectory_release"
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    for path in [
        aligned_keypoints_release_dir,
        aligned_ball_release_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    keypoint_release_log_df, keypoint_median_release = _build_release_shift_log(
        phases_df=phases_df,
        valid_trials=keypoint_trials,
        release_col=keypoint_release_col,
    )
    if keypoint_release_log_df.empty:
        raise ValueError(
            f"No valid keypoint release frames found for alignment using '{keypoint_release_col}'."
        )
    if overwrite_existing or not release_shift_table_path.exists():
        keypoint_release_log_df.to_csv(release_shift_table_path, index=False)

    aligned_relative_keypoints_release_frame_dfs = apply_shift_to_dataset(
        dfs=keypoints_dfs,
        log_df=keypoint_release_log_df,
        fps=int(cfg.get("player_tracking_fps", 60)),
        cols=KEYPOINT_COLS,
    )

    keypoint_aligned_release_map = _build_aligned_release_map(
        phases_df=phases_df,
        valid_trials=keypoint_trials,
        release_col=keypoint_release_col,
        log_df=keypoint_release_log_df,
    )
    aligned_relative_keypoints_release_frame_dfs = _reframe_to_release_zero(
        aligned_relative_keypoints_release_frame_dfs,
        keypoint_aligned_release_map,
    )

    _, skip_kp_release = _save_csv_dict(
        aligned_relative_keypoints_release_frame_dfs,
        aligned_keypoints_release_dir,
        overwrite_existing=overwrite_existing,
    )

    skipped_existing = skip_kp_release

    ball_source_dir = metrics_dir / "raw_ball_trajectories"
    ball_dfs = _to_base_name_dict(load_csv_folder(ball_source_dir)) if ball_source_dir.exists() else {}
    ball_cols: list[str] = []
    aligned_ball_dfs: dict[str, pd.DataFrame] = {}
    ball_release_log_df = pd.DataFrame(columns=["file", "shift"])
    ball_trials = set()
    ball_median_release = 0
    if ball_dfs:
        ball_trials = set(ball_dfs.keys()) & phases_trials
        sample_ball_df = next(iter(ball_dfs.values()))
        ball_cols = [c for c in sample_ball_df.select_dtypes(include=[np.number]).columns if c != "frame"]
        if ball_cols:
            ball_release_log_df, ball_median_release = _build_release_shift_log(
                phases_df=phases_df,
                valid_trials=ball_trials,
                release_col=ball_release_col,
            )
            if overwrite_existing or not ball_release_shift_table_path.exists():
                ball_release_log_df.to_csv(ball_release_shift_table_path, index=False)
            aligned_ball_dfs = apply_shift_to_dataset(
                dfs={k: v for k, v in ball_dfs.items() if k in ball_trials},
                log_df=ball_release_log_df,
                fps=int(cfg.get("ball_tracking_fps", 30)),
                shift_fps=int(cfg.get("ball_tracking_fps", 30)),
                target_fps=int(cfg.get("ball_tracking_fps", 30)),
                cols=ball_cols,
            )
            ball_aligned_release_map = _build_aligned_release_map(
                phases_df=phases_df,
                valid_trials=ball_trials,
                release_col=ball_release_col,
                log_df=ball_release_log_df,
            )
            aligned_ball_dfs = _reframe_to_release_zero(
                aligned_ball_dfs,
                ball_aligned_release_map,
            )
            _, skipped_ball = _save_csv_dict(
                aligned_ball_dfs,
                aligned_ball_release_dir,
                overwrite_existing=overwrite_existing,
            )
            skipped_existing += skipped_ball

    print(f"Aligned keypoint trials: {len(keypoint_trials)}")
    print(f"Keypoint alignment release column: {keypoint_release_col} (median={keypoint_median_release})")
    print(f"Saved keypoint release shift table: {release_shift_table_path}")
    if aligned_ball_dfs:
        print(f"Aligned ball trials: {len(aligned_ball_dfs)}")
        print(f"Ball alignment release column: {ball_release_col} (median={ball_median_release})")
        print(f"Saved ball release shift table: {ball_release_shift_table_path}")
        print(f"Saved aligned trajectories: {aligned_ball_release_dir}")
    print(f"Skipped existing output files: {skipped_existing}")

    return {
        "metrics_dir": str(metrics_dir),
        "aligned_keypoint_trials": len(aligned_relative_keypoints_release_frame_dfs),
        "release_shift_table_path": str(release_shift_table_path),
        "ball_release_shift_table_path": str(ball_release_shift_table_path),
        "skipped_existing_outputs": skipped_existing,
        "aligned_ball_trials": len(aligned_ball_dfs),
        "alignment_mode": "release_only",
        "keypoint_release_frame_column": keypoint_release_col,
        "ball_release_frame_column": ball_release_col,
    }


def _load_csv_dict_or_single(path: Path) -> dict[str, pd.DataFrame]:
    if path.is_dir():
        return load_csv_folder(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return {path.stem: pd.read_csv(path)}
    return {}


def _apply_shift_with_log(
    dfs: dict[str, pd.DataFrame],
    log_df: pd.DataFrame,
    cols: list[str],
    fps: int,
    *,
    shift_fps: int | None = None,
    target_fps: int | None = None,
) -> dict[str, pd.DataFrame]:
    if not dfs or log_df is None or log_df.empty:
        return {}
    if "file" not in log_df.columns or "shift" not in log_df.columns:
        return {}
    valid_cols = [c for c in cols if any(c in df.columns for df in dfs.values())]
    if not valid_cols:
        return {}
    return apply_shift_to_dataset(
        dfs=dfs,
        log_df=log_df,
        fps=fps,
        cols=valid_cols,
        shift_fps=shift_fps,
        target_fps=target_fps,
    )


def _common_numeric_columns(dfs_list: list[dict[str, pd.DataFrame]], exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    cols_accum: set[str] | None = None
    for dfs in dfs_list:
        if not dfs:
            continue
        per_dict_cols: set[str] = set()
        for df in dfs.values():
            numeric_cols = set(df.select_dtypes(include=[np.number]).columns) - exclude
            per_dict_cols = per_dict_cols | numeric_cols
        cols_accum = per_dict_cols if cols_accum is None else (cols_accum & per_dict_cols)
    if cols_accum is None:
        return []
    return sorted(cols_accum)


def run_alignment_viewer(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)

    cropped_phases_path = metrics_dir / "cropped_freethrow_phases.csv"
    phases_path = _format_path(cfg["paths"]["phases"], cfg)
    phases_df = pd.read_csv(phases_path) if phases_path.exists() else pd.DataFrame()
    if cropped_phases_path.exists():
        cropped_phases_df = pd.read_csv(cropped_phases_path)
    else:
        cropped_phases_df = phases_df.copy() if not phases_df.empty else None
        if cropped_phases_df is not None and "cropped_release_frame" not in cropped_phases_df.columns:
            if "raw_release_frame_stereo" in cropped_phases_df.columns:
                cropped_phases_df = cropped_phases_df.copy()
                cropped_phases_df["cropped_release_frame"] = pd.to_numeric(
                    cropped_phases_df["raw_release_frame_stereo"], errors="coerce"
                )
            elif "raw_release_frame" in cropped_phases_df.columns:
                cropped_phases_df = cropped_phases_df.copy()
                cropped_phases_df["cropped_release_frame"] = pd.to_numeric(
                    cropped_phases_df["raw_release_frame"], errors="coerce"
                )

    fps = int(cfg.get("player_tracking_fps", 60))

    keypoints_unaligned = _load_csv_dict_or_single(metrics_dir / "3d_keypoints_cropped")
    if not keypoints_unaligned:
        keypoints_path = _format_path(cfg["paths"]["keypoints_3d"], cfg)
        if keypoints_path.exists():
            keypoints_unaligned = _to_base_name_dict(load_csv_folder(keypoints_path))
    ball_unaligned = _load_csv_dict_or_single(metrics_dir / "cropped_ball_trajectory")
    if not ball_unaligned:
        ball_unaligned = _load_csv_dict_or_single(metrics_dir / "raw_ball_trajectories")

    release_log_path = metrics_dir / "alignment_release_shift_table.csv"
    release_log = pd.read_csv(release_log_path) if release_log_path.exists() else pd.DataFrame()
    if not release_log.empty and "file" in release_log.columns:
        release_log["file"] = release_log["file"].apply(extract_base_freethrow_name)

    keypoints_release = _load_csv_dict_or_single(metrics_dir / "3d_keypoints_aligned_release")
    if not keypoints_release:
        keypoints_release = _apply_shift_with_log(keypoints_unaligned, release_log, KEYPOINT_COLS, fps)

    ball_cols = []
    if ball_unaligned:
        sample_ball_df = next(iter(ball_unaligned.values()))
        ball_cols = [c for c in ("x", "y", "conf", "box_w", "box_h") if c in sample_ball_df.columns]
        if not ball_cols:
            ball_cols = [c for c in sample_ball_df.select_dtypes(include=[np.number]).columns if c != "frame"]

    ball_fps = int(cfg.get("ball_tracking_fps", 30))
    ball_release = _load_csv_dict_or_single(metrics_dir / "aligned_ball_trajectory_release")
    if not ball_release:
        ball_release = _apply_shift_with_log(
            ball_unaligned,
            release_log,
            ball_cols,
            fps=ball_fps,
            shift_fps=fps,
            target_fps=ball_fps,
        )

    keypoint_mode_sets = {
        "Aligned (Release)": keypoints_release,
    }
    ball_mode_sets = {
        "Aligned (Release)": ball_release,
    }

    keypoint_curves = _common_numeric_columns([keypoints_unaligned], exclude={"frame"})
    ball_curves = _common_numeric_columns([ball_unaligned], exclude={"frame"})

    if not keypoint_curves and not ball_curves:
        raise ValueError(
            "No aligned/unaligned keypoint/ball datasets found for viewer. Run align/extract steps first."
        )

    tab_specs = {
        "Keypoints": {
            "left": keypoints_unaligned,
            "right": keypoint_mode_sets["Aligned (Release)"],
            "left_label": "Unaligned",
            "curves": keypoint_curves,
            "ylabel": "Position",
            "show_release_left": False,
            "show_release_right": True,
        },
        "Ball": {
            "left": ball_unaligned,
            "right": ball_mode_sets["Aligned (Release)"],
            "left_label": "Unaligned",
            "curves": ball_curves,
            "ylabel": "Ball Metric",
            "show_release_left": False,
            "show_release_right": True,
        },
    }
    release_frames = np.array([], dtype=float)
    if cropped_phases_df is not None and "cropped_release_frame" in cropped_phases_df.columns:
        release_frames = pd.to_numeric(cropped_phases_df["cropped_release_frame"], errors="coerce").dropna().to_numpy(dtype=float)
    ball_release_frames = np.array([], dtype=float)
    if not phases_df.empty:
        if "raw_release_frame_ball_cam" in phases_df.columns:
            ball_release_frames = pd.to_numeric(
                phases_df["raw_release_frame_ball_cam"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
        elif "raw_release_frame_stereo" in phases_df.columns:
            ball_release_frames = (
                pd.to_numeric(phases_df["raw_release_frame_stereo"], errors="coerce").dropna().to_numpy(dtype=float)
                * (float(cfg.get("ball_tracking_fps", 30.0)) / max(float(cfg.get("player_tracking_fps", 60.0)), 1e-6))
            )
        elif "raw_release_frame" in phases_df.columns:
            ball_release_frames = (
                pd.to_numeric(phases_df["raw_release_frame"], errors="coerce").dropna().to_numpy(dtype=float)
                * (float(cfg.get("ball_tracking_fps", 30.0)) / max(float(cfg.get("player_tracking_fps", 60.0)), 1e-6))
            )

    viewer_result = run_curve_comparison_viewer(
        window_title="Alignment Review",
        tab_specs={
            **tab_specs,
            "Ball": {
                **tab_specs["Ball"],
                "release_frames_left": ball_release_frames,
                "release_frames_right": ball_release_frames,
            },
        },
        right_mode_label="Aligned (Release)",
        release_frames=release_frames,
    )

    return {
        "metrics_dir": str(metrics_dir),
        **viewer_result,
    }
