"""Alignment preprocessing pipeline mirroring notebooks/data_exploration.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.align_freethrows_utils import (
    align_by_lowest_frame,
    align_by_min_signed_area,
    align_by_min_unsigned_area,
    align_by_release_frame,
    apply_shift_to_dataset,
)
from utils.io_utils import load_csv_folder
from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import crop_to_freethrow, extract_base_freethrow_name


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

JOINT_COLS: list[str] = [
    "elbow_flex_l",
    "elbow_flex_r",
    "shoulder_flex_l",
    "shoulder_flex_r",
    "hip_flex_l",
    "hip_flex_r",
    "knee_flex_l",
    "knee_flex_r",
    "ankle_flex_l",
    "ankle_flex_r",
]

KEYPOINT_FLAGS: dict[str, bool] = {
    "nose": False,
    "left_eye_inner": False,
    "left_eye": False,
    "left_eye_outer": False,
    "right_eye_inner": False,
    "right_eye": False,
    "right_eye_outer": False,
    "left_ear": False,
    "right_ear": False,
    "mouth_left": False,
    "mouth_right": False,
    "left_shoulder": False,
    "right_shoulder": True,
    "left_elbow": False,
    "right_elbow": True,
    "left_wrist": False,
    "right_wrist": False,
    "left_pinky": False,
    "right_pinky": False,
    "left_index": False,
    "right_index": False,
    "left_thumb": False,
    "right_thumb": False,
    "left_hip": False,
    "right_hip": True,
    "left_knee": False,
    "right_knee": True,
    "left_ankle": False,
    "right_ankle": False,
    "left_heel": False,
    "right_heel": False,
    "left_foot_index": False,
    "right_foot_index": False,
}


def _expand_keypoint_flags(keypoint_flags: dict[str, bool]) -> list[str]:
    selected: list[str] = []
    for keypoint, enabled in keypoint_flags.items():
        if enabled:
            selected.extend([f"{keypoint}_x", f"{keypoint}_y", f"{keypoint}_z"])
    return selected


def _make_relative(dfs_dict: dict[str, pd.DataFrame], cols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for file_name, df in dfs_dict.items():
        rel_df = df.copy()
        for col in cols:
            if col in rel_df.columns and not rel_df.empty:
                rel_df[col] = rel_df[col] - rel_df[col].iloc[0]
        out[file_name] = rel_df
    return out


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _to_base_name_dict(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        base: df
        for name, df in dfs.items()
        for base in [extract_base_freethrow_name(name)]
        if base
    }


def run_alignment_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["metrics"], cfg)
    keypoints_path = _format_path(cfg["paths"]["keypoints_3d"], cfg)
    angles_path = _format_path(cfg["paths"]["angles"], cfg)

    phases_path = metrics_dir / "freethrow_phases.csv"
    outcomes_path = metrics_dir / "outcomes.csv"

    cropped_phases_path = metrics_dir / "cropped_freethrow_phases.csv"
    cropped_keypoints_dir = metrics_dir / "3d_keypoints_cropped"
    cropped_angles_dir = metrics_dir / "3d_angles_cropped"
    aligned_keypoints_release_dir = metrics_dir / "3d_keypoints_aligned_release"
    aligned_keypoints_unsigned_dir = metrics_dir / "3d_keypoints_aligned_unsigned_area"
    aligned_angles_release_dir = metrics_dir / "3d_angles_aligned_release"
    aligned_angles_unsigned_dir = metrics_dir / "3d_angles_aligned_unsigned_area"
    shift_table_path = metrics_dir / "alignment_shift_table.csv"
    release_shift_table_path = metrics_dir / "alignment_release_shift_table.csv"
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    for path in [
        cropped_keypoints_dir,
        cropped_angles_dir,
        aligned_keypoints_release_dir,
        aligned_keypoints_unsigned_dir,
        aligned_angles_release_dir,
        aligned_angles_unsigned_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if not phases_path.exists():
        raise FileNotFoundError(f"Missing phases file: {phases_path}")

    if outcomes_path.exists():
        _ = pd.read_csv(outcomes_path)

    raw_phases_df = pd.read_csv(phases_path)

    cropped_phases_df = pd.DataFrame(
        {
            "file": raw_phases_df["file"],
            "cropped_first_frame": 0,
            "cropped_release_frame": raw_phases_df["raw_release_frame"] - raw_phases_df["raw_windup_start"],
            "cropped_last_frame": raw_phases_df["raw_followthrough_end"] - raw_phases_df["raw_windup_start"],
        }
    )
    cropped_phases_df.to_csv(cropped_phases_path, index=False)

    raw_keypoints_dfs = _to_base_name_dict(load_csv_folder(keypoints_path))
    raw_angles_dfs = _to_base_name_dict(load_csv_folder(angles_path))

    if not raw_keypoints_dfs or not raw_angles_dfs:
        raise ValueError("No keypoint/angle CSVs found. Verify metrics paths and extraction steps.")

    cropped_keypoints_dfs = crop_to_freethrow(raw_keypoints_dfs, raw_phases_df)
    cropped_angles_dfs = crop_to_freethrow(raw_angles_dfs, raw_phases_df)

    selected_keypoints = _expand_keypoint_flags(KEYPOINT_FLAGS)
    cropped_relative_keypoints_dfs = _make_relative(cropped_keypoints_dfs, selected_keypoints)

    cropped_relative_keypoints_dfs = _to_base_name_dict(cropped_relative_keypoints_dfs)
    cropped_angles_dfs = _to_base_name_dict(cropped_angles_dfs)

    skipped_existing = 0

    for freethrow_name, df in cropped_relative_keypoints_dfs.items():
        out_path = cropped_keypoints_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)

    for freethrow_name, df in cropped_angles_dfs.items():
        out_path = cropped_angles_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)

    _, angles_lowest_frame_log_df = align_by_lowest_frame(cropped_angles_dfs)
    _, angles_release_frame_log_df = align_by_release_frame(cropped_angles_dfs, cropped_phases_df)
    _, angles_unsigned_area_log_df = align_by_min_unsigned_area(cropped_angles_dfs, "elbow_flex_r")
    _, angles_signed_area_log_df = align_by_min_signed_area(cropped_angles_dfs)

    if overwrite_existing or not shift_table_path.exists():
        angles_unsigned_area_log_df.to_csv(shift_table_path, index=False)
    if overwrite_existing or not release_shift_table_path.exists():
        angles_release_frame_log_df.to_csv(release_shift_table_path, index=False)

    angles_unsigned_area_log_df["file"] = angles_unsigned_area_log_df["file"].apply(extract_base_freethrow_name)
    angles_release_frame_log_df["file"] = angles_release_frame_log_df["file"].apply(extract_base_freethrow_name)

    valid_trials = (
        set(cropped_angles_dfs.keys())
        & set(cropped_relative_keypoints_dfs.keys())
        & set(angles_unsigned_area_log_df["file"])
    )

    cropped_angles_dfs = {k: v for k, v in cropped_angles_dfs.items() if k in valid_trials}
    cropped_relative_keypoints_dfs = {
        k: v for k, v in cropped_relative_keypoints_dfs.items() if k in valid_trials
    }

    angles_unsigned_area_log_df = angles_unsigned_area_log_df[
        angles_unsigned_area_log_df["file"].isin(valid_trials)
    ]
    angles_release_frame_log_df = angles_release_frame_log_df[
        angles_release_frame_log_df["file"].isin(valid_trials)
    ]

    aligned_relative_keypoints_release_frame_dfs = apply_shift_to_dataset(
        dfs=cropped_relative_keypoints_dfs,
        log_df=angles_release_frame_log_df,
        fps=int(cfg.get("player_tracking_fps", 60)),
        cols=KEYPOINT_COLS,
    )
    aligned_relative_keypoints_unsigned_area_dfs = apply_shift_to_dataset(
        dfs=cropped_relative_keypoints_dfs,
        log_df=angles_unsigned_area_log_df,
        fps=int(cfg.get("player_tracking_fps", 60)),
        cols=KEYPOINT_COLS,
    )
    aligned_angles_release_frame_dfs = apply_shift_to_dataset(
        dfs=cropped_angles_dfs,
        log_df=angles_release_frame_log_df,
        fps=int(cfg.get("player_tracking_fps", 60)),
        cols=JOINT_COLS,
    )
    aligned_angles_unsigned_area_dfs = apply_shift_to_dataset(
        dfs=cropped_angles_dfs,
        log_df=angles_unsigned_area_log_df,
        fps=int(cfg.get("player_tracking_fps", 60)),
        cols=JOINT_COLS,
    )

    for freethrow_name, df in aligned_relative_keypoints_release_frame_dfs.items():
        out_path = aligned_keypoints_release_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)
    for freethrow_name, df in aligned_relative_keypoints_unsigned_area_dfs.items():
        out_path = aligned_keypoints_unsigned_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)
    for freethrow_name, df in aligned_angles_release_frame_dfs.items():
        out_path = aligned_angles_release_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)
    for freethrow_name, df in aligned_angles_unsigned_area_dfs.items():
        out_path = aligned_angles_unsigned_dir / f"{freethrow_name}.csv"
        if out_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        df.to_csv(out_path, index=False)

    print(f"Valid trials: {len(valid_trials)}")
    print(f"Saved cropped phases: {cropped_phases_path}")
    print(f"Saved unsigned-area shift table: {shift_table_path}")
    print(f"Skipped existing output files: {skipped_existing}")

    return {
        "metrics_dir": str(metrics_dir),
        "valid_trials": len(valid_trials),
        "cropped_phases_path": str(cropped_phases_path),
        "unsigned_shift_table_path": str(shift_table_path),
        "release_shift_table_path": str(release_shift_table_path),
        "lowest_log_rows": len(angles_lowest_frame_log_df),
        "signed_log_rows": len(angles_signed_area_log_df),
        "skipped_existing_outputs": skipped_existing,
    }


def _load_csv_dict_or_single(path: Path) -> dict[str, pd.DataFrame]:
    if path.is_dir():
        return load_csv_folder(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return {path.stem: pd.read_csv(path)}
    return {}


def _align_by_min_squared_error(
    cropped_dfs: dict[str, pd.DataFrame],
    col: str,
    search: int = 20,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    aligned: dict[str, pd.DataFrame] = {}
    logs: list[dict[str, Any]] = []

    if not cropped_dfs:
        return aligned, pd.DataFrame(columns=["file", "shift", "squared_error"])

    min_len = min(len(df) for df in cropped_dfs.values())
    curves = np.vstack([df[col].values[:min_len] for df in cropped_dfs.values()])
    mean_curve = np.nanmean(curves, axis=0)

    for file, df in cropped_dfs.items():
        curve = df[col].values[:min_len]
        best_shift = 0
        best_score = np.inf

        for shift in range(-search, search + 1):
            if shift < 0:
                c = curve[-shift:min_len]
                m = mean_curve[: min_len + shift]
            elif shift > 0:
                c = curve[: min_len - shift]
                m = mean_curve[shift:min_len]
            else:
                c = curve
                m = mean_curve

            score = np.nansum((m - c) ** 2)
            if score < best_score:
                best_score = score
                best_shift = shift

        aligned_df = df.copy()
        if col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].shift(best_shift)
        aligned[file] = aligned_df
        logs.append({"file": file, "shift": best_shift, "squared_error": float(best_score)})

    return aligned, pd.DataFrame(logs)


def _apply_shift_with_log(
    dfs: dict[str, pd.DataFrame],
    log_df: pd.DataFrame,
    cols: list[str],
    fps: int,
) -> dict[str, pd.DataFrame]:
    if not dfs or log_df is None or log_df.empty:
        return {}
    if "file" not in log_df.columns or "shift" not in log_df.columns:
        return {}
    valid_cols = [c for c in cols if any(c in df.columns for df in dfs.values())]
    if not valid_cols:
        return {}
    return apply_shift_to_dataset(dfs=dfs, log_df=log_df, fps=fps, cols=valid_cols)


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
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    metrics_dir = _format_path(cfg["paths"]["metrics"], cfg)

    cropped_phases_path = metrics_dir / "cropped_freethrow_phases.csv"
    cropped_phases_df = pd.read_csv(cropped_phases_path) if cropped_phases_path.exists() else None

    fps = int(cfg.get("player_tracking_fps", 60))

    angles_unaligned = _load_csv_dict_or_single(metrics_dir / "3d_angles_cropped")
    keypoints_unaligned = _load_csv_dict_or_single(metrics_dir / "3d_keypoints_cropped")
    ball_unaligned = _load_csv_dict_or_single(metrics_dir / "cropped_ball_trajectory")
    if not ball_unaligned:
        ball_unaligned = _load_csv_dict_or_single(metrics_dir / "raw_ball_trajectory")

    release_log_path = metrics_dir / "alignment_release_shift_table.csv"
    unsigned_log_path = metrics_dir / "alignment_shift_table.csv"

    release_log = pd.read_csv(release_log_path) if release_log_path.exists() else pd.DataFrame()
    unsigned_log = pd.read_csv(unsigned_log_path) if unsigned_log_path.exists() else pd.DataFrame()
    if not release_log.empty and "file" in release_log.columns:
        release_log["file"] = release_log["file"].apply(extract_base_freethrow_name)
    if not unsigned_log.empty and "file" in unsigned_log.columns:
        unsigned_log["file"] = unsigned_log["file"].apply(extract_base_freethrow_name)

    signed_log = pd.DataFrame()
    squared_log = pd.DataFrame()
    if angles_unaligned:
        _, signed_log = align_by_min_signed_area(angles_unaligned, "elbow_flex_r")
        _, squared_log = _align_by_min_squared_error(angles_unaligned, "elbow_flex_r")

    angles_release = _load_csv_dict_or_single(metrics_dir / "3d_angles_aligned_release")
    if not angles_release:
        angles_release = _apply_shift_with_log(angles_unaligned, release_log, JOINT_COLS, fps)
    angles_unsigned = _load_csv_dict_or_single(metrics_dir / "3d_angles_aligned_unsigned_area")
    if not angles_unsigned:
        angles_unsigned = _apply_shift_with_log(angles_unaligned, unsigned_log, JOINT_COLS, fps)
    angles_signed = _apply_shift_with_log(angles_unaligned, signed_log, JOINT_COLS, fps)
    angles_squared = _apply_shift_with_log(angles_unaligned, squared_log, JOINT_COLS, fps)

    keypoints_release = _load_csv_dict_or_single(metrics_dir / "3d_keypoints_aligned_release")
    if not keypoints_release:
        keypoints_release = _apply_shift_with_log(keypoints_unaligned, release_log, KEYPOINT_COLS, fps)
    keypoints_unsigned = _load_csv_dict_or_single(metrics_dir / "3d_keypoints_aligned_unsigned_area")
    if not keypoints_unsigned:
        keypoints_unsigned = _apply_shift_with_log(keypoints_unaligned, unsigned_log, KEYPOINT_COLS, fps)
    keypoints_signed = _apply_shift_with_log(keypoints_unaligned, signed_log, KEYPOINT_COLS, fps)
    keypoints_squared = _apply_shift_with_log(keypoints_unaligned, squared_log, KEYPOINT_COLS, fps)

    ball_cols = []
    if ball_unaligned:
        sample_ball_df = next(iter(ball_unaligned.values()))
        ball_cols = [c for c in ("x", "y", "conf", "box_w", "box_h") if c in sample_ball_df.columns]
        if not ball_cols:
            ball_cols = [c for c in sample_ball_df.select_dtypes(include=[np.number]).columns if c != "frame"]

    ball_release = _apply_shift_with_log(ball_unaligned, release_log, ball_cols, fps)
    ball_unsigned = _load_csv_dict_or_single(metrics_dir / "aligned_ball_trajectory_unsigned_area")
    if not ball_unsigned:
        ball_unsigned = _apply_shift_with_log(ball_unaligned, unsigned_log, ball_cols, fps)
    ball_signed = _apply_shift_with_log(ball_unaligned, signed_log, ball_cols, fps)
    ball_squared = _apply_shift_with_log(ball_unaligned, squared_log, ball_cols, fps)

    angle_mode_sets = {
        "Aligned (Release)": angles_release,
        "Aligned (Unsigned Area)": angles_unsigned,
        "Aligned (Signed Area)": angles_signed,
        "Aligned (Squared Error)": angles_squared,
    }
    keypoint_mode_sets = {
        "Aligned (Release)": keypoints_release,
        "Aligned (Unsigned Area)": keypoints_unsigned,
        "Aligned (Signed Area)": keypoints_signed,
        "Aligned (Squared Error)": keypoints_squared,
    }
    ball_mode_sets = {
        "Aligned (Release)": ball_release,
        "Aligned (Unsigned Area)": ball_unsigned,
        "Aligned (Signed Area)": ball_signed,
        "Aligned (Squared Error)": ball_squared,
    }

    angle_curves = _common_numeric_columns([angles_unaligned], exclude={"frame"})
    keypoint_curves = _common_numeric_columns([keypoints_unaligned], exclude={"frame"})
    ball_curves = _common_numeric_columns([ball_unaligned], exclude={"frame"})

    if not angle_curves and not keypoint_curves and not ball_curves:
        raise ValueError(
            "No aligned/unaligned datasets found for viewer. Run align/extract steps first."
        )

    tab_specs = {
        "Angles": {
            "unaligned": angles_unaligned,
            "mode_sets": angle_mode_sets,
            "curves": angle_curves,
            "ylabel": "Angle",
            "show_release": True,
        },
        "Keypoints": {
            "unaligned": keypoints_unaligned,
            "mode_sets": keypoint_mode_sets,
            "curves": keypoint_curves,
            "ylabel": "Position",
            "show_release": True,
        },
        "Ball": {
            "unaligned": ball_unaligned,
            "mode_sets": ball_mode_sets,
            "curves": ball_curves,
            "ylabel": "Ball Metric",
            "show_release": False,
        },
    }

    curve_index = {tab: 0 for tab in tab_specs}
    tab_names = [t for t, spec in tab_specs.items() if spec["curves"]]
    if not tab_names:
        raise ValueError("Viewer tabs have no curves to display.")

    root = tk.Tk()
    root.title("Alignment Review")
    root.geometry("1500x900")

    top = tk.Frame(root)
    top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    title_var = tk.StringVar(value="Alignment Review")
    title_label = tk.Label(top, textvariable=title_var, font=("Helvetica", 14, "bold"))
    title_label.pack(side=tk.LEFT, padx=8)

    btn_prev = tk.Button(top, text="← Prev")
    btn_prev.pack(side=tk.RIGHT, padx=6)
    btn_next = tk.Button(top, text="Next →")
    btn_next.pack(side=tk.RIGHT, padx=6)

    mode_var = tk.StringVar(value="Aligned (Unsigned Area)")
    mode_box = ttk.Combobox(
        top,
        textvariable=mode_var,
        values=[
            "Aligned (Release)",
            "Aligned (Unsigned Area)",
            "Aligned (Signed Area)",
            "Aligned (Squared Error)",
        ],
        state="readonly",
        width=28,
    )
    mode_box.pack(side=tk.RIGHT, padx=12)
    mode_label = tk.Label(top, text="Right Plot:")
    mode_label.pack(side=tk.RIGHT)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    fig_map: dict[str, Any] = {}
    ax_map: dict[str, list[Any]] = {}
    canvas_map: dict[str, Any] = {}
    stats_map: dict[str, tuple[Any, Any]] = {}

    for tab in tab_names:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        stats_frame = tk.Frame(frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        left_stats = tk.Label(
            stats_frame,
            text="",
            justify=tk.LEFT,
            anchor="w",
            font=("Helvetica", 10),
        )
        left_stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        right_stats = tk.Label(
            stats_frame,
            text="",
            justify=tk.LEFT,
            anchor="w",
            font=("Helvetica", 10),
        )
        right_stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        fig_map[tab] = fig
        ax_map[tab] = list(axes)
        canvas_map[tab] = canvas
        stats_map[tab] = (left_stats, right_stats)

    def _format_dataset_stats(dfs: dict[str, pd.DataFrame], curve: str) -> str:
        total_points = 0
        finite_points = 0
        lengths: list[int] = []
        curves: list[np.ndarray] = []

        for df in dfs.values():
            if curve not in df.columns:
                continue
            y = df[curve].to_numpy(dtype=float)
            total_points += int(y.size)
            finite_mask = np.isfinite(y)
            finite_count = int(np.sum(finite_mask))
            finite_points += finite_count
            if finite_count == 0:
                continue
            y_valid = y[finite_mask]
            lengths.append(len(y_valid))
            curves.append(y_valid)

        n_visible = len(curves)
        if n_visible == 0:
            return "Visible throws: 0 | No finite data"

        stacked = np.concatenate(curves)
        max_len = max(len(c) for c in curves)
        padded = np.vstack([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in curves])
        pointwise_std = np.nanstd(padded, axis=0)
        finite_pct = (100.0 * finite_points / total_points) if total_points else 0.0

        return (
            f"Visible throws: {n_visible} | "
            f"Mean: {np.nanmean(stacked):.3f} | Std: {np.nanstd(stacked):.3f}\n"
            f"Avg len: {np.mean(lengths):.1f}f (min {np.min(lengths)} / max {np.max(lengths)}) | "
            f"Avg framewise std: {np.nanmean(pointwise_std):.3f} | Finite: {finite_pct:.1f}%"
        )

    def _plot_dataset(ax, dfs: dict[str, pd.DataFrame], curve: str, label: str, ylabel: str, show_release: bool):
        ax.set_title(label)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

        curves = []
        for df in dfs.values():
            if curve not in df.columns:
                continue
            y = df[curve].to_numpy(dtype=float)
            x = df["frame"].to_numpy(dtype=float) if "frame" in df.columns else np.arange(len(y), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            curves.append(y[valid])
            ax.plot(x[valid], y[valid], color="gray", alpha=0.18, linewidth=1)

        if curves:
            max_len = max(len(c) for c in curves)
            padded = np.vstack([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in curves])
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            x_mean = np.arange(len(mean_curve), dtype=float)
            ax.plot(x_mean, mean_curve, color="orange", linewidth=2, label="Mean")
            ax.fill_between(x_mean, mean_curve - std_curve, mean_curve + std_curve, color="orange", alpha=0.2)

        if show_release and cropped_phases_df is not None and "cropped_release_frame" in cropped_phases_df.columns:
            rel = pd.to_numeric(cropped_phases_df["cropped_release_frame"], errors="coerce").dropna()
            if len(rel):
                ax.axvline(float(rel.mean()), color="red", linestyle="--", linewidth=1.5, label="Avg release")

        ax.legend(loc="best", fontsize=8)

    def refresh():
        current_tab = notebook.tab(notebook.select(), "text")
        spec = tab_specs[current_tab]
        curves = spec["curves"]
        if not curves:
            return
        idx = curve_index[current_tab] % len(curves)
        curve = curves[idx]
        curve_index[current_tab] = idx
        selected_mode = mode_var.get()

        title_var.set(
            f"{current_tab}: {curve} ({idx + 1}/{len(curves)})  |  Right={selected_mode}  |  "
            "Use ←/→ or Prev/Next to change curve"
        )

        unaligned = spec["unaligned"]
        aligned = spec["mode_sets"].get(selected_mode, {})

        for ax in ax_map[current_tab]:
            ax.cla()

        _plot_dataset(
            ax=ax_map[current_tab][0],
            dfs=unaligned,
            curve=curve,
            label="Unaligned",
            ylabel=spec["ylabel"],
            show_release=spec["show_release"],
        )

        if aligned:
            _plot_dataset(
                ax=ax_map[current_tab][1],
                dfs=aligned,
                curve=curve,
                label=selected_mode,
                ylabel=spec["ylabel"],
                show_release=spec["show_release"],
            )
        else:
            ax = ax_map[current_tab][1]
            ax.set_title(selected_mode)
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No data for this mode",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        left_stats_label, right_stats_label = stats_map[current_tab]
        left_stats_label.configure(text="Unaligned Stats\n" + _format_dataset_stats(unaligned, curve))
        if aligned:
            right_stats_label.configure(text=f"{selected_mode} Stats\n" + _format_dataset_stats(aligned, curve))
        else:
            right_stats_label.configure(text=f"{selected_mode} Stats\nVisible throws: 0 | No data")

        fig_map[current_tab].suptitle(
            f"{current_tab} Alignment Review — {curve}",
            fontsize=12,
            fontweight="bold",
        )
        fig_map[current_tab].tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        canvas_map[current_tab].draw()

    def next_curve(event=None):
        current_tab = notebook.tab(notebook.select(), "text")
        curve_index[current_tab] += 1
        refresh()

    def prev_curve(event=None):
        current_tab = notebook.tab(notebook.select(), "text")
        curve_index[current_tab] -= 1
        refresh()

    btn_next.configure(command=next_curve)
    btn_prev.configure(command=prev_curve)

    root.bind("<Right>", next_curve)
    root.bind("<Left>", prev_curve)
    notebook.bind("<<NotebookTabChanged>>", lambda e: refresh())
    mode_box.bind("<<ComboboxSelected>>", lambda e: refresh())

    refresh()
    root.mainloop()

    return {
        "metrics_dir": str(metrics_dir),
        "tabs_loaded": tab_names,
        "angle_curve_count": len(angle_curves),
        "keypoint_curve_count": len(keypoint_curves),
        "ball_curve_count": len(ball_curves),
    }
