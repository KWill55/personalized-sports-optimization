"""Adaptive freethrow cropping around release-aligned frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.io_utils import PROJECT_ROOT, load_csv_folder
from utils.preprocess_utils import extract_base_freethrow_name
from utils.curve_comparison_viewer import run_curve_comparison_viewer


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
    out_dir.mkdir(parents=True, exist_ok=True)
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


def _resolve_ball_release_map(metrics_dir: Path) -> dict[str, int]:
    phases_path = metrics_dir / "freethrow_phases.csv"
    shifts_path = metrics_dir / "alignment_release_shift_table_ball_cam.csv"
    if not phases_path.exists() or not shifts_path.exists():
        return {}

    phases_df = pd.read_csv(phases_path)
    shifts_df = pd.read_csv(shifts_path)
    if "file" not in phases_df.columns or "file" not in shifts_df.columns or "shift" not in shifts_df.columns:
        return {}

    release_col = None
    for cand in ("raw_release_frame_ball_cam", "raw_release_frame"):
        if cand in phases_df.columns:
            release_col = cand
            break
    if release_col is None:
        return {}

    phases_df = phases_df.copy()
    shifts_df = shifts_df.copy()
    phases_df["base"] = phases_df["file"].apply(extract_base_freethrow_name)
    shifts_df["base"] = shifts_df["file"].apply(extract_base_freethrow_name)
    phases_df[release_col] = pd.to_numeric(phases_df[release_col], errors="coerce")
    shifts_df["shift"] = pd.to_numeric(shifts_df["shift"], errors="coerce")

    merged = phases_df.merge(shifts_df[["base", "shift"]], on="base", how="inner")
    merged = merged.dropna(subset=["base", release_col, "shift"])
    if merged.empty:
        return {}

    merged["aligned_release_frame_ball"] = np.rint(merged[release_col] + merged["shift"]).astype(int)
    return {
        str(r["base"]): int(r["aligned_release_frame_ball"])
        for _, r in merged.iterrows()
    }


def _crop_by_frame_bounds(
    df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> tuple[pd.DataFrame | None, str]:
    if df.empty:
        return None, "empty_df"
    if "frame" not in df.columns:
        return None, "missing_frame_col"

    frame = pd.to_numeric(df["frame"], errors="coerce")
    if frame.isna().all():
        return None, "invalid_frame_col"

    out = df[(frame >= int(start_frame)) & (frame <= int(end_frame))].copy()
    if out.empty:
        return None, "window_out_of_bounds"

    # Keep a fixed global frame window for every trial, but tolerate missing rows
    # by reindexing to the requested aligned frame range.
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["frame"]).copy()
    if out.empty:
        return None, "invalid_frame_col"
    out["frame"] = out["frame"].astype(int)

    full_index = pd.Index(np.arange(int(start_frame), int(end_frame) + 1, dtype=int), name="frame")
    out = out.set_index("frame")
    out = out[~out.index.duplicated(keep="first")]
    out = out.reindex(full_index).reset_index()
    out["aligned_frame"] = out["frame"]

    data_cols = [c for c in out.columns if c not in ("frame", "aligned_frame")]
    missing_count = int(out[data_cols].isna().all(axis=1).sum()) if data_cols else 0
    if missing_count > 0:
        return out, f"ok_with_gaps_{missing_count}"
    return out, "ok"


def _frame_coverage_counts(dfs: dict[str, pd.DataFrame]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for df in dfs.values():
        if df.empty or "frame" not in df.columns:
            continue
        frame_vals = pd.to_numeric(df["frame"], errors="coerce").dropna().astype(int).unique().tolist()
        for f in frame_vals:
            counts[int(f)] = counts.get(int(f), 0) + 1
    return counts


def _adaptive_window_from_coverage(
    dfs: dict[str, pd.DataFrame],
    min_trials_present: int,
) -> tuple[int, int] | None:
    counts = _frame_coverage_counts(dfs)
    if not counts:
        return None
    if int(counts.get(0, 0)) < int(min_trials_present):
        return None

    # Expand from release (0) until coverage falls below threshold.
    start = 0
    end = 0
    while int(counts.get(start - 1, 0)) >= int(min_trials_present):
        start -= 1
    while int(counts.get(end + 1, 0)) >= int(min_trials_present):
        end += 1
    return int(start), int(end)


def _crop_ball_release_plus_frames(
    df: pd.DataFrame,
    release_frame: int,
    post_frames: int,
) -> tuple[pd.DataFrame | None, str]:
    if df.empty:
        return None, "empty_df"

    if "frame" in df.columns:
        frames = pd.to_numeric(df["frame"], errors="coerce")
        if frames.isna().all():
            return None, "invalid_frame_col"
        min_frame = int(frames.min())
        max_frame = int(frames.max())
    else:
        min_frame = 0
        max_frame = len(df) - 1
        df = df.copy()
        df.insert(0, "frame", np.arange(len(df), dtype=int))

    start = int(release_frame)
    end = int(release_frame + post_frames)
    expected_len = post_frames + 1
    if start < min_frame or end > max_frame:
        return None, "window_out_of_bounds"

    out = df[(df["frame"] >= start) & (df["frame"] <= end)].copy()
    if len(out) != expected_len:
        return None, f"unexpected_window_len_{len(out)}"

    out.reset_index(drop=True, inplace=True)
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce") - int(release_frame)
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce").fillna(0).astype(int)
    out["aligned_frame"] = out["frame"]
    return out, "ok"


def _crop_ball_release_to_threshold(
    df: pd.DataFrame,
    release_frame: int,
    *,
    drop_below_release_px: float,
    persist_frames: int,
    release_search_radius: int,
) -> tuple[pd.DataFrame | None, str]:
    if df.empty:
        return None, "empty_df"
    if "frame" not in df.columns or "y" not in df.columns:
        return None, "missing_frame_or_y"

    frame = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    if len(frame) == 0:
        return None, "empty_df"

    # Find release row (or nearest valid around it).
    rel_idx = np.where(frame.astype(int) == int(release_frame))[0]
    if len(rel_idx) == 0:
        rel_idx = np.array([int(np.clip(release_frame, 0, len(df) - 1))], dtype=int)
    rel_idx = int(rel_idx[0])

    release_idx = None
    for off in range(0, max(0, int(release_search_radius)) + 1):
        for cand in (rel_idx - off, rel_idx + off):
            if 0 <= cand < len(df) and np.isfinite(y[cand]):
                release_idx = int(cand)
                break
        if release_idx is not None:
            break
    if release_idx is None:
        return None, "no_valid_release_y"

    release_y = float(y[release_idx])
    # In image coords, larger y means lower in frame.
    y_threshold = release_y + float(drop_below_release_px)

    # Use only valid points from release onward.
    tail_idx = np.arange(release_idx, len(df), dtype=int)
    valid_tail = tail_idx[np.isfinite(y[tail_idx])]
    if len(valid_tail) < 2:
        return None, "insufficient_valid_y_after_release"

    # Peak (highest ball) is minimum y after release.
    peak_local = int(np.argmin(y[valid_tail]))
    peak_idx = int(valid_tail[peak_local])

    p = max(1, int(persist_frames))
    end_idx = len(df) - 1
    found = False
    for i in range(peak_idx, len(df) - p + 1):
        window = y[i : i + p]
        if np.all(np.isfinite(window)) and np.all(window >= y_threshold):
            end_idx = int(i + p - 1)
            found = True
            break
    if not found:
        return None, "threshold_not_reached"

    out = df.iloc[release_idx : end_idx + 1].copy().reset_index(drop=True)
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce") - int(release_frame)
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce").fillna(0).astype(int)
    out["aligned_frame"] = out["frame"]
    return out, "ok"


def run_crop_freethrows_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)

    aligned_keypoints_dir = metrics_dir / "3d_keypoints_aligned_release"
    aligned_angles_dir = metrics_dir / "3d_angles_aligned_release"
    aligned_ball_dir = metrics_dir / "aligned_ball_trajectory_release"
    cropped_keypoints_dir = metrics_dir / "3d_keypoints_aligned_release_cropped"
    cropped_angles_dir = metrics_dir / "3d_angles_aligned_release_cropped"
    cropped_ball_dir = metrics_dir / "aligned_ball_trajectory_release_cropped"
    cropped_phases_path = metrics_dir / "cropped_freethrow_phases.csv"
    report_path = metrics_dir / "release_window_crop_report.csv"

    if not aligned_keypoints_dir.exists():
        raise FileNotFoundError(f"Missing aligned keypoints dir: {aligned_keypoints_dir}")

    keypoints_dfs = _to_base_name_dict(load_csv_folder(aligned_keypoints_dir))
    angles_dfs = _to_base_name_dict(load_csv_folder(aligned_angles_dir)) if aligned_angles_dir.exists() else {}
    ball_dfs = _to_base_name_dict(load_csv_folder(aligned_ball_dir)) if aligned_ball_dir.exists() else {}
    if not keypoints_dfs:
        raise ValueError("No aligned keypoint CSVs found for release-window cropping.")

    player_crop_mode = "adaptive_coverage"
    overwrite_existing = bool(cfg.get("overwrite_existing_outputs", False))

    ball_crop_mode = str(cfg.get("ball_crop_mode", "none")).strip().lower()
    ball_crop_post_frames = int(cfg.get("ball_crop_post_frames", 60))
    ball_drop_below_release_px = float(cfg.get("ball_crop_drop_below_release_px", 0.0))
    ball_persist_frames = int(cfg.get("ball_crop_persist_frames", 2))
    ball_release_search_radius = int(cfg.get("ball_crop_release_search_radius_frames", 8))
    adaptive_min_trials_cfg = cfg.get("adaptive_crop_min_trials")
    adaptive_min_pct = float(cfg.get("adaptive_crop_min_pct", 0.0))
    if adaptive_min_trials_cfg is None:
        adaptive_min_trials = max(1, int(np.ceil(adaptive_min_pct * len(keypoints_dfs)))) if adaptive_min_pct > 0 else max(1, min(10, len(keypoints_dfs)))
    else:
        adaptive_min_trials = max(1, int(adaptive_min_trials_cfg))

    adaptive_window = _adaptive_window_from_coverage(keypoints_dfs, adaptive_min_trials)
    if adaptive_window is None:
        raise ValueError(
            "Could not build adaptive crop window from aligned keypoint coverage. "
            f"Try lowering adaptive_crop_min_trials (current={adaptive_min_trials})."
        )
    bases = sorted(set(keypoints_dfs.keys()))
    if not bases:
        raise ValueError("No overlapping trials across aligned keypoints and release metadata.")

    cropped_keypoints: dict[str, pd.DataFrame] = {}
    cropped_angles: dict[str, pd.DataFrame] = {}
    cropped_ball: dict[str, pd.DataFrame] = {}
    report_rows: list[dict[str, Any]] = []
    w_start, w_end = adaptive_window

    for base in bases:
        kp_out, kp_reason = _crop_by_frame_bounds(keypoints_dfs[base], w_start, w_end)
        ang_out = None
        ang_reason = "angles_not_present"
        if base in angles_dfs:
            ang_out, ang_reason = _crop_by_frame_bounds(angles_dfs[base], w_start, w_end)

        keep = kp_out is not None
        if kp_out is not None:
            cropped_keypoints[base] = kp_out
        if ang_out is not None:
            cropped_angles[base] = ang_out

        ball_reason = "not_cropped_by_design"
        if ball_crop_mode in ("release_to_fixed_window", "release_plus_frames", "release_plus_60"):
            if base in ball_dfs:
                ball_out, ball_reason = _crop_ball_release_plus_frames(
                    ball_dfs[base],
                    release_frame=0,
                    post_frames=max(1, ball_crop_post_frames),
                )
                if ball_out is not None:
                    cropped_ball[base] = ball_out
            else:
                ball_reason = "missing_aligned_ball"
        elif ball_crop_mode in ("release_to_threshold", "release_to_below_release_height"):
            if base in ball_dfs:
                ball_out, ball_reason = _crop_ball_release_to_threshold(
                    ball_dfs[base],
                    release_frame=0,
                    drop_below_release_px=ball_drop_below_release_px,
                    persist_frames=ball_persist_frames,
                    release_search_radius=ball_release_search_radius,
                )
                if ball_out is not None:
                    cropped_ball[base] = ball_out
            else:
                ball_reason = "missing_aligned_ball"

        report_rows.append(
            {
                "file": f"{base}.avi",
                "aligned_release_frame": 0,
                "keypoints_crop_status": kp_reason,
                "angles_crop_status": ang_reason,
                "ball_crop_status": ball_reason,
                "kept": bool(keep),
            }
        )

    if not cropped_keypoints:
        raise RuntimeError("No trials survived fixed-window crop. Lower pre/post crop sizes.")

    kp_written, kp_skipped = _save_csv_dict(cropped_keypoints, cropped_keypoints_dir, overwrite_existing)
    ang_written, ang_skipped = (0, 0)
    if cropped_angles:
        ang_written, ang_skipped = _save_csv_dict(cropped_angles, cropped_angles_dir, overwrite_existing)
    ball_written, ball_skipped = (0, 0)
    if cropped_ball:
        ball_written, ball_skipped = _save_csv_dict(cropped_ball, cropped_ball_dir, overwrite_existing)
    kept_bases = sorted(set(cropped_keypoints.keys()))
    cropped_first = int(w_start)
    cropped_last = int(w_end)
    cropped_phases_df = pd.DataFrame(
        {
            "file": [f"{b}.avi" for b in kept_bases],
            "cropped_first_frame": cropped_first,
            "cropped_release_frame": 0,
            "cropped_last_frame": cropped_last,
            "window_pre_frames": abs(cropped_first),
            "window_post_frames": cropped_last,
            "player_crop_mode": player_crop_mode,
        }
    )
    cropped_phases_df.to_csv(cropped_phases_path, index=False)

    report_df = pd.DataFrame(report_rows).sort_values("file").reset_index(drop=True)
    report_df.to_csv(report_path, index=False)

    print("Release-window crop complete.")
    print(
        "Player crop mode: adaptive_coverage "
        f"(min_trials_present={adaptive_min_trials}, window={w_start}..{w_end})"
    )
    print(f"Cropped keypoints written: {kp_written} (skipped existing: {kp_skipped})")
    if cropped_angles:
        print(f"Cropped angles written: {ang_written} (skipped existing: {ang_skipped})")
    else:
        print("Cropped angles: skipped (aligned angles not present).")
    if ball_crop_mode in ("release_to_fixed_window", "release_plus_frames", "release_plus_60"):
        print(
            "Ball crop mode: release_to_fixed_window "
            f"(post_frames={max(1, ball_crop_post_frames)})"
        )
        print(f"Cropped ball written: {ball_written} (skipped existing: {ball_skipped})")
    elif ball_crop_mode in ("release_to_threshold", "release_to_below_release_height"):
        print(
            "Ball crop mode: release_to_threshold "
            f"(drop_below_release_px={ball_drop_below_release_px}, persist={ball_persist_frames})"
        )
        print(f"Cropped ball written: {ball_written} (skipped existing: {ball_skipped})")
    else:
        print("Ball trajectory: not cropped by release window (uses aligned full-length files).")
    print(f"Saved cropped phases: {cropped_phases_path}")
    print(f"Saved crop report: {report_path}")

    return {
        "metrics_dir": str(metrics_dir),
        "window_pre_frames": abs(int(w_start)),
        "window_post_frames": int(w_end),
        "player_crop_mode": player_crop_mode,
        "adaptive_crop_min_trials": adaptive_min_trials,
        "adaptive_window_start": int(w_start),
        "adaptive_window_end": int(w_end),
        "cropped_keypoints_dir": str(cropped_keypoints_dir),
        "cropped_angles_dir": str(cropped_angles_dir) if cropped_angles else "",
        "cropped_ball_dir": str(cropped_ball_dir) if cropped_ball else "",
        "cropped_phases_path": str(cropped_phases_path),
        "crop_report_path": str(report_path),
        "keypoints_written": kp_written,
        "angles_written": ang_written,
        "ball_written": ball_written,
        "kept_trials": len(kept_bases),
    }


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


def run_crop_viewer(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)

    angles_aligned_dir = metrics_dir / "3d_angles_aligned_release"
    angles_cropped_dir = metrics_dir / "3d_angles_aligned_release_cropped"
    keypoints_aligned_dir = metrics_dir / "3d_keypoints_aligned_release"
    keypoints_cropped_dir = metrics_dir / "3d_keypoints_aligned_release_cropped"
    ball_aligned_dir = metrics_dir / "aligned_ball_trajectory_release"
    ball_cropped_dir = metrics_dir / "aligned_ball_trajectory_release_cropped"
    cropped_phases_path = metrics_dir / "cropped_freethrow_phases.csv"

    angles_aligned = _to_base_name_dict(load_csv_folder(angles_aligned_dir)) if angles_aligned_dir.exists() else {}
    angles_cropped = _to_base_name_dict(load_csv_folder(angles_cropped_dir)) if angles_cropped_dir.exists() else {}
    keypoints_aligned = _to_base_name_dict(load_csv_folder(keypoints_aligned_dir)) if keypoints_aligned_dir.exists() else {}
    keypoints_cropped = _to_base_name_dict(load_csv_folder(keypoints_cropped_dir)) if keypoints_cropped_dir.exists() else {}
    ball_aligned = _to_base_name_dict(load_csv_folder(ball_aligned_dir)) if ball_aligned_dir.exists() else {}
    ball_cropped = _to_base_name_dict(load_csv_folder(ball_cropped_dir)) if ball_cropped_dir.exists() else {}

    angle_curves = _common_numeric_columns([angles_aligned], exclude={"frame"})
    keypoint_curves = _common_numeric_columns([keypoints_aligned], exclude={"frame"})
    ball_curves = _common_numeric_columns([ball_aligned], exclude={"frame"})

    if not angle_curves and not keypoint_curves and not ball_curves:
        raise ValueError("No aligned/cropped datasets found for crop viewer. Run crop first.")

    tab_specs = {
        "Angles": {
            "left": angles_aligned,
            "right": angles_cropped,
            "left_label": "Aligned (Pre-Crop)",
            "curves": angle_curves,
            "ylabel": "Angle",
            "show_release": True,
        },
        "Keypoints": {
            "left": keypoints_aligned,
            "right": keypoints_cropped,
            "left_label": "Aligned (Pre-Crop)",
            "curves": keypoint_curves,
            "ylabel": "Position",
            "show_release": True,
        },
        "Ball": {
            "left": ball_aligned,
            "right": ball_cropped,
            "left_label": "Aligned (Pre-Crop)",
            "curves": ball_curves,
            "ylabel": "Ball Metric",
            "show_release": True,
        },
    }

    release_frames = np.array([], dtype=float)
    if cropped_phases_path.exists():
        phases_df = pd.read_csv(cropped_phases_path)
        if "cropped_release_frame" in phases_df.columns:
            release_frames = pd.to_numeric(phases_df["cropped_release_frame"], errors="coerce").dropna().to_numpy(dtype=float)

    ball_release_map = _resolve_ball_release_map(metrics_dir)
    ball_release_frames_left = np.array(sorted(ball_release_map.values()), dtype=float) if ball_release_map else np.array([], dtype=float)
    ball_release_frames_right = np.array([0.0], dtype=float)

    viewer_result = run_curve_comparison_viewer(
        window_title="Crop Verification",
        tab_specs={
            **tab_specs,
            "Ball": {
                **tab_specs["Ball"],
                "release_frames_left": ball_release_frames_left,
                "release_frames_right": ball_release_frames_right,
            },
        },
        right_mode_label="Cropped",
        release_frames=release_frames,
    )
    return {
        "metrics_dir": str(metrics_dir),
        **viewer_result,
    }
