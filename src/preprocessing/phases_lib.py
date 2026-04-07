"""Rules based free-throw release frame labeling"""

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
        base = extract_base_freethrow_name(name)
        if base:
            out[base] = df
    return out


def run_phases_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    phases_path = _format_path(cfg["paths"]["phases"], cfg)
    side_release_path = metrics_dir / "side_ball_release_frames.csv"
    side_pose_dir = metrics_dir / "side_pose_2d"
    side_hands_compat_dir = metrics_dir / "side_hand_tracking"
    side_hands_dir = side_pose_dir if side_pose_dir.exists() else side_hands_compat_dir
    side_ball_dir = metrics_dir / "raw_ball_trajectories"
    
    if not side_hands_dir.exists():
        raise FileNotFoundError(
            f"Missing side pose/hand tracking folder: {side_hands_dir}. "
            "Run primary side pose extraction first."
        )
    if not side_ball_dir.exists():
        raise FileNotFoundError(f"Missing side ball trajectory folder: {side_ball_dir}")

    min_ball_conf = float(cfg.get("side_release_min_ball_conf", 0.20))
    distance_threshold_px = float(cfg.get("side_release_distance_threshold_px", 50.0))
    above_gap_px = float(cfg.get("side_release_ball_above_min_gap_px", 0.0))
    min_hits = int(cfg.get("side_release_min_hits", 2))
    player_fps = float(cfg.get("player_tracking_fps", 60.0))
    ball_fps = float(cfg.get("ball_tracking_fps", 30.0))
    frame_scale = player_fps / max(ball_fps, 1e-6)
    hand_map = {
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
    }

    def _release_from_side_ball_and_hands(ball_df: pd.DataFrame, hands_df: pd.DataFrame) -> int | None:
        if ball_df.empty:
            return None
        if hands_df.empty:
            return None

        n = min(len(ball_df), len(hands_df))
        bx = pd.to_numeric(ball_df.get("x", pd.Series([np.nan] * n)), errors="coerce").to_numpy(dtype=float)[:n]
        by = pd.to_numeric(ball_df.get("y", pd.Series([np.nan] * n)), errors="coerce").to_numpy(dtype=float)[:n]
        bconf = pd.to_numeric(ball_df.get("conf", pd.Series([np.nan] * n)), errors="coerce").to_numpy(dtype=float)[:n]
        ball_valid = np.isfinite(bx) & np.isfinite(by) & np.isfinite(bconf) & (bconf >= min_ball_conf)

        hit_idx: list[int] = []
        for i in range(n):
            if not ball_valid[i]:
                continue
            dists: list[float] = []
            ys: list[float] = []
            for k in hand_map:
                hx = pd.to_numeric(hands_df.iloc[i].get(f"{k}_x", np.nan), errors="coerce")
                hy = pd.to_numeric(hands_df.iloc[i].get(f"{k}_y", np.nan), errors="coerce")
                if np.isfinite(hx) and np.isfinite(hy):
                    dists.append(float(np.hypot(float(hx) - bx[i], float(hy) - by[i])))
                    ys.append(float(hy))
            if not dists or not ys:
                continue
            min_dist = float(np.min(dists))
            hand_top_y = float(np.min(ys))
            ball_above_hands = bool(by[i] < (hand_top_y - above_gap_px))
            if ball_above_hands and min_dist >= distance_threshold_px:
                hit_idx.append(i)

        if len(hit_idx) >= max(1, min_hits):
            return int(hit_idx[0])
        return None

    side_ball = _to_base_name_dict(load_csv_folder(side_ball_dir))
    side_hands = _to_base_name_dict(load_csv_folder(side_hands_dir))
    bases = sorted(set(side_ball.keys()) | set(side_hands.keys()))
    if not bases:
        raise ValueError(f"No side trial CSVs found in: {side_ball_dir} and {side_hands_dir}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for base in bases:
        try:
            release = _release_from_side_ball_and_hands(
                side_ball.get(base, pd.DataFrame()),
                side_hands.get(base, pd.DataFrame()),
            )
            frames_total = int(max(len(side_ball.get(base, pd.DataFrame())), len(side_hands.get(base, pd.DataFrame()))))
            rows.append(
                {
                    "file": f"{base}.avi",
                    "raw_release_frame": release,
                    "raw_release_frame_ball_cam": release,
                    "raw_release_frame_stereo": (int(round(float(release) * frame_scale)) if release is not None else np.nan),
                    "release_method": "side_ball_hand_distance_above_rule",
                    "frames_total": frames_total if frames_total > 0 else np.nan,
                    "phase_confidence": float(1.0) if release is not None else float(0.0),
                    "wrist_speed_threshold": np.nan,
                    "wrist_speed_peak": np.nan,
                }
            )
        except Exception as exc:
            failures.append({"file": base, "error": str(exc)})

    if not rows:
        raise RuntimeError("Side-camera phase labeling produced no rows.")

    phases_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    phases_df["raw_release_frame"] = pd.to_numeric(phases_df["raw_release_frame"], errors="coerce")
    phases_df["raw_release_frame_ball_cam"] = pd.to_numeric(phases_df["raw_release_frame_ball_cam"], errors="coerce")
    phases_df["raw_release_frame_stereo"] = pd.to_numeric(phases_df["raw_release_frame_stereo"], errors="coerce")
    phases_df[
        [
            "file",
            "raw_release_frame",
            "raw_release_frame_ball_cam",
            "raw_release_frame_stereo",
        ]
    ].to_csv(side_release_path, index=False)

    valid_phases_df = phases_df.dropna(subset=["raw_release_frame"]).copy()
    if valid_phases_df.empty:
        raise RuntimeError("No valid release frames found from side ball+hand rule.")

    valid_phases_df["raw_release_frame"] = valid_phases_df["raw_release_frame"].astype(int)
    valid_phases_df["raw_release_frame_ball_cam"] = valid_phases_df["raw_release_frame_ball_cam"].astype(int)
    valid_phases_df["raw_release_frame_stereo"] = valid_phases_df["raw_release_frame_stereo"].astype(int)
    valid_phases_df.to_csv(phases_path, index=False)

    failure_path = phases_path.with_name("freethrow_phases_failures.csv")
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)

    total = int(len(phases_df))
    valid = int(len(valid_phases_df))
    print(f"Labeled side-camera phases for {valid}/{total} clips (failed rows: {len(failures)}).")
    print(f"Saved phases: {phases_path}")
    print(f"Saved side release table: {side_release_path}")
    print(f"Release frame conversion scale (ball->stereo): {frame_scale:.3f}x")

    return {
        "phases_path": str(phases_path),
        "labeled_clips": valid,
        "total_trials_seen": total,
        "unlabeled_trials": total - valid,
        "failed_clips": len(failures),
        "failures_path": str(failure_path) if failures else "",
        "side_release_frames_path": str(side_release_path),
        "distance_threshold_px": distance_threshold_px,
        "min_hits": min_hits,
        "ball_to_stereo_frame_scale": frame_scale,
    }


def _sort_bases(bases: list[str]) -> list[str]:
    def _base_num(base: str) -> int:
        digits = "".join(ch for ch in base if ch.isdigit())
        return int(digits) if digits else 10**9

    return sorted(bases, key=lambda b: (_base_num(b), b))


def _safe_series(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col not in df.columns:
        return np.full(n, np.nan, dtype=float)
    out = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)), constant_values=np.nan)
    return out[:n]


def _candidate_video_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    out: dict[str, Path] = {}
    for key in ("player_tracking_sync", "player_tracking_left", "player_tracking_right", "ball_tracking_raw"):
        if key in paths:
            out[key] = _format_path(paths[key], cfg)
    return out


def _find_video_for_base(base: str, folder: Path | None) -> Path | None:
    if folder is None or not folder.exists():
        return None
    for ext in ("avi", "mp4", "mov", "mkv"):
        hits = sorted(folder.glob(f"{base}*.{ext}"))
        if hits:
            return hits[0]
    return None


def run_phase_verification_gui(cfg: dict[str, Any]) -> dict[str, Any]:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import Button, RadioButtons

    metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
    phases_path = _format_path(cfg["paths"]["phases"], cfg)
    keypoints_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)
    ball_dir = metrics_dir / "raw_ball_trajectories"
    if not ball_dir.exists():
        ball_dir = metrics_dir / "cropped_ball_trajectory"
    if not ball_dir.exists():
        ball_dir = metrics_dir / "aligned_ball_trajectory_unsigned_area"

    if not phases_path.exists():
        raise FileNotFoundError(f"Missing phases CSV: {phases_path}")

    phases_df = pd.read_csv(phases_path)
    if "file" not in phases_df.columns:
        raise ValueError(f"Invalid phases CSV format: {phases_path}")
    phases_df["base"] = phases_df["file"].apply(extract_base_freethrow_name)
    phases_df = phases_df[phases_df["base"].notna()].copy()
    if phases_df.empty:
        raise ValueError("No valid freethrow names found in phases CSV.")
    phases_df["is_2d"] = phases_df["file"].str.contains("_2d", case=False, na=False)
    phases_df = phases_df.sort_values(["base", "is_2d"])
    phases_df = phases_df.groupby("base", as_index=False).first()
    phases_by_base: dict[str, dict[str, Any]] = {
        row["base"]: row.to_dict() for _, row in phases_df.iterrows()
    }

    keypoints_dfs = _to_base_name_dict(load_csv_folder(keypoints_dir)) if keypoints_dir.exists() else {}
    ball_dfs = _to_base_name_dict(load_csv_folder(ball_dir)) if ball_dir.exists() else {}

    bases = _sort_bases(phases_df["base"].tolist())
    if not bases:
        raise ValueError("No clips available for phase verification.")

    video_dirs = _candidate_video_dirs(cfg)
    feed_names = ["player_tracking_left", "player_tracking_right", "ball_tracking_raw"]
    feed_labels = {
        "player_tracking_left": "Left Feed",
        "player_tracking_right": "Right Feed",
        "ball_tracking_raw": "Ball Feed",
    }
    player_fps = float(cfg.get("player_tracking_fps", 60.0))
    ball_fps = float(cfg.get("ball_tracking_fps", 30.0))
    ratio_ball_per_stereo = ball_fps / max(player_fps, 1e-6)

    state: dict[str, Any] = {
        "clip_idx": 0,
        "frame_idx": 0,
        "ball_frame_idx": 0,
        "mode": "Keypoints",
        "playing": False,
        "edit_target": "raw_release_frame",
        "dirty": set(),
        "caps": {k: None for k in feed_names},
        "clip_base": None,
        "clip_len": 1,
        "ball_clip_len": 1,
        "release_stereo_frame": 0,
        "release_ball_frame": 0,
        "n_clips": len(bases),
        "phase_row": None,
        "keypoints_df": pd.DataFrame(),
        "ball_df": pd.DataFrame(),
    }

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.2, 1.2, 1.1], hspace=0.35, wspace=0.25)
    video_axes = [
        fig.add_subplot(gs[0:2, 0:2]),
        fig.add_subplot(gs[0:2, 2:4]),
        fig.add_subplot(gs[0:2, 4:6]),
    ]
    data_ax = fig.add_subplot(gs[2, 0:6])

    btn_prev_ax = fig.add_axes([0.02, 0.93, 0.08, 0.045])
    btn_next_ax = fig.add_axes([0.11, 0.93, 0.08, 0.045])
    btn_play_ax = fig.add_axes([0.20, 0.93, 0.08, 0.045])
    btn_save_clip_ax = fig.add_axes([0.29, 0.93, 0.09, 0.045])
    btn_save_all_ax = fig.add_axes([0.39, 0.93, 0.09, 0.045])
    mode_ax = fig.add_axes([0.82, 0.91, 0.16, 0.09])
    btn_prev = Button(btn_prev_ax, "Prev Clip")
    btn_next = Button(btn_next_ax, "Next Clip")
    btn_play = Button(btn_play_ax, "Play/Pause")
    btn_save_clip = Button(btn_save_clip_ax, "Save Clip")
    btn_save_all = Button(btn_save_all_ax, "Save All")
    radio = RadioButtons(mode_ax, ("Keypoints", "Trajectory"), active=0)
    mode_ax.set_title("Bottom Plot", fontsize=9)
    fig.text(
        0.02,
        0.89,
        "Edit: R set release at current frame | [ ] nudge ±1 | { } nudge ±5 | Ctrl+S save all",
        fontsize=9,
    )

    timer = fig.canvas.new_timer(interval=33)

    def _release_caps() -> None:
        for key in feed_names:
            cap = state["caps"].get(key)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            state["caps"][key] = None

    def _normalize_release_frame(phase: dict[str, Any]) -> None:
        rf_st = int(phase.get("raw_release_frame_stereo", phase.get("raw_release_frame", 0)))
        rf_st = int(np.clip(rf_st, 0, max(0, state["clip_len"] - 1)))
        rf_ball = int(phase.get("raw_release_frame_ball_cam", round(rf_st * ratio_ball_per_stereo)))
        rf_ball = int(np.clip(rf_ball, 0, max(0, state["ball_clip_len"] - 1)))
        phase["raw_release_frame_stereo"] = rf_st
        phase["raw_release_frame_ball_cam"] = rf_ball
        phase["raw_release_frame"] = rf_st

    def _stereo_to_ball_frame(stereo_idx: int) -> int:
        sf = int(stereo_idx)
        rf_st = int(state.get("release_stereo_frame", 0))
        rf_ball = int(state.get("release_ball_frame", 0))
        bf = int(round((sf - rf_st) * ratio_ball_per_stereo + rf_ball))
        return int(np.clip(bf, 0, max(0, state["ball_clip_len"] - 1)))

    def _load_clip(idx: int) -> None:
        idx = int(np.clip(idx, 0, len(bases) - 1))
        state["clip_idx"] = idx
        base = bases[idx]
        state["clip_base"] = base

        row = phases_by_base.get(base)
        state["phase_row"] = pd.Series(row) if row else None
        state["keypoints_df"] = keypoints_dfs.get(base, pd.DataFrame())
        state["ball_df"] = ball_dfs.get(base, pd.DataFrame())

        lengths = [
            len(state["keypoints_df"]) if not state["keypoints_df"].empty else 0,
        ]
        ball_lengths = [len(state["ball_df"]) if not state["ball_df"].empty else 0]

        _release_caps()
        for feed in feed_names:
            vp = _find_video_for_base(base, video_dirs.get(feed))
            if vp is not None:
                cap = cv2.VideoCapture(str(vp))
                if cap.isOpened():
                    state["caps"][feed] = cap
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if feed == "ball_tracking_raw":
                        ball_lengths.append(n_frames)
                    else:
                        lengths.append(n_frames)

        state["clip_len"] = max(1, max(lengths) if lengths else 1)
        state["ball_clip_len"] = max(1, max(ball_lengths) if ball_lengths else 1)
        if row:
            _normalize_release_frame(row)
            state["release_stereo_frame"] = int(row.get("raw_release_frame_stereo", row.get("raw_release_frame", 0)))
            state["release_ball_frame"] = int(row.get("raw_release_frame_ball_cam", round(state["release_stereo_frame"] * ratio_ball_per_stereo)))
            state["frame_idx"] = int(np.clip(state["release_stereo_frame"], 0, state["clip_len"] - 1))
            state["ball_frame_idx"] = int(np.clip(state["release_ball_frame"], 0, state["ball_clip_len"] - 1))
        else:
            state["frame_idx"] = 0
            state["ball_frame_idx"] = 0
            state["release_stereo_frame"] = 0
            state["release_ball_frame"] = 0

    def _read_frame(cap: Any, frame_idx: int) -> np.ndarray | None:
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _draw_video_panel(ax: Any, img: np.ndarray | None, title: str) -> None:
        ax.clear()
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.set_facecolor("#111111")
            ax.text(0.5, 0.5, "No feed", transform=ax.transAxes, color="white", ha="center", va="center")
            return
        ax.imshow(img)

    def _phase_lines(ax: Any, current_frame: int, release_frame: int) -> None:
        row = state["phase_row"]
        if row is None:
            return
        ax.axvline(int(release_frame), color="#d62728", linestyle="--", linewidth=1.2, label="Release")
        ax.axvline(int(current_frame), color="black", linewidth=1.5, alpha=0.8, label="Current")

    def _draw_data_panel() -> None:
        data_ax.clear()
        mode = state["mode"]
        if mode == "Keypoints":
            df = state["keypoints_df"]
            if df.empty:
                data_ax.text(0.5, 0.5, "No 3D keypoint data", transform=data_ax.transAxes, ha="center", va="center")
            else:
                n = len(df)
                x = np.arange(n)
                rw_y = _safe_series(df, "right_wrist_y", n)
                rs_y = _safe_series(df, "right_shoulder_y", n)
                rh_y = _safe_series(df, "right_hip_y", n)
                data_ax.plot(x, rw_y, label="right_wrist_y", linewidth=1.8)
                data_ax.plot(x, rs_y, label="right_shoulder_y", linewidth=1.2)
                data_ax.plot(x, rh_y, label="right_hip_y", linewidth=1.2)
                data_ax.set_ylabel("3D position")
                data_ax.set_title("Keypoint Traces")

        else:
            df = state["ball_df"]
            if df.empty:
                data_ax.text(0.5, 0.5, "No ball trajectory data", transform=data_ax.transAxes, ha="center", va="center")
            else:
                n = len(df)
                x = np.arange(n)
                if "y" in df.columns:
                    data_ax.plot(x, _safe_series(df, "y", n), label="ball_y", linewidth=1.8)
                if "x" in df.columns:
                    data_ax.plot(x, _safe_series(df, "x", n), label="ball_x", linewidth=1.2)
                if "conf" in df.columns:
                    data_ax.plot(x, _safe_series(df, "conf", n), label="ball_conf", linewidth=1.0, alpha=0.8)
                data_ax.set_ylabel("trajectory")
                data_ax.set_title("Ball Trajectory")

        if mode == "Trajectory":
            _phase_lines(
                data_ax,
                current_frame=state["ball_frame_idx"],
                release_frame=int(state.get("release_ball_frame", 0)),
            )
        else:
            _phase_lines(
                data_ax,
                current_frame=state["frame_idx"],
                release_frame=int(state.get("release_stereo_frame", 0)),
            )
        data_ax.set_xlabel("Frame")
        data_ax.grid(alpha=0.25)
        handles, labels = data_ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            data_ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    def _render() -> None:
        base = state["clip_base"] or "unknown"
        state["ball_frame_idx"] = _stereo_to_ball_frame(state["frame_idx"])
        for i, feed in enumerate(feed_names):
            frame_idx = state["ball_frame_idx"] if feed == "ball_tracking_raw" else state["frame_idx"]
            frame = _read_frame(state["caps"].get(feed), frame_idx)
            _draw_video_panel(video_axes[i], frame, f"{feed_labels[feed]} (f={frame_idx})")
        _draw_data_panel()

        row = state["phase_row"]
        conf_txt = ""
        if row is not None and "phase_confidence" in row:
            conf_txt = f" | confidence={float(row['phase_confidence']):.2f}"
        dirty_flag = " *edited*" if base in state["dirty"] else ""
        target = state["edit_target"].replace("raw_", "").replace("_frame", "").replace("_", " ")
        fig.suptitle(
            f"Verify Phases — {base} ({state['clip_idx'] + 1}/{state['n_clips']}) | "
            f"stereo frame {state['frame_idx']}/{max(0, state['clip_len'] - 1)} | "
            f"ball frame {state['ball_frame_idx']}/{max(0, state['ball_clip_len'] - 1)} | "
            f"target={target}{conf_txt}{dirty_flag}",
            fontsize=12,
            fontweight="bold",
        )
        fig.canvas.draw_idle()

    def _step_frame(delta: int) -> None:
        state["frame_idx"] = int(np.clip(state["frame_idx"] + delta, 0, state["clip_len"] - 1))
        _render()

    def _prev_clip(event: Any = None) -> None:
        _load_clip(state["clip_idx"] - 1)
        _render()

    def _next_clip(event: Any = None) -> None:
        _load_clip(state["clip_idx"] + 1)
        _render()

    def _toggle_play(event: Any = None) -> None:
        state["playing"] = not state["playing"]

    def _set_marker(col: str, value: int) -> None:
        base = state["clip_base"]
        if base is None:
            return
        phase = phases_by_base.get(base)
        if phase is None:
            phase = {"base": base, "file": f"{base}.avi"}
            phases_by_base[base] = phase
        phase[col] = int(np.clip(value, 0, state["clip_len"] - 1))
        # Keep stereo and ball release columns synchronized around current mapping.
        phase["raw_release_frame_stereo"] = int(np.clip(value, 0, state["clip_len"] - 1))
        phase["raw_release_frame_ball_cam"] = _stereo_to_ball_frame(int(value))
        phase["raw_release_frame"] = int(phase["raw_release_frame_stereo"])
        _normalize_release_frame(phase)
        state["release_stereo_frame"] = int(phase["raw_release_frame_stereo"])
        state["release_ball_frame"] = int(phase["raw_release_frame_ball_cam"])
        state["phase_row"] = pd.Series(phase)
        state["dirty"].add(base)
        _render()

    def _nudge_marker(delta: int) -> None:
        base = state["clip_base"]
        if base is None:
            return
        phase = phases_by_base.get(base)
        if phase is None:
            return
        col = state["edit_target"]
        current = int(phase.get(col, state["frame_idx"]))
        _set_marker(col, current + delta)

    def _save_phases(save_only_current: bool) -> None:
        if save_only_current and state["clip_base"] is not None:
            selected_base = state["clip_base"]
            base_rows = [phases_by_base[selected_base]]
            existing = pd.read_csv(phases_path)
            existing["base"] = existing["file"].apply(extract_base_freethrow_name)
            keep = existing[existing["base"] != selected_base].copy()
            write_df = pd.concat([keep.drop(columns=["base"], errors="ignore"), pd.DataFrame(base_rows)], ignore_index=True)
        else:
            write_df = pd.DataFrame(phases_by_base.values())

        for col in ("raw_release_frame", "raw_release_frame_stereo", "raw_release_frame_ball_cam"):
            if col in write_df.columns:
                write_df[col] = pd.to_numeric(write_df[col], errors="coerce").fillna(0).astype(int)
        if "file" not in write_df.columns and "base" in write_df.columns:
            write_df["file"] = write_df["base"].astype(str) + ".avi"

        helper_cols = {
            "base",
            "is_2d",
            "raw_windup_start",
            "raw_followthrough_end",
            "windup_start",
            "followthrough_end",
        }
        ordered_front = [c for c in ["file", "raw_release_frame", "raw_release_frame_stereo", "raw_release_frame_ball_cam"] if c in write_df.columns]
        remaining = [c for c in write_df.columns if c not in set(ordered_front) | helper_cols]
        write_df = write_df[ordered_front + remaining]
        write_df = write_df.sort_values("file")
        phases_path.parent.mkdir(parents=True, exist_ok=True)
        write_df.to_csv(phases_path, index=False)

        if save_only_current and state["clip_base"] is not None:
            state["dirty"].discard(state["clip_base"])
        else:
            state["dirty"].clear()
        _render()

    def _on_timer() -> None:
        if not state["playing"]:
            return
        if state["frame_idx"] >= state["clip_len"] - 1:
            state["playing"] = False
            return
        state["frame_idx"] += 1
        _render()

    def _on_key(event: Any) -> None:
        if event.key in ("left", ","):
            _step_frame(-1)
        elif event.key in ("right", "."):
            _step_frame(1)
        elif event.key == "up":
            _step_frame(10)
        elif event.key == "down":
            _step_frame(-10)
        elif event.key in ("n",):
            _next_clip()
        elif event.key in ("p",):
            _prev_clip()
        elif event.key == " ":
            _toggle_play()
        elif event.key in ("r",):
            state["edit_target"] = "raw_release_frame"
            _set_marker("raw_release_frame", state["frame_idx"])
        elif event.key == "[":
            _nudge_marker(-1)
        elif event.key == "]":
            _nudge_marker(1)
        elif event.key == "{":
            _nudge_marker(-5)
        elif event.key == "}":
            _nudge_marker(5)
        elif event.key in ("ctrl+s", "cmd+s"):
            _save_phases(save_only_current=False)

    def _on_mode(label: str) -> None:
        state["mode"] = label
        _render()

    def _on_close(event: Any) -> None:
        _release_caps()
        try:
            timer.stop()
        except Exception:
            pass

    btn_prev.on_clicked(_prev_clip)
    btn_next.on_clicked(_next_clip)
    btn_play.on_clicked(_toggle_play)
    btn_save_clip.on_clicked(lambda e: _save_phases(save_only_current=True))
    btn_save_all.on_clicked(lambda e: _save_phases(save_only_current=False))
    radio.on_clicked(_on_mode)
    timer.add_callback(_on_timer)
    timer.start()
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)

    _load_clip(0)
    _render()
    plt.show()

    return {
        "verified_clips": len(bases),
        "phases_path": str(phases_path),
    }
