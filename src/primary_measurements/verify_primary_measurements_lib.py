"""Interactive GUI to verify primary measurements across all camera views."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from primary_measurements.pose_2d_lib import LANDMARK_NAMES
from utils.io_utils import PROJECT_ROOT
from utils.preprocess_utils import extract_base_freethrow_name
from utils.view_images import close_all_windows


POSE_CONNECTIONS: list[tuple[int, int]] = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (23, 24),
    (11, 23), (12, 24),
    (23, 25), (25, 27), (27, 31),
    (24, 26), (26, 28), (28, 32),
]

HAND_KEYPOINT_STEMS: list[str] = [
    "right_wrist", "left_wrist",
    "right_index", "left_index",
    "right_pinky", "left_pinky",
    "right_thumb", "left_thumb",
]


def _draw_side_hands(frame: np.ndarray, df: pd.DataFrame, frame_idx: int) -> None:
    if df.empty or frame_idx < 0 or frame_idx >= len(df):
        return
    row = df.iloc[frame_idx]

    pts: dict[str, tuple[int, int] | None] = {}
    for stem in HAND_KEYPOINT_STEMS:
        x = pd.to_numeric(row.get(f"{stem}_x", np.nan), errors="coerce")
        y = pd.to_numeric(row.get(f"{stem}_y", np.nan), errors="coerce")
        if np.isfinite(x) and np.isfinite(y):
            pts[stem] = (int(round(float(x))), int(round(float(y))))
        else:
            pts[stem] = None

    # Minimal hand skeleton cues.
    pairs = [
        ("left_wrist", "left_thumb"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_pinky"),
        ("right_wrist", "right_thumb"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_pinky"),
    ]
    for a, b in pairs:
        pa, pb = pts.get(a), pts.get(b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, (255, 120, 40), 2, cv2.LINE_AA)
    for p in pts.values():
        if p is not None:
            cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_num(df: pd.DataFrame, col: str, idx: int) -> float:
    if df.empty or col not in df.columns or idx < 0 or idx >= len(df):
        return float("nan")
    try:
        return float(pd.to_numeric(df.iloc[idx][col], errors="coerce"))
    except Exception:
        return float("nan")


def _xy_from_row(row: pd.Series, stem: str, w: int, h: int) -> tuple[int, int] | None:
    x = pd.to_numeric(row.get(f"{stem}_x", np.nan), errors="coerce")
    y = pd.to_numeric(row.get(f"{stem}_y", np.nan), errors="coerce")
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if 0.0 <= x <= 2.0 and 0.0 <= y <= 2.0:
        x *= w
        y *= h
    if not (0 <= x < w and 0 <= y < h):
        return None
    return int(round(float(x))), int(round(float(y)))


def _draw_pose(frame: np.ndarray, df: pd.DataFrame, frame_idx: int, vis_thresh: float) -> None:
    if df.empty or frame_idx < 0 or frame_idx >= len(df):
        return

    row = df.iloc[frame_idx]
    points: list[tuple[int, int] | None] = []
    for stem in LANDMARK_NAMES:
        pt = _xy_from_row(row, stem, frame.shape[1], frame.shape[0])
        vis = pd.to_numeric(row.get(f"{stem}_v", np.nan), errors="coerce")
        if np.isfinite(vis):
            points.append(pt if vis >= vis_thresh else None)
        else:
            points.append(pt)

    for s, e in POSE_CONNECTIONS:
        if 0 <= s < len(points) and 0 <= e < len(points):
            if points[s] and points[e]:
                cv2.line(frame, points[s], points[e], (255, 120, 40), 2, cv2.LINE_AA)
    for pt in points:
        if pt:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)


def _draw_ball_detection(frame: np.ndarray, df: pd.DataFrame, frame_idx: int, min_conf: float, color: tuple[int, int, int]) -> None:
    if df.empty or frame_idx < 0 or frame_idx >= len(df):
        return
    x = _safe_num(df, "x", frame_idx)
    y = _safe_num(df, "y", frame_idx)
    if not np.isfinite(x) or not np.isfinite(y):
        return
    conf = _safe_num(df, "conf", frame_idx) if "conf" in df.columns else np.nan
    if np.isfinite(conf) and conf < min_conf:
        return
    cv2.circle(frame, (int(round(x)), int(round(y))), 10, color, 2, cv2.LINE_AA)
    if np.isfinite(conf):
        cv2.putText(frame, f"{conf:.2f}", (int(round(x)) + 12, int(round(y)) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_side_trajectory(frame: np.ndarray, df: pd.DataFrame, frame_idx: int, min_conf: float) -> None:
    if df.empty or frame_idx < 0:
        return
    max_idx = min(frame_idx, len(df) - 1)
    pts: list[tuple[int, int]] = []
    for i in range(max_idx + 1):
        x = _safe_num(df, "x", i)
        y = _safe_num(df, "y", i)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        conf = _safe_num(df, "conf", i) if "conf" in df.columns else np.nan
        if np.isfinite(conf) and conf < min_conf:
            continue
        pts.append((int(round(x)), int(round(y))))
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 220, 220), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 7, (0, 255, 255), 2, cv2.LINE_AA)


def _video_length(path: Path) -> int:
    if not path.exists():
        return 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _open_cap(path: Path) -> cv2.VideoCapture | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    return cap


def _read_at(cap: cv2.VideoCapture | None, idx: int) -> np.ndarray | None:
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, idx)))
    ok, frame = cap.read()
    return frame if ok else None


def _placeholder_frame(text: str, w: int = 640, h: int = 360) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return frame


def _resize_h(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = float(target_h) / float(max(h, 1))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(frame, (nw, target_h), interpolation=cv2.INTER_AREA)


def _panel_with_title(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 30), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def _draw_global_text(canvas: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def _safe_pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.1f}%"


def _safe_dist(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.1f}px"


def _hand_ball_min_dist_px(
    ball_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    frame_idx: int,
    min_ball_conf: float,
) -> float:
    if ball_df.empty or hand_df.empty:
        return float("nan")
    if frame_idx < 0 or frame_idx >= len(ball_df) or frame_idx >= len(hand_df):
        return float("nan")

    bx = _safe_num(ball_df, "x", frame_idx)
    by = _safe_num(ball_df, "y", frame_idx)
    if not np.isfinite(bx) or not np.isfinite(by):
        return float("nan")
    if "conf" in ball_df.columns:
        conf = _safe_num(ball_df, "conf", frame_idx)
        if not np.isfinite(conf) or conf < min_ball_conf:
            return float("nan")

    dists: list[float] = []
    for stem in HAND_KEYPOINT_STEMS:
        hx = _safe_num(hand_df, f"{stem}_x", frame_idx)
        hy = _safe_num(hand_df, f"{stem}_y", frame_idx)
        if np.isfinite(hx) and np.isfinite(hy):
            dists.append(float(np.hypot(hx - bx, hy - by)))
    if not dists:
        return float("nan")
    return float(np.min(dists))


def _outcome_map(analysis_dir: Path) -> dict[str, str]:
    csv_path = analysis_dir / "outcomes.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        file_val = str(row.get("file", ""))
        base = extract_base_freethrow_name(file_val)
        if not base:
            continue
        label = str(row.get("outcome", "")).strip()
        if label:
            out[base] = label
    return out


def _coverage_map(metrics_dir: Path) -> dict[str, dict[str, float]]:
    path = metrics_dir / "detection_coverage_verification" / "detection_coverage_per_trial.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        base = extract_base_freethrow_name(str(row.get("file", "")))
        if not base:
            continue
        out[base] = {k: row[k] for k in df.columns if k.endswith("_pct")}
    return out


@dataclass
class TrialAssets:
    base: str
    left_video: Path
    right_video: Path
    side_video: Path
    left_2d: pd.DataFrame
    right_2d: pd.DataFrame
    side_ball: pd.DataFrame
    side_hands: pd.DataFrame
    keypoints_3d: pd.DataFrame
    left_n: int
    right_n: int
    side_n: int


class PrimaryMeasurementsVerificationGui:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.player_fps = float(cfg.get("player_tracking_fps", 60.0))
        self.ball_fps = float(cfg.get("ball_tracking_fps", 30.0))
        self.pose_vis_thresh = float(cfg.get("pose_visibility_threshold", 0.6))
        self.min_ball_conf = float(cfg.get("verification_min_ball_conf", 0.2))

        self.metrics_dir = _format_path(cfg["paths"]["primary_measurements"], cfg)
        self.analysis_dir = _format_path(cfg["paths"]["analysis"], cfg)
        self.left_dir = _format_path(cfg["paths"]["player_tracking_left"], cfg)
        self.right_dir = _format_path(cfg["paths"]["player_tracking_right"], cfg)
        self.side_dir = _format_path(cfg["paths"]["ball_tracking_raw"], cfg)
        self.kp2d_dir = _format_path(cfg["paths"]["keypoints_2d"], cfg)
        self.kp3d_dir = _format_path(cfg["paths"]["keypoints_3d"], cfg)
        self.side_ball_dir = self.metrics_dir / "raw_ball_trajectories"
        side_pose_dir = self.metrics_dir / "side_pose_2d"
        side_hands_compat_dir = self.metrics_dir / "side_hand_tracking"
        self.side_hands_dir = side_pose_dir if side_pose_dir.exists() else side_hands_compat_dir

        self.coverage = _coverage_map(self.metrics_dir)
        self.outcomes = _outcome_map(self.analysis_dir)
        self.trials = self._discover_trials()
        if not self.trials:
            raise ValueError("No freethrow trials found for verification GUI.")

        self.idx = 0
        self.frame_idx = 0
        self.playing = True
        self.playback_speed_mult = 1
        self.fast_mode = False
        self.show_pose = True
        self.show_ball_cam_pose = True
        self.show_side_ball = True
        self.show_pose_3d = True
        self.window_name = "Verify Primary Measurements"
        self.target_panel_h = int(cfg.get("verification_gui_panel_height", 360))

        self.left_cap: cv2.VideoCapture | None = None
        self.right_cap: cv2.VideoCapture | None = None
        self.side_cap: cv2.VideoCapture | None = None
        self.current: TrialAssets | None = None
        self._load_trial(self.idx)

    def _discover_trials(self) -> list[str]:
        bases: set[str] = set()
        for d in [self.left_dir, self.right_dir, self.side_dir]:
            if d.exists():
                for p in d.glob("*.avi"):
                    b = extract_base_freethrow_name(p.stem)
                    if b:
                        bases.add(b)
        return sorted(bases)

    def _load_trial(self, trial_idx: int) -> None:
        self._release_caps()
        base = self.trials[trial_idx]

        left_video = self.left_dir / f"{base}.avi"
        right_video = self.right_dir / f"{base}.avi"
        side_video = self.side_dir / f"{base}.avi"

        left_2d = _load_csv(self.kp2d_dir / f"{base}_left_2d.csv")
        right_2d = _load_csv(self.kp2d_dir / f"{base}_right_2d.csv")
        side_ball = _load_csv(self.side_ball_dir / f"{base}.csv")
        side_hands = _load_csv(self.side_hands_dir / f"{base}.csv")
        keypoints_3d = _load_csv(self.kp3d_dir / f"{base}_3d.csv")

        self.current = TrialAssets(
            base=base,
            left_video=left_video,
            right_video=right_video,
            side_video=side_video,
            left_2d=left_2d,
            right_2d=right_2d,
            side_ball=side_ball,
            side_hands=side_hands,
            keypoints_3d=keypoints_3d,
            left_n=_video_length(left_video),
            right_n=_video_length(right_video),
            side_n=_video_length(side_video),
        )
        self.left_cap = _open_cap(left_video)
        self.right_cap = _open_cap(right_video)
        self.side_cap = _open_cap(side_video)
        self.frame_idx = 0

    def _release_caps(self) -> None:
        for cap in [self.left_cap, self.right_cap, self.side_cap]:
            if cap is not None:
                cap.release()
        self.left_cap = None
        self.right_cap = None
        self.side_cap = None

    def _next_trial(self) -> None:
        self.idx = (self.idx + 1) % len(self.trials)
        self._load_trial(self.idx)

    def _prev_trial(self) -> None:
        self.idx = (self.idx - 1) % len(self.trials)
        self._load_trial(self.idx)

    def _build_canvas(self) -> np.ndarray:
        assert self.current is not None
        cur = self.current

        left_frame = _read_at(self.left_cap, self.frame_idx)
        right_frame = _read_at(self.right_cap, self.frame_idx)

        ball_frame_idx = int(round(self.frame_idx * self.ball_fps / max(self.player_fps, 1.0)))
        side_frame = _read_at(self.side_cap, ball_frame_idx)

        if left_frame is None:
            left_frame = _placeholder_frame(f"{cur.base} left missing")
        if right_frame is None:
            right_frame = _placeholder_frame(f"{cur.base} right missing")
        if side_frame is None:
            side_frame = _placeholder_frame(f"{cur.base} ball cam missing")

        if self.show_pose:
            _draw_pose(left_frame, cur.left_2d, self.frame_idx, self.pose_vis_thresh)
            _draw_pose(right_frame, cur.right_2d, self.frame_idx, self.pose_vis_thresh)
        if self.show_ball_cam_pose:
            _draw_side_hands(side_frame, cur.side_hands, ball_frame_idx)

        if self.show_side_ball:
            _draw_ball_detection(side_frame, cur.side_ball, ball_frame_idx, self.min_ball_conf, (0, 255, 255))
            _draw_side_trajectory(side_frame, cur.side_ball, ball_frame_idx, self.min_ball_conf)

        pose3d_frame = self._build_3d_pose_panel(cur.keypoints_3d, self.frame_idx, h=self.target_panel_h)

        side_hand_ball_dist = _hand_ball_min_dist_px(
            ball_df=cur.side_ball,
            hand_df=cur.side_hands,
            frame_idx=ball_frame_idx,
            min_ball_conf=self.min_ball_conf,
        )
        cv2.putText(
            side_frame,
            f"hand-ball: {_safe_dist(side_hand_ball_dist)}",
            (12, max(36, int(self.target_panel_h * 0.1))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        left_frame = _panel_with_title(_resize_h(left_frame, self.target_panel_h), "Stereo Left")
        right_frame = _panel_with_title(_resize_h(right_frame, self.target_panel_h), "Stereo Right")
        side_frame = _panel_with_title(_resize_h(side_frame, self.target_panel_h), "Ball Cam")
        pose3d_frame = _panel_with_title(_resize_h(pose3d_frame, self.target_panel_h), "3D Pose")
        panels = [left_frame, right_frame, side_frame]
        if self.show_pose_3d:
            panels.append(pose3d_frame)
        panel = cv2.hconcat(panels)

        top_pad = np.zeros((180, panel.shape[1], 3), dtype=np.uint8)
        canvas = cv2.vconcat([top_pad, panel])

        outcome = self.outcomes.get(cur.base, "unlabeled")
        cov = self.coverage.get(cur.base, {})
        lines = [
            f"Trial: {cur.base} ({self.idx + 1}/{len(self.trials)})  Outcome: {outcome}",
            f"Frame: player {self.frame_idx} | ball {ball_frame_idx}  Playing: {'yes' if self.playing else 'no'}  Speed: {self.playback_speed_mult}x  Fast[6/F]={'ON' if self.fast_mode else 'OFF'}",
            f"Overlay toggles: pose[1]={self.show_pose}  side-ball+traj[3]={self.show_side_ball}  ball-cam-pose[4]={self.show_ball_cam_pose}  3d-pose[5]={self.show_pose_3d}",
            f"Coverage: 2D L {_safe_pct(float(cov.get('two_d_left_valid_pct', np.nan)))} | 2D R {_safe_pct(float(cov.get('two_d_right_valid_pct', np.nan)))} | 3D {_safe_pct(float(cov.get('three_d_valid_pct', np.nan)))}",
            f"Coverage: side-ball {_safe_pct(float(cov.get('ball_side_valid_pct', np.nan)))}",
            f"Current min hand-ball dist: side {_safe_dist(side_hand_ball_dist)}",
            "Controls: Left/Right=prev/next trial, Space=play/pause, R=restart, A/D=frame -/+, 6/F=toggle 5x, Q=quit",
        ]
        _draw_global_text(canvas, lines)
        return canvas

    @staticmethod
    def _is_left_arrow(key: int) -> bool:
        return key in (81, 2424832, 2)

    @staticmethod
    def _is_right_arrow(key: int) -> bool:
        return key in (83, 2555904, 3)

    def _build_3d_pose_panel(self, df: pd.DataFrame, frame_idx: int, *, h: int) -> np.ndarray:
        w = int(max(320, round(h * 1.2)))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if df.empty or frame_idx < 0 or frame_idx >= len(df):
            cv2.putText(canvas, "No 3D keypoints", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
            return canvas

        row = df.iloc[frame_idx]
        points3d: list[tuple[float, float, float] | None] = []
        for stem in LANDMARK_NAMES:
            x = pd.to_numeric(row.get(f"{stem}_x", np.nan), errors="coerce")
            y = pd.to_numeric(row.get(f"{stem}_y", np.nan), errors="coerce")
            z = pd.to_numeric(row.get(f"{stem}_z", np.nan), errors="coerce")
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points3d.append((float(x), float(y), float(z)))
            else:
                points3d.append(None)

        valid = [p for p in points3d if p is not None]
        if not valid:
            cv2.putText(canvas, "3D frame invalid", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
            return canvas

        arr = np.array(valid, dtype=float)
        # Use x (horizontal) and y (vertical) projection; y is inverted to display up.
        xs = arr[:, 0]
        ys = -arr[:, 1]
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        pad = 30.0
        scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)

        pts2d: list[tuple[int, int] | None] = []
        for p in points3d:
            if p is None:
                pts2d.append(None)
                continue
            x2 = int(round((p[0] - min_x) * scale + pad))
            y2 = int(round((-p[1] - min_y) * scale + pad))
            pts2d.append((x2, y2))

        for s, e in POSE_CONNECTIONS:
            if 0 <= s < len(pts2d) and 0 <= e < len(pts2d):
                ps, pe = pts2d[s], pts2d[e]
                if ps is not None and pe is not None:
                    cv2.line(canvas, ps, pe, (120, 220, 120), 2, cv2.LINE_AA)
        for p in pts2d:
            if p is not None:
                cv2.circle(canvas, p, 3, (80, 180, 255), -1, cv2.LINE_AA)
        return canvas

    def run(self) -> dict[str, Any]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while True:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            assert self.current is not None
            cur = self.current
            max_player_frame = max(cur.left_n, cur.right_n, int(round(cur.side_n * self.player_fps / max(self.ball_fps, 1.0))))
            max_player_frame = max(max_player_frame, 1)
            self.frame_idx = int(np.clip(self.frame_idx, 0, max_player_frame - 1))

            canvas = self._build_canvas()
            cv2.imshow(self.window_name, canvas)

            base_playback_fps = max(self.player_fps, 1.0)
            delay = int(round(1000.0 / base_playback_fps)) if self.playing else 30
            key = cv2.waitKey(delay) & 0xFFFFFFFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                self.playing = not self.playing
            elif key in (ord("r"), ord("R")):
                self.frame_idx = 0
            elif key in (ord("f"), ord("F"), ord("6")):
                self.fast_mode = not self.fast_mode
                self.playback_speed_mult = 5 if self.fast_mode else 1
                print(f"[verify primary] playback speed set to {self.playback_speed_mult}x")
            elif key in (ord("1"),):
                self.show_pose = not self.show_pose
            elif key in (ord("3"),):
                self.show_side_ball = not self.show_side_ball
            elif key in (ord("4"),):
                self.show_ball_cam_pose = not self.show_ball_cam_pose
            elif key in (ord("5"),):
                self.show_pose_3d = not self.show_pose_3d
            elif key in (ord("a"), ord("A")):
                self.playing = False
                self.frame_idx = max(0, self.frame_idx - 1)
            elif key in (ord("d"), ord("D")):
                self.playing = False
                self.frame_idx = self.frame_idx + 1
            elif self._is_left_arrow(key):
                self._prev_trial()
            elif self._is_right_arrow(key):
                self._next_trial()

            if self.playing:
                self.frame_idx += max(1, int(self.playback_speed_mult))

            if self.frame_idx >= max_player_frame:
                self.playing = False
                self.frame_idx = max_player_frame - 1

        self.close()
        return {
            "trials_loaded": len(self.trials),
            "metrics_dir": str(self.metrics_dir),
            "coverage_csv": str(self.metrics_dir / "detection_coverage_verification" / "detection_coverage_per_trial.csv"),
        }

    def close(self) -> None:
        self._release_caps()
        close_all_windows()


def run_verify_primary_measurements_gui(cfg: dict[str, Any]) -> dict[str, Any]:
    gui = PrimaryMeasurementsVerificationGui(cfg)
    return gui.run()
