"""Golden-template comparison viewer for release-aligned freethrows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from utils.io_utils import PROJECT_ROOT, load_csv_folder
from utils.preprocess_utils import extract_base_freethrow_name


@dataclass(frozen=True)
class MetricKey:
    source: str
    column: str

    def label(self) -> str:
        return f"{self.source}::{self.column}"


def _format_path(template_or_path: str, cfg: dict[str, Any]) -> Path:
    return PROJECT_ROOT / Path(template_or_path.format(athlete=cfg["athlete"], session=cfg["session"]))


def _session_root(athlete: str, session: str) -> Path:
    return PROJECT_ROOT / "data" / athlete / session


def _metrics_dir(athlete: str, session: str) -> Path:
    session_root = _session_root(athlete, session)
    primary = session_root / "primary_measurements"
    if primary.exists():
        return primary
    return session_root / "metrics"


def _analysis_dir(athlete: str, session: str) -> Path:
    return _session_root(athlete, session) / "analysis"


def _secondary_dir(athlete: str, session: str) -> Path:
    return _session_root(athlete, session) / "secondary_measurements"


def _to_base_name_dict(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in dfs.items():
        base = extract_base_freethrow_name(name)
        if base:
            out[base] = df
    return out


def _load_release_map(metrics_dir: Path) -> dict[str, int]:
    phases_path = metrics_dir / "freethrow_phases.csv"
    shifts_path = metrics_dir / "alignment_release_shift_table.csv"
    if not phases_path.exists() or not shifts_path.exists():
        return {}

    phases = pd.read_csv(phases_path)
    shifts = pd.read_csv(shifts_path)
    if "file" not in phases.columns or "raw_release_frame" not in phases.columns:
        return {}
    if "file" not in shifts.columns or "shift" not in shifts.columns:
        return {}

    phases = phases.copy()
    shifts = shifts.copy()
    phases["base"] = phases["file"].apply(extract_base_freethrow_name)
    shifts["base"] = shifts["file"].apply(extract_base_freethrow_name)
    phases["raw_release_frame"] = pd.to_numeric(phases["raw_release_frame"], errors="coerce")
    shifts["shift"] = pd.to_numeric(shifts["shift"], errors="coerce")
    merged = phases.merge(shifts[["base", "shift"]], on="base", how="inner")
    merged = merged.dropna(subset=["base", "raw_release_frame", "shift"])
    if merged.empty:
        return {}

    merged["aligned_release"] = np.rint(merged["raw_release_frame"] + merged["shift"]).astype(int)
    return {str(r["base"]): int(r["aligned_release"]) for _, r in merged.iterrows()}


def _load_outcomes_map(analysis_dir: Path) -> dict[str, str]:
    candidates = [
        analysis_dir / "outcomes.csv",
        analysis_dir.parent / "primary_measurements" / "outcomes.csv",
    ]
    out_path = next((p for p in candidates if p.exists()), None)
    if out_path is None:
        return {}
    df = pd.read_csv(out_path)
    if "file" not in df.columns or "outcome" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        base = extract_base_freethrow_name(r["file"])
        if base:
            out[base] = str(r["outcome"])
    return out


def _load_raw_release_map(metrics_dir: Path) -> dict[str, int]:
    phases_path = metrics_dir / "freethrow_phases.csv"
    if not phases_path.exists():
        return {}
    phases = pd.read_csv(phases_path)
    if "file" not in phases.columns:
        return {}
    release_col = None
    for cand in ("raw_release_frame_stereo", "raw_release_frame"):
        if cand in phases.columns:
            release_col = cand
            break
    if release_col is None:
        return {}

    phases = phases.copy()
    phases["base"] = phases["file"].apply(extract_base_freethrow_name)
    phases[release_col] = pd.to_numeric(phases[release_col], errors="coerce")
    phases = phases.dropna(subset=["base", release_col])
    if phases.empty:
        return {}
    return {str(r["base"]): int(r[release_col]) for _, r in phases.iterrows()}


def _load_sources_for_session(
    athlete: str,
    session: str,
) -> tuple[dict[str, dict[str, Any]], Path]:
    metrics_dir = _metrics_dir(athlete, session)
    secondary_dir = _secondary_dir(athlete, session)

    def _pick(path_candidates: list[Path]) -> tuple[dict[str, pd.DataFrame], Path | None]:
        for p in path_candidates:
            if p.exists():
                return _to_base_name_dict(load_csv_folder(p)), p
        return {}, None

    keypoints_dfs, keypoints_path = _pick(
        [
            metrics_dir / "3d_keypoints_aligned_release",
            metrics_dir / "3d_keypoints",
        ]
    )
    angles_dfs, angles_path = _pick(
        [
            secondary_dir / "3d_angles",
            metrics_dir / "3d_angles_aligned_release",
            metrics_dir / "3d_angles",
        ]
    )
    ball_dfs, ball_path = _pick(
        [
            metrics_dir / "aligned_ball_trajectory_release",
            metrics_dir / "raw_ball_trajectories",
        ]
    )

    release_raw = _load_raw_release_map(metrics_dir)

    sources = {
        "keypoints": {
            "dfs": keypoints_dfs,
            "path": keypoints_path,
            "release_map": {},
            "already_relative": bool(keypoints_path and "aligned_release" in keypoints_path.name),
        },
        "angles": {
            "dfs": angles_dfs,
            "path": angles_path,
            "release_map": release_raw,
            "already_relative": bool(angles_path and "aligned_release" in angles_path.name),
        },
        "ball": {
            "dfs": ball_dfs,
            "path": ball_path,
            "release_map": {},
            "already_relative": bool(ball_path and "aligned_release" in ball_path.name),
        },
    }
    return sources, metrics_dir


def _load_metric_sources(metrics_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    sources = {
        "angles": metrics_dir / "3d_angles_aligned_release",
        "keypoints": metrics_dir / "3d_keypoints_aligned_release",
        "ball": metrics_dir / "aligned_ball_trajectory_release",
    }
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name, path in sources.items():
        if path.exists():
            out[name] = _to_base_name_dict(load_csv_folder(path))
        else:
            out[name] = {}
    return out


def _numeric_columns_union(dfs: dict[str, pd.DataFrame], *, exclude: set[str] | None = None) -> set[str]:
    exclude = exclude or set()
    cols: set[str] = set()
    for df in dfs.values():
        cols |= set(df.select_dtypes(include=[np.number]).columns)
    return {c for c in cols if c not in exclude}


def _build_metric_list(
    template_sources: dict[str, dict[str, pd.DataFrame]],
    athlete_sources: dict[str, dict[str, pd.DataFrame]],
) -> list[MetricKey]:
    metrics: list[MetricKey] = []
    for source in ("angles", "keypoints", "ball"):
        t_cols = _numeric_columns_union(template_sources.get(source, {}), exclude={"frame"})
        a_cols = _numeric_columns_union(athlete_sources.get(source, {}), exclude={"frame"})
        shared = sorted(t_cols & a_cols)
        for c in shared:
            metrics.append(MetricKey(source=source, column=c))
    return metrics


def _extract_relative_series(
    df: pd.DataFrame,
    column: str,
    release_frame: int,
    *,
    already_relative: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if column not in df.columns:
        return None
    y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if "frame" in df.columns:
        frame = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
    else:
        frame = np.arange(len(df), dtype=float)

    valid = np.isfinite(frame) & np.isfinite(y)
    if not np.any(valid):
        return None
    t = frame[valid] if already_relative else (frame[valid] - float(release_frame))
    yy = y[valid]
    order = np.argsort(t)
    return t[order], yy[order]


def _template_stats_for_metric(
    template_dfs: dict[str, pd.DataFrame],
    release_map: dict[str, int],
    column: str,
    *,
    already_relative: bool = False,
    min_curves: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for base, df in template_dfs.items():
        if (not already_relative) and (base not in release_map):
            continue
        rel_frame = int(release_map.get(base, 0))
        rel = _extract_relative_series(df, column, rel_frame, already_relative=already_relative)
        if rel is not None:
            series.append(rel)

    if len(series) < min_curves:
        return None

    t_min = int(max(np.floor(np.min(t)) for t, _ in series))
    t_max = int(min(np.ceil(np.max(t)) for t, _ in series))
    if t_max <= t_min:
        return None

    t_grid = np.arange(t_min, t_max + 1, dtype=float)
    stack: list[np.ndarray] = []
    for t, y in series:
        if len(np.unique(t)) < 2:
            continue
        interp = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
        stack.append(interp)
    if len(stack) < min_curves:
        return None

    mat = np.vstack(stack)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    valid = np.isfinite(mean) & np.isfinite(std)
    if not np.any(valid):
        return None
    return t_grid[valid], mean[valid], std[valid]


def _series_error_against_template(
    t_throw: np.ndarray,
    y_throw: np.ndarray,
    t_tpl: np.ndarray,
    y_tpl: np.ndarray,
) -> tuple[float, float, int]:
    if len(t_throw) < 2 or len(t_tpl) < 2:
        return float("nan"), float("nan"), 0
    t_lo = max(float(np.min(t_throw)), float(np.min(t_tpl)))
    t_hi = min(float(np.max(t_throw)), float(np.max(t_tpl)))
    if t_hi <= t_lo:
        return float("nan"), float("nan"), 0

    mask = (t_throw >= t_lo) & (t_throw <= t_hi)
    if np.sum(mask) < 2:
        return float("nan"), float("nan"), 0
    t_eval = t_throw[mask]
    y_eval = y_throw[mask]
    y_tpl_eval = np.interp(t_eval, t_tpl, y_tpl)
    diff = y_eval - y_tpl_eval
    rmse = float(np.sqrt(np.mean(diff * diff)))

    tpl_span = float(np.nanpercentile(y_tpl, 95) - np.nanpercentile(y_tpl, 5))
    denom = tpl_span if np.isfinite(tpl_span) and tpl_span > 1e-6 else 1.0
    pct_off = float(100.0 * rmse / denom)
    return rmse, pct_off, len(t_eval)


def _find_video_for_base(base: str, folder: Path | None) -> Path | None:
    if folder is None or not folder.exists():
        return None
    for ext in ("avi", "mp4", "mov", "mkv"):
        hits = sorted(folder.glob(f"{base}*.{ext}"))
        if hits:
            return hits[0]
    return None


def _read_frame(video_path: Path | None, frame_idx: int) -> np.ndarray | None:
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
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
        ax.text(0.5, 0.5, "No frame", transform=ax.transAxes, ha="center", va="center", color="white")
        return
    ax.imshow(img)


def run_golden_template_viewer(cfg: dict[str, Any]) -> dict[str, Any]:
    from matplotlib.widgets import Button

    viewer_mode = str(cfg.get("golden_template_mode", "golden")).strip().lower()
    is_consistency_mode = viewer_mode == "consistency"
    template_athlete = str(cfg.get("golden_template_athlete", "tyler_haws"))
    template_session = str(cfg.get("golden_template_session", "session_2025-10-9"))
    athlete = str(cfg.get("athlete"))
    session = str(cfg.get("session"))

    if not athlete or not session:
        raise ValueError("Current athlete/session must be set in project_config.yaml")

    t_sources, t_metrics_dir = _load_sources_for_session(template_athlete, template_session)
    a_sources, a_metrics_dir = _load_sources_for_session(athlete, session)
    outcomes = _load_outcomes_map(_analysis_dir(athlete, session))

    source_order = ["keypoints", "angles", "ball"]
    source_titles = {"keypoints": "Keypoints", "angles": "Angles", "ball": "Ball Trajectory"}
    metric_columns: dict[str, list[str]] = {}
    template_stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {s: {} for s in source_order}

    for source in source_order:
        t_dfs = t_sources[source]["dfs"]
        a_dfs = a_sources[source]["dfs"]
        shared = sorted(
            _numeric_columns_union(t_dfs, exclude={"frame"}) & _numeric_columns_union(a_dfs, exclude={"frame"})
        )
        if source == "ball":
            preferred = [c for c in ("x", "y") if c in shared]
            others = [c for c in shared if c not in ("x", "y")]
            shared = preferred + others
        metric_columns[source] = shared

        for col in shared:
            tpl = _template_stats_for_metric(
                template_dfs=t_dfs,
                release_map=t_sources[source]["release_map"],
                column=col,
                already_relative=bool(t_sources[source]["already_relative"]),
                min_curves=3,
            )
            if tpl is not None:
                template_stats[source][col] = tpl
        metric_columns[source] = [c for c in metric_columns[source] if c in template_stats[source]]

    if not any(metric_columns[s] for s in source_order):
        raise ValueError(
            "No shared template metrics available across keypoints/angles/ball. "
            "Make sure Tyler has those measurements computed in this session."
        )

    candidate_bases = sorted(
        set().union(*[set(a_sources[s]["dfs"].keys()) for s in source_order if a_sources[s]["dfs"]])
    )
    if not candidate_bases:
        raise ValueError("No athlete freethrows found for golden-template comparison.")

    template_out_dir = a_metrics_dir / ("consistency" if is_consistency_mode else "golden_template")
    template_out_dir.mkdir(parents=True, exist_ok=True)
    for source in source_order:
        for col, (t_rel, y_mean, y_std) in template_stats[source].items():
            pd.DataFrame({"t_rel": t_rel, "template_mean": y_mean, "template_std": y_std}).to_csv(
                template_out_dir / f"{source}__{col}.csv",
                index=False,
            )

    default_metric = {
        "keypoints": "right_wrist_y",
        "angles": "elbow_flex_r",
        "ball": "y",
    }
    metric_idx: dict[str, int] = {}
    for source in source_order:
        cols = metric_columns[source]
        if not cols:
            metric_idx[source] = 0
            continue
        if default_metric[source] in cols:
            metric_idx[source] = cols.index(default_metric[source])
        else:
            metric_idx[source] = 0

    state: dict[str, Any] = {"throw_idx": 0, "metric_idx": metric_idx}
    y_limits: dict[str, dict[str, tuple[float, float]]] = {s: {} for s in source_order}
    x_limits: dict[str, dict[str, tuple[float, float]]] = {s: {} for s in source_order}
    for source in source_order:
        for col in metric_columns[source]:
            t_tpl, y_mean, y_std = template_stats[source][col]
            t_finite = t_tpl[np.isfinite(t_tpl)]
            if t_finite.size == 0:
                x_limits[source][col] = (-1.0, 1.0)
            else:
                x_limits[source][col] = (float(np.min(t_finite)), float(np.max(t_finite)))
            y_ref = np.concatenate([y_mean - y_std, y_mean + y_std]).astype(float)
            finite = y_ref[np.isfinite(y_ref)]
            if finite.size == 0:
                y_limits[source][col] = (-1.0, 1.0)
                continue
            y_min = float(np.min(finite))
            y_max = float(np.max(finite))
            pad = max(1e-6, 0.08 * max(1e-6, y_max - y_min))
            y_limits[source][col] = (y_min - pad, y_max + pad)

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    fig.subplots_adjust(top=0.82, bottom=0.11)
    axes = {
        "keypoints": fig.add_subplot(gs[0, 0]),
        "angles": fig.add_subplot(gs[0, 1]),
        "ball": fig.add_subplot(gs[0, 2]),
    }

    button_map: dict[str, tuple[Button, Button]] = {}
    for i, source in enumerate(source_order):
        x0 = 0.07 + i * 0.315
        ax_prev = fig.add_axes([x0, 0.02, 0.06, 0.05])
        ax_next = fig.add_axes([x0 + 0.07, 0.02, 0.06, 0.05])
        b_prev = Button(ax_prev, "Prev")
        b_next = Button(ax_next, "Next")
        button_map[source] = (b_prev, b_next)

    header_left = (
        f"Consistency Baseline: {template_athlete}/{template_session}"
        if is_consistency_mode
        else f"Template: {template_athlete}/{template_session}"
    )
    header_right = (
        f"Athlete Session: {athlete}/{session}"
        if is_consistency_mode
        else f"Compare: {athlete}/{session}"
    )
    fig.text(0.01, 0.985, f"{header_left}  |  {header_right}", fontsize=10, fontweight="bold", va="top")
    fig.text(
        0.01,
        0.962,
        "Controls: \u2190/\u2192 throw, q quit | Per-graph Prev/Next buttons switch metric",
        fontsize=9,
        va="top",
    )

    def _render_source(ax: Any, source: str, base: str) -> tuple[float | None, int]:
        ax.clear()
        cols = metric_columns[source]
        if not cols:
            ax.set_title(f"{source_titles[source]}: no shared metrics")
            ax.grid(alpha=0.2)
            return None, 0

        col = cols[state["metric_idx"][source] % len(cols)]
        t_tpl, y_mean, y_std = template_stats[source][col]
        ax.plot(t_tpl, y_mean, color="orange", linewidth=2.0, label="Tyler Mean")
        ax.fill_between(
            t_tpl,
            y_mean - y_std,
            y_mean + y_std,
            color="orange",
            alpha=0.20,
            label="\u00b11 std",
        )

        rel_frame = int(a_sources[source]["release_map"].get(base, 0))
        df = a_sources[source]["dfs"].get(base, pd.DataFrame())
        rel_series = _extract_relative_series(
            df,
            col,
            rel_frame,
            already_relative=bool(a_sources[source]["already_relative"]),
        )

        in_band_pct_txt = "n/a"
        n_eval = 0
        if rel_series is not None and len(rel_series[0]) > 0:
            t_throw, y_throw = rel_series
            t_lo = max(float(np.min(t_throw)), float(np.min(t_tpl)))
            t_hi = min(float(np.max(t_throw)), float(np.max(t_tpl)))
            mask = (t_throw >= t_lo) & (t_throw <= t_hi)
            if np.any(mask):
                t_eval = t_throw[mask]
                y_eval = y_throw[mask]
                y_mean_eval = np.interp(t_eval, t_tpl, y_mean)
                y_std_eval = np.interp(t_eval, t_tpl, y_std)
                in_band = np.abs(y_eval - y_mean_eval) <= y_std_eval
                n_eval = int(len(t_eval))
                if np.any(in_band):
                    ax.scatter(t_eval[in_band], y_eval[in_band], s=14, color="#2ca02c", label="Within std", zorder=4)
                if np.any(~in_band):
                    ax.scatter(t_eval[~in_band], y_eval[~in_band], s=14, color="#d62728", label="Out of std", zorder=4)
                in_band_pct = 100.0 * float(np.mean(in_band))
                in_band_pct_txt = f"{in_band_pct:.1f}%"

        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Frames Relative To Release")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        if col in x_limits[source]:
            xmin, xmax = x_limits[source][col]
            ax.set_xlim(xmin, xmax)
        if col in y_limits[source]:
            ymin, ymax = y_limits[source][col]
            ax.set_ylim(ymin, ymax)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{source_titles[source]} | {col} | in-band: {in_band_pct_txt}", fontsize=10)
        if in_band_pct_txt == "n/a":
            return None, n_eval
        return float(in_band_pct_txt.replace("%", "")), n_eval

    def _render() -> None:
        base = candidate_bases[state["throw_idx"]]
        outcome = outcomes.get(base, "unknown")
        weighted_hits = 0.0
        weighted_total = 0
        for s in source_order:
            pct, n_eval = _render_source(axes[s], s, base)
            if pct is not None and n_eval > 0:
                weighted_hits += (pct / 100.0) * n_eval
                weighted_total += n_eval
        overall_txt = "n/a"
        if weighted_total > 0:
            overall_txt = f"{100.0 * (weighted_hits / weighted_total):.1f}%"
        title_prefix = "Form Consistency" if is_consistency_mode else "Golden Template Comparison"
        fig.suptitle(
            f"{title_prefix}  |  throw {state['throw_idx'] + 1}/{len(candidate_bases)} "
            f"({base}, outcome={outcome})  |  overall in-band={overall_txt}",
            fontsize=12,
            fontweight="bold",
            y=0.92,
        )
        fig.canvas.draw_idle()

    def _make_metric_handler(source: str, delta: int):
        def _fn(_: Any) -> None:
            cols = metric_columns[source]
            if not cols:
                return
            state["metric_idx"][source] = (state["metric_idx"][source] + delta) % len(cols)
            _render()

        return _fn

    for source in source_order:
        prev_b, next_b = button_map[source]
        prev_b.on_clicked(_make_metric_handler(source, -1))
        next_b.on_clicked(_make_metric_handler(source, +1))

    def _on_key(event: Any) -> None:
        if event.key == "right":
            state["throw_idx"] = (state["throw_idx"] + 1) % len(candidate_bases)
            _render()
        elif event.key == "left":
            state["throw_idx"] = (state["throw_idx"] - 1) % len(candidate_bases)
            _render()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _render()
    plt.show()

    return {
        "template_athlete": template_athlete,
        "template_session": template_session,
        "athlete": athlete,
        "session": session,
        "throws_available": len(candidate_bases),
        "template_output_dir": str(template_out_dir),
        "metrics_per_source": {k: len(v) for k, v in metric_columns.items()},
    }


def run_golden_template_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    return run_golden_template_viewer(cfg)
