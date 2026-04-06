"""Reusable two-panel curve comparison viewer for preprocessing verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def _plot_dataset(
    ax: Any,
    dfs: dict[str, pd.DataFrame],
    curve: str,
    label: str,
    ylabel: str,
    show_release: bool,
    release_frames: np.ndarray,
) -> None:
    ax.set_title(label)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    x_to_vals: dict[int, list[float]] = {}
    for df in dfs.values():
        if curve not in df.columns:
            continue
        y = df[curve].to_numpy(dtype=float)
        x = df["frame"].to_numpy(dtype=float) if "frame" in df.columns else np.arange(len(y), dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        xv = x[valid]
        yv = y[valid]
        ax.plot(xv, yv, color="gray", alpha=0.18, linewidth=1)
        for xi, yi in zip(xv, yv):
            k = int(round(float(xi)))
            x_to_vals.setdefault(k, []).append(float(yi))

    if x_to_vals:
        x_mean = np.array(sorted(x_to_vals.keys()), dtype=float)
        mean_curve = np.array([np.mean(x_to_vals[int(xi)]) for xi in x_mean], dtype=float)
        std_curve = np.array([np.std(x_to_vals[int(xi)]) for xi in x_mean], dtype=float)
        ax.plot(x_mean, mean_curve, color="orange", linewidth=2, label="Mean")
        ax.fill_between(
            x_mean,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="orange",
            alpha=0.2,
            label="±1 std dev",
        )

    if show_release:
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Release")

    ax.legend(loc="best", fontsize=8)


def run_curve_comparison_viewer(
    *,
    window_title: str,
    tab_specs: dict[str, dict[str, Any]],
    right_mode_label: str,
    release_frames: np.ndarray | None = None,
) -> dict[str, Any]:
    import tkinter as tk
    from tkinter import ttk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    release_frames = release_frames if release_frames is not None else np.array([], dtype=float)
    release_frames = np.asarray(release_frames, dtype=float)
    release_frames = release_frames[np.isfinite(release_frames)]

    tab_names = [t for t, spec in tab_specs.items() if spec.get("curves")]
    if not tab_names:
        raise ValueError("Viewer tabs have no curves to display.")

    curve_index = {tab: 0 for tab in tab_specs}

    root = tk.Tk()
    root.title(window_title)
    root.geometry("1500x900")

    top = tk.Frame(root)
    top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

    title_var = tk.StringVar(value=window_title)
    title_label = tk.Label(top, textvariable=title_var, font=("Helvetica", 14, "bold"))
    title_label.pack(side=tk.LEFT, padx=8)

    btn_prev = tk.Button(top, text="← Prev")
    btn_prev.pack(side=tk.RIGHT, padx=6)
    btn_next = tk.Button(top, text="Next →")
    btn_next.pack(side=tk.RIGHT, padx=6)

    mode_var = tk.StringVar(value=right_mode_label)
    mode_box = ttk.Combobox(
        top,
        textvariable=mode_var,
        values=[right_mode_label],
        state="disabled",
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
        left_stats = tk.Label(stats_frame, text="", justify=tk.LEFT, anchor="w", font=("Helvetica", 10))
        left_stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        right_stats = tk.Label(stats_frame, text="", justify=tk.LEFT, anchor="w", font=("Helvetica", 10))
        right_stats.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        fig_map[tab] = fig
        ax_map[tab] = list(axes)
        canvas_map[tab] = canvas
        stats_map[tab] = (left_stats, right_stats)

    def refresh() -> None:
        current_tab = notebook.tab(notebook.select(), "text")
        spec = tab_specs[current_tab]
        curves = spec["curves"]
        idx = curve_index[current_tab] % len(curves)
        curve = curves[idx]
        curve_index[current_tab] = idx

        title_var.set(
            f"{current_tab}: {curve} ({idx + 1}/{len(curves)})  |  Right={right_mode_label}  |  "
            "Use ←/→ or Prev/Next to change curve"
        )

        left_dfs = spec["left"]
        right_dfs = spec["right"]
        left_release_frames = spec.get("release_frames_left")
        right_release_frames = spec.get("release_frames_right")
        show_release_left = bool(spec.get("show_release_left", spec.get("show_release", False)))
        show_release_right = bool(spec.get("show_release_right", spec.get("show_release", False)))
        if left_release_frames is None:
            left_release_frames = release_frames
        if right_release_frames is None:
            right_release_frames = release_frames
        left_release_frames = np.asarray(left_release_frames, dtype=float)
        right_release_frames = np.asarray(right_release_frames, dtype=float)
        left_release_frames = left_release_frames[np.isfinite(left_release_frames)]
        right_release_frames = right_release_frames[np.isfinite(right_release_frames)]

        for ax in ax_map[current_tab]:
            ax.cla()

        _plot_dataset(
            ax=ax_map[current_tab][0],
            dfs=left_dfs,
            curve=curve,
            label=spec.get("left_label", "Left"),
            ylabel=spec["ylabel"],
            show_release=show_release_left,
            release_frames=left_release_frames,
        )
        if right_dfs:
            _plot_dataset(
                ax=ax_map[current_tab][1],
                dfs=right_dfs,
                curve=curve,
                label=right_mode_label,
                ylabel=spec["ylabel"],
                show_release=show_release_right,
                release_frames=right_release_frames,
            )
        else:
            ax = ax_map[current_tab][1]
            ax.set_title(right_mode_label)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data for this mode", transform=ax.transAxes, ha="center", va="center")

        left_stats_label, right_stats_label = stats_map[current_tab]
        left_stats_label.configure(text=f"{spec.get('left_label', 'Left')} Stats\n" + _format_dataset_stats(left_dfs, curve))
        right_stats_label.configure(text=f"{right_mode_label} Stats\n" + _format_dataset_stats(right_dfs, curve))

        fig_map[current_tab].suptitle(
            f"{current_tab} Review — {curve}",
            fontsize=12,
            fontweight="bold",
        )
        fig_map[current_tab].tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        canvas_map[current_tab].draw()

    def next_curve(event=None) -> None:
        current_tab = notebook.tab(notebook.select(), "text")
        curve_index[current_tab] += 1
        refresh()

    def prev_curve(event=None) -> None:
        current_tab = notebook.tab(notebook.select(), "text")
        curve_index[current_tab] -= 1
        refresh()

    btn_next.configure(command=next_curve)
    btn_prev.configure(command=prev_curve)
    root.bind("<Right>", next_curve)
    root.bind("<Left>", prev_curve)
    notebook.bind("<<NotebookTabChanged>>", lambda e: refresh())

    refresh()
    root.mainloop()

    return {
        "tabs_loaded": tab_names,
        "curve_counts": {tab: len(tab_specs[tab]["curves"]) for tab in tab_names},
    }
