import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FreeThrowVisualizer:
    """
    A fully generic visualization toolkit for plotting ANY time-series curves:
    - joint angles
    - keypoint coordinates (x, y, z)
    - ball metrics (height, speed)
    - derived biomechanical curves

    This is a draw-only class; it performs no computations.
    """

    def __init__(
        self,
        show_overlay=True,
        show_mean_std=True,
        y_label="Value",
        title_label="Curve",
    ):
        self.show_overlay = show_overlay
        self.show_mean_std = show_mean_std
        self.y_label = y_label         # e.g., "Angle (degrees)", "Position (meters)"
        self.title_label = title_label # e.g., "Joint", "Keypoint"

    # =====================================================
    # Internal helpers
    # =====================================================
    def _safe_display(self, obj):
        try:
            from IPython.display import display
            display(obj)
        except ImportError:
            print(obj)

    def _collect_curves(self, dfs_dict, curve_name):
        """
        Extract and pad a column across multiple free throws.
        Returns: 2D matrix (#throws × #frames)
        """
        curves = []
        for _, df in dfs_dict.items():
            if curve_name not in df.columns:
                continue
            curves.append(df[curve_name].to_numpy())

        if not curves:
            return None

        max_len = max(len(c) for c in curves)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in curves
        ]
        return np.vstack(padded)

    def _collect_column(self, dfs_dict, column_name):
        """
        Generic column collector with NaN padding.
        """
        cols = []
        for _, df in dfs_dict.items():
            if column_name not in df.columns:
                continue
            cols.append(df[column_name].to_numpy())

        if not cols:
            return None

        max_len = max(len(c) for c in cols)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in cols
        ]
        return np.vstack(padded)

    def _collect_curves_with_optional_x(self, dfs_dict, y_col, x_col=None):
        """
        Collect y curves (required) and optional x curves, padded to equal length.
        Returns (y_matrix, x_matrix or None, used_x_flag, warning_message or None)
        """
        y_list = []
        x_list = []
        missing_or_mismatch = False

        for _, df in dfs_dict.items():
            if y_col not in df.columns:
                continue
            y_vals = df[y_col].to_numpy()
            y_list.append(y_vals)

            if x_col:
                if x_col in df.columns:
                    x_vals = df[x_col].to_numpy()
                    if len(x_vals) != len(y_vals):
                        missing_or_mismatch = True
                    x_list.append(x_vals)
                else:
                    missing_or_mismatch = True
                    x_list.append(None)

        if not y_list:
            return None, None, False, None

        max_len = max(len(c) for c in y_list)
        y_padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in y_list
        ]

        x_matrix = None
        used_x = False
        warning = None

        if x_col:
            if missing_or_mismatch or len(x_list) != len(y_list) or any(v is None for v in x_list):
                warning = f"[WARN] x_col '{x_col}' missing/mismatched for some throws; using index instead."
            else:
                x_padded = [
                    np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                    for c in x_list
                ]
                x_matrix = np.vstack(x_padded)
                used_x = True

        return np.vstack(y_padded), x_matrix, used_x, warning

    # =====================================================
    # Plot primitives (use 'curve_name', not 'joint')
    # =====================================================
    def _plot_overlay(
        self,
        curves,
        curve_name,
        dataset_name,
        release_frame=None,
        ax=None,
        x_label=None,
        y_label=None,
        x_unit=None,
        y_unit=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        flip_y_axis=False,
        x_values=None,
    ):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
        x_matrix = None
        if x_values is not None:
            if x_values.ndim == 1 and len(x_values) == curves.shape[1]:
                x = x_values
            elif x_values.ndim == 2 and x_values.shape == curves.shape:
                x_matrix = x_values
                # representative x for mean/std plots
                x = np.nanmean(x_matrix, axis=0)
        mean_curve = np.nanmean(curves, axis=0)

        # Raw curves
        if x_matrix is not None:
            for xi, yi in zip(x_matrix, curves):
                valid = ~np.isnan(xi) & ~np.isnan(yi)
                ax.plot(xi[valid], yi[valid], color="gray", alpha=0.25)
        else:
            for yi in curves:
                valid = ~np.isnan(yi)
                ax.plot(x[valid], yi[valid], color="gray", alpha=0.25)

        # Mean
        ax.plot(x, mean_curve, color="orange", linewidth=2,
                label=f"Mean {self.title_label.lower()}")

        if release_frame is not None:
            ax.axvline(
                release_frame,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Avg Release ≈ {release_frame}",
            )

        x_axis_label = x_label or "Frame"
        y_axis_label = y_label or self.y_label

        if x_unit:
            x_axis_label = f"{x_axis_label} ({x_unit})"
        if y_unit:
            y_axis_label = f"{y_axis_label} ({y_unit})"

        ax.set_title(f"{dataset_name}: {curve_name.replace('_',' ').title()} – Overlay")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if flip_y_axis:
            if y_min is not None or y_max is not None:
                bottom, top = ax.get_ylim()
                ax.set_ylim(top, bottom)
            else:
                ax.invert_yaxis()

        ax.grid(alpha=0.3)

    def _plot_mean_std(
        self,
        curves,
        curve_name,
        dataset_name,
        release_frame=None,
        ax=None,
        x_label=None,
        y_label=None,
        x_unit=None,
        y_unit=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        flip_y_axis=False,
        x_values=None,
    ):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
        x_matrix = None
        if x_values is not None:
            if x_values.ndim == 1 and len(x_values) == curves.shape[1]:
                x = x_values
            elif x_values.ndim == 2 and x_values.shape == curves.shape:
                x_matrix = x_values
                x = np.nanmean(x_matrix, axis=0)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        ax.plot(x, mean_curve, color="orange", linewidth=2, label="Mean")
        ax.fill_between(
            x, mean_curve - std_curve, mean_curve + std_curve,
            color="orange", alpha=0.3, label="±1σ"
        )

        if release_frame is not None:
            ax.axvline(
                release_frame,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Avg Release ≈ {release_frame}",
            )

        x_axis_label = x_label or "Frame"
        y_axis_label = y_label or self.y_label

        if x_unit:
            x_axis_label = f"{x_axis_label} ({x_unit})"
        if y_unit:
            y_axis_label = f"{y_axis_label} ({y_unit})"

        ax.set_title(f"{dataset_name}: {curve_name.replace('_',' ').title()} – Mean ± σ")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if flip_y_axis:
            if y_min is not None or y_max is not None:
                bottom, top = ax.get_ylim()
                ax.set_ylim(top, bottom)
            else:
                ax.invert_yaxis()

        ax.grid(alpha=0.3)

    # =====================================================
    # Consistency dots
    # =====================================================
    def _plot_consistency_dots(self, y, mean, lower, upper, x, ax=None):

        if ax is None:
            ax = plt.gca()

        if len(y) < len(mean):
            y = np.pad(y, (0, len(mean) - len(y)), constant_values=np.nan)

        valid = ~np.isnan(y)
        inside = (y >= lower) & (y <= upper)

        ax.scatter(x[inside & valid], y[inside & valid],
                   color="green", s=20, label="Inside ±1σ")

        ax.scatter(x[~inside & valid], y[~inside & valid],
                   color="red", s=20, label="Outside ±1σ")
        
    def plot_single_curve(
        self,
        df,
        x_col,
        y_col,
        title="Curve",
        color="blue",
        marker="o",
        markersize=4,
        linewidth=2,
        equal_aspect=False,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        flip_y_axis=False,
        ax=None,
        x_label=None,
        y_label=None,
        x_unit=None,
        y_unit=None
    ):
        """
        Plot a single x–y curve from a DataFrame.

        Args:
            df: DataFrame containing the data
            x_col: column to use as x-axis
            y_col: column to use as y-axis
            title: title of the plot
            color: line color
            marker: marker style ('o', 'x', '.', etc.)
            markersize: size of markers
            linewidth: thickness of line
            equal_aspect: if True, treat x and y as spatial coordinates
            x_min/x_max: optional explicit x-axis bounds
            y_min/y_max: optional explicit y-axis bounds
            flip_y_axis: if True, invert y to match image coordinate systems
            ax: optional axis for subplot usage
            x_label: optional custom x-axis label
            y_label: optional custom y-axis label
            x_unit: optional x-axis unit suffix (e.g., 'frames')
            y_unit: optional y-axis unit suffix (e.g., 'meters')
        """

        if ax is None:
            plt.figure(figsize=(7, 4))
            ax = plt.gca()

        # Extract and mask NaNs
        x = df[x_col].values
        y = df[y_col].values
        valid = ~np.isnan(x) & ~np.isnan(y)

        ax.plot(
            x[valid],
            y[valid],
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=linewidth,
        )

        ax.set_title(title)
        x_axis_label = x_label or x_col.replace("_", " ").title()
        y_axis_label = y_label or y_col.replace("_", " ").title()

        if x_unit:
            x_axis_label = f"{x_axis_label} ({x_unit})"
        if y_unit:
            y_axis_label = f"{y_axis_label} ({y_unit})"

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.grid(alpha=0.3)

        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)

        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        if flip_y_axis:
            if y_min is not None or y_max is not None:
                bottom, top = ax.get_ylim()
                ax.set_ylim(top, bottom)
            else:
                ax.invert_yaxis()

        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        plt.show()


    # =====================================================
    # High-level multi-curve plotting
    # =====================================================
    
    def plot_multiple_curves(
        self,
        dfs_dict,
        curve_names=None,
        dataset_name="Example Name",
        cropped_phases_df=None,
        show_overlay=False,
        show_mean_std=False,
        show_consistency=False,
        show_release_line=False,
        stats_dict=None,
        throw_file=None,
        ax=None,
        x_label=None,
        y_label=None,
        x_unit=None,
        y_unit=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        flip_y_axis=False,
        x_col=None,
        y_col=None
    ):

        active_curve_names = curve_names
        if y_col is not None:
            active_curve_names = [y_col]
        if not active_curve_names:
            raise ValueError("Provide at least one curve via 'curve_names' or 'y_col'.")

        for curve_name in active_curve_names:
            curves, x_values, used_x, warning = self._collect_curves_with_optional_x(
                dfs_dict, y_col=curve_name, x_col=x_col
            )
            if curves is None:
                print(f"[WARN] No data found for {curve_name} in {dataset_name}")
                return
            if warning:
                print(warning)

            # Find release frame
            release_frame = None
            if show_release_line and cropped_phases_df is not None:
                valid = cropped_phases_df["cropped_release_frame"].dropna()
                release_frame = int(valid.mean()) if len(valid) else None

            created_own_fig = False
            if ax is None:
                plt.figure(figsize=(8, 4))
                ax = plt.gca()
                created_own_fig = True

            if show_overlay:
                self._plot_overlay(curves, curve_name, dataset_name,
                                   release_frame, ax=ax,
                                   x_label=x_label or (x_col.replace("_", " ").title() if used_x and x_col else None),
                                   y_label=y_label or curve_name.replace("_", " ").title(),
                                   x_unit=x_unit, y_unit=y_unit,
                                   x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max,
                                    flip_y_axis=flip_y_axis,
                                    x_values=x_values)

            if show_mean_std:
                self._plot_mean_std(curves, curve_name, dataset_name,
                                    release_frame, ax=ax,
                                    x_label=x_label or (x_col.replace("_", " ").title() if used_x and x_col else None),
                                    y_label=y_label or curve_name.replace("_", " ").title(),
                                    x_unit=x_unit, y_unit=y_unit,
                                    x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max,
                                    flip_y_axis=flip_y_axis,
                                    x_values=x_values)

            if show_consistency and stats_dict is not None and throw_file is not None:
                stats = stats_dict[curve_name]
                y = dfs_dict[throw_file][curve_name].values

                self._plot_consistency_dots(
                    y=y,
                    mean=stats["mean"],
                    lower=stats["lower"],
                    upper=stats["upper"],
                    x=stats["x"],
                    ax=ax
                )

            if created_own_fig:
                ax.legend()
                plt.tight_layout()
                plt.show()


    def interactive_review(
        self,
        curve_name,
        dfs_dict,
        stats_dict,
        title_prefix="Throw"
    ):
        """
        Interactive viewer for exploring individual throws for ANY curve
        (joints, keypoints, ball metrics, etc.).

        Args:
            curve_name:   string, e.g., 'elbow_flex_r' or 'right_wrist_x'
            dfs_dict:     dict[file_name → DataFrame]
            stats_dict:   output of compute_mean_std_for_alignment(...)
        """

        if curve_name not in stats_dict:
            raise ValueError(f"No stats found for curve '{curve_name}'")

        files = list(dfs_dict.keys())

        import ipywidgets as widgets
        from IPython.display import display

        def show(idx):
            file = files[idx]

            fig, ax = plt.subplots(figsize=(10, 5))

            self.plot_multiple_curves(
                dfs_dict=dfs_dict,
                curve_names=[curve_name],
                dataset_name=f"{title_prefix}: {file}",
                cropped_phases_df=None,
                show_overlay=False,
                show_mean_std=True,
                show_consistency=True,
                stats_dict=stats_dict,
                throw_file=file,
                ax=ax
            )

            ax.legend()
            plt.tight_layout()
            plt.show()

        slider = widgets.IntSlider(
            min=0,
            max=len(files) - 1,
            value=0,
            step=1,
            description="Throw Index"
        )

        display(widgets.interact(show, idx=slider))
