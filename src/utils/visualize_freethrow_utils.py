import numpy as np
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

    # =====================================================
    # Plot primitives (use 'curve_name', not 'joint')
    # =====================================================
    def _plot_overlay(self, curves, curve_name, dataset_name,
                      release_frame=None, ax=None):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
        mean_curve = np.nanmean(curves, axis=0)

        # Raw curves
        for y in curves:
            ax.plot(x, y, color="gray", alpha=0.25)

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

        ax.set_title(f"{dataset_name}: {curve_name.replace('_',' ').title()} – Overlay")
        ax.set_xlabel("Frame")
        ax.set_ylabel(self.y_label)
        ax.grid(alpha=0.3)

    def _plot_mean_std(self, curves, curve_name, dataset_name,
                       release_frame=None, ax=None):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
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

        ax.set_title(f"{dataset_name}: {curve_name.replace('_',' ').title()} – Mean ± σ")
        ax.set_xlabel("Frame")
        ax.set_ylabel(self.y_label)
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

    # =====================================================
    # High-level multi-curve plotting
    # =====================================================
    def plot_multiple_curves(
        self,
        dfs_dict,
        curve_names,
        dataset_name,
        cropped_phases_df=None,
        show_overlay=False,
        show_mean_std=False,
        show_consistency=False,
        show_release_line=False,
        stats_dict=None,
        throw_file=None,
        ax=None
    ):

        for curve_name in curve_names:
            curves = self._collect_curves(dfs_dict, curve_name)
            if curves is None:
                print(f"[WARN] No data found for {curve_name} in {dataset_name}")
                return

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
                                   release_frame, ax=ax)

            if show_mean_std:
                self._plot_mean_std(curves, curve_name, dataset_name,
                                    release_frame, ax=ax)

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