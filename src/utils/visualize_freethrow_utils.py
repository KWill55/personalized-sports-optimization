import numpy as np
import matplotlib.pyplot as plt

class FreeThrowVisualizer:
    """
    A unified visualization toolkit for plotting free-throw kinematic data.
    Draw-only class (no computation).
    """

    def __init__(self, show_overlay=True, show_mean_std=True):
        self.show_overlay = show_overlay
        self.show_mean_std = show_mean_std

    # =====================================================
    # Internal helpers
    # =====================================================
    def _safe_display(self, obj):
        try:
            from IPython.display import display
            display(obj)
        except ImportError:
            print(obj)

    def _collect_curves(self, dfs_dict, joint_name):
        """
        Extract and pad joint curves from dict[file_name → DataFrame].
        Returns a 2D array #clips × #frames.
        """
        curves = []
        for _, df in dfs_dict.items():
            if joint_name not in df.columns:
                continue
            curves.append(df[joint_name].to_numpy())

        if not curves:
            return None

        max_len = max(len(c) for c in curves)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in curves
        ]
        return np.vstack(padded)

    # =====================================================
    # Plot primitives (NOW accept ax)
    # =====================================================
    def _plot_overlay(self, curves, joint_name, dataset_name,
                      avg_release_frame=None, ax=None):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
        mean_curve = np.nanmean(curves, axis=0)

        for y in curves:
            ax.plot(x, y, color="gray", alpha=0.25)

        ax.plot(x, mean_curve, color="orange", linewidth=2,
                label="Mean trajectory")

        if avg_release_frame is not None:
            ax.axvline(avg_release_frame, color="red", linestyle="--",
                       linewidth=1.5, label=f"Avg Release ≈ {avg_release_frame}")

        ax.set_title(f"{dataset_name}: {joint_name.replace('_',' ').title()} – Overlay")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (degrees)")
        ax.grid(alpha=0.3)

    def _plot_mean_std(self, curves, joint_name, dataset_name,
                       avg_release_frame=None, ax=None):

        if ax is None:
            ax = plt.gca()

        x = np.arange(curves.shape[1])
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        ax.plot(x, mean_curve, color="orange", linewidth=2, label="Mean angle")
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color="orange", alpha=0.3, label="±1σ")

        if avg_release_frame is not None:
            ax.axvline(avg_release_frame, color="red", linestyle="--",
                       linewidth=1.5, label=f"Avg Release ≈ {avg_release_frame}")

        ax.set_title(f"{dataset_name}: {joint_name.replace('_',' ').title()} – Mean ± σ")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (degrees)")
        ax.grid(alpha=0.3)

    # =====================================================
    # Consistency-dots helper (NEW)
    # =====================================================
    def _plot_consistency_dots(self, y, mean, lower, upper, x, ax=None):
        """
        Plots green/red dots showing whether each frame is inside ±1σ.
        """
        if ax is None:
            ax = plt.gca()

        # Pad y to match lengths
        if len(y) < len(mean):
            y = np.pad(y, (0, len(mean) - len(y)), constant_values=np.nan)

        valid  = ~np.isnan(y)
        inside = (y >= lower) & (y <= upper)

        ax.scatter(x[inside & valid], y[inside & valid],
                   color="green", s=20, label="Inside ±1σ")

        ax.scatter(x[~inside & valid], y[~inside & valid],
                   color="red", s=20, label="Outside ±1σ")
        

    # =====================================================
    # Interactive throw reviewer (NEW)
    # =====================================================
    def interactive_review(
        self,
        joint_name,
        dfs_dict,
        stats_dict,
        title_prefix="Throw"
    ):
        """
        Interactive viewer for exploring individual throws with consistency dots.

        Args:
            joint_name:     e.g., 'elbow_flex_r'
            dfs_dict:       dict[file_name → DataFrame]
            stats_dict:     output of compute_mean_std_for_alignment(...)
        """

        if joint_name not in stats_dict:
            raise ValueError(f"No stats found for joint '{joint_name}'")

        files = list(dfs_dict.keys())

        import ipywidgets as widgets
        from IPython.display import display

        def show(idx):
            file = files[idx]

            # --- Create a single figure + axes ---
            fig, ax = plt.subplots(figsize=(10, 5))

            # --- Draw the visualization on THIS axis ---
            self.plot_multiple_joints(
                dfs_dict=dfs_dict,
                joint_names=[joint_name],
                dataset_name=f"{title_prefix} {file}",
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
            step=1,
            value=0,
            description="Throw Index:"
        )

        display(widgets.interact(show, idx=slider))


    # =====================================================
    # High-level multi-joint plotting
    # =====================================================
    def plot_multiple_joints(
        self,
        dfs_dict,
        joint_names,
        dataset_name,
        cropped_phases_df=None,
        show_overlay=False,
        show_mean_std=False,
        show_consistency=False,     # NEW
        show_release_line=False,
        stats_dict=None,            # NEW: pass in mean/std computed externally
        throw_file=None,            # NEW: which throw to highlight
        ax=None                     # NEW: unified axes
    ):
        """
        Draws overlay, mean±std, optional consistency dots.
        """

        for joint_name in joint_names:
            curves = self._collect_curves(dfs_dict, joint_name)
            if curves is None:
                print(f"[WARNING] No data found for {joint_name} in {dataset_name}")
                return

            # Determine release frame
            avg_release_frame = None
            if show_release_line and cropped_phases_df is not None:
                valid = cropped_phases_df["cropped_release_frame"].dropna()
                avg_release_frame = int(valid.mean()) if len(valid) else None

            # Use ax or create a new figure/axes if none provided
            created_own_fig = False
            if ax is None:
                plt.figure(figsize=(8, 4))
                ax = plt.gca()
                created_own_fig = True

            # Draw overlay
            if show_overlay:
                self._plot_overlay(curves, joint_name, dataset_name,
                                   avg_release_frame, ax=ax)

            # Draw mean ± std
            if show_mean_std:
                self._plot_mean_std(curves, joint_name, dataset_name,
                                    avg_release_frame, ax=ax)

            # Add consistency dots (external stats required)
            if show_consistency and stats_dict is not None and throw_file is not None:
                if joint_name not in stats_dict:
                    raise ValueError(f"No stats for joint {joint_name}")

                stats = stats_dict[joint_name]
                y = dfs_dict[throw_file][joint_name].values

                self._plot_consistency_dots(
                    y=y,
                    mean=stats["mean"],
                    lower=stats["lower"],
                    upper=stats["upper"],
                    x=stats["x"],
                    ax=ax
                )

            # Show if we created the figure
            if created_own_fig:
                ax.legend()
                plt.tight_layout()
                plt.show()
