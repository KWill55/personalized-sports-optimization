import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class FreeThrowVisualizer:
    """
    A unified visualization toolkit for plotting free-throw kinematic data.

    Supports:
        - Overlay trajectory plots
        - Mean ± standard deviation curves
        - Distribution curves with Gaussian overlays
        - Release-angle analysis
        - Duration distributions

    New visualization methods can easily be added.
    """

    def __init__(self, show_overlay=True, show_mean_std=True):
        self.show_overlay = show_overlay
        self.show_mean_std = show_mean_std

    # -------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------
    def _safe_display(self, obj):
        """Display nicely in Jupyter, fall back to print otherwise."""
        try:
            from IPython.display import display
            display(obj)
        except ImportError:
            print(obj)

    def _collect_curves(self, dfs_dict, joint_name):
        """
        Extract and pad joint curves from dict of DataFrames.

        Returns:
            curves -- 2D numpy array (#clips × #frames)
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

    def _plot_overlay(self, curves, joint_name, dataset_name, avg_release_frame=None):
        x = np.arange(curves.shape[1]) 
        mean_curve = np.nanmean(curves, axis=0)

        plt.figure(figsize=(8, 4))
        for y in curves:
            plt.plot(x, y, color="gray", alpha=0.25)

        plt.plot(x, mean_curve, color="orange", linewidth=2, label="Mean trajectory")

        if avg_release_frame is not None:
            plt.axvline(
                avg_release_frame,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Avg Release ≈ {avg_release_frame}"
            )

        plt.title(f"{dataset_name}: {joint_name.replace('_',' ').title()} – Overlay")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_mean_std(self, curves, joint_name, dataset_name, avg_release_frame=None):
        x = np.arange(curves.shape[1])
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)

        plt.figure(figsize=(8, 4))
        plt.plot(x, mean_curve, color="orange", linewidth=2, label="Mean angle")
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="orange",
            alpha=0.3,
            label="±1σ",
        )

        if avg_release_frame is not None:
            plt.axvline(
                avg_release_frame,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Avg Release ≈ {avg_release_frame}"
            )

        plt.title(f"{dataset_name}: {joint_name.replace('_',' ').title()} – Mean ± σ")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_multiple_joints(
            self,
            dfs_dict,
            joint_names,
            dataset_name,
            cropped_phases_df=None,
            show_overlay=False,
            show_mean_std=False,
            show_release_line=False,
        ):
            """
            High-level wrapper:
            - Extract curves
            - Optionally plot overlay
            - Optionally plot mean ± std
            - Optionally plot avg release-frame vertical line

            Args:
                dfs_dict: dict[str, DataFrame]
                joint_names: list of joint names (strings)
                dataset_name: label (e.g. 'Before Alignment')
                phases_df: optional phases DataFrame (for release line)
                show_release_line: whether to draw vertical release frame
            """
            for joint_name in joint_names:
                curves = self._collect_curves(dfs_dict, joint_name)
                
                if curves is None:
                    print(f"[WARNING] No data found for {joint_name} in {dataset_name}")
                    return

                # Retrieve average release frame if requested
                avg_release_frame = None
                if show_release_line and cropped_phases_df is not None:
                    valid = cropped_phases_df["cropped_release_frame"].dropna()
                    avg_release_frame = int(valid.mean())
                # TODO make a function that gets average release frame from phases somewhere else(this is done multiple places rn)

                # Pass to plotting helpers
                if show_overlay:
                    self._plot_overlay(curves, joint_name, dataset_name, avg_release_frame)

                if show_mean_std:
                    self._plot_mean_std(curves, joint_name, dataset_name, avg_release_frame)