import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def safe_display(obj):
    """Display in Jupyter; fallback to print otherwise."""
    try:
        from IPython.display import display
        display(obj)
    except ImportError:
        print(obj)


def analyze_freethrow_durations(cropped_angles_dfs: dict[str, pd.DataFrame],
                                fps: int = 60,
                                display_results: bool = False,
                                display_graph: bool = False):
    """
    Compute per-free-throw durations (in frames and seconds) and optionally
    display stats or a Gaussian overlay plot.
    
    Returns:
        elapsed_df: DataFrame with duration per throw
        summary: dict containing mean, std, ranges
    """
    free_throw_names = []
    frame_counts = []
    elapsed_seconds = []

    # Loop through each cropped DataFrame
    for file_name, df in cropped_angles_dfs.items():
        base_name = file_name.split("_angles")[0]

        if "frame" not in df.columns:
            continue

        num_frames = df["frame"].nunique()
        duration_s = num_frames / fps

        free_throw_names.append(base_name)
        frame_counts.append(num_frames)
        elapsed_seconds.append(duration_s)

    elapsed_df = pd.DataFrame({
        "FreeThrow": free_throw_names,
        "NumFrames": frame_counts,
        "Duration_sec": elapsed_seconds
    })

    # Summary statistics
    mean = np.mean(elapsed_df["Duration_sec"])
    std = np.std(elapsed_df["Duration_sec"], ddof=1)

    ranges = {
        "68%": (mean - std, mean + std),
        "95%": (mean - 2*std, mean + 2*std),
        "99.7%": (mean - 3*std, mean + 3*std)
    }

    summary = {
        "mean": mean,
        "std": std,
        "ranges": ranges
    }

    if display_results:
        print(elapsed_df)
        print(f"\nAverage duration: {mean:.3f}s")
        print(f"Std deviation:   {std:.3f}s\n")
        for label, (low, high) in ranges.items():
            pct = (elapsed_df["Duration_sec"].between(low, high)).mean() * 100
            print(f"{label} range: {low:.3f}s–{high:.3f}s ({pct:.1f}%)")

    if display_graph:
        data = elapsed_df["Duration_sec"]
        x = np.linspace(data.min() - 0.2, data.max() + 0.2, 400)
        y_gaussian = norm.pdf(x, mean, std)

        plt.hist(data, bins=15, density=True, alpha=0.5,
                 color="orange", edgecolor="black", label="Data")
        plt.plot(x, y_gaussian, color="blue", linewidth=2, label="Ideal Gaussian")

        plt.axvline(mean, color="red", linestyle="--", linewidth=1)
        plt.title("Free-Throw Duration vs Gaussian")
        plt.xlabel("Duration (sec)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return elapsed_df, summary

def analyze_release_timing(phases_df: pd.DataFrame,
                           display_results: bool = False,
                           display_graph: bool = False):
    """
    Analyze variation in the release frame across all free throws.

    Computes:
        - frames between windup_start and release_frame
        - mean and standard deviation
        - histogram and Gaussian overlay (optional)

    Args:
        phases_df: DataFrame with windup_start and release_frame columns.
        display_results: print summary results.
        display_graph: show histogram + Gaussian overlay.

    Returns:
        release_df: DataFrame with FramesUntilRelease column
        summary: dict with mean and std
    """

    # --- Collect durations ---
    release_durations = []
    for _, row in phases_df.iterrows():
        ws = row.get("windup_start")
        rf = row.get("release_frame")

        if pd.isna(ws) or pd.isna(rf):
            continue

        diff = rf - ws
        if diff > 0:
            release_durations.append(diff)

    # --- Build results DataFrame ---
    release_df = pd.DataFrame({"FramesUntilRelease": release_durations})

    if release_df.empty:
        print("⚠️ No valid release timing data found.")
        return release_df, {"mean": None, "std": None}

    mean_release = release_df["FramesUntilRelease"].mean()
    std_release = release_df["FramesUntilRelease"].std(ddof=1)

    summary = {
        "mean": mean_release,
        "std": std_release
    }

    # --- Print summary ---
    if display_results:
        print(release_df.head())
        print(f"✅ Computed release timing summary for {len(release_df)} free throws.\n")
        print(f"Average frames until release: {mean_release:.2f}")
        print(f"Standard deviation: {std_release:.2f} frames\n")

    # --- Plot histogram + Gaussian ---
    if display_graph:
        data = release_df["FramesUntilRelease"]
        x = np.linspace(data.min() - 5, data.max() + 5, 400)
        y_gaussian = norm.pdf(x, mean_release, std_release)

        plt.hist(
            data, bins=15, density=True, alpha=0.5,
            color="orange", edgecolor="black", label="Athlete Data (actual)"
        )

        plt.plot(x, y_gaussian, color="blue", linewidth=2, label="Ideal Gaussian")

        plt.axvline(
            mean_release, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean Release Frame ({mean_release:.1f})"
        )

        plt.title("Distribution of Frames Until Release vs Ideal Gaussian", fontsize=13)
        plt.xlabel("Frames until release (from windup start)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return release_df, summary


def analyze_release_angles(
    cropped_angles_dfs: dict[str, pd.DataFrame],
    cropped_phases_df: pd.DataFrame,
    metrics_path: Path,
    joints_to_plot: list[str] = None,
    save_summary: bool = False,
    display_summary: bool = False
):
    """
    Analyze joint angles at the release frame for each free throw.

    Args:
        cropped_angles_dfs: dict of {file_name → DataFrame}
        phases_df: DataFrame containing release_frame values per throw
        metrics_path: folder where summary CSV should be saved
        joints_to_plot: list of joint names to visualize
        save_summary: whether to export CSV
        display_summary: whether to print/display results

    Returns:
        release_summary_df: DataFrame containing (mean, std, CV, count)
    """

    # Default = analyze all common joints
    if joints_to_plot is None:
        joints_to_plot = [
            "elbow_flex_l", "elbow_flex_r",
            "shoulder_flex_l", "shoulder_flex_r",
            "hip_flex_l", "hip_flex_r",
            "knee_flex_l", "knee_flex_r",
            "ankle_flex_l", "ankle_flex_r"
        ]

    release_summaries = []

    # Precompute base names in phases_df
    phases_df = cropped_phases_df.copy()
    phases_df["base"] = phases_df["file"].str.split("_").str[0]

    for joint in joints_to_plot:
        all_values = []

        # --- Extract joint angle at release frame ---
        for file_name, df in cropped_angles_dfs.items():
            base_name = file_name.split("_angles")[0]
            match = phases_df[phases_df["base"] == base_name]

            if match.empty:
                continue

            release_frame = match["cropped_release_frame"].values[0]
            if pd.isna(release_frame):
                continue
            release_frame = int(release_frame)

            if "frame" not in df.columns or joint not in df.columns:
                continue

            row = df.loc[df["frame"] == release_frame]
            if row.empty:
                continue

            val = row[joint].values[0]
            if not pd.isna(val):
                all_values.append(val)

        if len(all_values) == 0:
            print(f"[WARNING] No release-frame data found for {joint}, skipping.")
            continue

        # --- Compute statistics ---
        arr = np.array(all_values)
        mean_val = arr.mean()
        std_val = arr.std(ddof=1)
        cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan

        release_summaries.append({
            "Joint": joint,
            "MeanAngle_deg": mean_val,
            "STD_deg": std_val,
            "CV_percent": cv,
            "NumShots": len(arr)
        })

        # --- Plot ---
        plt.figure(figsize=(7, 4))
        x = np.linspace(arr.min(), arr.max(), 400)
        y_gaussian = norm.pdf(x, mean_val, std_val)

        plt.hist(arr, bins=15, density=True, alpha=0.5,
                 color="orange", edgecolor="black", label=f"{joint} (release)")
        plt.plot(x, y_gaussian, color="blue", linewidth=2, label="Ideal Gaussian")
        plt.axvline(mean_val, color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {mean_val:.1f}°")

        plt.title(f"Release-Frame Distribution: {joint.replace('_',' ').title()}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # --- Build summary DataFrame ---
    release_summary_df = pd.DataFrame(release_summaries)

    if display_summary:
        safe_display(release_summary_df)

    if save_summary:
        metrics_path.mkdir(parents=True, exist_ok=True)
        save_path = metrics_path / "release_angle_summary.csv"
        release_summary_df.to_csv(save_path, index=False)
        print(f"✅ Saved release-angle summary to: {save_path}")

    return release_summary_df

