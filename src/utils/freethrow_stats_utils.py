import numpy as np

def compute_mean_std_for_alignment(dfs_dict, joint_names):
    """
    Computes mean, std, upper, lower curves for each joint
    across all throws in dfs_dict.
    
    Returns:
        dict[joint_name] = {
            "mean": ...,
            "std": ...,
            "upper": ...,
            "lower": ...,
            "curves": 2D array of padded curves,
            "x": x-axis array
        }
    """
    results = {}

    for joint in joint_names:
        curves = []
        # Collect curves
        for file, df in dfs_dict.items():
            if joint not in df.columns:
                continue
            curves.append(df[joint].to_numpy())

        if not curves:
            continue

        # Pad shorter throws with NaN
        max_len = max(len(c) for c in curves)
        padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in curves
        ]
        padded = np.vstack(padded)

        mean_curve = np.nanmean(padded, axis=0)
        std_curve  = np.nanstd(padded, axis=0)
        upper = mean_curve + std_curve
        lower = mean_curve - std_curve
        x = np.arange(max_len)

        results[joint] = {
            "curves": padded,
            "mean": mean_curve,
            "std": std_curve,
            "upper": upper,
            "lower": lower,
            "x": x,
        }

    return results