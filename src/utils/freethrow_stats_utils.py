import numpy as np

def compute_mean_std_for_alignment(dfs_dict, curve_names, x_col=None):
    """
    Computes mean, std, upper, lower curves for each requested column
    (joints, keypoints, trajectories, etc.) across all throws in dfs_dict.

    Args:
        dfs_dict:    dict[file_name -> DataFrame]
        curve_names: iterable of column names to summarize
        x_col:       optional x-axis column (e.g., "x" for trajectories).
                     If provided and present with matching length, the returned
                     x array is the mean x across throws; otherwise falls back
                     to frame index.

    Returns:
        dict[curve_name] = {
            "mean": ...,
            "std": ...,
            "upper": ...,
            "lower": ...,
            "curves": 2D array of padded curves,
            "x": x-axis array (mean of x_col if available, else index)
        }
    """
    results = {}

    for curve in curve_names:
        y_curves = []
        x_curves = []
        use_index = False

        for _, df in dfs_dict.items():
            if curve not in df.columns:
                continue

            y_vals = df[curve].to_numpy()
            y_curves.append(y_vals)

            if x_col:
                if x_col in df.columns and len(df[x_col]) == len(y_vals):
                    x_curves.append(df[x_col].to_numpy())
                else:
                    use_index = True

        if not y_curves:
            continue

        max_len = max(len(c) for c in y_curves)
        y_padded = [
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in y_curves
        ]
        y_padded = np.vstack(y_padded)

        x_axis = np.arange(max_len)
        if x_col and x_curves and not use_index and len(x_curves) == len(y_curves):
            x_padded = [
                np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                for c in x_curves
            ]
            x_axis = np.nanmean(np.vstack(x_padded), axis=0)

        mean_curve = np.nanmean(y_padded, axis=0)
        std_curve = np.nanstd(y_padded, axis=0)
        upper = mean_curve + std_curve
        lower = mean_curve - std_curve

        results[curve] = {
            "curves": y_padded,
            "mean": mean_curve,
            "std": std_curve,
            "upper": upper,
            "lower": lower,
            "x": x_axis,
        }

    return results
