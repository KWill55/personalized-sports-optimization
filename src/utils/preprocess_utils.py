"""
any operation that helps to take raw data and turn it into data that is ready for analysis, metrics, or modeling
"""

import pandas as pd
import numpy as np
import re

def extract_base_freethrow_name(name):
    match = re.search(r"(freethrow\d+)", str(name))
    return match.group(1) if match else None

def filter_angles_kalman(raw_angles_dfs: dict[str, pd.DataFrame],
                         process_var: float = 1e-3,
                         measurement_var: float = 1e-1) -> dict[str, pd.DataFrame]:
    """
    Apply a simple 1D Kalman filter to each angle column in every DataFrame.

    process_var (Q): how much you assume the angle changes between frames
    measurement_var (R): how noisy you think the input angles are
    """

    filtered = {}  # final dictionary of filtered DataFrames

    # Loop through every file's DataFrame
    for file_name, df in raw_angles_dfs.items():
        df = df.copy()

        # Get angle columns only (ignore 'frame')
        columns = [c for c in df.columns if c != "frame"]

        # Loop through each angle (e.g., elbow_flex_r, shoulder_flex_r...)
        for col in columns:
            z = df[col].values.astype(float)  # raw measurements as a NumPy array

            # ------------------------------------------------------
            # 1. Initialize Kalman filter storage for this column
            # ------------------------------------------------------
            x = np.zeros_like(z)     # filtered value estimate at each time
            P = np.zeros_like(z)     # uncertainty of that estimate

            x[0] = z[0]              # initial estimate = first measurement
            P[0] = 1.0               # start with moderate uncertainty

            # ------------------------------------------------------
            # 2. Run Kalman filter through all time steps
            # ------------------------------------------------------
            for t in range(1, len(z)):

                # ---------- PREDICT STEP ----------
                # Predict the state at time t based on t-1
                # We assume the angle is roughly constant from frame to frame.
                x_pred = x[t-1]

                # Predict uncertainty grows (since we moved forward in time)
                P_pred = P[t-1] + process_var


                # ---------- UPDATE STEP ----------
                # Compute Kalman Gain:
                # Tells us whether to trust the measurement or the prediction.
                K = P_pred / (P_pred + measurement_var)

                # Update the estimate:
                # Move prediction toward measurement, scaled by K.
                x[t] = x_pred + K * (z[t] - x_pred)

                # Update uncertainty:
                # After using a measurement, we become more confident.
                P[t] = (1 - K) * P_pred

            # Replace the column with filtered values
            df[col] = x

        filtered[file_name] = df

    return filtered



def combine_angles_dfs(angles_dfs: dict[str, pd.DataFrame], session_id: str) -> pd.DataFrame:
    """
    Convert dictionary of wide angle DataFrames → one long-form DataFrame.
    """
    dfs = []
    for file_name, df in angles_dfs.items():
        if "frame" not in df.columns:
            df = df.copy()
            df.insert(0, "frame", range(len(df)))

        # Convert from wide to long format
        long_df = df.melt(id_vars=["frame"], var_name="angle", value_name="value")
        long_df["file"] = file_name
        long_df["session"] = session_id
        dfs.append(long_df)

    return pd.concat(dfs, ignore_index=True)

def crop_to_freethrow(
    angles_dfs: dict[str, pd.DataFrame],
    phases_df: pd.DataFrame,
    start_col: str = "crop_start_frame",
    end_col: str = "crop_end_frame",
    phase_fps: float | int | None = None,
    data_fps: float | int | None = None,
) -> dict[str, pd.DataFrame]:
    cropped = {}
    successful_crops = 0

    phase_fps_val = float(phase_fps) if phase_fps else None
    data_fps_val = float(data_fps) if data_fps else None
    fps_ratio = 1.0
    if phase_fps_val and data_fps_val and phase_fps_val > 0 and data_fps_val > 0:
        fps_ratio = data_fps_val / phase_fps_val

    for file_name, df in angles_dfs.items():
        
        # Match by filename
        base_name = file_name.split("_")[0]
        # print(f" Base name for angles_dfs {file_name}: {base_name}")

        phase_row = phases_df[phases_df["file"].str.contains(base_name, na=False)] 

        if phase_row.empty:
            # print(f"[WARNING] No phase data found for {file_name}, skipping.")
            continue

        # Extract frame range safely
        if start_col not in phase_row.columns or end_col not in phase_row.columns:
            # print(f"[WARNING] Missing {start_col}/{end_col} for {file_name}, skipping.")
            continue

        start = phase_row[start_col].values[0]
        end = phase_row[end_col].values[0]

        if np.isnan(start) or np.isnan(end):
            # print(f"[WARNING] Missing phase values for {file_name}, skipping.")
            continue

        start = int(round(float(start) * fps_ratio))
        end = int(round(float(end) * fps_ratio))

        # Ensure frame column exists
        if "frame" not in df.columns:
            df = df.copy()
            df.insert(0, "frame", range(len(df)))

        if len(df) == 0:
            continue

        min_frame = int(df["frame"].min())
        max_frame = int(df["frame"].max())
        start = max(min_frame, start)
        end = min(max_frame, end)
        if end < start:
            continue

        # Crop DataFrame
        cropped_df = df[(df["frame"] >= start) & (df["frame"] <= end)].copy()
        if cropped_df.empty:
            continue
        cropped_df.reset_index(drop=True, inplace=True)

        cropped_df["frame"] = np.arange(len(cropped_df))

        cropped[file_name] = cropped_df
        successful_crops += 1

    print(f"\n✅ Successfully cropped {successful_crops} out of {len(angles_dfs)} freethrow sequences.")
    return cropped    

def align_by_reference(cropped_angles_dfs: dict[str, pd.DataFrame],
                       phases_df: pd.DataFrame,
                       outcomes_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:

    # --- Step 1. Filter to 'made' throws ---
    made_files = outcomes_df.loc[outcomes_df["outcome"] == "made", "file"].tolist()
    made_angles_dfs = {
        file: cropped_angles_dfs[key]
        for file in made_files
        for key in cropped_angles_dfs
        if file.split("_")[0] == key.split("_")[0]
    }

    print(f"[DEBUG] made_files: {len(made_files)}, matched: {len(made_angles_dfs)}")

    lowest_elbow_frames: dict[str, int] = {}
    release_frames: dict[str, int] = {}
    durations: dict[str, int] = {}

    # --- Step 2. Find dip & release frames for each make ---
    phases_df = phases_df.copy()
    phases_df["base_name"] = phases_df["file"].apply(extract_base_freethrow_name)

    for file, df in made_angles_dfs.items():
        base = extract_base_freethrow_name(file)
        match = phases_df[phases_df["base_name"] == base]
        if match.empty:
            print(f"[❌ No match for {base}]")
            continue

        release_frame = int(match["release_frame"].values[0])
        if "elbow_flex_r" not in df.columns:
            continue

        before_release = df[df["frame"] <= release_frame]
        if before_release.empty or before_release["elbow_flex_r"].dropna().empty:
            continue

        try:
            lowest = int(
                before_release.loc[before_release["elbow_flex_r"].idxmin(), "frame"]
            )
        except Exception:
            continue

        lowest_elbow_frames[file] = lowest
        release_frames[file] = release_frame
        durations[file] = release_frame - lowest

    if not durations:
        print("[❌] No valid dip→release durations found.")
        return {}, pd.DataFrame()

    # --- Step 3. Define canonical duration and build mean reference segment ---
    median_length = int(np.median(list(durations.values())))
    canon_len = median_length + 1  # +1 so 0..median_length inclusive

    segments_resampled = []

    for file, df in made_angles_dfs.items():
        if file not in lowest_elbow_frames:
            continue

        low = lowest_elbow_frames[file]
        rel = release_frames[file]

        seg = df.loc[
            (df["frame"] >= low) & (df["frame"] <= rel),
            "elbow_flex_r"
        ].to_numpy()

        if len(seg) < 5:
            continue

        # Original segment indexed from 0..(len(seg)-1)
        orig_x = np.linspace(0, median_length, num=len(seg))
        canon_x = np.linspace(0, median_length, num=canon_len)

        seg_resampled = np.interp(canon_x, orig_x, seg)
        segments_resampled.append(seg_resampled)

    if not segments_resampled:
        print("[❌] Failed to build mean reference segment.")
        return {}, pd.DataFrame()

    segments_resampled = np.vstack(segments_resampled)
    mean_reference_segment = np.mean(segments_resampled, axis=0)

    print(f"[INFO] Built mean reference segment from {segments_resampled.shape[0]} clips; "
          f"canonical length = {canon_len} frames.")

    # --- Step 4. Align all throws using that mean reference ---
    max_shift = 30  # search range in frames
    aligned_dfs: dict[str, pd.DataFrame] = {}
    alignment_log = []

    for file, df in made_angles_dfs.items():
        if file not in lowest_elbow_frames:
            continue

        low = lowest_elbow_frames[file]
        rel = release_frames[file]

        seg = df.loc[
            (df["frame"] >= low) & (df["frame"] <= rel),
            "elbow_flex_r"
        ].to_numpy()

        if len(seg) < 5:
            continue

        # Resample this segment to the same canonical grid as the mean reference
        orig_x = np.linspace(0, median_length, num=len(seg))
        canon_x = np.linspace(0, median_length, num=canon_len)
        seg_resampled = np.interp(canon_x, orig_x, seg)

        best_shift = 0
        min_diff = np.inf

        for shift in range(-max_shift, max_shift + 1):
            if shift > 0:
                ref_slice = mean_reference_segment[:-shift]
                seg_slice = seg_resampled[shift:]
            elif shift < 0:
                ref_slice = mean_reference_segment[-shift:]
                seg_slice = seg_resampled[:shift]
            else:
                ref_slice = mean_reference_segment
                seg_slice = seg_resampled

            min_len = min(len(ref_slice), len(seg_slice))
            if min_len <= 5:
                continue

            diff = np.mean(np.abs(ref_slice[:min_len] - seg_slice[:min_len]))
            if diff < min_diff:
                min_diff = diff
                best_shift = shift

        # Log diagnostics
        alignment_log.append({
            "file": file,
            "best_shift": best_shift,
            "min_diff": float(np.round(min_diff, 3)),
            "dip_to_release_len": durations[file],
            "median_len": median_length,
            "len_diff": durations[file] - median_length,
        })

        # Apply shift to the *entire* elbow series
        aligned_df = df.copy()
        aligned_df["elbow_flex_r"] = df["elbow_flex_r"].shift(best_shift)
        aligned_df["frame_aligned"] = np.arange(len(aligned_df))
        aligned_dfs[file] = aligned_df

    # --- Step 5. Pad to equal length (for easy stacking/plotting) ---
    max_len = max(len(df) for df in aligned_dfs.values())
    for f, df in aligned_dfs.items():
        diff = max_len - len(df)
        if diff > 0:
            pad_top = pd.DataFrame(np.nan, index=np.arange(diff // 2), columns=df.columns)
            pad_bottom = pd.DataFrame(np.nan, index=np.arange(diff - diff // 2), columns=df.columns)
            aligned_dfs[f] = pd.concat([pad_top, df, pad_bottom], ignore_index=True)

    alignment_log_df = pd.DataFrame(alignment_log)
    print("\n=== Alignment Diagnostics (min_diff) ===")
    print(alignment_log_df.sort_values("min_diff").head(5))
    print(alignment_log_df.sort_values("min_diff").tail(5))

    return aligned_dfs, alignment_log_df
