
from utils.preprocess_utils import extract_base_freethrow_name
import numpy as np
import pandas as pd


def align_by_lowest_frame(cropped_angles_dfs: dict[str, pd.DataFrame],
                          angle_col: str = "elbow_flex_r"):

    aligned = {}
    logs = []

    # 1. Compute reference dip frame (median)
    dips = {}
    for file, df in cropped_angles_dfs.items():
        if angle_col not in df.columns:
            continue
        vals = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
        finite_idx = np.where(np.isfinite(vals))[0]
        if len(finite_idx) == 0:
            continue
        local_min_idx = finite_idx[np.argmin(vals[finite_idx])]
        dips[file] = int(local_min_idx)

    if not dips:
        return {}, pd.DataFrame(columns=["file", "shift"])

    median_dip = int(round(float(np.median(list(dips.values())))))

    # 2. Align all throws by shifting dip to median_dip
    for file, df in cropped_angles_dfs.items():
        if file not in dips:
            continue
        shift = median_dip - dips[file]
        aligned_df = df.copy()
        if angle_col in aligned_df.columns:
            aligned_df[angle_col] = df[angle_col].shift(shift) # shift auto places front nans if needed

        aligned[file] = aligned_df
        logs.append({"file": file, "shift": shift})

    return aligned, pd.DataFrame(logs)


def align_by_release_frame(cropped_angles_dfs: dict[str, pd.DataFrame],
                           cropped_phases_df: pd.DataFrame,
                           angle_col: str = "elbow_flex_r"):

    aligned = {}
    logs = []

    cropped_phases_df = cropped_phases_df.copy()
    cropped_phases_df["base"] = cropped_phases_df["file"].apply(extract_base_freethrow_name)

    # 1. compute reference release frame (median)
    # TODO this for loop could be a separate function somewhere eventually to get release_frames from csv 
    releases = {}
    for file, df in cropped_angles_dfs.items():
        base = extract_base_freethrow_name(file)
        match = cropped_phases_df[cropped_phases_df["base"] == base]
        if match.empty:
            continue
        release_frame = int(match["cropped_release_frame"].values[0])
        releases[file] = release_frame

    if not releases:
        return {}, pd.DataFrame(columns=["file", "shift"])

    median_release = int(round(float(np.median(list(releases.values())))))

    # 2. Align all throws by shifting release to median release
    for file, df in cropped_angles_dfs.items():
        if file not in releases:
            continue
        shift = median_release - releases[file]
        aligned_df = df.copy()
        if angle_col in aligned_df.columns:
            aligned_df[angle_col] = df[angle_col].shift(shift) # shift auto places front nans if needed

        aligned[file] = aligned_df
        logs.append({"file": file, "shift": shift})

    return aligned, pd.DataFrame(logs)


def align_by_min_unsigned_area(cropped_dfs: dict[str, pd.DataFrame],
                               col: str,
                               search: int = 20):
    """
    col: column to align by in dataframe 
    """
    
    aligned = {}
    logs = []

    valid_dfs = {}
    for file, df in cropped_dfs.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).any():
            valid_dfs[file] = df

    if not valid_dfs:
        return {}, pd.DataFrame(columns=["file", "shift", "area"])

    # 1. Build mean curve (shortest common length)
    min_len = min(len(df) for df in valid_dfs.values())
    curves = np.vstack([
        pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[:min_len]
        for df in valid_dfs.values()
    ])
    mean_curve = np.nanmean(curves, axis=0)

    # 2. For each throw, find shift with minimum unsigned area
    for file, df in valid_dfs.items():
        curve = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[:min_len]
        best_shift = 0
        best_area = np.inf

        # find overlapping region between the two curves 
        for shift in range(-search, search + 1):
            if abs(shift) >= min_len:
                continue
            if shift < 0: # curve is to the left of median curve 
                c = curve[-shift:min_len]
                m = mean_curve[:min_len + shift]
            elif shift > 0: # curve is to the right of median curve 
                c = curve[:min_len - shift]
                m = mean_curve[shift:min_len]
            else: # no shift 
                c = curve
                m = mean_curve

            if len(c) == 0 or len(m) == 0 or len(c) != len(m):
                continue

            valid = np.isfinite(m) & np.isfinite(c)
            if not np.any(valid):
                continue
            area = float(np.sum(np.abs(m[valid] - c[valid])))
            if area < best_area:
                best_area = area
                best_shift = shift

        aligned_df = df.copy()
        if col in aligned_df.columns:
            aligned_df[col] = df[col].shift(best_shift)

        aligned[file] = aligned_df
        logs.append({"file": file, "shift": best_shift, "area": best_area})

    return aligned, pd.DataFrame(logs)

def align_by_min_signed_area(cropped_angles_dfs: dict[str, pd.DataFrame],
                                 angle_col: str = "elbow_flex_r",
                                 search: int = 20):

    aligned = {}
    logs = []

    valid_dfs = {}
    for file, df in cropped_angles_dfs.items():
        if angle_col not in df.columns:
            continue
        vals = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(vals).any():
            valid_dfs[file] = df

    if not valid_dfs:
        return {}, pd.DataFrame(columns=["file", "shift", "signed_segment_metric"])

    # 1. Build mean curve (shortest common length)
    min_len = min(len(df) for df in valid_dfs.values())
    curves = np.vstack([
        pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)[:min_len]
        for df in valid_dfs.values()
    ])
    mean_curve = np.nanmean(curves, axis=0)

    # 2. For each throw, search optimal shift
    for file, df in valid_dfs.items():
        curve = pd.to_numeric(df[angle_col], errors="coerce").to_numpy(dtype=float)[:min_len]
        best_shift = 0
        best_metric = np.inf

        for s in range(-search, search + 1):
            if abs(s) >= min_len:
                continue

            # compute overlap slices
            if s < 0:
                c = curve[-s:min_len]
                m = mean_curve[:min_len + s]
            elif s > 0:
                c = curve[:min_len - s]
                m = mean_curve[s:min_len]
            else:
                c = curve
                m = mean_curve

            if len(c) == 0 or len(m) == 0 or len(c) != len(m):
                continue

            # difference curve
            valid = np.isfinite(c) & np.isfinite(m)
            if not np.any(valid):
                continue
            c = c[valid]
            m = m[valid]
            d = c - m

            # find intersections (sign changes)
            sign_changes = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            segments = [0] + sign_changes.tolist() + [len(d)-1]

            # compute signed area segment-wise
            metric = 0.0
            for k in range(len(segments)-1):
                i = segments[k]
                j = segments[k+1]

                segment = d[i:j+1]
                if len(segment) == 0:
                    continue

                segment_sign = np.sign(np.nanmedian(segment))
                segment_area = segment_sign * np.sum(np.abs(segment))
                metric += segment_area

            # minimize absolute signed difference
            if abs(metric) < best_metric:
                best_metric = abs(metric)
                best_shift = s

        # apply shift
        aligned_df = df.copy()
        if angle_col in aligned_df.columns:
            aligned_df[angle_col] = df[angle_col].shift(best_shift)

        aligned[file] = aligned_df
        logs.append({
            "file": file,
            "shift": best_shift,
            "signed_segment_metric": best_metric
        })

    return aligned, pd.DataFrame(logs)

def apply_shift_to_dataset(
    dfs: dict[str, pd.DataFrame],
    log_df: pd.DataFrame,
    fps: int | float | None,
    cols: list[str],
    shift_fps: int | float | None = None,
    target_fps: int | float | None = None,
):
    shifted = {}
    shift_map = dict(zip(log_df["file"], log_df["shift"]))

    # Backward compatible defaults:
    # - existing calls used `fps` only (no scaling).
    # - when cross-FPS scaling is needed, pass shift_fps + target_fps explicitly.
    shift_fps_val = float(shift_fps) if shift_fps else (float(fps) if fps else None)
    target_fps_val = float(target_fps) if target_fps else (float(fps) if fps else None)

    for file, df in dfs.items():
        s = float(shift_map.get(file, 0))
        if shift_fps_val and target_fps_val and shift_fps_val > 0 and target_fps_val > 0:
            s = s * (target_fps_val / shift_fps_val)
        s_int = int(round(s))
        new_df = df.copy()
        for c in cols:
            if c in new_df.columns:
                new_df[c] = new_df[c].shift(s_int)
        shifted[file] = new_df

    return shifted

# new version that doesnt work yet 
# def apply_shift_to_dataset(
#     dfs: dict[str, pd.DataFrame],
#     log_df: pd.DataFrame,
#     fps: int,
#     frame_col: str = "frame",
# ) -> dict[str, pd.DataFrame]:
    
#     shifted_dfs = {}

#     for name, df in dfs.items():
#         if name not in log_df.index:
#             continue
        
#         shift_seconds = log_df.loc[name, "shift_seconds"]
#         shift_frames = int(round(shift_seconds * fps))

#         temp = df.copy()

#         # Create aligned frame number
#         temp["aligned_frame"] = temp[frame_col] + shift_frames

#         # Use aligned_frame as new index for plotting/merging
#         temp = temp.set_index("aligned_frame").sort_index()

#         shifted_dfs[name] = temp
    
#     return shifted_dfs


