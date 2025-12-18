
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
        dip_frame = df[angle_col].idxmin()
        dips[file] = dip_frame

    median_dip = int(np.median(list(dips.values())))

    # 2. Align all throws by shifting dip to median_dip
    for file, df in cropped_angles_dfs.items():
        shift = median_dip - dips[file]
        aligned_df = df.copy()
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

    median_release = int(np.median(list(releases.values())))

    # 2. Align all throws by shifting release to median release
    for file, df in cropped_angles_dfs.items():
        if file not in releases:
            continue
        shift = median_release - releases[file]
        aligned_df = df.copy()
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

    # 1. Build mean curve (shortest common length)
    min_len = min(len(df) for df in cropped_dfs.values())
    curves = np.vstack([
        df[col].values[:min_len]
        for df in cropped_dfs.values()
    ])
    mean_curve = np.nanmean(curves, axis=0)

    # 2. For each throw, find shift with minimum unsigned area
    for file, df in cropped_dfs.items():
        curve = df[col].values[:min_len]
        best_shift = 0
        best_area = np.inf

        # find overlapping region between the two curves 
        for shift in range(-search, search + 1):
            if shift < 0: # curve is to the left of median curve 
                c = curve[-shift:min_len]
                m = mean_curve[:min_len + shift]
            elif shift > 0: # curve is to the right of median curve 
                c = curve[:min_len - shift]
                m = mean_curve[shift:min_len]
            else: # no shift 
                c = curve
                m = mean_curve

            area = np.sum(np.abs(m - c))
            if area < best_area:
                best_area = area
                best_shift = shift

        aligned_df = df.copy()
        aligned_df[col] = df[col].shift(best_shift)

        aligned[file] = aligned_df
        logs.append({"file": file, "shift": best_shift, "area": best_area})

    return aligned, pd.DataFrame(logs)

def align_by_min_signed_area(cropped_angles_dfs: dict[str, pd.DataFrame],
                                 angle_col: str = "elbow_flex_r",
                                 search: int = 20):

    aligned = {}
    logs = []

    # 1. Build mean curve (shortest common length)
    min_len = min(len(df) for df in cropped_angles_dfs.values())
    curves = np.vstack([
        df[angle_col].values[:min_len]
        for df in cropped_angles_dfs.values()
    ])
    mean_curve = np.nanmean(curves, axis=0)

    # 2. For each throw, search optimal shift
    for file, df in cropped_angles_dfs.items():
        curve = df[angle_col].values[:min_len]
        best_shift = 0
        best_metric = np.inf

        for s in range(-search, search + 1):

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

            # difference curve
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
    fps: int,
    cols: list[str]
):
    shifted = {}
    shift_map = dict(zip(log_df["file"], log_df["shift"]))

    for file, df in dfs.items():
        s = shift_map.get(file, 0)
        new_df = df.copy()
        for c in cols:
            new_df[c] = new_df[c].shift(s)
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





