#!/usr/bin/env python3
"""
Detect outliers across multiple CSVs (e.g., many `merged.csv` files) and
produce both per‑file and aggregated reports.

This version has built‑in constants for:
- ROOTS: where to search for CSVs
- GLOB_PATTERN: which files to match
- OUTDIR_MODE: whether to write next to each CSV or into athlete directory

Outputs are the same as before.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# ==============================
# Defaults / constants (edit me)
# ==============================
# Where to search when no flags are provided
DEFAULT_ROOTS: list[Path] = [Path.cwd()]
# What filename/glob to look for under the roots
DEFAULT_GLOB: str = "**/merged.csv"
# Where to put outputs when not overridden by flags.
#   - "per_csv": next to each CSV (original behavior)
#   - "athlete": into that CSV's athlete directory (i.e., data/<athlete>/)
#   - absolute path string: write all outputs there
DEFAULT_OUTDIR_MODE: str = "athlete"  # choose from {"per_csv","athlete"} or set to an absolute path string

# ==========================
# Constants (edit these)
# ==========================
ROOTS = [Path("data")]          # directories to search
GLOB_PATTERN = "**/merged.csv"  # pattern under roots
OUTDIR_MODE = "athlete"         # "athlete" | "same" (output in athlete dir or next to file)

# ------------------------------
# Outlier methods
# ------------------------------

def outliers_iqr(x: pd.Series, mult: float = 1.5) -> pd.Series:
    x = x.astype(float)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=x.index)
    lo = q1 - mult * iqr
    hi = q3 + mult * iqr
    return (x < lo) | (x > hi)


def outliers_zscore(x: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(False, index=x.index)
    z = (x - mu) / sd
    return z.abs() > z_thresh


def outliers_mad(x: pd.Series, mad_thresh: float = 3.5) -> pd.Series:
    x = x.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=x.index)
    modified_z = 0.6745 * (x - med) / (mad if mad == 0 else mad)
    return modified_z.abs() > mad_thresh


METHODS = {
    "iqr": outliers_iqr,
    "zscore": outliers_zscore,
    "mad": outliers_mad,
}

# Default method/params
METHOD = "iqr"
PARAMS = {"mult": 1.5}

# Default ID columns
ID_COLS = ["athlete", "session", "throw_id", "file", "frame"]

# ------------------------------
# Utilities
# ------------------------------

def find_files(roots: List[Path], glob_pattern: str) -> List[Path]:
    files: List[Path] = []
    for r in roots:
        r = Path(r)
        files.extend([p for p in r.glob(glob_pattern) if p.is_file()])
    return sorted(set(files))


def athlete_dir_for(csv_path: Path) -> Path:
    """Return the athlete directory for a CSV assumed under data/<athlete>/<session>/... .
    Falls back to the file's parent if structure isn't recognized.
    """
    parts = list(csv_path.resolve().parts)
    try:
        idx = parts.index("data")
        # expect: .../data/<athlete>/<session>/...
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])  # up to and including athlete
    except ValueError:
        pass
    return csv_path.parent

def choose_metric_columns(df: pd.DataFrame, id_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    id_set = set([c for c in id_cols if c in df.columns])
    metrics = [c for c in numeric_cols if c not in id_set]
    return metrics


# ------------------------------
# Core processing per file
# ------------------------------

def process_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    metric_cols = choose_metric_columns(df, ID_COLS)
    if not metric_cols:
        print(f"[WARN] {path}: no numeric metric columns found; skipping.")
        return pd.DataFrame()

    flag_df = pd.DataFrame(index=df.index)
    for col in metric_cols:
        x = df[col]
        non_na = x.dropna()
        if len(non_na) < 5 or non_na.nunique() <= 1:
            flag_df[col] = False
            continue
        mask = METHODS[METHOD](x, **PARAMS)
        mask = mask.fillna(False)
        flag_df[col] = mask

    any_flag = flag_df.any(axis=1)
    id_subset = [c for c in ID_COLS if c in df.columns]

    rows_with_flags = df.loc[any_flag, id_subset + metric_cols].copy()
    rows_with_flags["outlier_metrics"] = rows_with_flags.apply(
        lambda row: ";".join([c for c in metric_cols if flag_df.loc[row.name, c]]), axis=1
    )

    per_file_summary = (
        flag_df[metric_cols]
        .sum()
        .rename("n_outliers")
        .to_frame()
        .assign(n=len(df))
        .assign(pct=lambda t: (t["n_outliers"] / t["n"]).replace([np.inf, -np.inf], np.nan))
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    per_file_summary.insert(0, "file", str(path))

    # Decide output dir
    if OUTDIR_MODE == "athlete":
        athlete = df["athlete"].iloc[0] if "athlete" in df.columns else "default"
        tgt_dir = Path("data") / athlete / "outliers"
    else:
        tgt_dir = path.parent

    tgt_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    wide_flags = df[id_subset].copy() if id_subset else pd.DataFrame(index=df.index)
    for col in metric_cols:
        wide_flags[f"{col}__is_outlier"] = flag_df[col]

    (tgt_dir / f"{stem}_outlier_flags.csv").write_text(wide_flags.to_csv(index=False))
    (tgt_dir / f"{stem}_outlier_rows.csv").write_text(rows_with_flags.to_csv(index=False))
    (tgt_dir / f"{stem}_outlier_summary.csv").write_text(per_file_summary.to_csv(index=False))

    print(f"[OK] {path}: {any_flag.sum()} rows with ≥1 outlier across {len(metric_cols)} metrics.")
    return per_file_summary


# ------------------------------
# Main
# ------------------------------

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute outliers for each metric column across multiple CSV files.")

    # Flags are now OPTIONAL; we keep them for flexibility
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--files", nargs="+", type=Path, help="Explicit CSV files to process")
    src.add_argument("--roots", nargs="+", type=Path, help="One or more directories to search")

    p.add_argument("--glob", default=None, help="Glob to match under roots (default from constant)")
    p.add_argument("--outdir", type=Path, default=None, help="Optional directory for all outputs (overrides constants)")

    p.add_argument("--method", choices=METHODS.keys(), default="iqr", help="Outlier method (default: iqr)")
    p.add_argument("--iqr-mult", type=float, default=1.5, help="IQR multiplier for Tukey fences (default: 1.5)")
    p.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold (default: 3.0)")
    p.add_argument("--mad-thresh", type=float, default=3.5, help="MAD threshold (default: 3.5)")

    p.add_argument("--id-cols", nargs="*", default=["athlete", "session", "throw_id", "file", "frame"],
                   help="Columns to carry through unchanged and NOT treat as metrics (default: common IDs)")

    args = p.parse_args(argv)

    # Resolve files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        roots = args.roots if args.roots else DEFAULT_ROOTS
        glob_pat = args.glob if args.glob else DEFAULT_GLOB
        files = find_files(roots, glob_pat)

    if not files:
        print("[ERROR] No input CSVs found.")
        return 2

    # Determine output strategy
    # CLI --outdir takes precedence; otherwise constant controls behavior
    explicit_outdir: Path | None = args.outdir
    outdir_mode = None
    constant_outdir: Path | None = None
    if explicit_outdir is None:
        if DEFAULT_OUTDIR_MODE in ("per_csv", "athlete"):
            outdir_mode = DEFAULT_OUTDIR_MODE
        else:
            # treat as absolute path string
            constant_outdir = Path(DEFAULT_OUTDIR_MODE).expanduser().resolve()
            constant_outdir.mkdir(parents=True, exist_ok=True)

    # Validate and prepare method params
    if args.method == "iqr":
        params = {"mult": args.iqr_mult}
    elif args.method == "zscore":
        params = {"z_thresh": args.z_thresh}
    elif args.method == "mad":
        params = {"mad_thresh": args.mad_thresh}
    else:
        print(f"[ERROR] Unknown method: {args.method}")
        return 2

    agg_rows = []
    for f in files:
        try:
            # Decide target directory for this file
            if explicit_outdir is not None:
                tgt = explicit_outdir
            elif outdir_mode == "per_csv":
                tgt = None  # process_one_csv will use file.parent
            elif outdir_mode == "athlete":
                tgt = athlete_dir_for(f)
            else:
                tgt = constant_outdir

            per = process_one_csv(
                path=f,
                method=args.method,
                params=params,
                id_cols=args.id_cols,
                outdir=tgt,
            )
            if not per.empty:
                agg_rows.append(per)
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}")

    if not agg_rows:
        print("[WARN] No summaries produced.")
        return 0

    agg = pd.concat(agg_rows, ignore_index=True)

    # Decide where to write aggregate summary
    if explicit_outdir is not None:
        agg_dir = explicit_outdir
    elif outdir_mode == "per_csv":
        agg_dir = files[0].parent
    elif outdir_mode == "athlete":
        agg_dir = athlete_dir_for(files[0])
    else:
        agg_dir = constant_outdir if constant_outdir else files[0].parent

    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / "aggregate_outlier_summary.csv"
    agg.to_csv(agg_path, index=False)

    # Pretty print top 10 most problematic metrics across all files
    piv = (
        agg.groupby("metric")["n_outliers"].sum().sort_values(ascending=False).head(10)
    )
    print("Top metrics by total outliers across all files:")
    for metric, n in piv.items():
        print(f"  {metric:32s}  {int(n)}")

    print(f"[OK] Wrote aggregate summary to: {agg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
