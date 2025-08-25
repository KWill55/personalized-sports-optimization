"""
prepare_phase_dataset.py

Purpose:
    Prepare ML-ready X, y from phase-level features + outcomes.
    - Merges on 'file' (or 'shot' -> renamed to 'file')
    - Robust label parsing (made/miss or 1/0)
    - Optional class balancing (downsample or upsample)
    - Missing-value handling (drop or median impute)
    - Drops non-numeric columns (e.g., file) before saving X
"""

from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.utils import resample
import sys


project_cfg_path = Path(__file__).resolve().parents[3] / "project_config.yaml"
project_cfg = yaml.safe_load(open(project_cfg_path, "r"))

ATHLETE = project_cfg["athlete"]
SESSION = project_cfg["session"]

DATASETS_DIR = Path(project_cfg["paths"]["datasets"].format(
    athlete=ATHLETE,
    session=SESSION
))

OUTCOMES_FILE = DATASETS_DIR / "outcomes.csv"

# ---------- Load feature config (to get the variant label) ----------
feature_cfg_path = Path(__file__).resolve().parents[3] / "feature_config.yaml"
feature_cfg = yaml.safe_load(open(feature_cfg_path, "r"))

MODEL_TYPE = str(feature_cfg.get("model_type", "unknown_model"))  # e.g., "summary_stats" or "time_series"
FEATURE_VERSION = str(feature_cfg.get("feature_version", "unknown_version"))  # e.g., "ang_vel_acc"

FEATURES_DIR = (
    Path(project_cfg["paths"]["features"].format(
        athlete=project_cfg["athlete"],
        session=project_cfg["session"],
    )))

FEATURES_FILE = FEATURES_DIR / MODEL_TYPE/ FEATURE_VERSION / f"phase_features_{FEATURE_VERSION}.csv"
VARIANT_DIR = FEATURES_DIR / MODEL_TYPE / FEATURE_VERSION
VARIANT_DIR.mkdir(parents=True, exist_ok=True)

X_OUT_PATH = VARIANT_DIR / "X.csv"
Y_OUT_PATH = VARIANT_DIR / "y.csv"
MERGED_OUT_PATH = VARIANT_DIR / "merged.csv"

# Label column name in outcomes file (set to None to auto-detect)
LABEL_COLUMN = "outcome"   # or None

# Processing options
BALANCE = "none"           # "none", "downsample", "upsample"
IMPUTE = "median"          # "median", "drop"


# ========================================
# FUNCTIONS
# ========================================

def normalize_label(val):
    """Map various label formats to 0/1."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        v = int(val)
        if v in (0, 1):
            return v
    s = str(val).strip().lower()
    if s in ("1", "make", "made", "hit", "true", "t", "yes", "y", "m"):
        return 1
    if s in ("0", "miss", "missed", "false", "f", "no", "n"):
        return 0
    return None

# ========================================
# MAIN
# ========================================

def main():
    features = pd.read_csv(FEATURES_FILE)
    outcomes = pd.read_csv(OUTCOMES_FILE)

    # Align id column name
    if "file" not in outcomes.columns:
        if "shot" in outcomes.columns:
            outcomes = outcomes.rename(columns={"shot": "file"})
        else:
            raise ValueError("Outcomes CSV must have 'file' or 'shot'.")

    # Decide label column
    label_col = LABEL_COLUMN
    if label_col is None:
        candidates = [c for c in outcomes.columns if c.lower() not in ("file",)]
        if not candidates:
            raise ValueError("Could not infer label column; set LABEL_COLUMN.")
        label_col = candidates[0]

    # Normalize labels
    outcomes["_label"] = outcomes[label_col].apply(normalize_label)
    outcomes = outcomes.dropna(subset=["_label"]).copy()
    outcomes["_label"] = outcomes["_label"].astype(int)

    # Merge
    merged = features.merge(outcomes[["file", "_label"]], on="file", how="inner")

    # Handle missing values
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if IMPUTE == "median":
        for col in numeric_cols:
            if merged[col].isna().any():
                merged[col] = merged[col].fillna(merged[col].median())
    else:  # drop
        merged = merged.dropna()

    # Optional balancing
    if BALANCE != "none":
        made = merged[merged["_label"] == 1]
        miss = merged[merged["_label"] == 0]
        if BALANCE == "downsample":
            if len(made) and len(miss):
                if len(made) < len(miss):
                    miss_bal = resample(miss, replace=False, n_samples=len(made), random_state=42)
                    merged = pd.concat([made, miss_bal], axis=0)
                else:
                    made_bal = resample(made, replace=False, n_samples=len(miss), random_state=42)
                    merged = pd.concat([made_bal, miss], axis=0)
        elif BALANCE == "upsample":
            if len(made) and len(miss):
                if len(made) < len(miss):
                    made_bal = resample(made, replace=True, n_samples=len(miss), random_state=42)
                    merged = pd.concat([made_bal, miss], axis=0)
                else:
                    miss_bal = resample(miss, replace=True, n_samples=len(made), random_state=42)
                    merged = pd.concat([made, miss_bal], axis=0)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Recompute numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if "_label" not in numeric_cols:
        raise RuntimeError("Internal error: label column missing after processing.")

    X_cols = [c for c in numeric_cols if c != "_label"]
    X = merged[X_cols].copy()
    y = merged["_label"].copy()

    # Save
    Path(X_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(Y_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MERGED_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    X.to_csv(X_OUT_PATH, index=False)
    y.to_csv(Y_OUT_PATH, index=False, header=True)
    merged.to_csv(MERGED_OUT_PATH, index=False)

    print("✅ Prepared dataset")
    print(f" - Features file: {FEATURES_FILE}")
    print(f" - Outcomes file: {OUTCOMES_FILE}")
    print(f" - Saved to: {VARIANT_DIR}")
    print(f" - Balance: {BALANCE}, Impute: {IMPUTE}")
    print(f" - Rows: {len(merged)}, X shape: {X.shape}")
    print(f" - y distribution:\n{y.value_counts()}")

if __name__ == "__main__":
    main()
