"""
Reusable functions for input/output 
"""

import pandas as pd


def load_csv_folder(folder) -> dict[str, pd.DataFrame]:
    """Return dict mapping filename (no extension) → DataFrame."""
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder}")
        return {}
    files = sorted(folder.glob("*.csv"))
    return {f.stem: pd.read_csv(f) for f in files}