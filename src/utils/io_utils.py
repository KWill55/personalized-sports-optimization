"""
Reusable functions for input/output 
"""

import pandas as pd
import yaml
from pathlib import Path
import re


def load_csv_folder(folder: Path) -> dict[str, pd.DataFrame]:
    """Return dict mapping filename (no extension) → DataFrame."""
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder}")
        return {}
    files = sorted(folder.glob("*.csv"))
    return {f.stem: pd.read_csv(f) for f in files}

def load_config(config_path: str | Path) -> dict:
    """Load the YAML config as a dictionary."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    