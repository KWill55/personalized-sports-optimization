"""
Reusable functions for input/output 
"""

import pandas as pd
import yaml
from pathlib import Path
import re
from tkinter import Tk, filedialog, Label, Button
import os
from pathlib import Path
import sys

# Find project root (directory containing project_config.yaml)
def find_project_root():
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "project_config.yaml").exists():
            return parent
    raise FileNotFoundError("project_config.yaml not found")

PROJECT_ROOT = find_project_root()

# Add src/ to import path
sys.path.append(str(PROJECT_ROOT / "src"))


def pick_folder(initial_dir="."):
    """Open a folder picker and return the selected folder as a Path."""
    initial_dir = Path(initial_dir).resolve()
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    folder = filedialog.askdirectory(initialdir=str(initial_dir), title="Select Folder")
    root.destroy()

    if not folder:
        return None  # User canceled

    return Path(folder)


def load_csv_folder(folder: Path) -> dict[str, pd.DataFrame]:
    """Return dict mapping filename (no extension) → DataFrame."""
    if not folder.exists():
        print(f"⚠️ Folder not found: {folder}")
        return {}
    files = sorted(folder.glob("*.csv"))
    return {f.stem: pd.read_csv(f) for f in files}

    
# def load_config(config_filename: str = "project_config.yaml") -> dict:
#     """
#     Load the project YAML config from the repo root.
#     Returns a dictionary with all config values.
#     """
#     base_dir = Path(__file__).resolve().parents[1]
#     config_path = base_dir / config_filename

#     with open(config_path, "r") as f:
#         cfg = yaml.safe_load(f)

#     return cfg

def load_config(config_filename: str = "project_config.yaml") -> dict:
    """
    Load the YAML config file from anywhere in the repository.
    Automatically searches parent directories until the config is found.
    """
    # Start at the directory of the current file
    current = Path(__file__).resolve()

    # Walk upward until we find the config file
    for parent in [current, *current.parents]:
        candidate = parent / config_filename
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Could not find '{config_filename}' in this folder or any parent folder."
    )
