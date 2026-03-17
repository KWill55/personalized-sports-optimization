from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/preprocessing/alignment_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from preprocessing.alignment_lib import run_alignment_pipeline, run_alignment_viewer
from utils.io_utils import load_config


class AlignmentNode:
    """Orchestrates alignment preprocessing."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_alignment_pipeline(self.cfg)

    def view(self) -> dict[str, Any]:
        return run_alignment_viewer(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = AlignmentNode()
    result = node.run()
    print(f"Alignment pipeline complete: {result}")
