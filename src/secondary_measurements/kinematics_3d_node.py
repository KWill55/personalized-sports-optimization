from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/secondary_measurements/kinematics_3d_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from secondary_measurements.kinematics_3d_lib import run_kinematics_3d_pipeline
from utils.io_utils import load_config


class Kinematics3dNode:
    """Orchestrates 3D kinematics from already-computed 3D keypoints."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_kinematics_3d_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = Kinematics3dNode()
    result = node.run()
    print(f"3D kinematics pipeline complete: {result}")

