from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/player_tracking/pose_3d_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from player_tracking.pose_3d_lib import run_pose_3d_pipeline
from utils.io_utils import load_config


class Pose3dNode:
    """Orchestrates 3D pose reconstruction and kinematics outputs."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_pose_3d_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = Pose3dNode()
    result = node.run()
    print(f"3D pose pipeline complete: {result}")
