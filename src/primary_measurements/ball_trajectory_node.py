from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/primary_measurements/ball_trajectory_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from primary_measurements.ball_trajectory_lib import run_ball_trajectory_pipeline
from utils.io_utils import load_config


class BallTrajectoryNode:
    """Orchestrates raw ball trajectory extraction."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_ball_trajectory_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = BallTrajectoryNode()
    result = node.run()
    print(f"Ball trajectory extraction complete: {result}")


# Backward-compatible alias for older imports.
BallDetectionNode = BallTrajectoryNode
