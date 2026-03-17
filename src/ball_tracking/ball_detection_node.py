from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Support direct execution: python3 src/ball_tracking/ball_detection_node.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ball_tracking.ball_detection_lib import run_ball_detection_pipeline
from utils.io_utils import load_config


class BallDetectionNode:
    """Orchestrates ball detection/tracking preprocessing."""

    def __init__(self, cfg: dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> dict[str, Any]:
        return run_ball_detection_pipeline(self.cfg)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    node = BallDetectionNode()
    result = node.run()
    print(f"Ball detection pipeline complete: {result}")
