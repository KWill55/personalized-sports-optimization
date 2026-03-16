from __future__ import annotations

from typing import Any, Dict

from ball_tracking.ball_detection_lib import run_ball_detection_pipeline
from utils.io_utils import load_config


class BallDetectionNode:
    """
    Future BallDetection Node -- orchestrates ball tracking process.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None, config_filename: str = "project_config.yaml") -> None:
        self.cfg = cfg if cfg is not None else load_config(config_filename)

    def run(self) -> None:
        run_ball_detection_pipeline()

    def close(self) -> None:
        pass


if __name__ == "__main__":
    BallDetectionNode().run()
